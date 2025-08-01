use std::{
    cmp,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use candle_core::{D, DType, Device, IndexOp, ModuleT, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, ops::softmax};
use ndarray::linspace;
use plotly::{Layout, Plot, Scatter, common::Mode, layout::Axis};

use crate::{config::Config, data_loader::DataLoader, gpt::GPT, gpt_data_loader::GPTDataLoader, gpt_model::GPTModel, gpt_dataset_v1::GPTDatasetV1, tokenizer::{Decoder, Encoder, Tokenizer}};

mod config;
mod data_loader;
mod feed_forward;
mod ff_layer;
mod gelu;
mod gpt;
mod gpt_data_batcher;
mod gpt_data_loader;
mod gpt_dataset_iter;
mod gpt_model;
mod layer_norm;
mod multi_head_attention;
mod gpt_dataset_v1;
mod sequential_transformers;
mod tokenizer;
mod transformer_block;

fn main() {
    // Create Variable Manager
    let dev = Device::cuda_if_available(0).unwrap();
    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

    // Prepare Model
    let config = Config::bnbusdt();
    let model = GPTModel::new(config, vb.pp("model")).unwrap();
    if Path::new("load.safetensors").exists() {
        vm.load("load.safetensors").unwrap();
    }

    // Training Optimizer
    let optimizer = AdamW::new(
        vm.all_vars(),
        ParamsAdamW {
            lr: 0.0004,
            weight_decay: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    // Train Model
    let tokenizer = Tokenizer::new("data/vocab.txt");
    let (eval_freq, eval_iter, num_epochs) = (5_usize, 5_usize, 20_usize);
    let (train_loader, val_loader) = get_data_loader(&tokenizer, &config, false).unwrap();
    let start_context = &[524.5, 524.2, 524.5, 523.6, 522.7, 524.2, 526.5, 526.1];
    let (train_losses, val_losses, tokens_seen) = train_model_simple(
        &model,
        &train_loader,
        &val_loader,
        optimizer,
        &dev,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        &tokenizer,
        None,
    )
    .unwrap();

    // Evaluate
    let token_ids = generate_text_simple(
        &model,
        text_to_token_ids(start_context, &tokenizer, &dev).unwrap(),
        25,
        config.context_length,
    )
    .unwrap();
    println!(
        "Output text:\n{:?}",
        token_ids_to_text(token_ids, &tokenizer)
    );

    // Save Weight
    println!("Saving weights to `./save.safetensors`");
    vm.save("save.safetensors").unwrap();

    // Save Loss Plot
    println!("Saving loss plot to `./loss_plot.html`");
    let epochs_seen = Vec::from_iter(linspace(0_f32, num_epochs as f32, train_losses.len()));
    let tokens_seen = tokens_seen
        .into_iter()
        .map(|el| el as f32)
        .collect::<Vec<_>>();
    let save_path = Path::new("loss_plot.html").to_path_buf();
    plot_losses(
        epochs_seen,
        tokens_seen,
        train_losses,
        val_losses,
        save_path,
    )
    .unwrap();
}

pub fn calc_loss_batch<M: GPT + ModuleT>(
    input_batch: &Tensor,
    target_batch: &Tensor,
    model: &M,
    device: &Device,
    train: bool,               // added to compute loss for train or otherwise
    ignore_index: Option<i64>, // introduced for ch07 instruction finetuning
) -> Result<Tensor> {
    let input_batch = input_batch.to_device(device)?;
    let target_batch = target_batch.to_device(device)?;
    let logits = model.forward_t(&input_batch, train)?;

    // flatten
    let logits_flat = logits.flatten(0, 1)?;
    let targets_flat = target_batch.flatten_all()?;

    // handle ignore_index if set to a value — in such cases, we expect targets
    // to be Tensor of DType::I64
    let (logits_flat, targets_flat) = if let Some(ignore_val) = ignore_index {
        // get indices to keep
        let keep = targets_flat
            .to_vec1::<i64>()? // has to be i64 to include ignore_index of -100
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != ignore_val)
            .map(|(ix, _)| ix as u32)
            .collect::<Vec<_>>();
        let keep = Tensor::new(&keep[..], device)?;

        (
            logits_flat.index_select(&keep, 0)?,
            targets_flat.index_select(&keep, 0)?,
        )
    } else {
        (logits_flat, targets_flat)
    };

    let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
    Ok(loss)
}

pub fn calc_loss_loader<
    M: GPT + ModuleT,
    L: DataLoader<Batcher = impl Iterator<Item = Result<(Tensor, Tensor)>>>,
>(
    data_loader: &L,
    model: &M,
    device: &Device,
    num_batches: Option<usize>,
    ignore_index: Option<i64>, // introduced for ch07 instruction finetuning
) -> Result<f32> {
    let mut total_loss = 0_f32;
    let mut count = 0_usize;

    let mut data_batcher = data_loader.batcher();
    match num_batches {
        None => {
            while let Some(Ok((input_batch, target_batch))) = data_batcher.next() {
                let loss = calc_loss_batch(
                    &input_batch,
                    &target_batch,
                    model,
                    device,
                    false,
                    ignore_index,
                )?;
                total_loss += loss.to_scalar::<f32>()?;
                count += 1_usize;
            }
            Ok(total_loss / count as f32)
        }
        Some(n) => {
            while let Some(Ok((input_batch, target_batch))) = data_batcher.next() {
                let loss = calc_loss_batch(
                    &input_batch,
                    &target_batch,
                    model,
                    device,
                    false,
                    ignore_index,
                )?;
                total_loss += loss.to_scalar::<f32>()?;
                count += 1_usize;
                if count >= n {
                    break;
                }
            }
            Ok(total_loss / std::cmp::min(n, count) as f32)
        }
    }
}

pub fn create_dataloader_v1(
    tokenizer: &Tokenizer,
    txt: &[f64],
    batch_size: usize,
    max_length: usize,
    stride: usize,
    shuffle: bool,
    drop_last: bool,
) -> GPTDataLoader {
    let dataset = GPTDatasetV1::new(txt, tokenizer, max_length, stride);
    GPTDataLoader::new(dataset, batch_size, shuffle, drop_last)
}

pub fn evaluate_model<
    M: GPT + ModuleT,
    L: DataLoader<Batcher = impl Iterator<Item = Result<(Tensor, Tensor)>>>,
>(
    model: &M,
    train_loader: &L,
    val_loader: &L,
    device: &Device,
    eval_iter: usize,
    ignore_index: Option<i64>, // introduced for ch07 instruction finetuning
) -> Result<(f32, f32)> {
    let train_loss = calc_loss_loader(train_loader, model, device, Some(eval_iter), ignore_index)?;
    let val_loss = calc_loss_loader(val_loader, model, device, Some(eval_iter), ignore_index)?;
    Ok((train_loss, val_loss))
}

pub fn generate_and_print_sample<M: GPT + ModuleT>(
    model: &M,
    tokenizer: &Tokenizer,
    device: &Device,
    start_context: &[f64],
) -> Result<()> {
    let context_size = model.context_size();
    let encoded = text_to_token_ids(start_context, tokenizer, device)?;
    let token_ids = generate_text_simple(model, encoded, 50, context_size)?;
    let decoded_text = token_ids_to_text(token_ids, tokenizer).unwrap();
    println!("{decoded_text:?}");
    Ok(())
}

pub fn generate_text_simple<M: GPT + ModuleT>(
    model: &M,
    idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> Result<Tensor> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let (_b, seq_len) = idx.dims2()?;
        let start_token_index = cmp::max(0isize, seq_len as isize - context_size as isize) as usize;
        let idx_cond = idx.i((.., start_token_index..seq_len))?;
        let logits = model.forward_t(&idx_cond, false)?;
        let (_b, c, _vocab_size) = logits.dims3()?;
        let logits = logits.i((.., c - 1, ..))?;
        let probas = softmax(&logits, 1)?;
        let idx_next = probas.argmax_keepdim(D::Minus1)?;
        idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?;
    }
    Ok(idx)
}

pub fn get_data_loader(
    tokenizer: &Tokenizer,
    config: &Config,
    verbose: bool,
) -> anyhow::Result<(GPTDataLoader, GPTDataLoader)> {
    let file = File::open("data/data.txt")?;
    let reader = BufReader::new(file);

    let data: Vec<f64> = reader
        .lines()
        .filter_map(std::io::Result::ok)
        .filter_map(|line| line.parse().ok())
        .collect();

    let tokens = tokenizer.encode(&data);
    if verbose {
        println!("Data: {:?}", data.len());
        println!("Tokens: {:?}", tokens.len());
    }

    // establish train and val data
    let train_ratio = 0.90_f32;
    let split_idx = (train_ratio * data.len() as f32) as usize;
    let train_data = &data[..split_idx];
    let val_data = &data[split_idx..];

    let batch_size = 2_usize;
    let max_length = config.context_length;
    let stride = config.context_length;

    let train_loader = create_dataloader_v1(
        &tokenizer, train_data, batch_size, max_length, stride, true, true,
    );
    let val_loader = create_dataloader_v1(
        &tokenizer, val_data, batch_size, max_length, stride, false, false,
    );

    Ok((train_loader, val_loader))
}

pub fn plot_losses<P: AsRef<Path>>(
    epochs_seen: Vec<f32>,
    tokens_seen: Vec<f32>,
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
    save_path: P,
) -> Result<()> {
    let trace1 = Scatter::new(epochs_seen.clone(), train_losses.clone())
        .show_legend(false)
        .opacity(0_f64)
        .mode(Mode::LinesMarkers);
    let trace2 = Scatter::new(epochs_seen, val_losses.clone())
        .show_legend(false)
        .opacity(0_f64)
        .mode(Mode::LinesMarkers);
    let trace3 = Scatter::new(tokens_seen.clone(), train_losses)
        .name("Training loss")
        .x_axis("x2")
        .mode(Mode::LinesMarkers);
    let trace4 = Scatter::new(tokens_seen, val_losses)
        .name("Validation loss")
        .x_axis("x2")
        .mode(Mode::LinesMarkers);

    let layout = Layout::new()
        .x_axis(Axis::new().title("Epochs"))
        .x_axis2(
            Axis::new()
                .title("Tokens Seen")
                .side(plotly::common::AxisSide::Top),
        )
        .y_axis(Axis::new().title("Loss"));
    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.add_trace(trace3);
    plot.add_trace(trace4);
    plot.set_layout(layout);
    plot.write_html(save_path);
    Ok(())
}

pub fn text_to_token_ids(text: &[f64], tokenizer: &Tokenizer, dev: &Device) -> Result<Tensor> {
    let encoded = tokenizer.encode(text);
    let num_tokens = encoded.len();
    // encoded tensor
    Tensor::from_vec(encoded, (1_usize, num_tokens), dev)
}

pub fn token_ids_to_text(token_ids: Tensor, tokenizer: &Tokenizer) -> anyhow::Result<Vec<f64>> {
    let flat = token_ids.squeeze(0)?;
    Ok(tokenizer.decode(flat.to_vec1::<u32>()?))
}

#[allow(clippy::too_many_arguments)]
pub fn train_model_simple<
    T,
    L: DataLoader<Batcher = impl Iterator<Item = Result<(Tensor, Tensor)>>>,
    M: GPT + ModuleT,
>(
    model: &M,
    train_loader: &L,
    val_loader: &L,
    mut optimizer: T,
    device: &Device,
    num_epochs: usize,
    eval_freq: usize,
    eval_iter: usize,
    start_context: &[f64],
    tokenizer: &Tokenizer,
    ignore_index: Option<i64>, // introduced for ch07 instruction finetuning
) -> Result<(Vec<f32>, Vec<f32>, Vec<usize>)>
where
    T: Optimizer,
{
    // retvals
    let mut train_losses: Vec<f32> = vec![];
    let mut val_losses: Vec<f32> = vec![];
    let mut track_tokens_seen: Vec<usize> = vec![];

    let (mut tokens_seen, mut global_step) = (0usize, 0_usize);

    for epoch in 0..num_epochs {
        let mut train_batcher = train_loader.batcher();
        while let Some(Ok((input_batch, target_batch))) = train_batcher.next() {
            let loss = calc_loss_batch(
                &input_batch,
                &target_batch,
                model,
                device,
                true,
                ignore_index,
            )?;
            optimizer.backward_step(&loss)?;
            tokens_seen += input_batch.elem_count();

            if global_step % eval_freq == 0 {
                let (train_loss, val_loss) = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                    ignore_index,
                )?;
                train_losses.push(train_loss);
                val_losses.push(val_loss);
                track_tokens_seen.push(tokens_seen);
                println!(
                    "Ep {} (Step {}) \
                    Train loss: {}, \
                    Val loss: {}",
                    epoch + 1,
                    global_step,
                    train_loss,
                    val_loss
                );
            }
            global_step += 1;
        }
        generate_and_print_sample(model, tokenizer, device, start_context)?
    }

    Ok((train_losses, val_losses, track_tokens_seen))
}
