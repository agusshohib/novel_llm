use std::{cmp, path::Path};

use candle_core::{D, DType, Device, IndexOp, ModuleT, Result, Tensor};
use candle_nn::{VarBuilder, VarMap, ops::softmax};

use crate::{
    config::Config,
    gpt::GPT,
    gpt_model::GPTModel,
    tokenizer::{Decoder, Encoder, Tokenizer},
};

mod config;
mod data_loader;
mod feed_forward;
mod ff_layer;
mod gelu;
mod gpt;
mod gpt_data_batcher;
mod gpt_data_loader;
mod gpt_dataset_iter;
mod gpt_dataset_v1;
mod gpt_model;
mod layer_norm;
mod multi_head_attention;
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

    // Evaluate
    let start_context = &[
        660.3, 660.5, 660.7, 660.6, 660.9, 661.5, 662.2, 662.5, 661.9, 661.0, 660.5, 659.8, 658.9,
        658.5, 659.0, 659.3, 658.5, 658.4, 659.0, 659.4, 659.6, 659.5, 659.0, 659.2, 659.5, 659.2,
        659.1, 659.6, 659.6, 659.6, 659.8, 660.0, 660.2, 659.8, 659.8, 660.0, 659.9, 660.3, 660.9,
        660.8, 660.6, 660.8, 661.0, 660.8, 660.5, 659.9, 659.7, 660.5, 660.4, 660.4, 660.9, 661.1,
        661.2, 661.3, 661.1, 660.5, 660.7, 660.9, 660.6, 660.9, 661.4, 660.9, 659.9, 659.8, 659.1,
        658.6, 658.3, 658.0, 658.2, 657.8, 658.4, 658.9, 658.6, 659.8, 660.9, 660.8, 660.8, 661.2,
        661.6, 661.6, 661.8, 661.6, 661.2, 660.8, 660.9, 660.7, 660.4, 660.2, 659.4, 658.5, 658.2,
        658.8, 659.7, 659.3, 658.9, 659.4, 659.9, 660.2, 660.1, 660.2, 660.6, 660.5, 660.3, 660.6,
        661.2, 661.9, 661.5, 660.7, 660.7, 661.2, 661.6, 661.7, 661.8, 661.5, 661.5, 661.5, 661.5,
        661.5, 661.5, 661.8, 662.0, 662.2, 662.3, 663.1, 664.1, 664.0, 663.8, 663.5, 663.4, 663.2,
        662.9, 662.7, 662.5, 662.6, 662.2, 661.8, 661.4, 660.5, 660.2, 660.3, 660.2, 660.1, 659.8,
        659.8, 660.4, 660.8, 661.0, 661.3, 661.5, 661.4, 661.7, 662.3, 662.4, 662.6, 662.6, 662.4,
        662.3, 662.1, 662.2, 661.7, 661.6, 661.8, 662.3, 662.7, 662.4, 662.7, 663.0, 662.9, 662.9,
        662.7, 662.4, 662.3, 662.6, 663.2, 664.5, 665.1, 664.9, 665.2, 666.4, 668.0, 669.0, 669.5,
        669.0, 668.6, 667.6, 667.1, 667.7, 667.4, 667.4, 668.5, 669.5, 669.5, 669.9, 669.3, 669.0,
        669.3, 669.0, 669.5, 669.7, 669.2, 669.5, 669.3, 668.5, 668.5, 669.1, 669.9, 670.4, 670.3,
        670.3, 670.9, 670.9, 670.4, 670.3, 670.4, 670.9, 672.2, 672.6, 672.4, 672.4, 672.3, 672.7,
        672.7, 672.4, 671.3, 670.3, 671.2, 671.7, 671.6, 671.7, 671.9, 672.6, 673.0, 672.5, 672.4,
        671.9, 671.4, 671.4, 670.7, 670.0, 669.9, 670.1, 670.5, 670.6, 670.1, 670.0, 669.7,
    ];
    let tokenizer = Tokenizer::new("data/vocab.txt");
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
