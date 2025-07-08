use candle_core::{Module, Result, Tensor};
use candle_nn::{Dropout, Embedding, Linear, ModuleT, VarBuilder, embedding, linear_b};

use crate::{
    config::Config,
    gpt::GPT,
    layer_norm::LayerNorm,
    sequential_transformers::{SequentialTransformers, seqtransformers},
    transformer_block::TransformerBlock,
};

#[derive(Debug, Clone)]
pub struct GPTModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    trf_blocks: SequentialTransformers,
    final_norm: LayerNorm,
    out_head: Linear,
}

impl GPTModel {
    pub fn new(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        let tok_emb = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("tok_emb"))?;
        let pos_emb = embedding(cfg.context_length, cfg.emb_dim, vb.pp("pos_emb"))?;
        let drop_emb = Dropout::new(cfg.drop_rate);
        let mut trf_blocks = seqtransformers();
        for ix in 0..cfg.n_layers {
            trf_blocks =
                trf_blocks.add(TransformerBlock::new(cfg, vb.pp(format!("trf.{}", ix))).unwrap());
        }
        let final_norm = LayerNorm::new(cfg.emb_dim, vb.pp("final_norm"))?;
        let out_head = linear_b(cfg.emb_dim, cfg.vocab_size, false, vb.pp("out_head"))?;
        Ok(Self {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        })
    }
}

impl GPT for GPTModel {
    fn context_size(&self) -> usize {
        self.pos_emb.embeddings().dims()[0]
    }
}

impl ModuleT for GPTModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (_batch_size, seq_len) = xs.dims2()?;
        let tok_embeds = self.tok_emb.forward(xs)?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_embeds = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let mut x = tok_embeds.broadcast_add(&pos_embeds)?;
        x = self.drop_emb.forward(&x, train)?;
        x = self.trf_blocks.forward_t(&x, train)?;
        x = self.final_norm.forward(&x)?;

        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }
}
