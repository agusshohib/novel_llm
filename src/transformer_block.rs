use candle_core::{Module, ModuleT, Result, Tensor};
use candle_nn::{Dropout, VarBuilder};

use crate::{
    config::Config, feed_forward::FeedForward, layer_norm::LayerNorm,
    multi_head_attention::MultiHeadAttention,
};

#[derive(Clone, Debug)]
pub struct TransformerBlock {
    att: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_shortcut: Dropout,
}

impl TransformerBlock {
    pub fn new(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        let att = MultiHeadAttention::new(
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.drop_rate,
            cfg.n_heads,
            cfg.qkv_bias,
            vb.pp("mha"),
        )?;
        let ff = FeedForward::new(cfg, vb.pp("ff"))?;
        let norm1 = LayerNorm::new(cfg.emb_dim, vb.pp("norm1"))?;
        let norm2 = LayerNorm::new(cfg.emb_dim, vb.pp("norm2"))?;
        let drop_shortcut = Dropout::new(cfg.drop_rate);
        Ok(Self {
            att,
            ff,
            norm1,
            norm2,
            drop_shortcut,
        })
    }
}

impl ModuleT for TransformerBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let shortcut = xs.to_owned();
        let mut x = xs.to_owned();
        x = self.norm1.forward(&x)?;
        x = self.att.forward_t(&x, train)?;
        x = self.drop_shortcut.forward(&x, train)?;
        x = (x + shortcut)?;

        let shortcut = x.clone();
        x = self.norm2.forward(&x)?;
        x = self.ff.forward(&x)?;
        x = self.drop_shortcut.forward(&x, train)?;
        x = (x + shortcut)?;
        Ok(x)
    }
}
