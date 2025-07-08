use candle_core::{Module, Result, Tensor};
use candle_nn::{VarBuilder, linear_b};

use crate::{config::Config, ff_layer::FFLayer, gelu::GELU};

#[derive(Clone, Debug)]
pub struct FeedForward {
    layers: Vec<FFLayer>,
}

impl FeedForward {
    pub fn new(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        let layers = vec![
            FFLayer::Linear(linear_b(
                cfg.emb_dim,
                4_usize * cfg.emb_dim,
                true,
                vb.pp("first_layer"),
            )?),
            FFLayer::GELU(GELU),
            FFLayer::Linear(linear_b(
                4_usize * cfg.emb_dim,
                cfg.emb_dim,
                true,
                vb.pp("second_layer"),
            )?),
        ];
        Ok(Self { layers })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}
