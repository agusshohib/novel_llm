use crate::gelu::GELU;
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

#[derive(Clone, Debug)]
pub enum FFLayer {
    Linear(Linear),
    GELU(GELU),
}

impl Module for FFLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            FFLayer::GELU(g) => g.forward(xs),
            FFLayer::Linear(l) => l.forward(xs),
        }
    }
}
