use crate::transformer_block::TransformerBlock;
use candle_core::{ModuleT, Result, Tensor};

#[derive(Clone, Debug)]
pub struct SequentialTransformers {
    layers: Vec<TransformerBlock>,
}

impl SequentialTransformers {
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: TransformerBlock) -> Self {
        self.layers.push(layer);
        self
    }
}

impl ModuleT for SequentialTransformers {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, train)?
        }
        Ok(xs)
    }
}

pub fn seqtransformers() -> SequentialTransformers {
    SequentialTransformers { layers: vec![] }
}
