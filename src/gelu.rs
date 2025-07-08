use core::f64::consts::PI;

use candle_core::{DType::F32, Module, Result, Tensor};

#[derive(Clone, Debug)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (0.5_f64 * xs)?.mul(
            &((2_f64 / PI).sqrt() * (xs + (xs.mul(xs)?.mul(xs)? * 0.044715f64)?)?)?
                .tanh()?
                .broadcast_add(&Tensor::ones((1,), F32, xs.device())?)?,
        )
    }
}
