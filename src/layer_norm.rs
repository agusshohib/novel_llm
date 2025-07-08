use candle_core::{Module, Result, Tensor, D};
use candle_nn::VarBuilder;

const EPS: f32 = 1e-5;

#[derive(Clone, Debug)]
pub struct LayerNorm {
    eps: f32,
    scale: Tensor,
    shift: Tensor,
}

impl LayerNorm {
    pub fn new(emb_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let scale = vb.get_with_hints(emb_dim, "scale", candle_nn::Init::Const(1.))?;
        let shift = vb.get_with_hints(emb_dim, "shift", candle_nn::Init::Const(0.))?;
        Ok(Self {
            eps: EPS,
            scale,
            shift,
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let var = xs.var_keepdim(D::Minus1)?;
        let norm_xs = xs.broadcast_sub(&mean)?.broadcast_div(
            &(var.broadcast_add(&Tensor::new(&[self.eps], xs.device())?)?).sqrt()?,
        )?;
        let out_norm = norm_xs
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)?;
        Ok(out_norm)
    }
}
