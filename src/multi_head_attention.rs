use candle_core::{Device, ModuleT, Result, Tensor, D};
use candle_nn::{linear_b, ops::softmax, Dropout, Linear, VarBuilder};

#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    num_heads: usize,
    d_out: usize,
    head_dim: usize,
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    out_proj: Linear,
    scaling: f64,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        drop_p: f32,
        num_heads: usize,
        qkv_bias: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        if d_out % num_heads != 0 {
            panic!("`d_out` must be divisible by `num_heads`.")
        }
        let head_dim = d_out / num_heads;

        let w_query = linear_b(d_in, d_out, qkv_bias, vb.pp("query"))?;
        let w_key = linear_b(d_in, d_out, qkv_bias, vb.pp("key"))?;
        let w_value = linear_b(d_in, d_out, qkv_bias, vb.pp("value"))?;
        let out_proj = linear_b(d_out, d_out, true, vb.pp("out_proj"))?;
        let scaling = 1. / (head_dim as f64).sqrt();
        let dropout = Dropout::new(drop_p);

        Ok(Self {
            num_heads,
            d_out,
            head_dim,
            w_query,
            w_key,
            w_value,
            out_proj,
            scaling,
            dropout,
        })
    }
}

impl ModuleT for MultiHeadAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b, num_tokens, _d_in) = xs.dims3()?;
        let queries = self.w_query.forward_t(xs, train)?;
        let keys = self.w_key.forward_t(xs, train)?;
        let values = self.w_value.forward_t(xs, train)?;

        // reshapes to facilitate getting attn scores each of the individual heads
        // with one matrix multiplication
        let queries = queries
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let keys = keys
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let values = values
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn_scores = queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)?;

        let mask = get_mask(num_tokens, xs.device())?;
        let masked = masked_fill(
            &attn_scores,
            &mask.broadcast_left((b, self.num_heads)).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // scale
        let mut attn_weights = softmax(&(masked * self.scaling)?, D::Minus1)?;
        // dropout
        attn_weights = self.dropout.forward(&attn_weights, train)?;

        // context vectors
        let context_vec = attn_weights.matmul(&values)?.transpose(1, 2)?;
        let context_vec = context_vec
            .reshape((b, num_tokens, self.d_out))?
            .contiguous()?;

        // projection
        self.out_proj.forward_t(&context_vec, train)
    }
}

pub fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u32::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}
