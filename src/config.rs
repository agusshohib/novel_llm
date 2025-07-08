#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

impl Config {
    pub fn gpt2_124m() -> Self {
        Self {
            vocab_size: 50_257,
            context_length: 1_024,
            emb_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }
}
