use std::{
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Clone, Debug)]
pub struct Tokenizer {
    vocab: Vec<f64>,
}

impl Tokenizer {
    pub fn new(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);

        let vocab: Vec<f64> = reader
            .lines()
            .filter_map(std::io::Result::ok)
            .filter_map(|line| line.parse().ok())
            .collect();

        Tokenizer { vocab }
    }
}

pub trait Encoder {
    fn encode(&self, txt: &[f64]) -> Vec<u32>;
}

pub trait Decoder {
    fn decode(&self, tokens: Vec<u32>) -> Vec<f64>;
}

impl Encoder for Tokenizer {
    fn encode(&self, txt: &[f64]) -> Vec<u32> {
        let result: Vec<u32> = txt
            .iter()
            .map(|&val| _find_closest_index(&self.vocab, val))
            .collect::<Vec<u32>>();

        result
    }
}

impl Decoder for Tokenizer {
    fn decode(&self, tokens: Vec<u32>) -> Vec<f64> {
        let result: Vec<f64> = tokens
            .iter()
            .map(|&i| self.vocab[i as usize])
            .collect();

        result
    }
}

fn _find_closest_index(vocab: &[f64], value: f64) -> u32 {
    vocab
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - value)
                .abs()
                .partial_cmp(&((**b - value).abs()))
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap()
        .try_into()
        .unwrap()
}
