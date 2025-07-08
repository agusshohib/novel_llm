use std::{ops::Deref, rc::Rc};

use tiktoken_rs::CoreBPE;

#[derive(Clone)]
pub struct GPTDatasetV1(Rc<GPTDatasetV1_>);

pub struct GPTDatasetV1_ {
    input_ids: Vec<Vec<u32>>,
    target_ids: Vec<Vec<u32>>,
}

impl GPTDatasetV1 {
    pub fn new(txt: &str, tokenizer: CoreBPE, max_length: usize, stride: usize) -> Self {
        let token_ids = tokenizer.encode_with_special_tokens(txt);

        let mut input_ids: Vec<Vec<u32>> = Vec::default();
        let mut target_ids: Vec<Vec<u32>> = Vec::default();
        // get input_ids and target_ids
        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_chunk = &token_ids[i..(i + max_length)];
            let target_chunk = &token_ids[(i + 1_usize)..(i + max_length + 1_usize)];
            input_ids.push(input_chunk.to_vec());
            target_ids.push(target_chunk.to_vec());
        }

        let dataset_ = GPTDatasetV1_ {
            input_ids,
            target_ids,
        };

        Self(Rc::new(dataset_))
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn get_pair_at_index(&self, idx: usize) -> (&Vec<u32>, &Vec<u32>) {
        (&self.input_ids[idx], &self.target_ids[idx])
    }
}

impl Deref for GPTDatasetV1 {
    type Target = GPTDatasetV1_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
