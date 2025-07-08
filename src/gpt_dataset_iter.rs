use candle_core::{Device, Result, Tensor};
use rand::{rng, seq::SliceRandom};

use crate::gpt_dataset_v1::GPTDatasetV1;

pub struct GPTDatasetIter {
    dataset: GPTDatasetV1,
    remaining_indices: Vec<usize>,
}

impl GPTDatasetIter {
    pub fn new(dataset: GPTDatasetV1, shuffle: bool) -> Self {
        let mut remaining_indices = (0..dataset.len()).rev().collect::<Vec<_>>();
        if shuffle {
            remaining_indices.shuffle(&mut rng());
        }
        Self {
            dataset,
            remaining_indices,
        }
    }
}

impl Iterator for GPTDatasetIter {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.remaining_indices.pop() {
            let (input_ids, target_ids) = self.dataset.get_pair_at_index(idx);

            // turn into Tensors and return
            let dev = Device::cuda_if_available(0).unwrap();
            let input_tensor = Tensor::new(&input_ids[..], &dev);
            let target_tensor = Tensor::new(&target_ids[..], &dev);
            Some(candle_core::error::zip(input_tensor, target_tensor))
        } else {
            None
        }
    }
}
