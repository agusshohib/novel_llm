use candle_datasets::Batcher;

use crate::{
    data_loader::DataLoader, gpt_data_batcher::GPTDataBatcher, gpt_dataset_iter::GPTDatasetIter,
    gpt_dataset_v1::GPTDatasetV1,
};

pub struct GPTDataLoader {
    dataset: GPTDatasetV1,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl GPTDataLoader {
    pub fn new(dataset: GPTDatasetV1, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
        }
    }
}

impl DataLoader for GPTDataLoader {
    type Batcher = GPTDataBatcher;

    fn batcher(&self) -> GPTDataBatcher {
        let iter = GPTDatasetIter::new(self.dataset.clone(), self.shuffle);
        Batcher::new_r2(iter)
            .batch_size(self.batch_size)
            .return_last_incomplete_batch(!self.drop_last)
    }
}
