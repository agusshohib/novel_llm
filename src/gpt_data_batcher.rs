use candle_datasets::{batcher::IterResult2, Batcher};

use crate::gpt_dataset_iter::GPTDatasetIter;

pub type GPTDataBatcher = Batcher<IterResult2<GPTDatasetIter>>;
