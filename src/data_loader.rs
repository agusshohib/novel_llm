pub trait DataLoader {
    type Batcher;

    fn batcher(&self) -> Self::Batcher;
}
