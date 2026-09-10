//! Ordered, bounded worker pipeline for the per-morsel logical rewrite.
//!
//! The rewrite (file split, DV keep-mask, select) used to run inside
//! `Iterator::next` on the consumer thread, between polars' single-slot
//! batch channel and the FFI export. Here it runs on a few worker threads
//! while the consumer exports the previous morsel; results come back in
//! input order and at most `in_flight` morsels are ever ahead of the
//! consumer, so a row limit reads little more than it needs.

use std::collections::BTreeMap;
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

use polars::prelude::DataFrame;

pub(crate) struct OrderedPipeline<T: Send + 'static> {
    results: Receiver<(usize, anyhow::Result<T>)>,
    next_seq: usize,
    parked: BTreeMap<usize, anyhow::Result<T>>,
}

impl<T: Send + 'static> OrderedPipeline<T> {
    /// Feeds `source` through `work` on `workers` threads. The feeder and
    /// the workers stop on their own once this pipeline is dropped: their
    /// sends fail, which also drops `source` and cancels the read behind it.
    pub(crate) fn new<I, F>(source: I, workers: usize, in_flight: usize, work: F) -> Self
    where
        I: Iterator<Item = anyhow::Result<DataFrame>> + Send + 'static,
        F: Fn(DataFrame) -> anyhow::Result<T> + Send + Sync + 'static,
    {
        let work = Arc::new(work);
        let (task_tx, task_rx) = sync_channel::<(usize, anyhow::Result<DataFrame>)>(in_flight);
        let task_rx = Arc::new(Mutex::new(task_rx));
        let (result_tx, results) = sync_channel::<(usize, anyhow::Result<T>)>(in_flight);

        thread::Builder::new()
            .name("pldl-rewrite-feed".into())
            .spawn(move || {
                for item in source.enumerate() {
                    if task_tx.send(item).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn rewrite feeder");

        for index in 0..workers.max(1) {
            let task_rx = task_rx.clone();
            let result_tx = result_tx.clone();
            let work = work.clone();
            thread::Builder::new()
                .name(format!("pldl-rewrite-{index}"))
                .spawn(move || {
                    loop {
                        // Hold the lock only while waiting for a task, so the
                        // other workers can pick up the next one meanwhile.
                        let task = task_rx.lock().expect("rewrite queue poisoned").recv();
                        let Ok((seq, item)) = task else {
                            break;
                        };
                        let out = item.and_then(|df| work(df));
                        if result_tx.send((seq, out)).is_err() {
                            break;
                        }
                    }
                })
                .expect("spawn rewrite worker");
        }

        Self {
            results,
            next_seq: 0,
            parked: BTreeMap::new(),
        }
    }
}

impl<T: Send + 'static> Iterator for OrderedPipeline<T> {
    type Item = anyhow::Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.parked.remove(&self.next_seq) {
                self.next_seq += 1;
                return Some(item);
            }
            match self.results.recv() {
                Ok((seq, item)) => {
                    self.parked.insert(seq, item);
                }
                // Every worker has exited: the source is drained.
                Err(_) => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::{NamedFrom, Series};

    use super::*;

    fn frame(value: i64) -> DataFrame {
        DataFrame::new(1, vec![Series::new("v".into(), vec![value]).into()]).unwrap()
    }

    /// Workers finish out of order; the consumer must still see input order.
    #[test]
    fn preserves_input_order() {
        let source = (0..64i64).map(|i| Ok(frame(i)));
        let out: Vec<i64> = OrderedPipeline::new(source, 3, 2, |df| {
            let v = df.column("v").unwrap().i64().unwrap().get(0).unwrap();
            if v % 5 == 0 {
                std::thread::sleep(std::time::Duration::from_millis(3));
            }
            Ok(v)
        })
        .map(|r| r.unwrap())
        .collect();
        assert_eq!(out, (0..64).collect::<Vec<_>>());
    }

    /// An error travels in sequence like any other result.
    #[test]
    fn errors_keep_their_position() {
        let source = (0..4i64).map(|i| Ok(frame(i)));
        let out: Vec<Result<i64, String>> = OrderedPipeline::new(source, 2, 2, |df| {
            let v = df.column("v").unwrap().i64().unwrap().get(0).unwrap();
            if v == 2 {
                anyhow::bail!("boom")
            }
            Ok(v)
        })
        .map(|r| r.map_err(|e| e.to_string()))
        .collect();
        assert_eq!(out[0], Ok(0));
        assert_eq!(out[1], Ok(1));
        assert_eq!(out[2], Err("boom".to_string()));
        assert_eq!(out[3], Ok(3));
    }

    /// Dropping the pipeline early must not hang: the feeder's send fails
    /// once the workers are gone.
    #[test]
    fn early_drop_stops_the_feeder() {
        let source = (0..1_000_000i64).map(|i| Ok(frame(i)));
        let mut pipeline = OrderedPipeline::new(source, 2, 2, Ok);
        assert!(pipeline.next().is_some());
        drop(pipeline);
    }
}
