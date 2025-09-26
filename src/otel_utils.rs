// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![allow(dead_code)]

use crate::config;
use crate::daemon::AtomicHistogram;
use crate::nccl_metadata::NcclOpKey;
use crate::step_tracker::EventStep;

use opentelemetry::metrics::Histogram as OtelHistogram;
use opentelemetry::{global, KeyValue};
use opentelemetry_sdk::metrics::{new_view, Aggregation, Instrument, InstrumentKind, Stream};
use opentelemetry_sdk::Resource;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

pub static RESOURCE: LazyLock<Resource> =
    LazyLock::new(|| Resource::builder().with_service_name("CoMMA").build());

pub fn init_meter_provider(config: &config::Config) -> Option<()> {
    let resource = &*RESOURCE;
    let otlp_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .build()
        .ok()?;
    let mut histogram_instrument = Instrument::new().name("*latency");
    histogram_instrument.kind = Some(InstrumentKind::Histogram);
    let mask = Stream::new().aggregation(Aggregation::Base2ExponentialHistogram {
        max_size: config.otel_latency_histogram_max_size,
        max_scale: config.otel_latency_histogram_max_scale as _,
        record_min_max: true,
    });

    let mut meter_provider_builder = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_resource(resource.clone())
        .with_periodic_exporter(otlp_exporter);
    if let Ok(view) = new_view(histogram_instrument, mask) {
        meter_provider_builder = meter_provider_builder.with_view(view);
    }
    let meter_provider = meter_provider_builder.build();
    global::set_meter_provider(meter_provider.clone());
    Some(())
}

#[derive(Debug)]
pub struct LatencyHistogram {
    inner: OtelHistogram<u64>,
    op_key: NcclOpKey,
    high_fidelity: bool,
    local_counter: AtomicUsize,
    counter: Arc<AtomicUsize>,
}

impl LatencyHistogram {
    pub fn new(
        inner: OtelHistogram<u64>,
        op_key: NcclOpKey,
        high_fidelity: bool,
        counter: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            inner,
            op_key,
            high_fidelity,
            local_counter: AtomicUsize::new(0),
            counter,
        }
    }

    #[cfg(test)]
    fn high_fidelity(&self) -> bool {
        self.high_fidelity
    }
}

impl AtomicHistogram<EventStep> for LatencyHistogram {
    fn record(&self, step: &EventStep) {
        if let NcclOpKey::NetSend(comm_hash, src, dst) = &self.op_key {
            let attributes: &[_] = if self.high_fidelity {
                &[
                    KeyValue::new("nccl.communicator.hash", format!("0x{:016x}", comm_hash)),
                    KeyValue::new("nccl.source.rank", *src as i64),
                    KeyValue::new("nccl.destination.rank", *dst as i64),
                ]
            } else {
                &[KeyValue::new("nccl.metric.aggregated", true)]
            };
            self.inner.record(step.dur_ns as _, attributes);
        }
        self.local_counter.fetch_add(1, Ordering::Relaxed);
    }
}

impl std::ops::Drop for LatencyHistogram {
    fn drop(&mut self) {
        self.counter.fetch_add(
            self.local_counter.load(Ordering::Acquire),
            Ordering::Relaxed,
        );
    }
}

#[derive(Debug)]
struct HistogramInfo {
    high_fidelity: bool, // This histogram is within top-k and should set all attributes
    counter: Arc<AtomicUsize>,
}

#[derive(Debug)]
pub struct HistogramManager {
    histogram: OtelHistogram<u64>,
    info: HashMap<NcclOpKey, HistogramInfo>,
    num_active: usize,
    cardinality_limit: usize,
}

impl HistogramManager {
    pub fn new(name: &str, unit: &str, cardinality_limit: usize) -> Self {
        let meter = opentelemetry::global::meter("CoMMA");
        Self {
            histogram: meter
                .u64_histogram(String::from(name))
                .with_unit(String::from(unit))
                .build(),
            info: HashMap::new(),
            num_active: 0,
            cardinality_limit,
        }
    }

    pub fn get_histogram(&mut self, key: NcclOpKey) -> LatencyHistogram {
        let info = self.info.entry(key.clone()).or_insert_with(|| {
            let mut high_fidelity = false;
            if self.num_active < self.cardinality_limit {
                high_fidelity = true;
                self.num_active += 1;
            }
            HistogramInfo {
                high_fidelity,
                counter: Arc::new(AtomicUsize::new(0)),
            }
        });
        LatencyHistogram::new(
            self.histogram.clone(),
            key,
            info.high_fidelity,
            info.counter.clone(),
        )
    }

    // sort the histograms by number of entries to get the new "Top K"
    pub fn update_priority(&mut self) {
        let mut info_vec: Vec<_> = self.info.iter_mut().collect();
        info_vec.sort_by_key(|e| std::cmp::Reverse(e.1.counter.load(Ordering::Acquire)));
        self.num_active = 0;
        for (i, e) in info_vec.iter_mut().enumerate() {
            e.1.high_fidelity = i < self.cardinality_limit;
            self.num_active += 1
        }
    }

    #[cfg(test)]
    fn num_active(&self) -> usize {
        self.num_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;

    #[tokio::test]
    async fn histogram_manager() {
        let config = config::CONFIG.clone();
        assert!(init_meter_provider(&config).is_some());

        const MAX_CARDINALITY: usize = 16;
        const N_HISTOGRAM: usize = 32;
        let mut manager = HistogramManager::new("test", "s", MAX_CARDINALITY);

        let mut histograms = Vec::new();
        for i in 0..N_HISTOGRAM {
            let key = NcclOpKey::NetSend(0x123, 42, i);
            histograms.push(manager.get_histogram(key));
        }
        assert_eq!(manager.num_active(), MAX_CARDINALITY);

        let mut num_high_fidelity = 0;
        for h in histograms.iter() {
            if h.high_fidelity() {
                num_high_fidelity += 1;
            }
        }
        assert_eq!(num_high_fidelity, MAX_CARDINALITY);

        // now we feed random amount of telemetry to them
        // we use shuffle to avoid equals.
        // this simplifies the validation logic.
        let mut rng = rand::rng();
        let mut n_telemetry: Vec<usize> = (0..histograms.len()).collect();
        n_telemetry.shuffle(&mut rng);
        for (i, h) in histograms.iter().enumerate() {
            let n = n_telemetry[i];
            for _ in 0..n {
                let step = EventStep {
                    step: 0,
                    size: 65536,
                    start_time: 1234567,
                    fifo_wait_dur_ns: None,
                    dur_ns: 512,
                };
                h.record(&step);
            }
        }

        let mut n_telemetry: Vec<_> = n_telemetry.into_iter().enumerate().collect();
        n_telemetry.sort_by_key(|e| std::cmp::Reverse(e.1));

        std::mem::drop(histograms);

        manager.update_priority();

        for (i, (hist_idx, _)) in n_telemetry.iter().enumerate() {
            let key = NcclOpKey::NetSend(0x123, 42, *hist_idx);
            let h = manager.get_histogram(key);
            assert_eq!(h.high_fidelity(), i < MAX_CARDINALITY);
        }
    }
}
