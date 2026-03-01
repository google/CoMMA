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

use crate::config;
use crate::daemon::AtomicHistogram;
use crate::event;
use crate::event::ProfilerEvent as _;
use crate::nccl_metadata;
use crate::nccl_metadata::NcclOpKey;
use crate::profiler::Profiler;
use crate::step_tracker::EventStep;

use opentelemetry::context::Context as OtelContext;
use opentelemetry::metrics::Histogram as OtelHistogram;
use opentelemetry::trace::{
    Span, SpanBuilder, SpanContext, SpanKind, TraceContextExt as _, TraceFlags, TraceState, Tracer,
};
use opentelemetry::{global, KeyValue, SpanId, TraceId};
use opentelemetry_sdk::metrics::{new_view, Aggregation, Instrument, InstrumentKind, Stream};
use opentelemetry_sdk::Resource;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, SystemTime};

pub static RESOURCE: LazyLock<Resource> =
    LazyLock::new(|| Resource::builder().with_service_name("CoMMA").build());

pub static METER_PROVIDER: OnceLock<opentelemetry_sdk::metrics::SdkMeterProvider> = OnceLock::new();

pub static TRACER_PROVIDER: OnceLock<opentelemetry_sdk::trace::SdkTracerProvider> = OnceLock::new();

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
    let meter_provider = METER_PROVIDER.get_or_init(|| meter_provider_builder.build());
    global::set_meter_provider(meter_provider.clone());
    Some(())
}

pub fn init_tracer_provider(_config: &config::Config) -> Option<()> {
    let resource = &*RESOURCE;
    let otlp_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .build()
        .ok()?;

    // Create a tracer provider with the exporter
    let provider = TRACER_PROVIDER.get_or_init(|| {
        opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_batch_exporter(otlp_exporter)
            .build()
    });
    global::set_tracer_provider(provider.clone());
    Some(())
}

#[derive(Debug)]
pub struct LatencyHistogram {
    inner: OtelHistogram<u64>,
    op_key: NcclOpKey,
    hostname: Option<String>,
    high_fidelity: bool,
    local_counter: AtomicUsize,
    counter: Arc<AtomicUsize>,
}

impl LatencyHistogram {
    pub fn new(
        inner: OtelHistogram<u64>,
        op_key: NcclOpKey,
        hostname: Option<String>,
        high_fidelity: bool,
        counter: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            inner,
            op_key,
            hostname,
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
            let hostname: String = self.hostname.clone().unwrap_or(String::from(""));
            let attributes: &[_] = if self.high_fidelity {
                &[
                    KeyValue::new("nccl.communicator.hash", format!("0x{:016x}", comm_hash)),
                    KeyValue::new("nccl.source.rank", *src as i64),
                    KeyValue::new("nccl.destination.rank", *dst as i64),
                    KeyValue::new("nccl.hostname", hostname),
                ]
            } else {
                &[
                    KeyValue::new("nccl.metric.aggregated", true),
                    KeyValue::new("nccl.hostname", hostname),
                ]
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

fn get_hostname_libc() -> Option<String> {
    // hostname should be no longer than HOST_NAME_MAX, which is typically smaller than 256
    let mut buf = [0 as libc::c_char; 257];

    // SAFETY: gethostname will not overflow the buffer
    let result = unsafe { libc::gethostname(buf.as_mut_ptr(), buf.len() - 1) };

    // If truncation happens (though it shouldn't), gethostname() won't write the trailing null.
    // So we write it to be safe
    buf[256] = 0;

    if result == 0 {
        // Safely convert the null-terminated C string into a Rust String
        let c_str = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
        Some(c_str.to_string_lossy().into_owned())
    } else {
        // If it fails, grab the last OS error (errno)
        None
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
    hostname: Option<String>,
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
            hostname: get_hostname_libc(),
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
            self.hostname.clone(),
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

fn coll_type_to_num(t: nccl_metadata::NcclOpType) -> u32 {
    use nccl_metadata::NcclOpType as T;
    match t {
        T::Broadcast => 1,
        T::Reduce => 2,
        T::AllGather => 3,
        T::ReduceScatter => 4,
        T::AllReduce => 5,
        _ => 0xabcd, // we don't use zero as zero span ID is invalid
    }
}

fn ncclop_otel_name(op_type: nccl_metadata::NcclOpType) -> &'static str {
    use nccl_metadata::NcclOpType;

    match op_type {
        NcclOpType::Broadcast => "ncclBroadcast",
        NcclOpType::Reduce => "ncclReduce",
        NcclOpType::AllGather => "ncclAllGather",
        NcclOpType::ReduceScatter => "ncclReduceScatter",
        NcclOpType::AllReduce => "ncclAllReduce",
        NcclOpType::Send => "ncclSend",
        NcclOpType::Recv => "ncclRecv",
        _ => "unknown nccl op",
    }
}

#[derive(Clone)]
struct NcclOpAttr {
    name: String,
    start_time: SystemTime,
    duration: Duration,
    attributes: Vec<KeyValue>,
}

fn gen_ncclop_attributes(profiler: &Profiler, op: &event::NcclOp) -> Option<NcclOpAttr> {
    let basic_info = op.basic_info();
    let start_time = op.child_start_time()?;
    let end_time = basic_info.end_time()?;
    let duration = end_time - start_time;
    let start_time = profiler.init_time + (start_time - profiler.init_instant);
    let name;
    let descr = op.get_descr();
    let mut attributes = vec![
        KeyValue::new("nccl.comm.hash", format!("0x{:016x}", op.comm_hash())),
        KeyValue::new("nccl.rank", basic_info.rank() as i64),
        KeyValue::new("nccl.size.bytes", op.byte_count() as i64),
    ];
    if let Some(coll) = descr.try_cast_to_coll() {
        name = ncclop_otel_name(coll.op_type());
        attributes.append(&mut vec![
            KeyValue::new(
                "nccl.collective.algo",
                nccl_metadata::algo::name(coll.algo()),
            ),
            KeyValue::new(
                "nccl.collective.proto",
                nccl_metadata::proto::name(coll.proto()),
            ),
            KeyValue::new("nccl.collective.n_max_channel", coll.n_max_channel() as i64),
        ]);
    } else if let Some(p2p) = descr.try_cast_to_p2p() {
        name = if p2p.is_send() {
            "ncclSend"
        } else {
            "ncclRecv"
        };
        attributes.push(KeyValue::new("nccl.p2p.peer.rank", p2p.peer() as i64));
    } else {
        return None;
    }

    Some(NcclOpAttr {
        name: name.to_string(),
        start_time,
        duration,
        attributes,
    })
}

pub fn add_ncclop_trace<T>(tracer: &mut T, profiler: &Profiler, op: &event::NcclOp) -> Option<()>
where
    T: Tracer,
{
    let attr = gen_ncclop_attributes(profiler, op)?;

    let maybe_ctx = op.get_coll_descr().map(|coll| {
        // create a parent
        let trace_id = {
            let comm_hash = op.comm_hash();
            let encoded = comm_hash as u128;
            TraceId::from_bytes(encoded.to_be_bytes())
        };
        let span_id = {
            let coll_type = coll_type_to_num(coll.op_type());
            let seq_num = coll.seq_num();
            let encoded = ((coll_type as u64) << 32) | seq_num;
            SpanId::from_bytes(encoded.to_be_bytes())
        };
        if op.basic_info().rank() == 0 {
            // rank 0 should build the "parent span"
            let attr = attr.clone();
            let builder = SpanBuilder {
                trace_id: Some(trace_id),
                span_id: Some(span_id),
                name: attr.name.into(),
                start_time: Some(attr.start_time),
                end_time: Some(attr.start_time + attr.duration), // use same duration as rank 0
                span_kind: Some(SpanKind::Server),
                attributes: Some(attr.attributes),
                ..Default::default()
            };
            let mut parent = tracer.build(builder);
            parent.end_with_timestamp(attr.start_time + attr.duration);
        }
        let parent = SpanContext::new(
            trace_id,
            span_id,
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        );
        OtelContext::new().with_remote_span_context(parent)
    });

    let builder = SpanBuilder {
        name: attr.name.into(),
        start_time: Some(attr.start_time),
        end_time: Some(attr.start_time + attr.duration),
        span_kind: Some(SpanKind::Server),
        attributes: Some(attr.attributes),
        ..Default::default()
    };
    let mut span = if let Some(ctx) = maybe_ctx {
        tracer.build_with_context(builder, &ctx)
    } else {
        tracer.build(builder)
    };
    span.end_with_timestamp(attr.start_time + attr.duration);
    Some(())
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
