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

// copybara:strip_begin(otel)
use criterion::{criterion_group, criterion_main, Criterion};
use nccl_profiler::otel_utils::{
    DurationHistogram, EventStep, GapTracker, HistogramManager, NcclOpKey,
};
use opentelemetry_sdk::metrics::{
    new_view, Aggregation, Instrument, InstrumentKind, ManualReader, SdkMeterProvider, Stream,
};

// install a real meter provider with the same base2 exponential histogram
// aggregation as init_meter_provider() so record costs are representative
fn init_meter_provider() {
    let mut builder = SdkMeterProvider::builder().with_reader(ManualReader::builder().build());
    for name in [
        "*latency",
        "nccl.collective.duration",
        "nccl.collective.gap",
    ] {
        let mut histogram_instrument = Instrument::new().name(name);
        histogram_instrument.kind = Some(InstrumentKind::Histogram);
        let mask = Stream::new().aggregation(Aggregation::Base2ExponentialHistogram {
            max_size: 160,
            max_scale: 20,
            record_min_max: true,
        });
        if let Ok(view) = new_view(histogram_instrument, mask) {
            builder = builder.with_view(view);
        }
    }
    opentelemetry::global::set_meter_provider(builder.build());
}

fn criterion_benchmark(c: &mut Criterion) {
    init_meter_provider();

    let step = EventStep {
        step: 0,
        size: 65536,
        start_time: 1234567,
        fifo_wait_dur_ns: None,
        dur_ns: 512,
    };

    let mut send_manager = HistogramManager::new("nccl.net_send.latency", "ns", 16);
    let send_histogram = send_manager.get_histogram(NcclOpKey::NetSend(0x123, 0, 1));
    c.bench_function("net_send latency record", |b| {
        b.iter(|| send_histogram.record(&step))
    });

    let mut recv_manager = HistogramManager::new("nccl.net_recv.latency", "ns", 16);
    let recv_histogram = recv_manager.get_histogram(NcclOpKey::NetRecv(0x123, 0, 1));
    c.bench_function("net_recv latency record", |b| {
        b.iter(|| recv_histogram.record(&step))
    });

    let mut duration_histogram = DurationHistogram::new("nccl.collective.duration", "ns", 16);
    c.bench_function("collective duration record", |b| {
        b.iter(|| duration_histogram.record(4096, 0x123, "ncclAllReduce", 0, 1 << 22))
    });

    // the clock closure mirrors the shipped default path of
    // Profiler::recent_timer_ns (an Instant read per call)
    let gap_tracker = GapTracker::new();
    gap_tracker.init_instrument();
    let t0 = std::time::Instant::now();
    let now_ns = move || t0.elapsed().as_nanos() as u64;
    c.bench_function("gap idle transition", |b| {
        b.iter(|| {
            gap_tracker.activity_begin(now_ns);
            gap_tracker.activity_end(now_ns());
        })
    });

    gap_tracker.activity_begin(now_ns);
    c.bench_function("gap nested activity", |b| {
        b.iter(|| {
            gap_tracker.activity_begin(now_ns);
            gap_tracker.activity_end(now_ns());
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
// copybara:strip_end
