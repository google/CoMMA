// commited by sputti-czi

use crate::config;
use crate::daemon::AtomicHistogram;
use crate::event;
use crate::event::ProfilerEvent as _;
use crate::nccl_metadata;
use crate::profiler::Profiler;

pub use crate::nccl_metadata::NcclOpKey;
pub use crate::step_tracker::EventStep;

use opentelemetry::context::Context as OtelContext;
use opentelemetry::metrics::Gauge;
use opentelemetry::metrics::Histogram as OtelHistogram;
use opentelemetry::trace::{
    Span, SpanBuilder, SpanContext, SpanKind, TraceContextExt as _, TraceFlags, TraceState, Tracer,
};
use opentelemetry::{global, KeyValue, SpanId, TraceId};
use opentelemetry_sdk::metrics::{new_view, Aggregation, Instrument, InstrumentKind, Stream};
use opentelemetry_sdk::Resource;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
    let mut meter_provider_builder = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_resource(resource.clone())
        .with_periodic_exporter(otlp_exporter);
    for name in [
        "*latency",
        "nccl.collective.duration",
        "nccl.collective.gap",
    ] {
        let mut histogram_instrument = Instrument::new().name(name);
        histogram_instrument.kind = Some(InstrumentKind::Histogram);
        let mask = Stream::new().aggregation(Aggregation::Base2ExponentialHistogram {
            max_size: config.otel_latency_histogram_max_size,
            max_scale: config.otel_latency_histogram_max_scale as _,
            record_min_max: true,
        });
        if let Ok(view) = new_view(histogram_instrument, mask) {
            meter_provider_builder = meter_provider_builder.with_view(view);
        }
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
    attributes: Vec<KeyValue>,
    local_counter: AtomicUsize,
    counter: Arc<AtomicUsize>,
}

impl LatencyHistogram {
    pub fn new(
        inner: OtelHistogram<u64>,
        op_key: NcclOpKey,
        hostname: String,
        high_fidelity: bool,
        counter: Arc<AtomicUsize>,
    ) -> Self {
        // attribute sets are built once per connection so the per-step
        // record path performs no allocation
        let attributes = if !high_fidelity {
            vec![
                KeyValue::new("nccl.metric.aggregated", true),
                KeyValue::new("nccl.hostname", hostname),
            ]
        } else {
            match &op_key {
                NcclOpKey::NetSend(comm_hash, src, dst) => vec![
                    KeyValue::new("nccl.communicator.hash", format!("0x{:016x}", comm_hash)),
                    KeyValue::new("nccl.source.rank", *src as i64),
                    KeyValue::new("nccl.destination.rank", *dst as i64),
                    KeyValue::new("nccl.hostname", hostname),
                ],
                NcclOpKey::NetRecv(comm_hash, local, peer) => vec![
                    KeyValue::new("nccl.communicator.hash", format!("0x{:016x}", comm_hash)),
                    KeyValue::new("nccl.source.rank", *peer as i64),
                    KeyValue::new("nccl.destination.rank", *local as i64),
                    KeyValue::new("nccl.hostname", hostname),
                ],
                _ => Vec::new(),
            }
        };
        Self {
            inner,
            attributes,
            local_counter: AtomicUsize::new(0),
            counter,
        }
    }

    pub fn record(&self, step: &EventStep) {
        if !self.attributes.is_empty() {
            self.inner.record(step.dur_ns as _, &self.attributes);
        }
        self.local_counter.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn high_fidelity(&self) -> bool {
        !self
            .attributes
            .contains(&KeyValue::new("nccl.metric.aggregated", true))
    }

    #[cfg(test)]
    fn attributes(&self) -> &[KeyValue] {
        &self.attributes
    }
}

impl AtomicHistogram<EventStep> for LatencyHistogram {
    fn record(&self, step: &EventStep) {
        LatencyHistogram::record(self, step);
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
    hostname: String,
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
            hostname: get_hostname_libc().unwrap_or_default(),
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
            if e.1.high_fidelity {
                self.num_active += 1
            }
        }
    }

    pub fn close_comm(&mut self, comm_hash: u64) {
        let removed_active = self
            .info
            .iter()
            .filter(|(key, info)| key.get_comm_hash() == comm_hash && info.high_fidelity)
            .count();
        self.info.retain(|key, _| key.get_comm_hash() != comm_hash);
        self.num_active -= removed_active;
    }

    #[cfg(test)]
    pub(crate) fn num_active(&self) -> usize {
        self.num_active
    }
}

const SIZE_CLASS_SMALL_MAX: usize = 1 << 20;
const SIZE_CLASS_MEDIUM_MAX: usize = 16 << 20;

fn size_class(byte_count: usize) -> &'static str {
    if byte_count < SIZE_CLASS_SMALL_MAX {
        "lt1m"
    } else if byte_count <= SIZE_CLASS_MEDIUM_MAX {
        "1m_16m"
    } else {
        "gt16m"
    }
}

type DurationKey = (
    /* comm_hash = */ u64,
    /* op name = */ &'static str,
    /* size class = */ &'static str,
    /* rank = */ usize,
);

#[derive(Debug)]
struct DurationInfo {
    attributes: Option<Vec<KeyValue>>, // Some iff this key is within top-k
    count: usize,
}

#[derive(Debug)]
pub struct DurationHistogram {
    inner: OtelHistogram<u64>,
    hostname: String,
    info: HashMap<DurationKey, DurationInfo>,
    aggregated_attributes: Vec<KeyValue>,
    num_active: usize,
    cardinality_limit: usize,
}

impl DurationHistogram {
    pub fn new(name: &str, unit: &str, cardinality_limit: usize) -> Self {
        let meter = opentelemetry::global::meter("CoMMA");
        let hostname = get_hostname_libc().unwrap_or_default();
        Self {
            inner: meter
                .u64_histogram(String::from(name))
                .with_unit(String::from(unit))
                .build(),
            aggregated_attributes: vec![
                KeyValue::new("nccl.metric.aggregated", true),
                KeyValue::new("nccl.hostname", hostname.clone()),
            ],
            hostname,
            info: HashMap::new(),
            num_active: 0,
            cardinality_limit,
        }
    }

    fn build_attributes(hostname: &str, key: &DurationKey) -> Vec<KeyValue> {
        vec![
            KeyValue::new("nccl.comm.hash", format!("0x{:016x}", key.0)),
            KeyValue::new("nccl.collective.name", key.1),
            KeyValue::new("nccl.size.class", key.2),
            KeyValue::new("nccl.rank", key.3 as i64),
            KeyValue::new("nccl.hostname", String::from(hostname)),
        ]
    }

    pub fn record(
        &mut self,
        dur_ns: u64,
        comm_hash: u64,
        op_name: &'static str,
        rank: usize,
        byte_count: usize,
    ) {
        let key = (comm_hash, op_name, size_class(byte_count), rank);
        let info = match self.info.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let attributes = (self.num_active < self.cardinality_limit)
                    .then(|| Self::build_attributes(&self.hostname, e.key()));
                if attributes.is_some() {
                    self.num_active += 1;
                }
                e.insert(DurationInfo {
                    attributes,
                    count: 0,
                })
            }
        };
        info.count += 1;
        match info.attributes.as_ref() {
            Some(attributes) => self.inner.record(dur_ns, attributes),
            None => self.inner.record(dur_ns, &self.aggregated_attributes),
        }
    }

    // sort the keys by number of records to get the new "Top K"; counts are
    // per-interval so keys of idle communicators age out of the top-k
    pub fn update_priority(&mut self) {
        let hostname = &self.hostname;
        let mut info_vec: Vec<_> = self.info.iter_mut().collect();
        info_vec.sort_by_key(|e| std::cmp::Reverse(e.1.count));
        self.num_active = 0;
        for (i, (key, info)) in info_vec.into_iter().enumerate() {
            if i < self.cardinality_limit {
                if info.attributes.is_none() {
                    info.attributes = Some(Self::build_attributes(hostname, key));
                }
                self.num_active += 1;
            } else {
                info.attributes = None;
            }
            info.count = 0;
        }
    }

    pub fn close_comm(&mut self, comm_hash: u64) {
        let removed_active = self
            .info
            .iter()
            .filter(|(key, info)| key.0 == comm_hash && info.attributes.is_some())
            .count();
        self.info.retain(|key, _| key.0 != comm_hash);
        self.num_active -= removed_active;
    }

    #[cfg(test)]
    fn num_active(&self) -> usize {
        self.num_active
    }

    #[cfg(test)]
    fn high_fidelity(
        &self,
        comm_hash: u64,
        op_name: &'static str,
        rank: usize,
        byte_count: usize,
    ) -> bool {
        self.info
            .get(&(comm_hash, op_name, size_class(byte_count), rank))
            .is_some_and(|info| info.attributes.is_some())
    }
}

pub fn record_ncclop_duration(
    histogram: &mut DurationHistogram,
    _profiler: &Profiler,
    op: &event::NcclOp,
) -> Option<()> {
    let duration = op.child_duration()?;
    let descr = op.get_descr();
    let name = if let Some(coll) = descr.try_cast_to_coll() {
        ncclop_otel_name(coll.op_type())
    } else if let Some(p2p) = descr.try_cast_to_p2p() {
        if p2p.is_send() {
            "ncclSend"
        } else {
            "ncclRecv"
        }
    } else {
        return None;
    };
    histogram.record(
        duration.as_nanos() as _,
        op.comm_hash(),
        name,
        op.basic_info().rank(),
        op.byte_count(),
    );
    Some(())
}

const GAP_IDLE_NEVER: u64 = 0;
const GAP_IN_FLIGHT_MASK: u64 = (1 << 32) - 1;
const GAP_BEGIN_UNIT: u64 = 1 << 32;

/// Tracks intervals where this process has no NCCL activity in flight.
///
/// An activity window is an op enqueue (`NcclOp` start to stop), a proxy op
/// or a kernel channel; the union of these windows spans each operation from
/// its API call to the end of its network / kernel work, so the recorded gaps
/// are the complement of NCCL engagement and interval-union-correct under
/// overlap by construction.
///
/// The in-flight count and idle timestamp are lock-free atomics, so activity
/// begin / end cost two atomic operations each on the caller's thread.
#[derive(Debug)]
pub struct GapTracker {
    // low 32 bits: activity windows in flight; high 32 bits: total begins,
    // so check_stalled can tell a leaked window apart from a busy process
    state: AtomicU64,
    idle_since_ns: AtomicU64,
    last_state: AtomicU64,
    stall_logged: AtomicBool,
    histogram: OnceLock<OtelHistogram<u64>>,
    attributes: [KeyValue; 2],
}

impl GapTracker {
    pub fn new() -> Self {
        Self {
            state: AtomicU64::new(0),
            idle_since_ns: AtomicU64::new(GAP_IDLE_NEVER),
            last_state: AtomicU64::new(0),
            stall_logged: AtomicBool::new(false),
            histogram: OnceLock::new(),
            // the gap is process-wide (activity windows span communicators
            // and threads), so it is labeled with a process-stable identity
            // rather than a comm-local rank
            attributes: [
                KeyValue::new("nccl.hostname", get_hostname_libc().unwrap_or_default()),
                // SAFETY: `getpid()` takes no input and does not modify rust-managed state
                KeyValue::new("nccl.pid", unsafe { libc::getpid() } as i64),
            ],
        }
    }

    /// Must be called after the meter provider is installed; before that,
    /// activity_begin / activity_end only maintain the in-flight count.
    pub fn init_instrument(&self) {
        let meter = opentelemetry::global::meter("CoMMA");
        let _ = self.histogram.get_or_init(|| {
            meter
                .u64_histogram("nccl.collective.gap")
                .with_unit("ns")
                .build()
        });
    }

    pub fn activity_begin<T>(&self, now_ns: T)
    where
        T: FnOnce() -> u64,
    {
        let Some(gap_ns) = self.begin_transition(now_ns) else {
            return;
        };
        let Some(histogram) = self.histogram.get() else {
            return;
        };
        histogram.record(gap_ns, &self.attributes);
    }

    pub fn activity_end(&self, now_ns: u64) {
        // stamp before decrementing so a concurrent 0 -> 1 observer never
        // reads a timestamp from a previous idle period; fetch_max keeps the
        // idle start at the latest end under concurrent activity ends
        self.idle_since_ns.fetch_max(now_ns, Ordering::AcqRel);
        self.state.fetch_sub(1, Ordering::AcqRel);
    }

    fn begin_transition<T>(&self, now_ns: T) -> Option<u64>
    where
        T: FnOnce() -> u64,
    {
        let prev = self.state.fetch_add(GAP_BEGIN_UNIT | 1, Ordering::AcqRel);
        if prev & GAP_IN_FLIGHT_MASK != 0 {
            return None;
        }
        let idle_since = self.idle_since_ns.load(Ordering::Acquire);
        if idle_since == GAP_IDLE_NEVER {
            return None;
        }
        Some(now_ns().saturating_sub(idle_since))
    }

    /// Called periodically off the hot path. If NCCL abandons a started
    /// event on an error path the in-flight count never returns to zero and
    /// the metric goes silent; warn once when windows stay in flight with no
    /// begin or end for a whole check interval.
    pub fn check_stalled(&self) -> bool {
        let state = self.state.load(Ordering::Acquire);
        let prev = self.last_state.swap(state, Ordering::Relaxed);
        let stalled = state & GAP_IN_FLIGHT_MASK != 0
            && state == prev
            && !self.stall_logged.swap(true, Ordering::Relaxed);
        if stalled {
            log::warn!(
                "{} NCCL activity window(s) stuck in flight; \
                nccl.collective.gap will report no further gaps",
                state & GAP_IN_FLIGHT_MASK
            );
        }
        stalled
    }

    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        (self.state.load(Ordering::Acquire) & GAP_IN_FLIGHT_MASK) as usize
    }
}

impl Default for GapTracker {
    fn default() -> Self {
        Self::new()
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

pub fn record_ncclop_seqnum(
    gauge: &mut Gauge<i64>,
    _profiler: &Profiler,
    op: &event::NcclOp,
) -> Option<()> {
    let descr = op.get_descr();
    if let Some(coll) = descr.try_cast_to_coll() {
        let name = ncclop_otel_name(coll.op_type());
        let hostname: String = get_hostname_libc().unwrap_or(String::from(""));
        gauge.record(
            coll.seq_num() as i64,
            &[
                KeyValue::new("nccl.comm.hash", format!("0x{:016x}", op.comm_hash())),
                KeyValue::new("nccl.collective.name", name),
                KeyValue::new("nccl.rank", coll.rank() as i64),
                KeyValue::new("nccl.hostname", hostname),
            ],
        );
    }
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
        assert_eq!(manager.num_active(), MAX_CARDINALITY);

        for (i, (hist_idx, _)) in n_telemetry.iter().enumerate() {
            let key = NcclOpKey::NetSend(0x123, 42, *hist_idx);
            let h = manager.get_histogram(key);
            assert_eq!(h.high_fidelity(), i < MAX_CARDINALITY);
        }
    }

    #[test]
    fn latency_histogram_attributes() {
        let mut manager = HistogramManager::new("test.latency", "ns", 16);

        let send = manager.get_histogram(NcclOpKey::NetSend(0x123, 4, 8));
        assert!(send
            .attributes()
            .contains(&KeyValue::new("nccl.source.rank", 4_i64)));
        assert!(send
            .attributes()
            .contains(&KeyValue::new("nccl.destination.rank", 8_i64)));

        // for recv connections the local rank is the destination and the
        // peer is the remote sender
        let recv = manager.get_histogram(NcclOpKey::NetRecv(0x123, 4, 8));
        assert!(recv
            .attributes()
            .contains(&KeyValue::new("nccl.source.rank", 8_i64)));
        assert!(recv
            .attributes()
            .contains(&KeyValue::new("nccl.destination.rank", 4_i64)));

        let aggregated = LatencyHistogram::new(
            manager.histogram.clone(),
            NcclOpKey::NetRecv(0x123, 4, 8),
            String::from("host"),
            false,
            Arc::new(AtomicUsize::new(0)),
        );
        assert!(aggregated
            .attributes()
            .contains(&KeyValue::new("nccl.metric.aggregated", true)));
    }

    #[test]
    fn latency_histogram_close_comm() {
        const MAX_CARDINALITY: usize = 2;
        let mut manager = HistogramManager::new("test.latency.close", "ns", MAX_CARDINALITY);

        manager.get_histogram(NcclOpKey::NetSend(0x1, 0, 1));
        manager.get_histogram(NcclOpKey::NetRecv(0x1, 0, 1));
        let low = manager.get_histogram(NcclOpKey::NetSend(0x2, 0, 1));
        assert_eq!(manager.num_active(), MAX_CARDINALITY);
        assert!(!low.high_fidelity());

        // closing a comm frees its slots for later connections
        manager.close_comm(0x1);
        assert_eq!(manager.num_active(), 0);
        let h = manager.get_histogram(NcclOpKey::NetRecv(0x2, 0, 1));
        assert!(h.high_fidelity());
    }

    #[test]
    fn size_class_boundaries() {
        assert_eq!(size_class(0), "lt1m");
        assert_eq!(size_class((1 << 20) - 1), "lt1m");
        assert_eq!(size_class(1 << 20), "1m_16m");
        assert_eq!(size_class(16 << 20), "1m_16m");
        assert_eq!(size_class((16 << 20) + 1), "gt16m");
    }

    #[test]
    fn duration_histogram_cardinality() {
        const MAX_CARDINALITY: usize = 4;
        let mut histogram = DurationHistogram::new("test.duration", "ns", MAX_CARDINALITY);

        for comm_hash in 0..(MAX_CARDINALITY as u64 * 2) {
            histogram.record(1024, comm_hash, "ncclAllReduce", 0, 65536);
        }
        assert_eq!(histogram.num_active(), MAX_CARDINALITY);
        assert!(histogram.high_fidelity(0, "ncclAllReduce", 0, 65536));
        assert!(!histogram.high_fidelity(MAX_CARDINALITY as u64, "ncclAllReduce", 0, 65536));

        // repeated keys within the cap do not consume more slots, and the
        // rank is part of the key
        histogram.record(1024, 0, "ncclAllReduce", 0, 65536);
        histogram.record(1024, 0, "ncclAllReduce", 1, 65536);
        assert_eq!(histogram.num_active(), MAX_CARDINALITY);
        assert!(!histogram.high_fidelity(0, "ncclAllReduce", 1, 65536));
    }

    #[test]
    fn duration_histogram_update_priority() {
        const MAX_CARDINALITY: usize = 2;
        let mut histogram = DurationHistogram::new("test.duration.priority", "ns", MAX_CARDINALITY);

        // comms 0 and 1 take the slots first; 2 and 3 arrive later but are
        // more active in this interval
        for comm_hash in 0..4 {
            histogram.record(1024, comm_hash, "ncclAllReduce", 0, 65536);
        }
        for _ in 0..8 {
            histogram.record(1024, 2, "ncclAllReduce", 0, 65536);
            histogram.record(1024, 3, "ncclAllReduce", 0, 65536);
        }
        assert!(histogram.high_fidelity(0, "ncclAllReduce", 0, 65536));
        assert!(!histogram.high_fidelity(2, "ncclAllReduce", 0, 65536));

        histogram.update_priority();
        assert_eq!(histogram.num_active(), MAX_CARDINALITY);
        assert!(!histogram.high_fidelity(0, "ncclAllReduce", 0, 65536));
        assert!(!histogram.high_fidelity(1, "ncclAllReduce", 0, 65536));
        assert!(histogram.high_fidelity(2, "ncclAllReduce", 0, 65536));
        assert!(histogram.high_fidelity(3, "ncclAllReduce", 0, 65536));
    }

    #[test]
    fn duration_histogram_close_comm() {
        const MAX_CARDINALITY: usize = 2;
        let mut histogram = DurationHistogram::new("test.duration.close", "ns", MAX_CARDINALITY);

        histogram.record(1024, 1, "ncclAllReduce", 0, 65536);
        histogram.record(1024, 1, "ncclAllGather", 0, 65536);
        histogram.record(1024, 2, "ncclAllReduce", 0, 65536);
        assert_eq!(histogram.num_active(), MAX_CARDINALITY);
        assert!(!histogram.high_fidelity(2, "ncclAllReduce", 0, 65536));

        // closing a comm frees its slots for later comms
        histogram.close_comm(1);
        assert_eq!(histogram.num_active(), 0);
        histogram.record(1024, 2, "ncclAllReduce", 0, 1 << 24);
        assert!(histogram.high_fidelity(2, "ncclAllReduce", 0, 1 << 24));
    }

    #[test]
    fn gap_tracker_transitions() {
        let tracker = GapTracker::new();

        // no idle interval exists before the first activity completes
        assert_eq!(tracker.begin_transition(|| 100), None);
        tracker.activity_end(200);
        assert_eq!(tracker.begin_transition(|| 500), Some(300));

        // overlapping activity: only the 0 -> 1 transition observes a gap,
        // and the gap starts when the last in-flight activity ends
        assert_eq!(tracker.begin_transition(|| 600), None);
        tracker.activity_end(800);
        tracker.activity_end(700); // out-of-order end keeps the max stamp
        assert_eq!(tracker.begin_transition(|| 1000), Some(200));
        tracker.activity_end(1100);

        // a gap can never be negative
        assert_eq!(tracker.begin_transition(|| 1050), Some(0));
    }

    #[test]
    fn gap_tracker_stall_detection() {
        let tracker = GapTracker::new();
        assert!(!tracker.check_stalled());

        // a leaked window is reported once, after a whole quiet interval
        assert_eq!(tracker.begin_transition(|| 100), None);
        assert!(!tracker.check_stalled());
        assert!(tracker.check_stalled());
        assert!(!tracker.check_stalled());

        // ongoing activity is never mistaken for a stall
        let tracker = GapTracker::new();
        assert_eq!(tracker.begin_transition(|| 100), None);
        assert!(!tracker.check_stalled());
        tracker.activity_end(200);
        assert_eq!(tracker.begin_transition(|| 300), Some(100));
        assert!(!tracker.check_stalled());
    }
}
