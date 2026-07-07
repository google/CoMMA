# OpenTelemetry Support in CoMMA

CoMMA (Collective coMMunication Analyzer) profiler supports exporting telemetry data via OpenTelemetry (OTel). It supports both metrics (latency histograms and status gauges) and tracing (NCCL operation spans).

## Configuration

OTel support is configured via environment variables.

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NCCL_PROFILER_OTEL_ENABLE` | Boolean | `false` | Enables OpenTelemetry support. |
| `NCCL_PROFILER_OTEL_TRACE_NCCLOP` | Boolean | `false` | (Experimental) Enables tracing for NCCL operations. Requires `NCCL_PROFILER_OTEL_ENABLE` to be `true`. |
| `NCCL_PROFILER_OTEL_METRICS_MAX_CARDINALITY` | Integer | `0` | Maximum number of unique metric streams (cardinality limit) for high-fidelity tracking. If set to `0`, all metrics use low-fidelity aggregation. |
| `NCCL_PROFILER_OTEL_METRICS_CARDINALITY_GROUPING_INTERVAL` | Duration | `3600s` | Interval at which CoMMA updates the priority ("Top K") of metrics for cardinality management. |
| `NCCL_PROFILER_OTEL_LATENCY_HISTOGRAM_MAX_SIZE` | Integer | `160` | Maximum size parameter for OTel Base2 Exponential Histogram. |
| `NCCL_PROFILER_OTEL_LATENCY_HISTOGRAM_MAX_SCALE` | Integer | `20` | Maximum scale parameter for OTel Base2 Exponential Histogram. |

### Duration Format
Duration fields (like `NCCL_PROFILER_OTEL_METRICS_CARDINALITY_GROUPING_INTERVAL`) support values with units:
- `d`: Days
- `h`: Hours
- `m`: Minutes
- `s`: Seconds
- `ms`: Milliseconds
- `us`: Microseconds
- `ns`: Nanoseconds

Example: `1h30m` or `10s`.

## Metrics

CoMMA registers a meter provider under the service name `CoMMA`. It exports the following metrics:

### `nccl.net.send.latency` (Histogram, Unit: `ns`)

This metric records the latency of network send operations (also known as `isend()` or "proxy step").

To prevent high cardinality issues, CoMMA uses a "Top K" cardinality management strategy if `NCCL_PROFILER_OTEL_METRICS_MAX_CARDINALITY` is configured to a value greater than 0:

- **High-Fidelity Streams (Top K)**: The most frequent network send operations (up to the configured limit) are recorded with full attributes:
  - `nccl.communicator.hash`: Hexadecimal string identifying the NCCL communicator.
  - `nccl.source.rank`: Source rank of the transfer.
  - `nccl.destination.rank`: Destination rank of the transfer.
  - `nccl.hostname`: Hostname of the node.
- **Aggregated Streams**: Streams exceeding the cardinality limit are aggregated together to save memory and export bandwidth. They are recorded with:
  - `nccl.metric.aggregated`: Set to `true`.
  - `nccl.hostname`: Hostname of the node.

The "Top K" list is dynamically updated at the interval defined by `NCCL_PROFILER_OTEL_METRICS_CARDINALITY_GROUPING_INTERVAL`, and keys of closed communicators are evicted immediately.

Note that the OTel SDK exports with cumulative temporality and keeps a stream for every attribute set it has ever recorded until process exit — including streams for keys that were later demoted from the "Top K" set. CoMMA's cardinality limit bounds how many keys record with high-fidelity attributes at any time, but under communicator churn the SDK-side stream count grows with the total number of distinct attribute sets ever promoted. Keep `NCCL_PROFILER_OTEL_METRICS_MAX_CARDINALITY` modest when communicators churn frequently. This applies to every "Top K" managed metric below.

### `nccl.net_recv.latency` (Histogram, Unit: `ns`)

This metric records the latency of network receive operations. Receive steps are only generated when `NCCL_PROFILER_TRACK_RECV_STEPS` is enabled (subject to `NCCL_PROFILER_P2P_RECV_SAMPLE_RATE` for point-to-point operations).

It uses the same "Top K" cardinality management strategy and attributes as `nccl.net_send.latency`. For receive connections, `nccl.source.rank` is the remote sender and `nccl.destination.rank` is the local rank.

A receive step spans buffer post to data arrival, so its duration includes any delay before the remote sender was ready — unlike send steps, which measure post-clearance transfer time. The two metrics are therefore not symmetric; receive latencies are most meaningful compared across edges rather than against send latencies.

### `nccl.collective.duration` (Histogram, Unit: `ns`)

This metric records, for each completed NCCL operation, the span of its network / kernel activity: from the start of its first proxy op (or kernel channel, when `NCCL_PROFILER_TRACK_KERNEL_CH` is enabled) to the end of its last one. This is the same operation lifetime CoMMA computes for traces and summaries. Four classes of operations are not recorded:

- Operations that produce no proxy op or kernel channel activity, e.g. single-node NVLink-only collectives with kernel channel tracking disabled.
- Point-to-point operations not selected by sampling (`NCCL_PROFILER_P2P_SAMPLE_RATE`, `NCCL_PROFILER_P2P_RECV_SAMPLE_RATE`). With the default recv sample rate of `0.1`, `ncclRecv` durations are a 10% sample: bucket shapes are unbiased, but counts and sums under-report by 10x, and asymmetrically versus `ncclSend`.
- Operations reclaimed by the hang timeout (`NCCL_PROFILER_NCCLOP_TIMEOUT`) before completing.
- Operations on the small-message fast paths: point-to-point operations at or below `NCCL_PROFILER_SMALL_MSG_THRESHOLD` (default 64 KiB) are never tracked, and with the default `NCCL_PROFILER_SKIP_SMALL_COLLECTIVE=true` collectives other than AllReduce at or below the threshold are not tracked either. The `lt1m` size class therefore only covers operations outside these fast paths.

Attributes:
- `nccl.comm.hash`: Hexadecimal string identifying the NCCL communicator.
- `nccl.collective.name`: Name of the operation (e.g., `ncclAllReduce`, `ncclSend`).
- `nccl.rank`: Rank of the process within the communicator.
- `nccl.size.class`: Coarse operation size class: `lt1m` (< 1 MiB), `1m_16m` (1-16 MiB), or `gt16m` (> 16 MiB). This preserves the latency-bound vs bandwidth-bound split with bounded cardinality.
- `nccl.hostname`: Hostname of the node.

Distinct attribute sets are capped by `NCCL_PROFILER_OTEL_METRICS_MAX_CARDINALITY` with the same "Top K" strategy as the latency metrics: keys are re-ranked by activity at every `NCCL_PROFILER_OTEL_METRICS_CARDINALITY_GROUPING_INTERVAL`, keys of closed communicators are evicted, and operations outside the top K are recorded with `nccl.metric.aggregated` set to `true`.

### `nccl.collective.gap` (Histogram, Unit: `ns`)

This metric records the duration of intervals where the process has no NCCL activity in flight. Activity is the union of operation enqueue windows, proxy op windows (network transfers), and kernel channel windows (when `NCCL_PROFILER_TRACK_KERNEL_CH` is enabled), so an operation keeps the process busy from its API call until its network / kernel work actually completes, not merely until it is enqueued. Each idle interval is recorded once, when the activity that ends it starts. Overlapping activity never contributes to a gap.

Attributes:
- `nccl.hostname`: Hostname of the node.
- `nccl.pid`: Process id. Together with the hostname this identifies the process; a comm-local rank would be ambiguous for a metric that spans communicators.

The gap is a per-process property by definition, so it deliberately carries no communicator dimension.

Caveats — what counts as activity depends on the tracking configuration:
- The metric assumes the default `NCCL_PROFILER_TRACK_NCCLOP=true`. With operation tracking disabled, only proxy op windows (and kernel channel windows, when enabled) count as activity, and everything else is reported as gap.
- Operations whose execution produces no tracked activity window (see the duration carve-outs above) are only counted while they are enqueued, so their execution time appears as gap. Enable kernel channel tracking for full coverage of NVLink-only collectives.
- Operations on the small-message fast paths (see the duration carve-outs above) are covered only while enqueued — small point-to-point operations not at all — so their transfer time, possibly tens of microseconds each, is reported as gap. Interpret gap totals with care for workloads dominated by messages below `NCCL_PROFILER_SMALL_MSG_THRESHOLD`.
- Point-to-point operations not selected by sampling (`NCCL_PROFILER_P2P_SAMPLE_RATE`, `NCCL_PROFILER_P2P_RECV_SAMPLE_RATE`) produce no activity window at all, so their entire enqueue and transfer time is reported as gap. With the default recv sample rate of `0.1`, 90% of receive transfers count as idle time; set the sample rates to `1.0` before interpreting gap totals for p2p-heavy workloads such as pipeline parallelism.
- Sub-microsecond gaps can appear between an operation's enqueue window and the start of its proxy activity; they land in the lowest buckets and carry negligible weight in gap-time totals.
- If NCCL aborts a plan launch on an error path it may never stop the events it started; the in-flight count then stays above zero and the metric reports no further gaps for the process lifetime. CoMMA logs a one-time warning when it detects a stuck in-flight count. This only occurs after NCCL errors, which the job surfaces on its own.

### `nccl.collective.seq_num` (Gauge)

This metric records the sequence number of the last collective operation executed.

Attributes:
- `nccl.comm.hash`: Hexadecimal string identifying the NCCL communicator.
- `nccl.collective.name`: Name of the collective operation (e.g., `ncclAllReduce`, `ncclBroadcast`).
- `nccl.rank`: Rank of the process.
- `nccl.hostname`: Hostname of the node.

## Tracing (Experimental)

> [!WARNING]
> Tracing support in CoMMA is currently experimental and may be subject to future changes.

When tracing is enabled (via `NCCL_PROFILER_OTEL_TRACE_NCCLOP`), CoMMA exports spans for NCCL operations.

### Span Correlation Across Ranks

For collective operations, CoMMA attempts to correlate spans across different ranks participating in the same collective:

- A deterministic `TraceId` is generated using the communicator hash.
- A deterministic `SpanId` is generated using the collective operation type and its sequence number.
- Rank 0 of the communicator creates the parent span (Server span) with the duration of the operation on rank 0.
- Other ranks create child spans linked to this parent span using the remote span context.

This allows visualization of the collective operation as a single distributed trace.

### Span Attributes

Spans contain the following attributes:

- **Common Attributes**:
  - `nccl.comm.hash`: Communicator hash.
  - `nccl.rank`: Rank of the process.
  - `nccl.size.bytes`: Size of the data transferred.

- **Collective Operation Attributes** (e.g., `ncclAllReduce`, `ncclBroadcast`):
  - `nccl.collective.algo`: NCCL algorithm used (e.g., Tree, Ring).
  - `nccl.collective.proto`: NCCL protocol used (e.g., LL, LL128, Simple).
  - `nccl.collective.n_max_channel`: Number of channels used.

- **P2P Operation Attributes** (`ncclSend`, `ncclRecv`):
  - `nccl.p2p.peer.rank`: Peer rank involved in the transfer.
