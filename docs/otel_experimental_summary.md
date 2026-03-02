### **Experimental OpenTelemetry Integration (Summary of Recent Changes)**

**⚠️ IMPORTANT:** This branch is **experimental**.

This branch contains series of changes implements an OTLP (OpenTelemetry Protocol) integration to provide visibility into NCCL operations.

---

#### **1. Network Metrics & Cardinality Control**
Added a histogram metric (`nccl.net_send.latency`) for the completion time of each `isend()` operation.
To prevent "cardinality explosion" in large clusters, we use a **Top-K Histogram Manager** for network latency histograms.
*   **How it works:** Only the most active network paths retain full metadata (source/destination ranks), while less frequent paths are aggregated.
*   **Caveat:** This works best with skewed traffic. If pairwise traffic is **perfectly even**, the Top-K set may "jitter," making it harder to track specific rank pairs consistently. Also when the reshuffle duration is set to be very short, this method may end up creating more time series.

#### **2. Distributed Tracing for NCCL Collectives**
Enabled cluster-wide visualization of collectives by treating each communicator as a single Trace.
*   **The Mechanism:** Rank 0 coordinates a parent span for each collective, which all other ranks reference. This allows you to visualize the timing, duration, and rank-to-rank skew of a single operation (e.g., an AllReduce) across all nodes in a tracing backend.
*   **Caveat:** The implementation relies on stable metadata alignment across ranks. Large-scale tracing can generate significant data volume; ensure your OTLP collector is prepared for the throughput.

#### **3. Collective Sequence Monitoring**
Added a Gauge metric (`nccl.collective.seq_num`) to track the progress of collectives.
*   **Purpose:** By monitoring sequence numbers across ranks, you can immediately identify the specific node or rank that lags behind the rest of the cluster.

