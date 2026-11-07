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

use crate::event;
use crate::event::ProfilerEvent as _;
use crate::fixed_batch;
use crate::nccl_metadata;
use crate::profiler;
use crate::profiler::Communicator;
use crate::profiler::Profiler;
use crate::shm_fifo;
use crate::slab;
use crate::spsc;
use crate::step_tracker::EventStep;

use log::error;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub trait Daemon {
    fn new(profiler: &'static Profiler) -> Self;
}

const RETRY_MS: u64 = 100;

pub trait Export {
    fn export(&self, ctx: &mut PollingContext, maybe_retry_ms: Option<u64>);
    fn get_latency_histogram(
        &self,
        _key: &nccl_metadata::NcclOpKey,
    ) -> Option<impl std::iter::IntoIterator<Item = Arc<dyn AtomicHistogram<EventStep>>>> {
        None::<Vec<_>>
    }
}

pub trait AtomicHistogram<T>: std::fmt::Debug + Sync + Send {
    fn record(&self, data: &T);
}

#[derive(Debug)]
pub struct FifoReceiver<T> {
    last_fetch: Instant,
    receiver: spsc::Receiver<T>,
    pending_msg: VecDeque<T>,
}

impl<T> FifoReceiver<T> {
    pub fn new(receiver: spsc::Receiver<T>) -> Self {
        Self {
            last_fetch: Instant::now(),
            receiver,
            pending_msg: VecDeque::new(),
        }
    }

    pub fn fetch_from_sender(&mut self) {
        self.receiver.try_fetch_from_sender();
    }

    pub fn recv_many(
        &mut self,
        max_idle: std::time::Duration,
        max_recv: usize,
        force_fetch: bool,
    ) -> usize {
        let mut fetched = false;

        let now = Instant::now();
        if force_fetch || now - self.last_fetch > max_idle {
            self.fetch_from_sender();
            fetched = true;
        }
        let mut n_recv = 0;
        while let Some(msg) = self.receiver.recv() {
            fetched = true;
            self.pending_msg.push_back(msg);
            n_recv += 1;
            if n_recv > max_recv {
                break;
            }
        }

        if fetched {
            self.last_fetch = Instant::now();
        }
        n_recv
    }

    pub fn process_many<H>(&mut self, max_process: usize, mut handler: H)
    where
        H: FnMut(T),
    {
        let mut n_processed = 0;
        while let Some(msg) = self.pending_msg.pop_front() {
            handler(msg);
            n_processed += 1;
            if n_processed >= max_process {
                break;
            }
        }
    }
}

#[derive(Debug)]
pub enum ControlMessage {
    NewThread(ThreadControl),
}

#[derive(Debug)]
pub enum Message {
    Group(Box<event::Group>),
    NcclOp(slab::AllocatedNode<event::NcclOp>),
    ProxyOpLite(
        /* start time */ Instant,
        /* duration ns */ u64,
        event::ProxyOpInfo,
    ),
    ProxyOpExtra(event::ProxyOpExtra),
    ProxyOp(/* id = */ u32),
    StepBatch(
        event::ProxyOpInfo,
        slab::AllocatedNode<StepBatch>,
        /* is_last =*/ bool,
    ),
    KernelCh(
        /* end time */ Instant,
        /* parent handle */ usize,
        profiler::KernelChInfo,
    ),
    CommOpen(Communicator),
    CommClose(/* comm_hash = */ u64),
}

#[derive(Debug)]
pub enum InterProcessMessage {
    ProxyOpStart(
        /* handle: */ usize,
        /* pid: */ libc::pid_t,
        /* thread idx: */ usize,
        /* id: */ u32,
        Instant,
    ),
    ProxyOpEnd(/* handle: */ usize, Instant),
    ReplyProxyOpComm(/* thread idx */ usize, /* proxy id = */ u32, u64),
}

pub const STEP_BATCH_SZ: usize = 64;
pub type StepBatch = fixed_batch::Batch<EventStep, STEP_BATCH_SZ>;

/// This separate Telemetry type should make a copy of the Message type.
/// Doing so decouples:
///   1. event management of those shared profiler API handler,
///      which has higher performance requirements
///   2. resource used by the telemetry exporter, which may be blocking for I/O
pub enum Telemetry {
    Group(Box<event::Group>),
    NcclOpIssued(Box<event::NcclOp>), // copybara:strip(hang detection)
    NcclOp(Box<event::NcclOp>),
    ProxyOp(Box<event::ProxyOp>),
    CommOpen(Communicator),
    CommClose(/* comm_hash = */ u64),
}

pub struct PollingContext<'a> {
    profiler: &'a Profiler,
    pub ncclops: HashMap<usize, Box<event::NcclOp>>,
    reclaim_heap: BinaryHeap<Reverse<(Instant, usize)>>,
    pub pending_telemetry: VecDeque<Telemetry>,
    stop: Arc<AtomicBool>,

    free_ncclop: slab::FreeList<event::NcclOp>,
    free_proxyop: slab::FreeList<event::ProxyOp>,
    free_step_batch: slab::FreeList<StepBatch>,

    peer_rank_fifo: HashMap<libc::pid_t, shm_fifo::mpsc::Sender<InterProcessMessage>>,
    pending_ipc_msg: HashMap<libc::pid_t, VecDeque<InterProcessMessage>>,
}

impl<'a> PollingContext<'a> {
    pub fn new(profiler: &'a Profiler, stop_signal: Arc<AtomicBool>) -> Self {
        Self {
            profiler,
            ncclops: HashMap::new(),
            reclaim_heap: BinaryHeap::new(),
            pending_telemetry: VecDeque::new(),
            stop: stop_signal,
            free_ncclop: slab::FreeList::default(),
            free_proxyop: slab::FreeList::default(),
            free_step_batch: slab::FreeList::default(),
            peer_rank_fifo: HashMap::new(),
            pending_ipc_msg: HashMap::new(),
        }
    }

    fn get_ncclop(&mut self, id: usize) -> Option<&mut event::NcclOp> {
        self.ncclops.get_mut(&id).map(|b| b.as_mut())
    }

    fn get_ipc_fifo(
        &mut self,
        pid: libc::pid_t,
    ) -> Option<&mut shm_fifo::mpsc::Sender<InterProcessMessage>> {
        if self.profiler.config.track_interprocess_proxyop {
            use std::collections::hash_map::Entry;
            match self.peer_rank_fifo.entry(pid) {
                Entry::Vacant(e) => {
                    let tx = shm_fifo::mpsc::Sender::new(&ipc_shm_path(pid));
                    if tx.is_err() {
                        return None;
                    }
                    Some(e.insert(tx.unwrap()))
                }
                Entry::Occupied(e) => Some(e.into_mut()),
            }
        } else {
            None
        }
    }

    fn try_commit_ipc(&mut self) {
        if self.profiler.config.track_interprocess_proxyop {
            for (_, fifo) in self.peer_rank_fifo.iter_mut() {
                let _ = fifo.try_commit();
            }
        }
    }

    fn append_ipc_message(&mut self, pid: libc::pid_t, msg: InterProcessMessage) {
        self.pending_ipc_msg.entry(pid).or_default().push_back(msg);
    }

    // copybara:strip_begin(gpuviz)
    fn try_add_histogram<E>(
        &mut self,
        thread_state: &mut ThreadState,
        exporter: &mut E,
        ncclop_id: usize,
        proxyop_id: u32,
    ) -> Option<()>
    where
        E: Export,
    {
        let ncclop = self.get_ncclop(ncclop_id)?;
        let op_key = ncclop.op_key();
        let comm_hash = op_key.get_comm_hash();
        let proxyop = thread_state.proxyops.get_mut(&proxyop_id)?;
        if proxyop.info().rank != op_key.local_rank() {
            static LOG_TIMER: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
            if let Ok(mut lg) = LOG_TIMER.lock() {
                let should_log = if let Some(t) = *lg {
                    t.elapsed().as_secs() >= 60
                } else {
                    true
                };

                if should_log {
                    log::error!(
                        "Found rank mismatch between proxyop and its parent: {} vs {}",
                        proxyop.info().rank,
                        op_key.local_rank()
                    );
                    *lg = Some(Instant::now());
                }
            }
        }
        self.try_add_histogram_from_comm_hash(thread_state, exporter, proxyop_id, comm_hash)
    }

    fn try_add_histogram_from_comm_hash<E>(
        &mut self,
        thread_state: &mut ThreadState,
        exporter: &mut E,
        proxyop_id: u32,
        comm_hash: u64,
    ) -> Option<()>
    where
        E: Export,
    {
        let proxyop = if let Some(p) = thread_state.proxyops.get_mut(&proxyop_id) {
            p
        } else {
            &mut thread_state
                .pending_comm_hash_proxyops
                .get_mut(&proxyop_id)?
                .0
        };
        if proxyop.has_step_histograms() {
            return None;
        }
        let conn_key = proxyop.info().op_key(comm_hash);
        if let Some(h_list) = exporter.get_latency_histogram(&conn_key) {
            proxyop.set_step_histograms(h_list.into_iter().map(|h| h as _).collect());
        }
        Some(())
    }
    // copybara:strip_end

    fn should_wait_for_comm_hash(&self, proxyop: &event::ProxyOp) -> bool {
        self.profiler.config.track_interprocess_proxyop
            && proxyop.aggregate_steps()
            && proxyop.info().pid != self.profiler.pid
            && proxyop.info().parent().is_some()
            && !proxyop.has_step_histograms()
    }

    fn add_proxyop<E>(
        &mut self,
        thread_state: &mut ThreadState,
        exporter: &mut E,
        info: &event::ProxyOpInfo,
        start_time: Instant,
    ) where
        E: Export,
    {
        let id = info.id;
        let mut op = Box::new(event::ProxyOp::from_info(info, start_time));
        let mut aggregate_steps = false;
        // only aggregate steps when:
        // 0. aggregate_steps flag is set
        // 1. we know the parent (and therefore comm hash)
        // 2. this proxyop is originated from current process OR
        //    we are tracking interprocess proxyop
        if info.parent().is_some() {
            aggregate_steps = self.profiler.config.aggregate_steps
                && (info.pid == self.profiler.pid
                    || self.profiler.config.track_interprocess_proxyop);
        }
        op.init_step_tracking(self.profiler.config.track_steps, aggregate_steps);
        self.handle_proxyop_start(thread_state, info, op.basic_info().start_time());

        thread_state.proxyops.insert(id, op);
        // copybara:strip_begin(gpuviz)
        if let Some(parent) = info.parent() {
            if aggregate_steps && info.pid == self.profiler.pid {
                if self.ncclops.contains_key(&parent) {
                    self.try_add_histogram(thread_state, exporter, parent, id);
                } else if let Some(comm_hash) = info.comm_hash {
                    // The parent ncclop is no longer tracked (e.g. it was reclaimed due to
                    // timeout before this proxyop arrived). Fall back to the comm hash recorded
                    // on the proxyop itself so the steps are still aggregated.
                    self.try_add_histogram_from_comm_hash(thread_state, exporter, id, comm_hash);
                }
            }
        }
        // copybara:strip_end
    }

    fn handle_proxyop_start(
        &mut self,
        thread_state: &mut ThreadState,
        info: &event::ProxyOpInfo,
        time: Instant,
    ) {
        if let Some(parent) = info.parent() {
            let pid = info.pid;
            if pid == self.profiler.pid {
                if let Some(ncclop) = self.get_ncclop(parent) {
                    ncclop_update(ncclop, time, None);
                }
            } else {
                let msg = InterProcessMessage::ProxyOpStart(
                    parent,
                    self.profiler.pid,
                    thread_state.idx,
                    info.id,
                    time,
                );
                self.append_ipc_message(pid, msg);
            }
        }
    }

    #[allow(clippy::vec_box)]
    fn handle_proxyop_end(
        &mut self,
        end_time: Instant,
        info: &event::ProxyOpInfo,
        proxyops: Vec<Box<event::ProxyOp>>,
        send_ipc: bool,
    ) {
        let config = &self.profiler.config;
        let record_proxyop = config.track_proxyop || config.track_steps;
        if let Some(parent_handle) = info.parent() {
            if info.pid == self.profiler.pid {
                if let Some(ncclop) = self.ncclops.get_mut(&parent_handle) {
                    ncclop_update(ncclop, end_time, Some(end_time));
                    if record_proxyop {
                        for p in proxyops {
                            ncclop.add_proxyop(*p);
                        }
                    }
                    let deadline = end_time + config.ncclop_completion_delay;
                    self.reclaim_heap.push(Reverse((deadline, parent_handle)));
                    return;
                }
                // The parent ncclop is no longer tracked (e.g. it was reclaimed due to timeout
                // before this proxyop ended). Export the orphan proxyop on its own below.
            } else if send_ipc {
                let msg = InterProcessMessage::ProxyOpEnd(parent_handle, end_time);
                self.append_ipc_message(info.pid, msg);
            }
        }
        if record_proxyop {
            for p in proxyops {
                self.pending_telemetry.push_back(Telemetry::ProxyOp(p));
            }
        }
    }

    fn finalize_pending_comm_hash_proxyop(&mut self, proxyop: Box<event::ProxyOp>) {
        let info = proxyop.info().clone();
        let end_time = proxyop.basic_info().end_time().unwrap_or_else(Instant::now);
        self.handle_proxyop_end(end_time, &info, vec![proxyop], false);
    }

    /// Construct a Telemetry type from Message.
    /// Resource used by `msg` should be release ASAP so the profiler API handlers won't be blocked on
    /// them
    fn handle_fifo_message<E>(
        &mut self,
        msg: Message,
        thread_state: &mut ThreadState,
        exporter: &mut E,
    ) where
        E: Export,
    {
        match msg {
            Message::Group(group) => {
                if self.profiler.config.track_group {
                    self.pending_telemetry.push_back(Telemetry::Group(group));
                }
            }
            Message::NcclOp(op) => {
                let id = op.id();
                let start_time = op.basic_info().start_time();
                let op = self.free_ncclop.take_and_free(op);
                let deadline = start_time + self.profiler.config.ncclop_timeout;
                self.reclaim_heap.push(Reverse((deadline, id)));
                let _ = self.ncclops.insert(id, Box::new(op.clone()));
                // copybara:strip_begin(hang detection)
                self.pending_telemetry
                    .push_back(Telemetry::NcclOpIssued(Box::new(op)));
                // copybara:strip_end
                if self.free_ncclop.num_free() >= slab::FREELIST_BATCH {
                    self.free_ncclop.try_publish(&self.profiler.free_ncclop);
                }
            }
            Message::ProxyOpLite(start_time, dur_ns, info) => {
                self.handle_proxyop_start(thread_state, &info, start_time);
                let end_time = start_time + Duration::from_nanos(dur_ns);
                self.handle_proxyop_end(end_time, &info, Vec::new(), true);
            }
            Message::ProxyOpExtra(extra) => {
                thread_state.aux_msg.push(Message::ProxyOpExtra(extra));
            }
            Message::ProxyOp(id) => {
                let proxyop = thread_state.proxyops.remove(&id);
                let extra = if let Some(Message::ProxyOpExtra(_)) = thread_state.aux_msg.last() {
                    let msg = thread_state.aux_msg.pop().unwrap();
                    if let Message::ProxyOpExtra(extra) = msg {
                        Some(extra)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(mut proxyop) = proxyop {
                    if let Some(extra) = extra {
                        proxyop.add_extra_info(extra);
                    }
                    let info = proxyop.info().clone();
                    let end_time = proxyop.basic_info().end_time().unwrap_or_else(Instant::now);

                    let mut ops = Vec::new();
                    if self.should_wait_for_comm_hash(&proxyop) {
                        thread_state
                            .pending_comm_hash_proxyops
                            .insert(id, (proxyop, Instant::now()));
                    } else {
                        ops.push(proxyop);
                    }
                    self.handle_proxyop_end(end_time, &info, ops, true);
                } else {
                    // error!("proxyop {} not found!", id);
                }
            }
            Message::StepBatch(info, mut steps, is_last) => {
                if !thread_state.proxyops.contains_key(&info.id) {
                    self.add_proxyop(
                        thread_state,
                        exporter,
                        &info,
                        self.profiler.init_instant + Duration::from_nanos(steps[0].start_time),
                    );
                }
                let proxyop = thread_state.proxyops.get_mut(&info.id).unwrap();
                let init_instant = self.profiler.init_instant;
                for step in steps.drain() {
                    let start_time = init_instant + Duration::from_nanos(step.start_time as _);
                    let end_time = init_instant + Duration::from_nanos(step.end_time_ns() as _);
                    proxyop.try_set_start_time(start_time);
                    proxyop.set_end_time(end_time);
                    proxyop.add_step(step);
                }
                if is_last {
                    let mut proxyop = thread_state.proxyops.remove(&info.id).unwrap();
                    let extra = if let Some(Message::ProxyOpExtra(_)) = thread_state.aux_msg.last()
                    {
                        let msg = thread_state.aux_msg.pop().unwrap();
                        if let Message::ProxyOpExtra(extra) = msg {
                            Some(extra)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(extra) = extra {
                        proxyop.add_extra_info(extra);
                    }
                    let end_time = proxyop.basic_info().end_time().unwrap_or_else(Instant::now);

                    let mut ops = Vec::new();
                    if self.should_wait_for_comm_hash(&proxyop) {
                        thread_state
                            .pending_comm_hash_proxyops
                            .insert(info.id, (proxyop, Instant::now()));
                    } else {
                        ops.push(proxyop);
                    }
                    self.handle_proxyop_end(end_time, &info, ops, true);
                }
                self.free_step_batch.free(steps);
                if self.free_step_batch.num_free() >= slab::FREELIST_BATCH {
                    self.free_step_batch
                        .try_publish(&self.profiler.free_step_batch);
                }
            }
            Message::KernelCh(end_time, parent, info) => {
                let init_instant = self.profiler.init_instant;
                let dur = Duration::from_nanos(info.stop_timer.saturating_sub(info.start_timer));
                let start_time = end_time
                    .checked_sub(dur)
                    .unwrap_or(init_instant)
                    .max(init_instant);
                if let Some(ncclop) = self.get_ncclop(parent) {
                    ncclop_update(ncclop, start_time, Some(end_time));
                    ncclop.add_kernel_ch(info);
                    let deadline = end_time + self.profiler.config.ncclop_completion_delay;
                    self.reclaim_heap.push(Reverse((deadline, parent)));
                }
            }
            Message::CommOpen(comm) => {
                self.pending_telemetry.push_back(Telemetry::CommOpen(comm));
            }
            Message::CommClose(comm_hash) => {
                self.pending_telemetry
                    .push_back(Telemetry::CommClose(comm_hash));
            }
        }
    }

    fn handle_ipc_message<E>(
        &mut self,
        /* copybara:strip_begin(gpuviz) */
        threads: &mut [ThreadControl],
        exporter: &mut E,
        /* copybara:strip_end_and_replace
        _threads: &mut [ThreadControl],
        _exporter: &mut E,
         */
        msg: InterProcessMessage,
    ) where
        E: Export,
    {
        match msg {
            InterProcessMessage::ProxyOpStart(handle, pid, thread_idx, id, time) => {
                if let Some(ncclop) = self.get_ncclop(handle) {
                    ncclop_update(ncclop, time, None);
                    let comm = ncclop.op_key().get_comm_hash();
                    let reply = InterProcessMessage::ReplyProxyOpComm(thread_idx, id, comm);
                    self.append_ipc_message(pid, reply);
                }
            }
            InterProcessMessage::ProxyOpEnd(handle, time) => {
                if let Some(ncclop) = self.get_ncclop(handle) {
                    ncclop_update(ncclop, time, Some(time));
                    let deadline = time + self.profiler.config.ncclop_completion_delay;
                    self.reclaim_heap.push(Reverse((deadline, handle)));
                }
            }
            /* copybara:strip_begin(gpuviz) */
            InterProcessMessage::ReplyProxyOpComm(thread_idx, id, comm) => {
                let t = &mut threads[thread_idx];
                if self.profiler.config.aggregate_steps {
                    let _ = self.try_add_histogram_from_comm_hash(
                        &mut t.daemon_state,
                        exporter,
                        id,
                        comm,
                    );
                    // If the proxyop has already ended and was waiting for this comm_hash,
                    // finalize it now.
                    if let Some((proxyop, _)) =
                        t.daemon_state.pending_comm_hash_proxyops.remove(&id)
                    {
                        self.finalize_pending_comm_hash_proxyop(proxyop);
                    }
                }
            } /* copybara:strip_end_and_replace
              InterProcessMessage::ReplyProxyOpComm(_, _, _) => {}
              */
        }
    }

    pub fn reclaim_expired_ncclops(&mut self, now: Instant) {
        let comp_delay = self.profiler.config.ncclop_completion_delay;
        let timeout = self.profiler.config.ncclop_timeout;

        while let Some(&Reverse((deadline, id))) = self.reclaim_heap.peek() {
            if deadline <= now {
                self.reclaim_heap.pop();
                let actual_deadline = match self.ncclops.get(&id) {
                    Some(op) => match op.basic_info().end_time() {
                        Some(et) => Some(et + comp_delay),
                        None => Some(op.basic_info().start_time() + timeout),
                    },
                    None => None,
                };
                if let Some(actual_dl) = actual_deadline {
                    if now >= actual_dl {
                        if let Some(op) = self.ncclops.remove(&id) {
                            self.pending_telemetry.push_back(Telemetry::NcclOp(op));
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    pub fn reclaim_all_ncclops_in_map(&mut self) {
        self.pending_telemetry
            .extend(self.ncclops.drain().map(|(_, op)| Telemetry::NcclOp(op)));
        self.reclaim_heap.clear();
    }
}

/// control data used by each thread.
///
/// This struct is used by the daemon thread to communicate with each thread.
#[derive(Debug)]
pub struct ThreadControl {
    pub ncclop_fifo: FifoReceiver<Message>,
    pub fifo: FifoReceiver<Message>,
    daemon_state: ThreadState,
}

#[derive(Debug, Default)]
struct ThreadState {
    idx: usize,
    proxyops: HashMap<u32, Box<event::ProxyOp>>,
    pending_comm_hash_proxyops: HashMap<u32, (Box<event::ProxyOp>, Instant)>,
    aux_msg: Vec<Message>,
}

impl ThreadControl {
    pub fn new(ncclop_fifo: FifoReceiver<Message>, fifo: FifoReceiver<Message>) -> Self {
        Self {
            ncclop_fifo,
            fifo,
            daemon_state: ThreadState::default(),
        }
    }
}

fn ncclop_update(op: &mut event::NcclOp, st: Instant, et: Option<Instant>) {
    op.update_child_start_time(st);
    if let Some(t) = et {
        op.basic_info_mut().update_end_time(t);
    }
}

fn ipc_shm_path(pid: libc::pid_t) -> String {
    format!("nccl-profiler-{}", pid)
}

const FIFO_FETCH_INTERVAL: Duration = Duration::from_secs(1);
const FIFO_PROCESS_BATCH: usize = 512;
const FIFO_RECV_BATCH: usize = profiler::EVENT_QUEUE_SZ;
const IPC_RECV_BATCH: usize = 512;
const IPC_SEND_BATCH: usize = 64;

pub fn polling_loop<E>(ctx: &mut PollingContext, mut exporter: E)
where
    E: Export,
{
    let mut threads: Vec<ThreadControl> = Vec::new();

    let mut ipc_fifo_rx: Option<shm_fifo::mpsc::Receiver<InterProcessMessage>> =
        if ctx.profiler.config.track_interprocess_proxyop {
            let path = ipc_shm_path(ctx.profiler.pid);
            let rx = shm_fifo::mpsc::Receiver::new(&path, 1024);
            if rx.is_err() {
                error!("failed to create shm fifo {}", path);
            }
            rx.ok()
        } else {
            None
        };

    let ncclop_reclaim_interval = ctx.profiler.config.ncclop_reclaim_interval;
    let mut last_reclaim = Instant::now();

    while !ctx.stop.load(Ordering::Acquire) {
        while let Some(ctrl_msg) = ctx.profiler.ctrl_fifo.pop() {
            match ctrl_msg {
                ControlMessage::NewThread(mut ctrl) => {
                    ctrl.daemon_state.idx = threads.len();
                    threads.push(ctrl);
                }
            }
        }

        let mut n_recv = 0;
        for thread in threads.iter_mut() {
            n_recv += thread
                .fifo
                .recv_many(FIFO_FETCH_INTERVAL, FIFO_RECV_BATCH, false);
        }

        let mut ipc_msg = Vec::new();
        if let Some(rx) = ipc_fifo_rx.as_mut() {
            while let Some(msg) = rx.recv() {
                ipc_msg.push(msg);
                if ipc_msg.len() > IPC_RECV_BATCH {
                    break;
                }
            }
        }

        n_recv += ipc_msg.len();

        for thread in threads.iter_mut() {
            thread
                .ncclop_fifo
                .recv_many(FIFO_FETCH_INTERVAL, FIFO_RECV_BATCH, n_recv > 0);
            thread.ncclop_fifo.process_many(FIFO_PROCESS_BATCH, |msg| {
                ctx.handle_fifo_message(msg, &mut thread.daemon_state, &mut exporter);
            });
        }

        for thread in threads.iter_mut() {
            thread.fifo.process_many(FIFO_PROCESS_BATCH, |msg| {
                ctx.handle_fifo_message(msg, &mut thread.daemon_state, &mut exporter);
            });
        }

        for msg in ipc_msg {
            ctx.handle_ipc_message(&mut threads, &mut exporter, msg);
        }

        let mut pending_ipc_msg = std::mem::take(&mut ctx.pending_ipc_msg);
        pending_ipc_msg.retain(|pid, msgs| {
            if let Some(tx) = ctx.get_ipc_fifo(*pid) {
                let mut n_sent = 0;
                while let Some(msg) = msgs.pop_front() {
                    if let Err(m) = tx.send(msg) {
                        msgs.push_front(m);
                        break;
                    }
                    n_sent += 1;
                    if n_sent >= IPC_SEND_BATCH {
                        break;
                    }
                }
                return !msgs.is_empty();
            }
            true
        });

        ctx.pending_ipc_msg = pending_ipc_msg;

        ctx.try_commit_ipc();

        let comm_hash_ipc_timeout = ctx.profiler.config.comm_hash_ipc_timeout;
        for thread in threads.iter_mut() {
            thread.daemon_state.pending_comm_hash_proxyops =
                std::mem::take(&mut thread.daemon_state.pending_comm_hash_proxyops)
                    .into_iter()
                    .flat_map(|(id, (proxyop, start_time))| {
                        if start_time.elapsed() > comm_hash_ipc_timeout {
                            ctx.finalize_pending_comm_hash_proxyop(proxyop);
                            None
                        } else {
                            Some((id, (proxyop, start_time)))
                        }
                    })
                    .collect();
        }

        if ctx.free_proxyop.num_free() >= slab::FREELIST_BATCH {
            ctx.free_proxyop.try_publish(&ctx.profiler.free_proxyop);
        }

        let now = Instant::now();
        if now.duration_since(last_reclaim) >= ncclop_reclaim_interval {
            ctx.reclaim_expired_ncclops(now);
            last_reclaim = now;
        }

        while ctx.ncclops.len() > ctx.profiler.config.max_tracked_ncclop {
            if let Some(Reverse((_, id))) = ctx.reclaim_heap.pop() {
                if let Some(op) = ctx.ncclops.remove(&id) {
                    ctx.pending_telemetry.push_back(Telemetry::NcclOp(op));
                }
            } else {
                break;
            }
        }

        exporter.export(ctx, None);
    }

    for thread in threads.iter_mut() {
        thread
            .ncclop_fifo
            .recv_many(FIFO_FETCH_INTERVAL, usize::MAX, true);
        thread.ncclop_fifo.process_many(usize::MAX, |msg| {
            ctx.handle_fifo_message(msg, &mut thread.daemon_state, &mut exporter);
        });
    }

    for thread in threads.iter_mut() {
        thread.fifo.recv_many(FIFO_FETCH_INTERVAL, usize::MAX, true);
        thread.fifo.process_many(usize::MAX, |msg| {
            ctx.handle_fifo_message(msg, &mut thread.daemon_state, &mut exporter);
        });
    }

    for thread in threads.iter_mut() {
        for (_, (proxyop, _)) in thread.daemon_state.pending_comm_hash_proxyops.drain() {
            ctx.finalize_pending_comm_hash_proxyop(proxyop);
        }
    }

    ctx.reclaim_all_ncclops_in_map();

    exporter.export(ctx, Some(RETRY_MS));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::Version;
    use crate::profiler_shim;

    fn make_test_context<'a>(profiler: &'a Profiler) -> PollingContext<'a> {
        let stop = Arc::new(AtomicBool::new(false));
        PollingContext::new(profiler, stop)
    }

    fn dummy_op(id: usize, start_time: Instant) -> event::NcclOp {
        let descr = profiler_shim::tests::dummy_coll_descr();
        event::NcclOp::from_descr(&descr, start_time, id, None)
    }

    #[test]
    fn test_reclaim_timeout() {
        let mut profiler = Profiler::new(Version::V1);
        profiler.config.ncclop_timeout = Duration::from_secs(10);
        profiler.config.ncclop_completion_delay = Duration::from_secs(2);

        let mut ctx = make_test_context(&profiler);
        let t0 = Instant::now();

        // Op 1 created at t0 (deadline = t0 + 10s)
        let op = dummy_op(1, t0);
        let deadline = t0 + profiler.config.ncclop_timeout;
        ctx.reclaim_heap.push(Reverse((deadline, 1)));
        ctx.ncclops.insert(1, Box::new(op));

        // At t0 + 5s: not expired yet
        ctx.reclaim_expired_ncclops(t0 + Duration::from_secs(5));
        assert_eq!(ctx.ncclops.len(), 1);
        assert_eq!(ctx.pending_telemetry.len(), 0);

        // At t0 + 10s: expired!
        ctx.reclaim_expired_ncclops(t0 + Duration::from_secs(10));
        assert_eq!(ctx.ncclops.len(), 0);
        assert_eq!(ctx.pending_telemetry.len(), 1);
        match ctx.pending_telemetry.pop_front().unwrap() {
            Telemetry::NcclOp(reclaimed) => assert_eq!(reclaimed.id(), 1),
            _ => panic!("Expected Telemetry::NcclOp"),
        }
    }

    #[test]
    fn test_reclaim_early_completion_via_proxyop() {
        let mut profiler = Profiler::new(Version::V1);
        profiler.config.ncclop_timeout = Duration::from_secs(10);
        profiler.config.ncclop_completion_delay = Duration::from_secs(2);

        let mut ctx = make_test_context(&profiler);
        let t0 = Instant::now();

        // Op 1 created at t0 (initial deadline = t0 + 10s)
        let op = dummy_op(1, t0);
        let timeout_deadline = t0 + profiler.config.ncclop_timeout;
        ctx.reclaim_heap.push(Reverse((timeout_deadline, 1)));
        ctx.ncclops.insert(1, Box::new(op));

        // At t0 + 1s, proxy op finishes: end_time = t0 + 1s
        let end_time = t0 + Duration::from_secs(1);
        ncclop_update(ctx.get_ncclop(1).unwrap(), end_time, Some(end_time));
        let comp_deadline = end_time + profiler.config.ncclop_completion_delay; // t0 + 3s
        ctx.reclaim_heap.push(Reverse((comp_deadline, 1)));

        // At t0 + 2.5s: comp_delay (2s after t0+1s) has not elapsed yet
        ctx.reclaim_expired_ncclops(t0 + Duration::from_millis(2500));
        assert_eq!(ctx.ncclops.len(), 1);
        assert_eq!(ctx.pending_telemetry.len(), 0);

        // At t0 + 3.1s: comp_delay has elapsed! Should reclaim early before 10s timeout
        ctx.reclaim_expired_ncclops(t0 + Duration::from_millis(3100));
        assert_eq!(ctx.ncclops.len(), 0);
        assert_eq!(ctx.pending_telemetry.len(), 1);

        // Later at t0 + 10.5s: the original timeout token pops from the heap
        ctx.reclaim_expired_ncclops(t0 + Duration::from_millis(10500));
        // Should NOT emit duplicate telemetry!
        assert_eq!(ctx.pending_telemetry.len(), 1);
    }

    #[test]
    fn test_reclaim_multiple_proxyops_extending_endtime() {
        let mut profiler = Profiler::new(Version::V1);
        profiler.config.ncclop_timeout = Duration::from_secs(10);
        profiler.config.ncclop_completion_delay = Duration::from_secs(2);

        let mut ctx = make_test_context(&profiler);
        let t0 = Instant::now();

        // Op 1 created at t0
        let op = dummy_op(1, t0);
        ctx.reclaim_heap
            .push(Reverse((t0 + profiler.config.ncclop_timeout, 1)));
        ctx.ncclops.insert(1, Box::new(op));

        // Proxy op 1 ends at t0 + 1s (deadline = t0 + 3s)
        let et1 = t0 + Duration::from_secs(1);
        ncclop_update(ctx.get_ncclop(1).unwrap(), et1, Some(et1));
        ctx.reclaim_heap
            .push(Reverse((et1 + profiler.config.ncclop_completion_delay, 1)));

        // Proxy op 2 ends at t0 + 2s (deadline = t0 + 4s)
        let et2 = t0 + Duration::from_secs(2);
        ncclop_update(ctx.get_ncclop(1).unwrap(), et2, Some(et2));
        ctx.reclaim_heap
            .push(Reverse((et2 + profiler.config.ncclop_completion_delay, 1)));

        // At t0 + 3.5s: token 1 (deadline t0+3s) pops, but end_time is t0+2s so actual deadline is t0+4s.
        // Op 1 must NOT be reclaimed!
        ctx.reclaim_expired_ncclops(t0 + Duration::from_millis(3500));
        assert_eq!(ctx.ncclops.len(), 1);
        assert_eq!(ctx.pending_telemetry.len(), 0);

        // At t0 + 4.1s: token 2 (deadline t0+4s) pops. Now actual deadline has passed!
        ctx.reclaim_expired_ncclops(t0 + Duration::from_millis(4100));
        assert_eq!(ctx.ncclops.len(), 0);
        assert_eq!(ctx.pending_telemetry.len(), 1);
    }

    #[test]
    fn test_reclaim_capacity_overflow() {
        let mut profiler = Profiler::new(Version::V1);
        profiler.config.max_tracked_ncclop = 2;
        profiler.config.ncclop_timeout = Duration::from_secs(10);

        let mut ctx = make_test_context(&profiler);
        let t0 = Instant::now();

        // Insert 3 ops (limit is 2)
        for id in 1..=3 {
            let op = dummy_op(id, t0 + Duration::from_millis(id as u64 * 10));
            let deadline = t0 + Duration::from_secs(10) + Duration::from_millis(id as u64 * 10);
            ctx.reclaim_heap.push(Reverse((deadline, id)));
            ctx.ncclops.insert(id, Box::new(op));
        }

        assert_eq!(ctx.ncclops.len(), 3);

        // Simulate overflow check
        while ctx.ncclops.len() > ctx.profiler.config.max_tracked_ncclop {
            if let Some(Reverse((_, id))) = ctx.reclaim_heap.pop() {
                if let Some(op) = ctx.ncclops.remove(&id) {
                    ctx.pending_telemetry.push_back(Telemetry::NcclOp(op));
                }
            } else {
                break;
            }
        }

        assert_eq!(ctx.ncclops.len(), 2);
        assert_eq!(ctx.pending_telemetry.len(), 1);
        // The oldest op (id = 1) was force-reclaimed
        match ctx.pending_telemetry.pop_front().unwrap() {
            Telemetry::NcclOp(reclaimed) => assert_eq!(reclaimed.id(), 1),
            _ => panic!("Expected Telemetry::NcclOp"),
        }
        assert!(ctx.ncclops.contains_key(&2));
        assert!(ctx.ncclops.contains_key(&3));
    }

    #[test]
    fn test_reclaim_all_ncclops_in_map() {
        let profiler = Profiler::new(Version::V1);
        let mut ctx = make_test_context(&profiler);
        let t0 = Instant::now();

        for id in 1..=5 {
            let op = dummy_op(id, t0);
            ctx.reclaim_heap
                .push(Reverse((t0 + Duration::from_secs(10), id)));
            ctx.ncclops.insert(id, Box::new(op));
        }

        assert_eq!(ctx.ncclops.len(), 5);
        ctx.reclaim_all_ncclops_in_map();
        assert_eq!(ctx.ncclops.len(), 0);
        assert_eq!(ctx.reclaim_heap.len(), 0);
        assert_eq!(ctx.pending_telemetry.len(), 5);
    }

    // copybara:strip_begin(gpuviz)
    #[derive(Debug, Default)]
    struct CountingHistogram {
        count: std::sync::atomic::AtomicUsize,
    }

    impl AtomicHistogram<EventStep> for CountingHistogram {
        fn record(&self, _data: &EventStep) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[derive(Default)]
    struct MockExporter {
        requested_keys: std::sync::Mutex<Vec<nccl_metadata::NcclOpKey>>,
        histogram: Arc<CountingHistogram>,
    }

    impl Export for MockExporter {
        fn export(&self, _ctx: &mut PollingContext, _maybe_retry_ms: Option<u64>) {}

        fn get_latency_histogram(
            &self,
            key: &nccl_metadata::NcclOpKey,
        ) -> Option<impl std::iter::IntoIterator<Item = Arc<dyn AtomicHistogram<EventStep>>>>
        {
            self.requested_keys.lock().unwrap().push(key.clone());
            Some(vec![
                self.histogram.clone() as Arc<dyn AtomicHistogram<EventStep>>
            ])
        }
    }

    fn send_last_step_batch(
        ctx: &mut PollingContext,
        thread_state: &mut ThreadState,
        exporter: &mut MockExporter,
        info: event::ProxyOpInfo,
    ) {
        let mut step_batch_list = slab::FreeList::new_list(1);
        // SAFETY: alloc() always calls the closure with a valid pointer
        let mut steps = step_batch_list
            .alloc(|p| unsafe { StepBatch::init(p) }, None, false)
            .unwrap();
        steps.push(EventStep {
            step: 0,
            size: 65536,
            start_time: 123,
            fifo_wait_dur_ns: None,
            dur_ns: 256,
        });
        ctx.handle_fifo_message(
            Message::StepBatch(info, steps, /* is_last = */ true),
            thread_state,
            exporter,
        );
    }

    fn proxyop_info(pid: libc::pid_t, parent: usize, comm_hash: Option<u64>) -> event::ProxyOpInfo {
        event::ProxyOpInfo {
            rank: 2,
            parent: crate::event_ffi::ProxyParent::NcclOp(parent),
            pid,
            id: 7,
            peer: 3,
            is_send: true,
            comm_hash,
        }
    }

    #[test]
    fn test_proxyop_falls_back_to_own_comm_hash_when_parent_reclaimed() {
        let mut profiler = Profiler::new(Version::V4);
        profiler.config.aggregate_steps = true;
        profiler.config.track_steps = false;
        let pid = profiler.pid;

        let mut ctx = make_test_context(&profiler);
        let mut thread_state = ThreadState::default();
        let mut exporter = MockExporter::default();

        // Parent ncclop (handle 1) is not in ctx.ncclops, e.g. it was already reclaimed.
        let info = proxyop_info(pid, 1, Some(0xabcd));
        send_last_step_batch(&mut ctx, &mut thread_state, &mut exporter, info);

        assert_eq!(
            *exporter.requested_keys.lock().unwrap(),
            vec![nccl_metadata::NcclOpKey::NetSend(0xabcd, 2, 3)]
        );
        assert_eq!(exporter.histogram.count.load(Ordering::Relaxed), 1);
        assert!(thread_state.proxyops.is_empty());
    }

    #[test]
    fn test_proxyop_prefers_parent_comm_hash() {
        let mut profiler = Profiler::new(Version::V4);
        profiler.config.aggregate_steps = true;
        profiler.config.track_steps = false;
        let pid = profiler.pid;

        let mut ctx = make_test_context(&profiler);
        let mut thread_state = ThreadState::default();
        let mut exporter = MockExporter::default();

        let descr = profiler_shim::tests::dummy_coll_descr();
        let ncclop = event::NcclOp::from_descr(&descr, Instant::now(), 1, Some(0x1234));
        ctx.ncclops.insert(1, Box::new(ncclop));

        let info = proxyop_info(pid, 1, Some(0xabcd));
        send_last_step_batch(&mut ctx, &mut thread_state, &mut exporter, info);

        let keys = exporter.requested_keys.lock().unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].get_comm_hash(), 0x1234);
        assert_eq!(exporter.histogram.count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_proxyop_without_parent_or_comm_hash_is_not_aggregated() {
        let mut profiler = Profiler::new(Version::V4);
        profiler.config.aggregate_steps = true;
        profiler.config.track_steps = false;
        let pid = profiler.pid;

        let mut ctx = make_test_context(&profiler);
        let mut thread_state = ThreadState::default();
        let mut exporter = MockExporter::default();

        let info = proxyop_info(pid, 1, None);
        send_last_step_batch(&mut ctx, &mut thread_state, &mut exporter, info);

        assert!(exporter.requested_keys.lock().unwrap().is_empty());
        assert_eq!(exporter.histogram.count.load(Ordering::Relaxed), 0);
    }
    // copybara:strip_end

    #[test]
    fn test_orphan_local_proxyop_is_exported_standalone() {
        let mut profiler = Profiler::new(Version::V4);
        profiler.config.track_proxyop = true;
        let pid = profiler.pid;

        let mut ctx = make_test_context(&profiler);
        let info = event::ProxyOpInfo {
            rank: 2,
            parent: crate::event_ffi::ProxyParent::NcclOp(1),
            pid,
            id: 7,
            peer: 3,
            is_send: true,
            comm_hash: Some(0xabcd),
        };
        let now = Instant::now();

        // Parent ncclop (handle 1) is not tracked: proxyop should be exported on its own.
        let proxyop = Box::new(event::ProxyOp::from_info(&info, now));
        ctx.handle_proxyop_end(now, &info, vec![proxyop], true);
        assert_eq!(ctx.pending_telemetry.len(), 1);
        match ctx.pending_telemetry.pop_front().unwrap() {
            Telemetry::ProxyOp(p) => {
                let record = p.trace_record(|_| 0);
                assert_eq!(record["comm_hash"], "0x000000000000abcd");
                assert_eq!(record["parent"], "<ncclop 1>");
            }
            _ => panic!("Expected Telemetry::ProxyOp"),
        }
        assert!(ctx.reclaim_heap.is_empty());

        // Parent ncclop is tracked: proxyop should be attached to it instead.
        ctx.ncclops.insert(1, Box::new(dummy_op(1, now)));
        let proxyop = Box::new(event::ProxyOp::from_info(&info, now));
        ctx.handle_proxyop_end(now, &info, vec![proxyop], true);
        assert!(ctx.pending_telemetry.is_empty());
        assert_eq!(ctx.reclaim_heap.len(), 1);
        let record = ctx.ncclops[&1].trace_record(|_| 0);
        assert_eq!(record["proxyops"].as_array().unwrap().len(), 1);
    }
}
