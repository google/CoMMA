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
use crate::profiler::Profiler;
use crate::shm_fifo;
use crate::slab;
use crate::spsc;
use crate::step_tracker::EventStep;

use crate::gpuviz;

use log::error;

use std::collections::{BTreeMap, HashMap, VecDeque};
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
        /* start time */ Instant,
        /* duration ns */ u64,
        /* parent handle */ usize,
    ),
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
    CommClose(/* comm_hash = */ u64),
}

pub struct PollingContext<'a> {
    profiler: &'a Profiler,
    pub ncclops: BTreeMap<usize, Box<event::NcclOp>>,
    pub pending_telemetry: VecDeque<Telemetry>,
    stop: Arc<AtomicBool>,

    free_ncclop: slab::FreeList<event::NcclOp>,
    free_proxyop: slab::FreeList<event::ProxyOp>,
    free_step_batch: slab::FreeList<StepBatch>,

    peer_rank_fifo: HashMap<libc::pid_t, shm_fifo::mpsc::Sender<InterProcessMessage>>,
    pending_ipc_msg: HashMap<libc::pid_t, VecDeque<InterProcessMessage>>,

    gpuviz: Option<gpuviz::HistogramManager<Arc<gpuviz::Connection>>>, // copybara:strip(gpuviz)
}

impl<'a> PollingContext<'a> {
    pub fn new(profiler: &'a Profiler, stop_signal: Arc<AtomicBool>) -> Self {
        Self {
            profiler,
            ncclops: BTreeMap::new(),
            pending_telemetry: VecDeque::new(),
            stop: stop_signal,
            free_ncclop: slab::FreeList::default(),
            free_proxyop: slab::FreeList::default(),
            free_step_batch: slab::FreeList::default(),
            peer_rank_fifo: HashMap::new(),
            pending_ipc_msg: HashMap::new(),
            // copybara:strip_begin(gpuviz)
            gpuviz: profiler
                .gpuviz_lib
                .as_ref()
                .map(|lib| gpuviz::HistogramManager::new(lib.clone())),
            // copybara:strip_end
        }
    }

    fn reclaim_ncclop(&mut self, op: event::NcclOp) {
        self.pending_telemetry
            .push_back(Telemetry::NcclOp(Box::new(op)));
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
        let proxyop = thread_state.proxyops.get_mut(&proxyop_id)?;
        if proxyop.has_step_histograms() {
            return None;
        }
        let conn_key = proxyop.info().op_key(comm_hash);
        let mut histograms: Vec<Arc<dyn AtomicHistogram<EventStep>>> = Vec::new();
        if let Some(h_list) = exporter.get_latency_histogram(&conn_key) {
            for h in h_list {
                histograms.push(h as _);
            }
        }
        if let Some(gpuviz) = self.gpuviz.as_mut() {
            if self.profiler.config.track_step_fifo_wait {
                histograms.push(
                    gpuviz
                        .get_connection(&conn_key, Some(gpuviz::ConnectionType::Net))
                        .cloned()
                        .ok()?,
                );
                histograms.push(
                    gpuviz
                        .get_connection(&conn_key, Some(gpuviz::ConnectionType::E2e))
                        .cloned()
                        .ok()?,
                );
            } else {
                histograms.push(gpuviz.get_connection(&conn_key, None).cloned().ok()?);
            };
        }
        proxyop.set_step_histograms(histograms);
        Some(())
    }
    // copybara:strip_end

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
                self.try_add_histogram(thread_state, exporter, parent, id);
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
    ) {
        let config = &self.profiler.config;
        let record_proxyop = config.track_proxyop || config.track_steps;
        if let Some(parent_handle) = info.parent() {
            if info.pid == self.profiler.pid {
                if let Some(ncclop) = self.get_ncclop(parent_handle) {
                    ncclop_update(ncclop, end_time, Some(end_time));
                    if record_proxyop {
                        for p in proxyops {
                            ncclop.add_proxyop(*p);
                        }
                    }
                }
                return;
            } else {
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
                let op = self.free_ncclop.take_and_free(op);
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
                self.handle_proxyop_end(end_time, &info, Vec::new());
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
                    let ops = vec![proxyop];
                    self.handle_proxyop_end(end_time, &info, ops);
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
                    let ops = vec![proxyop];
                    self.handle_proxyop_end(end_time, &info, ops);
                }
                self.free_step_batch.free(steps);
                if self.free_step_batch.num_free() >= slab::FREELIST_BATCH {
                    self.free_step_batch
                        .try_publish(&self.profiler.free_step_batch);
                }
            }
            Message::KernelCh(start_time, duration, parent) => {
                if let Some(ncclop) = self.get_ncclop(parent) {
                    ncclop_update(
                        ncclop,
                        start_time,
                        Some(start_time + Duration::from_nanos(duration)),
                    );
                }
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
                }
            } /* copybara:strip_end_and_replace
              InterProcessMessage::ReplyProxyOpComm(_, _, _) => {}
              */
        }
    }

    fn try_reclaim_ncclops_from_map<P>(&mut self, mut should_reclaim: P)
    where
        P: FnMut(&Box<event::NcclOp>) -> ReclaimAction,
    {
        let mut to_keep = Vec::new();
        while let Some((k, v)) = self.ncclops.pop_first() {
            match should_reclaim(&v) {
                ReclaimAction::Keep => to_keep.push((k, v)),
                ReclaimAction::Reclaim => self.reclaim_ncclop(*v),
                ReclaimAction::Stop => {
                    self.ncclops.insert(k, v);
                    break;
                }
            }
        }
        for (k, v) in to_keep {
            self.ncclops.insert(k, v);
        }
    }

    pub fn reclaim_all_ncclops_in_map(&mut self) {
        self.try_reclaim_ncclops_from_map(|_| ReclaimAction::Reclaim);
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

fn should_reclaim_ncclop(op: &event::NcclOp, comp_delay: Duration, timeout: Duration) -> bool {
    let end_time = op.basic_info().end_time();
    if let Some(et) = end_time {
        let now = Instant::now();
        if now - et > comp_delay {
            return true;
        }
    }
    let start_time = op.basic_info().start_time();
    if start_time.elapsed() > timeout {
        return true;
    }
    false
}

#[derive(Debug)]
enum ReclaimAction {
    Keep,
    Reclaim,
    Stop,
}

fn ipc_shm_path(pid: libc::pid_t) -> String {
    format!("nccl-profiler-{}", pid)
}

const FIFO_FETCH_INTERVAL: Duration = Duration::from_secs(1);
const FIFO_PROCESS_BATCH: usize = 512;
const FIFO_RECV_BATCH: usize = profiler::EVENT_QUEUE_SZ;
const IPC_RECV_BATCH: usize = 512;
const IPC_SEND_BATCH: usize = 64;
const NCCLOP_RECLAIM_BATCH: usize = 512;

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

    let ncclop_timeout = ctx.profiler.config.ncclop_timeout;
    let ncclop_comp_delay = ctx.profiler.config.ncclop_completion_delay;

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

        if ctx.free_proxyop.num_free() >= slab::FREELIST_BATCH {
            ctx.free_proxyop.try_publish(&ctx.profiler.free_proxyop);
        }

        let mut n_processed = 0;
        ctx.try_reclaim_ncclops_from_map(|op| {
            if n_processed > NCCLOP_RECLAIM_BATCH {
                return ReclaimAction::Stop;
            }
            n_processed += 1;
            if should_reclaim_ncclop(op, ncclop_comp_delay, ncclop_timeout) {
                ReclaimAction::Reclaim
            } else {
                ReclaimAction::Keep
            }
        });

        if ctx.ncclops.len() > ctx.profiler.config.max_tracked_ncclop {
            // force reclaim
            let mut num_to_reclaim = ctx.ncclops.len() - ctx.profiler.config.max_tracked_ncclop;
            ctx.try_reclaim_ncclops_from_map(|_| {
                if num_to_reclaim > 0 {
                    num_to_reclaim -= 1;
                    ReclaimAction::Reclaim
                } else {
                    ReclaimAction::Stop
                }
            });
        }

        exporter.export(ctx, None)
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

    ctx.reclaim_all_ncclops_in_map();

    exporter.export(ctx, Some(RETRY_MS));
}
