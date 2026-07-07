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

use crate::clock;
use crate::cloud_daemon;
use crate::config;
use crate::daemon;
use crate::event;
use crate::event_ffi;
use crate::event_ffi::AsFFI as _;
use crate::gpuviz;
use crate::nccl_metadata;
use crate::nccl_metadata::ProxyOp;
use crate::nccl_metadata::{Coll as _, Event as _, NcclOp as _, P2p as _, ProxyStep as _};
use crate::otel_utils;
use crate::profiler_shim;
use crate::slab;
use crate::spsc;
use crate::step_tracker::StepTracker;
use crate::NcclResult;

use crossbeam::queue::ArrayQueue;
use rand::{rngs::SmallRng, RngCore as _, SeedableRng as _};

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime};

/// version of NCCL profiler API
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    V1,
    V2,
    V3,
    V4,
}

/// struct that holds global states for profiler
#[derive(Debug)]
pub struct Profiler {
    pub config: config::Config,
    pub version: Version,
    pub pid: libc::pid_t,
    pub init_time: SystemTime,
    pub init_instant: Instant,
    pub gpuviz_lib: Option<Arc<gpuviz::GpuViz>>, // copybara:strip(gpuviz)
    pub ctrl_fifo: ArrayQueue<daemon::ControlMessage>,
    daemon: Mutex<Option<cloud_daemon::CloudDaemon>>,

    ncclop_cnt: AtomicU64,
    pub gap_tracker: Option<otel_utils::GapTracker>,
    pub remote_net_bytes: Option<Arc<AtomicUsize>>,
    pub free_ncclop: slab::AtomicFreeList<event::NcclOp>,
    pub free_proxyop: slab::AtomicFreeList<event::ProxyOp>,
    pub free_step_batch: slab::AtomicFreeList<daemon::StepBatch>,
    pub cached_clock: clock::CachedClock,
}

pub const EVENT_QUEUE_SZ: usize = 8192;
const CTRL_FIFO_SZ: usize = 256;

impl Profiler {
    pub fn new(version: Version) -> Self {
        let config = &*config::CONFIG;
        let init_time = SystemTime::now();
        let init_instant = Instant::now();
        Self {
            config: config.clone(),
            version,
            // SAFETY: `getpid()` takes no input and does not modify rust-managed state
            pid: unsafe { libc::getpid() },
            init_time,
            init_instant,
            // copybara:strip_begin(gpuviz)
            gpuviz_lib: if config.telemetry_mode > 0 && config.use_gpuviz {
                if let Ok(lib) = gpuviz::GpuViz::dylib(
                    init_time
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map_or(0, |d| d.as_nanos() as _),
                ) {
                    Some(Arc::new(lib))
                } else {
                    None
                }
            } else {
                None
            },
            // copybara:strip_end
            ctrl_fifo: ArrayQueue::new(CTRL_FIFO_SZ),
            daemon: Mutex::new(None),

            ncclop_cnt: AtomicU64::new(0),
            gap_tracker: if config.otel_enable {
                Some(otel_utils::GapTracker::new())
            } else {
                None
            },
            remote_net_bytes: if config.heartbeat_collective_progress {
                Some(Arc::new(AtomicUsize::new(0)))
            } else {
                None
            },
            free_ncclop: slab::AtomicFreeList::default(),
            free_proxyop: slab::AtomicFreeList::default(),
            free_step_batch: slab::AtomicFreeList::default(),
            cached_clock: clock::CachedClock::new(init_instant),
        }
    }

    pub fn recent_timer_instant(&self) -> Instant {
        if self.config.use_cached_clock {
            self.cached_clock.recent()
        } else {
            Instant::now()
        }
    }

    pub fn recent_timer_ns(&self) -> u64 {
        if self.config.use_cached_clock {
            self.cached_clock.recent_ns()
        } else {
            self.init_instant.elapsed().as_nanos() as _
        }
    }

    pub fn instant_to_timestamp(&self, t: Instant) -> std::time::Duration {
        let t0 = self.init_instant;
        let init_time = self.init_time;
        let mut dt = t - t0;
        if let Ok(d) = init_time.duration_since(SystemTime::UNIX_EPOCH) {
            dt += d;
        }
        dt
    }

    pub fn spawn_daemon(&'static self) {
        let mut lg = self.daemon.lock().unwrap();
        let daemon = daemon::Daemon::new(self);
        *lg = Some(daemon);
    }

    pub fn join_daemon(&'static self) {
        let mut lg = self.daemon.lock().unwrap();
        // dropping the handle would cause the runtime to stop and the
        // worker will be joined
        if let Some(_handle) = lg.take() {
            // dropping the handle would stop the daemon
        }
    }

    pub fn register_thread(&self, thread_ctrl: daemon::ThreadControl) {
        self.ctrl_fifo
            .push(daemon::ControlMessage::NewThread(thread_ctrl))
            .unwrap();
    }

    pub fn init_thread_state(&'static self) -> Box<ThreadLocalState<'static>> {
        let (state, control) = thread_local_state(self);
        self.register_thread(control);
        Box::new(state)
    }
}

pub fn thread_local_state(profiler: &'_ Profiler) -> (ThreadLocalState<'_>, daemon::ThreadControl) {
    let (mut ncclop_tx, ncclop_rx) = spsc::channel(EVENT_QUEUE_SZ);
    let (mut fifo_tx, fifo_rx) = spsc::channel(EVENT_QUEUE_SZ);
    ncclop_tx.set_batch(std::cmp::max(1, profiler.config.fifo_batch_size));
    fifo_tx.set_batch(std::cmp::max(1, profiler.config.fifo_batch_size));

    let n_free_proxystep = if profiler.version == Version::V4 {
        slab::FREELIST_BATCH
    } else {
        0
    };

    let thread_local = ThreadLocalState {
        ncclop_fifo: ncclop_tx,
        fifo: fifo_tx,
        profiler,
        ncclop_refcnt: HashMap::new(),
        ncclop_free_list: slab::FreeList::new_list(slab::FREELIST_BATCH * 4),
        proxyop_free_list: slab::FreeList::new_list(slab::FREELIST_BATCH * 4),
        proxystep_free_list: slab::FreeList::new_list(n_free_proxystep),
        steps_free_list: slab::FreeList::new_list(slab::FREELIST_BATCH * 4),
        kernelch_free_list: slab::FreeList::new_list(slab::FREELIST_BATCH),
        proxyop_id: 0,
        rng: SmallRng::from_rng(&mut rand::rng()),
    };
    let thread_control = daemon::ThreadControl::new(
        daemon::FifoReceiver::new(ncclop_rx),
        daemon::FifoReceiver::new(fifo_rx),
    );
    (thread_local, thread_control)
}

pub fn with_thread_state<F, R>(f: F) -> R
where
    F: FnOnce(&mut ThreadLocalState) -> R,
{
    THREAD_STATE.with_borrow_mut(|state| {
        if state.is_none() {
            let profiler = PROFILER.get().unwrap();
            *state = Some(profiler.init_thread_state())
        }
        f(state.as_mut().unwrap())
    })
}

/// thread_local state for each threads
///
/// Generally, NCCL profiler would be called from two different threads:
/// 1. the workload thread where group start/end and nccl collectives are issued.
/// 2. the proxy thread where network ops are issued
#[derive(Debug)]
pub struct ThreadLocalState<'a> {
    pub ncclop_fifo: spsc::Sender<daemon::Message>,
    pub fifo: spsc::Sender<daemon::Message>,
    pub profiler: &'a Profiler,

    pub ncclop_refcnt: HashMap<(libc::pid_t, usize), (usize, Instant, bool)>,
    pub ncclop_free_list: slab::FreeList<event::NcclOp>,
    pub proxyop_free_list: slab::FreeList<ProxyOpLocalData>,
    pub proxystep_free_list: slab::FreeList<event::ProxyStep>,
    pub steps_free_list: slab::FreeList<daemon::StepBatch>,
    pub kernelch_free_list: slab::FreeList<KernelCh>,

    proxyop_id: u32,
    rng: SmallRng,
}

impl ThreadLocalState<'_> {
    pub fn send_to_daemon(&mut self, mut msg: daemon::Message, use_barrier: bool) {
        while let Err(v) = self.fifo.send(msg, use_barrier) {
            msg = v;
        }
    }

    pub fn send_ncclop_to_daemon(&mut self, mut msg: daemon::Message, use_barrier: bool) {
        while let Err(v) = self.ncclop_fifo.send(msg, use_barrier) {
            msg = v;
        }
    }

    pub fn next_proxyop_id(&mut self) -> u32 {
        let id = self.proxyop_id;
        self.proxyop_id = self.proxyop_id.wrapping_add(1);
        id
    }

    pub fn inc_ncclop_ref(&mut self, pid: libc::pid_t, op: usize) -> usize {
        let cnt = self
            .ncclop_refcnt
            .entry((pid, op))
            .or_insert_with(|| (0, self.profiler.recent_timer_instant(), false));
        cnt.0 += 1;
        cnt.0 - 1
    }

    pub fn dec_ncclop_ref(&mut self, pid: libc::pid_t, op: usize) -> Option<Instant> {
        if let Some(e) = self.ncclop_refcnt.get_mut(&(pid, op)) {
            e.0 -= 1;
            if e.0 == 0 {
                let e = self.ncclop_refcnt.remove(&(pid, op)).unwrap();
                Some(e.1)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn update_ncclop_start_time(&mut self, pid: libc::pid_t, op: usize) {
        if let Some(e) = self.ncclop_refcnt.get_mut(&(pid, op)) {
            if !e.2 {
                e.1 = self.profiler.recent_timer_instant();
                e.2 = true;
            }
        }
    }

    pub fn new_ncclop<E>(
        &mut self,
        descr: &E,
        time: Instant,
        is_lite: bool,
        comm_hash_override: Option<u64>,
    ) -> Option<event::Event>
    where
        E: nccl_metadata::Event,
    {
        let id = self.profiler.ncclop_cnt.fetch_add(1, Ordering::Relaxed);
        let op = event::NcclOp::from_descr(descr, time, id as _, comm_hash_override);
        self.ncclop_free_list
            .alloc_new(op, Some(&self.profiler.free_ncclop), true)
            .map(|op| {
                let id = op.id();
                self.send_ncclop_to_daemon(daemon::Message::NcclOp(op), true);
                if is_lite {
                    event::Event::NcclOpLite(id)
                } else {
                    event::Event::NcclOp(id)
                }
            })
            .ok()
    }

    #[inline(always)]
    pub fn rnd_decision(&mut self, true_prob: f64) -> bool {
        if true_prob >= 1.0 {
            true
        } else if true_prob <= 0.0 {
            false
        } else {
            let bar = u32::MAX as f64 * true_prob;
            self.rng.next_u32() <= bar as u32
        }
    }
}

/// struct that holds states for communicator
#[derive(Debug, Clone)]
pub struct Communicator {
    pub comm_hash: Option<u64>,
    pub total_net_bytes: Arc<AtomicUsize>,
}

impl Communicator {
    pub fn new() -> Self {
        Self {
            comm_hash: None,
            total_net_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[derive(Debug)]
#[repr(align(16))]
pub struct KernelCh {
    pub parent_op: Option<usize>,
}

#[derive(Debug)]
#[repr(align(16))]
pub struct ProxyOpLocalData {
    pub info: event::ProxyOpInfo,
    pub extra: event::ProxyOpExtra,
    pub step_tracker: StepTracker,
    pub n_steps: usize,
    pub parent_op: Option<usize>,
    pub steps: Option<slab::AllocatedNode<daemon::StepBatch>>,
    pub rng_seed: u32,
}

impl ProxyOpLocalData {
    pub fn new<E>(
        id: u32,
        descr: &E,
        is_v1: bool,
        track_fifo_wait: bool,
        comm: Option<Communicator>,
    ) -> Self
    where
        E: nccl_metadata::ProxyOp,
    {
        let info = event::ProxyOpInfo::from_descr(descr, id, comm);
        let is_send = info.is_send;
        Self {
            info,
            extra: event::ProxyOpExtra::from_descr(descr),
            step_tracker: StepTracker::new(is_send, is_v1, track_fifo_wait),
            n_steps: 0,
            parent_op: None,
            steps: None,
            rng_seed: 0,
        }
    }

    pub fn get_steps_mut<'a>(
        &'a mut self,
        thread_state: &mut ThreadLocalState,
    ) -> &'a mut daemon::StepBatch {
        if self.steps.is_none() {
            self.steps = Some(
                thread_state
                    .steps_free_list
                    .alloc(
                        // SAFETY: alloc() function always call the closure with valid pointer
                        |p| unsafe { daemon::StepBatch::init(p) },
                        Some(&thread_state.profiler.free_step_batch),
                        true,
                    )
                    .unwrap(),
            );
        }
        self.steps.as_mut().unwrap()
    }
}

thread_local! {
    pub(crate) static THREAD_STATE: RefCell<Option<Box<ThreadLocalState<'static>>>> = const { RefCell::new(None) };
}

static INIT_FLAG: Mutex<usize> = Mutex::new(0);
static PROFILER: OnceLock<Profiler> = OnceLock::new();

pub fn init_handler(
    e_activation_mask: &mut i32,
    version: Version,
) -> NcclResult<Box<Communicator>> {
    let mut mask = 0;
    if config::CONFIG.telemetry_mode > 0 {
        let mut lg = INIT_FLAG.lock().unwrap();
        if *lg == 0 {
            env_logger::init();
            // after this point we should be able to use all the log macros

            // get_or_init: creates Profiler only on first init, reuses on subsequent cycles
            PROFILER.get_or_init(|| Profiler::new(version));
            // Always spawn daemon when INIT_FLAG=0 (daemon was stopped on last finalize)
            PROFILER.get().unwrap().spawn_daemon();
        }
        *lg += 1;

        mask |= profiler_shim::ncclProfileGroup;
        mask |= profiler_shim::ncclProfileColl;
        mask |= profiler_shim::ncclProfileP2p;
        mask |= profiler_shim::ncclProfileProxyOp;
        if config::CONFIG.track_kernel_ch {
            mask |= profiler_shim::ncclProfileKernelCh;
        }
    }
    *e_activation_mask = mask as i32;
    Ok(Box::new(Communicator::new()))
}

pub fn init_handler_v4(
    e_activation_mask: &mut i32,
    _comm_name: *const libc::c_char,
    comm_hash: u64,
    _n_nodes: i32,
    _n_ranks: i32,
    _rank: i32,
    version: Version,
) -> NcclResult<Box<Communicator>> {
    let mut mask = 0;
    if config::CONFIG.telemetry_mode > 0 {
        let mut lg = INIT_FLAG.lock().unwrap();
        if *lg == 0 {
            // get_or_init: creates Profiler only on first init, reuses on subsequent cycles
            PROFILER.get_or_init(|| Profiler::new(version));
            // Always spawn daemon when INIT_FLAG=0 (daemon was stopped on last finalize)
            PROFILER.get().unwrap().spawn_daemon();
        }
        *lg += 1;

        mask |= profiler_shim::ncclProfileGroup;
        mask |= profiler_shim::ncclProfileColl;
        mask |= profiler_shim::ncclProfileP2p;
        mask |= profiler_shim::ncclProfileProxyOp;

        if config::CONFIG.track_kernel_ch {
            mask |= profiler_shim::ncclProfileKernelCh;
        }

        if config::CONFIG.track_steps | config::CONFIG.aggregate_steps {
            mask |= profiler_shim::ncclProfileProxyStep;
        }
    }
    *e_activation_mask = mask as i32;
    let mut comm = Box::new(Communicator::new());
    comm.comm_hash = Some(comm_hash);
    if config::CONFIG.heartbeat_collective_progress {
        with_thread_state(|thread_state| {
            thread_state.send_to_daemon(daemon::Message::CommOpen(*comm.clone()), true)
        });
    }
    Ok(comm)
}

pub fn start_event_handler<E>(
    descr: &E,
    comm: *const Communicator,
) -> NcclResult<Option<event::Event>>
where
    E: nccl_metadata::Version + nccl_metadata::Event,
{
    let config: &config::Config = &config::CONFIG;
    let event = match descr.type_() as u32 {
        profiler_shim::ncclProfileGroup => {
            if config.track_ncclop {
                Some(event::Event::new_group(descr, Instant::now()))
            } else if config.track_proxyop {
                Some(event::Event::new_dummyop(42))
            } else {
                None
            }
        }
        profiler_shim::ncclProfileColl => {
            // SAFETY: just checked that event type is collective
            let descr = unsafe { descr.cast_to_coll() };
            // SAFETY: for coll and p2p, NCCL guarantees the comm pointer is valid
            let comm = unsafe { &*comm };
            let byte_count = descr.byte_count();
            let op_type = descr.op_type();
            if config.track_ncclop {
                use nccl_metadata::NcclOpType;
                let skip_step_tracking = !(config.track_steps || config.aggregate_steps);
                let skip_small_msg =
                    config.skip_small_collective || config.skip_small_collective_steps;
                if skip_small_msg
                    && !std::matches!(op_type, NcclOpType::AllReduce)
                    && byte_count <= config.small_msg_threshold
                {
                    if config.skip_small_collective {
                        let comm_hash = comm
                            .comm_hash
                            .unwrap_or_else(|| descr.comm_hash().unwrap_or(42));
                        Some(event::Event::SmallNcclOp(comm_hash as usize))
                    } else {
                        Some(with_thread_state(|thread_state| {
                            thread_state
                                .new_ncclop(
                                    descr,
                                    Instant::now(),
                                    /* is_lite = */ true,
                                    comm.comm_hash,
                                )
                                .unwrap_or_else(|| event::Event::new_dummyop(42))
                        }))
                    }
                } else if skip_step_tracking
                    || (config.skip_nvls
                        && std::matches!(
                            descr.algo(),
                            nccl_metadata::algo::NVLS | nccl_metadata::algo::NVLS_TREE
                        ))
                {
                    Some(with_thread_state(|thread_state| {
                        thread_state
                            .new_ncclop(
                                descr,
                                Instant::now(),
                                /* is_lite = */ true,
                                comm.comm_hash,
                            )
                            .unwrap_or_else(|| event::Event::new_dummyop(42))
                    }))
                } else {
                    Some(with_thread_state(|thread_state| {
                        thread_state
                            .new_ncclop(
                                descr,
                                Instant::now(),
                                /* is_lite = */ false,
                                comm.comm_hash,
                            )
                            .unwrap_or_else(|| event::Event::new_dummyop(42))
                    }))
                }
            } else if config.track_proxyop {
                let comm_hash = comm
                    .comm_hash
                    .unwrap_or_else(|| descr.comm_hash().unwrap_or(42));
                Some(event::Event::new_dummyop(comm_hash as usize))
            } else {
                None
            }
        }
        profiler_shim::ncclProfileP2p => {
            // SAFETY: just checked that event type is p2p
            let descr = unsafe { descr.cast_to_p2p() };
            // SAFETY: for coll and p2p, NCCL guarantees the comm pointer is valid
            let comm = unsafe { &*comm };
            let should_sample = with_thread_state(|thread_state| {
                if thread_state.rnd_decision(thread_state.profiler.config.p2p_sample_rate) {
                    if descr.is_send() {
                        true
                    } else {
                        thread_state.rnd_decision(thread_state.profiler.config.p2p_recv_sample_rate)
                    }
                } else {
                    false
                }
            });

            if should_sample {
                if config.track_ncclop {
                    let byte_count = descr.byte_count();
                    if byte_count > config.small_msg_threshold {
                        with_thread_state(|thread_state| {
                            Some(
                                thread_state
                                    .new_ncclop(
                                        descr,
                                        thread_state.profiler.recent_timer_instant(), //Instant::now()
                                        /* is_lite = */ false,
                                        comm.comm_hash,
                                    )
                                    .unwrap_or_else(|| event::Event::new_dummyop(42)),
                            )
                        })
                    } else {
                        None
                    }
                } else if config.track_proxyop {
                    let comm_hash = comm
                        .comm_hash
                        .unwrap_or_else(|| descr.comm_hash().unwrap_or(42));
                    Some(event::Event::new_dummyop(comm_hash as usize))
                } else {
                    None
                }
            } else {
                None
            }
        }
        profiler_shim::ncclProfileProxyOp => {
            // SAFETY: just checked that event type is proxyop
            let descr = unsafe { descr.cast_to_proxyop() };
            let parent_ffi = descr.parent_obj();
            let parent_type = event_ffi::get_handle_type(parent_ffi);

            if parent_ffi.is_null() {
                None
            } else if parent_type == event_ffi::Type::SmallNcclOp {
                // TODO: Add fast path to handle small (one network op) message
                None
            } else {
                with_thread_state(|thread_state| {
                    let profiler = thread_state.profiler;
                    let proxyop_id = thread_state.next_proxyop_id();
                    let comm: Option<Communicator> =
                        // SAFETY: comm is valid when pids match
                        if config.heartbeat_collective_progress && profiler.pid == descr.pid() {
                            unsafe { Some((*comm).clone()) }
                        } else {
                            None
                        };
                    let mut local_data = ProxyOpLocalData::new(
                        proxyop_id,
                        descr,
                        E::version() == Version::V1,
                        profiler.config.track_step_fifo_wait,
                        comm,
                    );
                    let is_send = local_data.info.is_send;
                    let mut skip_step_tracking = parent_type == event_ffi::Type::NcclOpLite;

                    if !skip_step_tracking && !is_send && !profiler.config.track_recv_steps {
                        skip_step_tracking = true;
                    }

                    let parent = event_ffi::ProxyParent::from_ffi(parent_ffi);
                    if let event_ffi::ProxyParent::NcclOp(ncclop) = parent {
                        local_data.parent_op = Some(ncclop);
                    }
                    let mut data = thread_state
                        .proxyop_free_list
                        .alloc_new(local_data, None, true)
                        .unwrap();

                    if skip_step_tracking {
                        if let event_ffi::ProxyParent::NcclOp(ncclop) = parent {
                            thread_state.inc_ncclop_ref(data.info.pid, ncclop);
                            Some(event::Event::ProxyOpLite(data))
                        } else {
                            None
                        }
                    } else {
                        data.rng_seed = thread_state.rng.next_u32();
                        Some(event::Event::ProxyOp(data))
                    }
                })
            }
        }
        profiler_shim::ncclProfileKernelCh => {
            let parent_ffi = descr.parent_obj();
            let parent_type = event_ffi::get_handle_type(parent_ffi);
            if parent_ffi.is_null() || parent_type == event_ffi::Type::SmallNcclOp {
                None
            } else {
                let parent = event_ffi::ProxyParent::from_ffi(parent_ffi);
                if let event_ffi::ProxyParent::NcclOp(ncclop) = parent {
                    with_thread_state(|thread_state| {
                        thread_state.inc_ncclop_ref(thread_state.profiler.pid, ncclop);
                        let kernelch = thread_state
                            .kernelch_free_list
                            .alloc_new(
                                KernelCh {
                                    parent_op: Some(ncclop),
                                },
                                None,
                                true,
                            )
                            .unwrap();
                        Some(event::Event::KernelCh(kernelch))
                    })
                } else {
                    None
                }
            }
        }
        profiler_shim::ncclProfileProxyStep => {
            // SAFETY: just checked that event type is proxystep
            let descr = unsafe { descr.cast_to_proxystep() };
            let parent_ffi = descr.parent_obj();
            if parent_ffi.is_null() {
                None
            } else {
                // SAFETY: When not set to null,
                // NCCL always sets the parent_obj to a valid handle returned by profiler
                // API. So `event::Event::from_ffi()` would always be called on a valid handle
                let parent = unsafe { event::Event::from_ffi(parent_ffi) };
                if let Some(event::Event::ProxyOp(op)) = parent {
                    with_thread_state(|thread_state| {
                        let data = thread_state
                            .proxystep_free_list
                            .alloc_new(
                                event::ProxyStep::new(
                                    descr.step(),
                                    slab::AllocatedNode::into_raw(op),
                                ),
                                None,
                                true,
                            )
                            .unwrap();
                        Some(event::Event::ProxyStep(data))
                    })
                } else {
                    let _ = parent.map(event::Event::into_ffi);
                    None
                }
            }
        }
        _ => panic!("unknown event type"),
    };
    if event.as_ref().is_some_and(is_nccl_activity) {
        with_thread_state(|thread_state| {
            let profiler = thread_state.profiler;
            if let Some(gap_tracker) = profiler.gap_tracker.as_ref() {
                gap_tracker.activity_begin(|| profiler.recent_timer_ns());
            }
        });
    }
    Ok(event)
}

// events that represent NCCL activity for gap tracking: an op being enqueued
// or its network / kernel work in progress; the proxy op and kernel channel
// windows keep an op in flight until its work actually completes, well after
// NCCL stops the op event itself at enqueue time
fn is_nccl_activity(event: &event::Event) -> bool {
    std::matches!(
        event,
        event::Event::NcclOp(_)
            | event::Event::NcclOpLite(_)
            | event::Event::SmallNcclOp(_)
            | event::Event::ProxyOp(_)
            | event::Event::ProxyOpLite(_)
            | event::Event::KernelCh(_)
    )
}

pub fn stop_event_handler(event: event::Event) -> NcclResult<()> {
    let end_nccl_activity = is_nccl_activity(&event);
    with_thread_state(|thread_state| {
        match event {
            event::Event::Group(group) => {
                thread_state.send_to_daemon(daemon::Message::Group(group), true);
            }
            event::Event::ProxyOpLite(data) => {
                thread_state.fifo.prefetch_next();
                if let Some(ncclop) = data.parent_op {
                    if let Some(start_time) = thread_state.dec_ncclop_ref(data.info.pid, ncclop) {
                        let msg = daemon::Message::ProxyOpLite(
                            start_time,
                            start_time.elapsed().as_nanos() as u64,
                            data.info.clone(),
                        );
                        thread_state.send_to_daemon(msg, true);
                    }
                }
                thread_state.proxyop_free_list.free(data);
            }
            event::Event::ProxyOp(mut data) => {
                thread_state.fifo.prefetch_next();
                if let Some(step) = data.step_tracker.finalize() {
                    data.get_steps_mut(thread_state).push(step);
                }
                if thread_state.profiler.config.track_proxyop {
                    // if we are tracking proxyop also send the extra info about this proxyop
                    let msg = daemon::Message::ProxyOpExtra(data.extra.clone());
                    thread_state.send_to_daemon(msg, false);
                }
                if let Some(steps) = data.steps.take() {
                    let msg = daemon::Message::StepBatch(data.info.clone(), steps, true);
                    thread_state.send_to_daemon(msg, true);
                } else {
                    let msg = daemon::Message::ProxyOp(data.info.id);
                    thread_state.send_to_daemon(msg, true);
                }
                thread_state.proxyop_free_list.free(data);
            }
            event::Event::ProxyStep(mut data) => {
                if data.end_time.is_none() {
                    data.end_time = Some(thread_state.profiler.recent_timer_instant());
                }
                let step =
                    data.finalize(|t| (*t - thread_state.profiler.init_instant).as_nanos() as u64);
                // SAFETY: NCCL guarantees that the proxyop event handle is
                // live at this moment and the proxystep is on the same thread
                // as the parent event handle.
                // Therefore, dereference this pointer is safe as
                // 1. the pointer is valid
                // 2. there is no other threads accessing it
                let parent = unsafe { &mut *data.parent };
                let steps = parent.get_steps_mut(thread_state);
                steps.push(step);
                if steps.is_full() {
                    let steps = parent.steps.take().unwrap();
                    let msg = daemon::Message::StepBatch(parent.info.clone(), steps, false);
                    thread_state.send_to_daemon(msg, false);
                }
                thread_state.proxystep_free_list.free(data);
            }
            event::Event::KernelCh(kernelch) => {
                thread_state.fifo.prefetch_next();
                if let Some(ncclop) = kernelch.parent_op {
                    if let Some(start_time) =
                        thread_state.dec_ncclop_ref(thread_state.profiler.pid, ncclop)
                    {
                        let msg = daemon::Message::KernelCh(
                            start_time,
                            start_time.elapsed().as_nanos() as u64,
                            ncclop,
                        );
                        thread_state.send_to_daemon(msg, true);
                    }
                }
                thread_state.kernelch_free_list.free(kernelch);
            }
            event::Event::Dummy(_) => (),
            event::Event::SmallNcclOp(_) => (),
            event::Event::NcclOpLite(_) => {}
            event::Event::NcclOp(_) => {}
        }
        if end_nccl_activity {
            let profiler = thread_state.profiler;
            if let Some(gap_tracker) = profiler.gap_tracker.as_ref() {
                gap_tracker.activity_end(profiler.recent_timer_ns());
            }
        }
    });
    Ok(())
}

pub fn record_event_state_handler<S>(
    event: &mut event::Event,
    e_state: profiler_shim::ncclProfilerEventState_v1_t,
    e_state_args: &S,
) -> NcclResult<()>
where
    S: nccl_metadata::ProxyOpState,
{
    if let event::Event::ProxyOp(data) = event {
        with_thread_state(|thread_state| {
            // let n_steps = data.n_steps;
            let maybe_step = data.step_tracker.update_step(e_state, e_state_args, || {
                thread_state.profiler.recent_timer_ns()
            });

            if let Some(step) = maybe_step {
                data.n_steps += 1;
                let steps = data.get_steps_mut(thread_state);
                steps.push(step);
                if steps.is_full() {
                    let steps = data.steps.take().unwrap();
                    let msg = daemon::Message::StepBatch(data.info.clone(), steps, false);
                    thread_state.send_to_daemon(msg, false);
                }
            }
        });
    }
    Ok(())
}

pub fn record_proxystep_event_state_handler<S>(
    event: &mut event::Event,
    e_state: profiler_shim::ncclProfilerEventState_v4_t,
    e_state_args: &S,
) -> NcclResult<()>
where
    S: nccl_metadata::ProxyStepState,
{
    if let event::Event::ProxyStep(data) = event {
        with_thread_state(|thread_state| match e_state {
            profiler_shim::proxy_event_state::v4::SEND_PEER_WAIT => {
                if data.start_time.is_none() && thread_state.profiler.config.track_step_fifo_wait {
                    data.start_time = Some(thread_state.profiler.recent_timer_instant());
                }
            }
            profiler_shim::proxy_event_state::v4::SEND_WAIT => {
                let now = thread_state.profiler.recent_timer_instant();
                if thread_state.profiler.config.track_step_fifo_wait {
                    data.fifo_ready_time = Some(now);
                } else {
                    data.start_time = Some(now);
                }
                data.size = e_state_args.trans_size();
                if let Some(ref remote_net_bytes) = thread_state.profiler.remote_net_bytes {
                    // SAFETY: NCCL guarantees that the proxyop event handle is
                    // live at this moment and the proxystep is on the same thread
                    // as the parent event handle.
                    let parent = unsafe { &*data.parent };
                    if let Some(ref comm) = parent.info.comm {
                        comm.total_net_bytes.fetch_add(data.size, Ordering::Release);
                    } else {
                        remote_net_bytes.fetch_add(data.size, Ordering::Release);
                    }
                }
            }
            profiler_shim::proxy_event_state::v4::RECV_WAIT => {
                data.start_time = Some(thread_state.profiler.recent_timer_instant());
            }
            profiler_shim::proxy_event_state::v4::RECV_FLUSH_WAIT => {
                data.size = e_state_args.trans_size();
                data.end_time = Some(thread_state.profiler.recent_timer_instant());
            }
            _ => (),
        });
    }
    Ok(())
}

pub fn record_proxyop_event_state_handler_v4(
    event: &mut event::Event,
    e_state: profiler_shim::ncclProfilerEventState_v1_t,
) -> NcclResult<()> {
    if let event::Event::ProxyOpLite(data) = event {
        if e_state != profiler_shim::proxy_event_state::v4::PROXYOP_IN_PROGRESS {
            return Ok(());
        }
        with_thread_state(|thread_state| {
            if let Some(ncclop) = data.parent_op {
                thread_state.update_ncclop_start_time(data.info.pid, ncclop);
            }
        });
    }
    Ok(())
}

#[allow(clippy::boxed_local)]
pub fn finalize_handler(comm: Box<Communicator>) -> NcclResult<()> {
    if config::CONFIG.telemetry_mode > 0 {
        let mut lg = INIT_FLAG.lock().unwrap();
        if let Some(comm_hash) = comm.comm_hash {
            with_thread_state(|thread_state| {
                thread_state.send_to_daemon(daemon::Message::CommClose(comm_hash), true);
            });
        }
        if *lg == 1 {
            let profiler = PROFILER.get().unwrap();
            profiler.join_daemon();
        }
        *lg -= 1;
    }
    Ok(())
}
