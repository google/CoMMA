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

pub mod clock;
mod cloud_daemon;
mod config;
mod daemon;
mod event;
mod event_ffi;
mod fixed_batch;
mod gcp_acs_proto; // copybara:strip(oss_protobuf)
mod gpuviz;
mod histogram;
mod nccl_metadata;
mod otel_utils;
mod profiler;
pub mod profiler_shim;
mod shm_fifo;
mod slab;
mod spsc;
mod step_tracker;

use std::sync::OnceLock;

use event_ffi::AsFFI as _;
use profiler_shim::{ncclResult_t, EventDescrV1, EventDescrV2, EventDescrV3, EventDescrV4};

/// # Safety
///
/// the input must not have interior NUL bytes
macro_rules! static_cstr {
    ($l:expr) => {
        ::std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($l, "\0").as_bytes())
    };
}

pub type NcclResult<T> = Result<T, ncclResult_t>;

struct NcclLogger(profiler_shim::ncclDebugLogger_t);

impl NcclLogger {
    fn get_target(target: &str) -> libc::c_int {
        match target {
            "init" => profiler_shim::ncclDebugLogSubSys_NCCL_INIT,
            "env" => profiler_shim::ncclDebugLogSubSys_NCCL_ENV,
            "init+net" => {
                profiler_shim::ncclDebugLogSubSys_NCCL_INIT
                    | profiler_shim::ncclDebugLogSubSys_NCCL_NET
            }
            _ => profiler_shim::ncclDebugLogSubSys_NCCL_NET,
        }
    }

    fn get_level(lvl: &log::Level) -> libc::c_uint {
        use log::Level::*;
        match lvl {
            Error => profiler_shim::ncclDebugLogLevel_NCCL_LOG_WARN,
            Warn => profiler_shim::ncclDebugLogLevel_NCCL_LOG_WARN,
            Info => profiler_shim::ncclDebugLogLevel_NCCL_LOG_INFO,
            Debug => profiler_shim::ncclDebugLogLevel_NCCL_LOG_TRACE,
            Trace => profiler_shim::ncclDebugLogLevel_NCCL_LOG_TRACE,
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the input string is null-terminated
    unsafe fn convert_to_cstr(s: &str) -> &std::ffi::CStr {
        std::ffi::CStr::from_bytes_with_nul_unchecked(s.as_bytes())
    }
}

impl log::Log for NcclLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        // leave it to the NCCL logger function to deside
        true
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let level = Self::get_level(&record.level());
            let target = Self::get_target(record.target());
            let log_fn = self.0.unwrap();
            unsafe {
                let file = format!("{}\0", record.file().unwrap_or("<???>"));
                let file_cstr = Self::convert_to_cstr(&file);
                let msg = format!("{}{}", record.args(), '\0');
                let msg_cstr = Self::convert_to_cstr(&msg);
                log_fn(
                    level,
                    target as _,
                    file_cstr.as_ptr(),
                    record.line().unwrap_or(0) as _,
                    msg_cstr.as_ptr(),
                );
            }
        }
    }

    fn flush(&self) {}
}

static LOGGER: OnceLock<NcclLogger> = OnceLock::new();
static LOGGER_LOCK: std::sync::Mutex<bool> = std::sync::Mutex::new(false);

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_init_v1(
    context: *mut *mut libc::c_void,
    e_activation_mask: *mut i32,
) -> ncclResult_t {
    match profiler::init_handler(&mut *e_activation_mask, profiler::Version::V1) {
        Ok(comm) => {
            *context = Box::into_raw(comm) as _;
            profiler_shim::ncclResult_t_ncclSuccess
        }
        Err(e) => e,
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_init_v2(
    context: *mut *mut libc::c_void,
    e_activation_mask: *mut i32,
) -> ncclResult_t {
    match profiler::init_handler(&mut *e_activation_mask, profiler::Version::V2) {
        Ok(comm) => {
            *context = Box::into_raw(comm) as _;
            profiler_shim::ncclResult_t_ncclSuccess
        }
        Err(e) => e,
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_init_v3(
    context: *mut *mut libc::c_void,
    e_activation_mask: *mut i32,
) -> ncclResult_t {
    match profiler::init_handler(&mut *e_activation_mask, profiler::Version::V3) {
        Ok(comm) => {
            *context = Box::into_raw(comm) as _;
            profiler_shim::ncclResult_t_ncclSuccess
        }
        Err(e) => e,
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_init_v4(
    context: *mut *mut libc::c_void,
    e_activation_mask: *mut i32,
    comm_name: *const libc::c_char,
    comm_hash: u64,
    n_nodes: i32,
    n_ranks: i32,
    rank: i32,
    log_fn: profiler_shim::ncclDebugLogger_t,
) -> ncclResult_t {
    {
        let mut lg = LOGGER_LOCK.lock().unwrap();
        if !*lg {
            let logger = LOGGER.get_or_init(|| NcclLogger(log_fn));
            log::set_logger(logger)
                .map(|()| log::set_max_level(log::LevelFilter::Trace))
                .unwrap();
            *lg = true;
        }
    }
    match profiler::init_handler_v4(
        &mut *e_activation_mask,
        comm_name,
        comm_hash,
        n_nodes,
        n_ranks,
        rank,
        profiler::Version::V4,
    ) {
        Ok(comm) => {
            *context = Box::into_raw(comm) as _;
            profiler_shim::ncclResult_t_ncclSuccess
        }
        Err(e) => e,
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_start_event_v1(
    context: *mut libc::c_void,
    e_handle: *mut *mut libc::c_void,
    e_descr: *mut profiler_shim::ncclProfilerEventDescr_v1_t,
) -> ncclResult_t {
    let descr = &*(e_descr as *const EventDescrV1);
    match profiler::start_event_handler(descr, context as _) {
        Ok(event) => {
            *e_handle = event.map_or(std::ptr::null_mut(), event::Event::into_ffi);
            profiler_shim::ncclResult_t_ncclSuccess
        }
        Err(e) => e,
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_start_event_v2(
    context: *mut libc::c_void,
    e_handle: *mut *mut libc::c_void,
    e_descr: *mut profiler_shim::ncclProfilerEventDescr_v2_t,
) -> ncclResult_t {
    let descr = &*(e_descr as *const EventDescrV2);
    match profiler::start_event_handler(descr, context as _) {
        Ok(event) => {
            *e_handle = event.map_or(std::ptr::null_mut(), event::Event::into_ffi);
            profiler_shim::ncclResult_t_ncclSuccess
        }
        Err(e) => e,
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_start_event_v3(
    context: *mut libc::c_void,
    e_handle: *mut *mut libc::c_void,
    e_descr: *mut profiler_shim::ncclProfilerEventDescr_v3_t,
) -> ncclResult_t {
    let descr = &*(e_descr as *const EventDescrV3);
    match profiler::start_event_handler(descr, context as _) {
        Ok(event) => {
            *e_handle = event.map_or(std::ptr::null_mut(), event::Event::into_ffi);
            profiler_shim::ncclResult_t_ncclSuccess
        }
        Err(e) => e,
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_start_event_v4(
    context: *mut libc::c_void,
    e_handle: *mut *mut libc::c_void,
    e_descr: *mut profiler_shim::ncclProfilerEventDescr_v4_t,
) -> ncclResult_t {
    let descr = &*(e_descr as *const EventDescrV4);
    match profiler::start_event_handler(descr, context as _) {
        Ok(event) => {
            *e_handle = event.map_or(std::ptr::null_mut(), event::Event::into_ffi);
            profiler_shim::ncclResult_t_ncclSuccess
        }
        Err(e) => e,
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_stop_event(e_handle: *mut libc::c_void) -> ncclResult_t {
    if e_handle.is_null() {
        return profiler_shim::ncclResult_t_ncclSuccess;
    }
    if let Some(event) = event::Event::from_ffi(e_handle) {
        if let Err(e) = profiler::stop_event_handler(event) {
            return e;
        }
    }
    profiler_shim::ncclResult_t_ncclSuccess
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_record_event_state_impl<P>(
    e_handle: *mut libc::c_void,
    e_state: profiler_shim::ncclProfilerEventState_v1_t,
    e_state_args: &P,
) -> ncclResult_t
where
    P: nccl_metadata::ProxyOpState,
{
    let handle_type = event_ffi::get_handle_type(e_handle);
    if std::matches!(handle_type, event_ffi::Type::ProxyOp) {
        if let Some(mut event) = event::Event::from_ffi(e_handle) {
            if let Err(e) = profiler::record_event_state_handler(&mut event, e_state, e_state_args)
            {
                return e;
            }
            let _ = event::Event::into_ffi(event);
        }
    }
    profiler_shim::ncclResult_t_ncclSuccess
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_record_event_state_v1(
    e_handle: *mut libc::c_void,
    e_state: profiler_shim::ncclProfilerEventState_v1_t,
    e_state_args: *mut profiler_shim::ncclProfilerEventStateArgs_v1_t,
) -> ncclResult_t {
    profiler_record_event_state_impl(
        e_handle,
        e_state,
        nccl_metadata::ProxyOpStateV1::cast_from_union(&*e_state_args),
    )
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_record_event_state_v2(
    e_handle: *mut libc::c_void,
    e_state: profiler_shim::ncclProfilerEventState_v2_t,
    e_state_args: *mut profiler_shim::ncclProfilerEventStateArgs_v2_t,
) -> ncclResult_t {
    profiler_record_event_state_impl(
        e_handle,
        e_state,
        nccl_metadata::ProxyOpStateV2::cast_from_union(&*e_state_args),
    )
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_record_event_state_v3(
    e_handle: *mut libc::c_void,
    e_state: profiler_shim::ncclProfilerEventState_v3_t,
    e_state_args: *mut profiler_shim::ncclProfilerEventStateArgs_v3_t,
) -> ncclResult_t {
    profiler_record_event_state_impl(
        e_handle,
        e_state,
        nccl_metadata::ProxyOpStateV3::cast_from_union(&*e_state_args),
    )
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_record_event_state_v4(
    e_handle: *mut libc::c_void,
    e_state: profiler_shim::ncclProfilerEventState_v4_t,
    e_state_args: *mut profiler_shim::ncclProfilerEventStateArgs_v4_t,
) -> ncclResult_t {
    let handle_type = event_ffi::get_handle_type(e_handle);
    match handle_type {
        event_ffi::Type::ProxyStep => {
            if let Some(mut event) = event::Event::from_ffi(e_handle) {
                let step_state = nccl_metadata::ProxyStepStateV4::cast_from_union(&*e_state_args);
                if let Err(e) =
                    profiler::record_proxystep_event_state_handler(&mut event, e_state, step_state)
                {
                    return e;
                }
                let _ = event::Event::into_ffi(event);
            }
        }

        event_ffi::Type::ProxyOpLite => {
            if let Some(mut event) = event::Event::from_ffi(e_handle) {
                if let Err(e) = profiler::record_proxyop_event_state_handler_v4(&mut event, e_state)
                {
                    return e;
                }
                let _ = event::Event::into_ffi(event);
            }
        }

        _ => (),
    }
    profiler_shim::ncclResult_t_ncclSuccess
}

#[allow(clippy::missing_safety_doc)]
unsafe extern "C" fn profiler_finalize(context: *mut libc::c_void) -> ncclResult_t {
    if let Err(e) = profiler::finalize_handler(Box::from_raw(context.cast())) {
        e
    } else {
        profiler_shim::ncclResult_t_ncclSuccess
    }
}

/// # Safety
///
/// Type defined by NCCL (C / C++). These types are safe to share among threads.
unsafe impl Sync for profiler_shim::ncclProfiler_v1_t {}
unsafe impl Sync for profiler_shim::ncclProfiler_v2_t {}
unsafe impl Sync for profiler_shim::ncclProfiler_v3_t {}
unsafe impl Sync for profiler_shim::ncclProfiler_v4_t {}

#[allow(non_upper_case_globals)]
#[no_mangle]
pub static ncclProfiler_v1: profiler_shim::ncclProfiler_v1_t = profiler_shim::ncclProfiler_v1_t {
    // SAFETY: string has no interior NUL bytes
    name: unsafe { static_cstr!("GCP_NCCL_PROFILER_V1").as_ptr() },
    init: Some(profiler_init_v1),
    startEvent: Some(profiler_start_event_v1),
    stopEvent: Some(profiler_stop_event),
    recordEventState: Some(profiler_record_event_state_v1),
    finalize: Some(profiler_finalize),
};

#[allow(non_upper_case_globals)]
#[no_mangle]
pub static ncclProfiler_v2: profiler_shim::ncclProfiler_v2_t = profiler_shim::ncclProfiler_v2_t {
    // SAFETY: string has no interior NUL bytes
    name: unsafe { static_cstr!("GCP_NCCL_PROFILER_V2").as_ptr() },
    init: Some(profiler_init_v2),
    startEvent: Some(profiler_start_event_v2),
    stopEvent: Some(profiler_stop_event),
    recordEventState: Some(profiler_record_event_state_v2),
    finalize: Some(profiler_finalize),
};

#[allow(non_upper_case_globals)]
#[no_mangle]
pub static ncclProfiler_v3: profiler_shim::ncclProfiler_v3_t = profiler_shim::ncclProfiler_v3_t {
    // SAFETY: string has no interior NUL bytes
    name: unsafe { static_cstr!("GCP_NCCL_PROFILER_V3").as_ptr() },
    init: Some(profiler_init_v3),
    startEvent: Some(profiler_start_event_v3),
    stopEvent: Some(profiler_stop_event),
    recordEventState: Some(profiler_record_event_state_v3),
    finalize: Some(profiler_finalize),
};

#[allow(non_upper_case_globals)]
#[no_mangle]
pub static ncclProfiler_v4: profiler_shim::ncclProfiler_v4_t = profiler_shim::ncclProfiler_v4_t {
    // SAFETY: string has no interior NUL bytes
    name: unsafe { static_cstr!("GCP_NCCL_PROFILER_V4").as_ptr() },
    init: Some(profiler_init_v4),
    startEvent: Some(profiler_start_event_v4),
    stopEvent: Some(profiler_stop_event),
    recordEventState: Some(profiler_record_event_state_v4),
    finalize: Some(profiler_finalize),
};

// Helper for testing the Profiler type
#[cfg(test)]
fn scoped_profiler_test<F>(profiler: profiler::Profiler, f: F)
where
    F: FnOnce(&profiler::Profiler, &mut profiler::ThreadLocalState<'static>),
{
    // SAFETY: leak the boxed Profiler object until daemon thread ends.
    // spawn_daemon() and join_daemon() call below limits the lifetime
    // of daemon thread to be shorter than the lifetime of the profiler object.

    let profiler_boxed = Box::new(profiler);
    let profiler_ptr = Box::into_raw(profiler_boxed);
    let profiler: &'static profiler::Profiler = unsafe { &*profiler_ptr };
    profiler.spawn_daemon();
    let mut thread_state = profiler.init_thread_state();
    f(profiler, &mut thread_state);
    profiler.join_daemon();
    let _ = unsafe { Box::from_raw(profiler_ptr) };
}
