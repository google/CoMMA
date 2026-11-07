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

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]
#![allow(improper_ctypes)]
#![allow(unused_imports)]
include!(concat!(env!("OUT_DIR"), "/profiler_shim_inner.rs"));

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct EventDescrV1(pub ncclProfilerEventDescr_v1_t);

impl AsRef<ncclProfilerEventDescr_v1_t> for EventDescrV1 {
    #[inline(always)]
    fn as_ref(&self) -> &ncclProfilerEventDescr_v1_t {
        &self.0
    }
}

// Safety invariant: The `type` field of the inner `ncclProfilerEventDescr_v2_t`
// must accurately reflect the active union variant in that struct.
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct EventDescrV2(pub ncclProfilerEventDescr_v2_t);

impl AsRef<ncclProfilerEventDescr_v2_t> for EventDescrV2 {
    #[inline(always)]
    fn as_ref(&self) -> &ncclProfilerEventDescr_v2_t {
        &self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct EventDescrV3(pub ncclProfilerEventDescr_v3_t);

impl AsRef<ncclProfilerEventDescr_v3_t> for EventDescrV3 {
    #[inline(always)]
    fn as_ref(&self) -> &ncclProfilerEventDescr_v3_t {
        &self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct EventDescrV4(pub ncclProfilerEventDescr_v4_t);

impl AsRef<ncclProfilerEventDescr_v4_t> for EventDescrV4 {
    #[inline(always)]
    fn as_ref(&self) -> &ncclProfilerEventDescr_v4_t {
        &self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct EventDescrV7(pub ncclProfilerEventDescr_v7_t);

impl AsRef<ncclProfilerEventDescr_v7_t> for EventDescrV7 {
    #[inline(always)]
    fn as_ref(&self) -> &ncclProfilerEventDescr_v7_t {
        &self.0
    }
}

pub type EventDescr = EventDescrV2;

// alias of `ncclProfilerEventState_vX_t` to make the name shorter
pub mod proxy_event_state {
    use super::*;

    pub mod v1 {
        use super::*;

        // NCCL uses the same definition for v1 and v2
        // Here we create alias to avoid mixing v1 and v2 constants and cause
        // confusion.
        pub const SEND_TRANSMITTED: u32 =
            ncclProfilerEventState_t_ncclProfilerProxyOpSendTransmitted;
        pub const SEND_DONE: u32 = ncclProfilerEventState_t_ncclProfilerProxyOpSendDone;

        pub const RECV_POSTED: u32 = ncclProfilerEventState_t_ncclProfilerProxyOpRecvPosted;
        pub const RECV_RECEIVED: u32 = ncclProfilerEventState_t_ncclProfilerProxyOpRecvReceived;
    }

    pub mod v2 {
        use super::*;

        pub const SEND_REM_FIFO_WAIT: u32 =
            ncclProfilerEventState_t_ncclProfilerProxyOpSendRemFifoWait;
        pub const SEND_TRANSMITTED: u32 =
            ncclProfilerEventState_t_ncclProfilerProxyOpSendTransmitted;
        pub const SEND_DONE: u32 = ncclProfilerEventState_t_ncclProfilerProxyOpSendDone;

        pub const RECV_POSTED: u32 = ncclProfilerEventState_t_ncclProfilerProxyOpRecvPosted;
        pub const RECV_RECEIVED: u32 = ncclProfilerEventState_t_ncclProfilerProxyOpRecvReceived;
    }

    pub mod v4 {
        use super::*;
        pub const PROXYOP_IN_PROGRESS: u32 =
            ncclProfilerEventState_t_ncclProfilerProxyOpInProgress_v4;
        pub const SEND_PEER_WAIT: u32 =
            ncclProfilerEventState_t_ncclProfilerProxyStepSendPeerWait_v4;
        pub const SEND_WAIT: u32 = ncclProfilerEventState_t_ncclProfilerProxyStepSendWait;

        pub const RECV_WAIT: u32 = ncclProfilerEventState_t_ncclProfilerProxyStepRecvWait;
        pub const RECV_FLUSH_WAIT: u32 =
            ncclProfilerEventState_t_ncclProfilerProxyStepRecvFlushWait;
    }

    pub mod v7 {
        pub use super::v4::*;
        use super::*;
        pub const KERNEL_PHASE_STOP: u32 = ncclProfilerEventState_t_ncclProfilerKernelPhaseStop;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn dummy_group_descr() -> EventDescr {
        let mut descr: ncclProfilerEventDescr_v2_t = unsafe { std::mem::zeroed() };
        descr.type_ = ncclProfileGroup as _;
        descr.parentObj = std::ptr::null_mut();
        descr.rank = 0;
        EventDescrV2(descr)
    }

    pub(crate) fn dummy_coll_descr() -> EventDescr {
        let mut descr: ncclProfilerEventDescr_v2_t = unsafe { std::mem::zeroed() };
        descr.type_ = ncclProfileColl as _;
        descr.parentObj = std::ptr::null_mut();
        descr.rank = 1;
        unsafe {
            let coll = &mut descr.__bindgen_anon_1.coll;
            coll.commHash = 0x1234;
            coll.seqNumber = 123;
            coll.func = c"AllGather".as_ptr();
            coll.count = 65536;
            coll.datatype = c"ncclInt8".as_ptr();
        }
        EventDescrV2(descr)
    }

    pub(crate) fn dummy_proxyop_descr() -> EventDescr {
        let mut descr: ncclProfilerEventDescr_v2_t = unsafe { std::mem::zeroed() };
        descr.type_ = ncclProfileProxyOp as _;
        descr.parentObj = std::ptr::null_mut();
        descr.rank = 2;
        unsafe {
            let proxyop = &mut descr.__bindgen_anon_1.proxyOp;
            proxyop.pid = 42;
            proxyop.channelId = 123;
            proxyop.peer = 2;
            proxyop.nSteps = 65536;
            proxyop.chunkSize = 1 << 20;
            proxyop.isSend = 1;
        }
        EventDescrV2(descr)
    }

    pub(crate) fn dummy_kernelch_descr_v7(parent_obj: *mut std::ffi::c_void) -> EventDescrV7 {
        let mut descr: ncclProfilerEventDescr_v7_t = unsafe { std::mem::zeroed() };
        descr.type_ = ncclProfileKernelCh as _;
        descr.parentObj = parent_obj;
        descr.rank = 0;
        unsafe {
            let kernel_ch = &mut descr.__bindgen_anon_1.kernelCh;
            kernel_ch.channelId = 1;
            kernel_ch.pTimer = 1000;
        }
        EventDescrV7(descr)
    }

    pub(crate) fn dummy_kernelphase_descr_v7(
        parent_obj: *mut std::ffi::c_void,
        phase_id: u8,
        p_timer: u64,
    ) -> EventDescrV7 {
        let mut descr: ncclProfilerEventDescr_v7_t = unsafe { std::mem::zeroed() };
        descr.type_ = ncclProfileKernelPhase as _;
        descr.parentObj = parent_obj;
        descr.rank = 0;
        unsafe {
            let kernel_phase = &mut descr.__bindgen_anon_1.kernelPhase;
            kernel_phase.channelId = 1;
            kernel_phase.phaseId = phase_id;
            kernel_phase.phaseName = match phase_id {
                0 => c"initial_sync".as_ptr(),
                1 => c"compute".as_ptr(),
                2 => c"final_sync".as_ptr(),
                _ => c"unknown".as_ptr(),
            };
            kernel_phase.pTimer = p_timer;
        }
        EventDescrV7(descr)
    }
}
