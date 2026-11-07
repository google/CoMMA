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

use crate::event::{Event, Group, ProxyStep};
use crate::profiler::{self, KernelCh, ProxyOpLocalData};
use crate::slab;

use static_assertions::const_assert;

pub type Handle = *mut libc::c_void;

pub const N_TYPE_BITS: u32 = 4;
pub const HANDLE_TYPE_MASK: usize = (1 << N_TYPE_BITS) - 1;

pub const N_PHASE_ID_BITS: u32 = 2;
pub const KERNEL_PHASE_SHIFT: u32 = N_TYPE_BITS; // 4
pub const KERNEL_PHASE_MASK: usize = ((1 << N_PHASE_ID_BITS) - 1) << KERNEL_PHASE_SHIFT; // 0b11_0000
pub const KERNEL_HANDLE_PTR_MASK: usize = !((1 << (N_TYPE_BITS + N_PHASE_ID_BITS)) - 1); // !0b111111

pub fn make_kernel_phase_handle(ptr: usize, phase_id: u8) -> Handle {
    let phase_bits = ((phase_id as usize) << KERNEL_PHASE_SHIFT) & KERNEL_PHASE_MASK;
    (ptr | phase_bits | Type::KernelPhase.to_bits()) as _
}

pub fn parse_kernel_phase_handle(handle: Handle) -> (usize, u8) {
    let val = handle as usize;
    let ptr = val & KERNEL_HANDLE_PTR_MASK;
    let phase_id = ((val & KERNEL_PHASE_MASK) >> KERNEL_PHASE_SHIFT) as u8;
    (ptr, phase_id)
}

/// # Safety
///
/// `handle` must be a valid `KernelPhase` handle pointing to a live `KernelCh` on the current thread.
pub unsafe fn parse_kernel_phase_event<'a>(handle: usize) -> Option<(&'a mut KernelCh, u8)> {
    let (kernelch_ptr, phase_id) = parse_kernel_phase_handle(handle as Handle);
    if kernelch_ptr != 0 && (phase_id as usize) < profiler::MAX_KERNEL_PHASES {
        Some((&mut *(kernelch_ptr as *mut KernelCh), phase_id))
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Group,
    NcclOpLite, // NCCL op without step tracking
    NcclOp,
    ProxyOpLite,
    ProxyOp,
    Dummy,
    SmallNcclOp,
    ProxyStep,
    KernelCh,
    KernelPhase,
}

impl Type {
    fn to_bits(self) -> usize {
        match self {
            Type::Group => 0b00,
            Type::NcclOpLite => 0b01,
            Type::NcclOp => 0b10,
            Type::ProxyOpLite => 0b11,
            Type::ProxyOp => 0b100,
            Type::Dummy => 0b101,
            Type::SmallNcclOp => 0b110,
            Type::ProxyStep => 0b111,
            Type::KernelCh => 0b1000,
            Type::KernelPhase => 0b1001,
        }
    }

    fn from_bits(bits: usize) -> Self {
        match bits {
            0b00 => Type::Group,
            0b01 => Type::NcclOpLite,
            0b10 => Type::NcclOp,
            0b11 => Type::ProxyOpLite,
            0b100 => Type::ProxyOp,
            0b101 => Type::Dummy,
            0b110 => Type::SmallNcclOp,
            0b111 => Type::ProxyStep,
            0b1000 => Type::KernelCh,
            0b1001 => Type::KernelPhase,
            _ => panic!("unknown bit pattern"),
        }
    }
}

fn handle(handle: usize, t: Type) -> Handle {
    (handle | t.to_bits()) as _
}

pub fn get_handle_type(handle: Handle) -> Type {
    let val = handle as usize;
    let type_bits = val & HANDLE_TYPE_MASK;
    Type::from_bits(type_bits)
}

pub(crate) fn get_handle_inner(handle: Handle) -> usize {
    (handle as usize) & !HANDLE_TYPE_MASK
}

fn to_ncclop_handle(handle: Handle) -> usize {
    get_handle_inner(handle) >> N_TYPE_BITS
}

pub trait AsFFI: Sized {
    fn into_ffi(self) -> Handle;

    /// # Safety
    ///
    /// `handle` parameter must be a valid handle returned by calling `into_ffi()`
    unsafe fn from_ffi(handle: Handle) -> Option<Self>;
}

const_assert!(std::mem::align_of::<Group>() >= (1 << N_TYPE_BITS));
const_assert!(std::mem::align_of::<ProxyStep>() >= (1 << N_TYPE_BITS));
const_assert!(std::mem::align_of::<KernelCh>() >= (1 << (N_TYPE_BITS + N_PHASE_ID_BITS)));
const_assert!(std::mem::align_of::<ProxyOpLocalData>() >= (1 << N_TYPE_BITS));

impl AsFFI for Event {
    fn into_ffi(self) -> Handle {
        match self {
            Event::Group(group) => {
                let ptr = Box::into_raw(group);
                handle(ptr as _, Type::Group)
            }
            /*
            Event::NcclOpLite(op) => {
                let slab_handle = slab::Entry::into_handle(op) << N_TYPE_BITS;
                handle(slab_handle as _, Type::NcclOpLite)
            }
            Event::NcclOp(op) => {
                let slab_handle = slab::Entry::into_handle(op) << N_TYPE_BITS;
                handle(slab_handle as _, Type::NcclOp)
            }
            */
            Event::NcclOpLite(id) => handle(id << N_TYPE_BITS, Type::NcclOpLite),
            Event::NcclOp(id) => handle(id << N_TYPE_BITS, Type::NcclOp),
            Event::ProxyOpLite(op) => {
                let ptr = slab::AllocatedNode::into_raw(op);
                handle(ptr as _, Type::ProxyOpLite)
            }
            Event::KernelCh(op) => {
                let ptr = slab::AllocatedNode::into_raw(op);
                handle(ptr as _, Type::KernelCh)
            }
            Event::KernelPhase(h) => h as Handle,
            /*
            Event::ProxyOp(id) => {
                let v = (id as usize) << N_TYPE_BITS;
                handle(v, Type::ProxyOp)
            }
            */
            Event::ProxyOp(op) => {
                let ptr = slab::AllocatedNode::into_raw(op);
                handle(ptr as _, Type::ProxyOp)
            }
            Event::Dummy(val) => {
                let v = val << N_TYPE_BITS;
                handle(v, Type::Dummy)
            }
            Event::SmallNcclOp(val) => {
                let v = val << N_TYPE_BITS;
                handle(v, Type::SmallNcclOp)
            }
            Event::ProxyStep(step) => {
                let ptr = slab::AllocatedNode::into_raw(step);
                handle(ptr as _, Type::ProxyStep)
            }
        }
    }

    unsafe fn from_ffi(handle: Handle) -> Option<Self> {
        let handle_type = get_handle_type(handle);
        match handle_type {
            Type::Group => {
                let ptr = get_handle_inner(handle) as _;
                Some(Event::Group(Box::from_raw(ptr)))
            }
            /*
            Type::NcclOpLite => {
                let handle = to_ncclop_handle(handle);
                slab::Entry::from_handle(handle).map(Event::NcclOp)
            }
            Type::NcclOp => {
                let handle = to_ncclop_handle(handle);
                slab::Entry::from_handle(handle).map(Event::NcclOp)
            }
            */
            Type::NcclOpLite => {
                let handle = to_ncclop_handle(handle);
                Some(Event::NcclOpLite(handle))
            }
            Type::NcclOp => {
                let handle = to_ncclop_handle(handle);
                Some(Event::NcclOp(handle))
            }
            Type::ProxyOpLite => {
                let handle = get_handle_inner(handle) as _;
                let proxyop = slab::AllocatedNode::from_raw(handle);
                Some(Event::ProxyOpLite(proxyop))
            }
            Type::KernelCh => {
                let handle = get_handle_inner(handle) as _;
                let proxyop = slab::AllocatedNode::from_raw(handle);
                Some(Event::KernelCh(proxyop))
            }
            Type::KernelPhase => Some(Event::KernelPhase(handle as usize)),
            /*
            Type::ProxyOp => {
                let handle = get_handle_inner(handle);
                let id = handle >> N_TYPE_BITS;
                Some(Event::ProxyOp(id as u32))
            }
            */
            Type::ProxyOp => {
                let handle = get_handle_inner(handle) as _;
                let proxyop = slab::AllocatedNode::from_raw(handle);
                Some(Event::ProxyOp(proxyop))
            }
            Type::Dummy => {
                let handle = get_handle_inner(handle);
                let v = handle >> N_TYPE_BITS;
                Some(Event::Dummy(v))
            }
            Type::SmallNcclOp => {
                let handle = get_handle_inner(handle);
                let v = handle >> N_TYPE_BITS;
                Some(Event::SmallNcclOp(v))
            }
            Type::ProxyStep => {
                let handle = get_handle_inner(handle) as _;
                let proxystep = slab::AllocatedNode::from_raw(handle);
                Some(Event::ProxyStep(proxystep))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProxyParent {
    NcclOp(/* slab handle */ usize),
    Dummy(/* comm_hash */ usize),
    Null,
}

impl ProxyParent {
    pub fn from_ffi(handle: Handle) -> Self {
        if !handle.is_null() {
            if get_handle_type(handle) != Type::Dummy {
                Self::NcclOp(to_ncclop_handle(handle))
            } else {
                Self::Dummy(get_handle_inner(handle))
            }
        } else {
            Self::Null
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::event::ProfilerEvent as _;
    use crate::profiler;
    use crate::profiler_shim::tests::{dummy_group_descr, dummy_proxyop_descr};

    use std::time::Instant;

    #[test]
    fn type_bits_inv() {
        let types = [
            Type::Group,
            Type::NcclOp,
            Type::ProxyOp,
            Type::Dummy,
            Type::KernelCh,
            Type::KernelPhase,
        ];
        for t in types {
            assert_eq!(Type::from_bits(t.to_bits()), t);
        }
    }

    #[test]
    fn ffi_handle_inv() {
        // note: the list of handles should have last N_TYPE_BITS zeroed
        let handles = [0, 48, 1024, 0xabcd1230, 0xfffffff0];
        let types = [
            Type::Group,
            Type::NcclOp,
            Type::ProxyOp,
            Type::Dummy,
            Type::KernelCh,
            Type::KernelPhase,
        ];
        for h in handles {
            for t in types {
                let ffi = handle(h, t);
                assert_eq!(get_handle_inner(ffi), h);
                assert_eq!(get_handle_type(ffi), t);
            }
        }
    }

    #[test]
    fn group_event_mock() {
        let t0 = Instant::now();
        let group_descr = dummy_group_descr();
        let event = Event::new_group(&group_descr, t0);
        let handle = Event::into_ffi(event);

        // SAFETY: handle is valid return value of into_ffi()
        if let Some(Event::Group(mut group)) = unsafe { Event::from_ffi(handle) } {
            assert_eq!(
                group.basic_info().rank(),
                group_descr.as_ref().rank as usize
            );
            assert_eq!(group.basic_info().start_time(), t0);

            let t1 = Instant::now();
            group.basic_info_mut().update_end_time(t1);
            let record = group.trace_record(|t| (t - t0).as_micros() as _);
            assert_eq!(record["cat"], "GROUP");
            assert_eq!(record["dur"], (t1 - t0).as_micros() as u64);
        } else {
            panic!("failed to get group from ffi");
        }
    }

    #[test]
    fn proxyop_event_mock() {
        use crate::nccl_metadata::Version as _;

        let proxyop_descr = dummy_proxyop_descr();
        let event = Event::ProxyOp(slab::AllocatedNode::new(profiler::ProxyOpLocalData::new(
            42,
            // SAFETY: proxyop_descr is a proxyop descriptor
            unsafe { proxyop_descr.cast_to_proxyop() },
            false,
            false,
            /* comm_hash = */ None,
            /* net_bytes = */ None,
        )));
        let handle = Event::into_ffi(event);
        // SAFETY: handle is valid return value of into_ffi()
        if let Some(Event::ProxyOp(proxydata)) = unsafe { Event::from_ffi(handle) } {
            assert_eq!(proxydata.info.id, 42);
            assert!(proxydata.info.is_send);
        } else {
            panic!("failed get proxyop from ffi");
        }
    }

    #[test]
    fn kernel_phase_handle_encoding() {
        let ptrs = [0x1000_0040, 0xabcd_0000, 0x1234_5680];
        for ptr in ptrs {
            for phase_id in 0..=3 {
                let handle = make_kernel_phase_handle(ptr, phase_id);
                assert_eq!(get_handle_type(handle), Type::KernelPhase);
                let (parsed_ptr, parsed_phase) = parse_kernel_phase_handle(handle);
                assert_eq!(parsed_ptr, ptr);
                assert_eq!(parsed_phase, phase_id);
            }
        }
    }

    #[test]
    fn kernel_phase_event_mock() {
        let kch_event = Event::KernelCh(slab::AllocatedNode::new(KernelCh {
            parent_op: Some(0x17e9),
            info: profiler::KernelChInfo {
                channel_id: 3,
                start_timer: 1000,
                stop_timer: 0,
                phases: Default::default(),
            },
        }));
        let kch_ffi = Event::into_ffi(kch_event);
        assert_eq!(get_handle_type(kch_ffi), Type::KernelCh);
        let ptr = get_handle_inner(kch_ffi);

        let phase_id = 1;
        let handle = make_kernel_phase_handle(ptr, phase_id);
        let event = Event::KernelPhase(handle as usize);
        let ffi_handle = Event::into_ffi(event);
        assert_eq!(get_handle_type(ffi_handle), Type::KernelPhase);
        // SAFETY: ffi_handle is valid return value of into_ffi()
        if let Some(Event::KernelPhase(h)) = unsafe { Event::from_ffi(ffi_handle) } {
            let (parsed_ptr, parsed_phase) = parse_kernel_phase_handle(h as Handle);
            assert_eq!(parsed_ptr, ptr);
            assert_eq!(parsed_phase, phase_id);

            // SAFETY: h points to the live KernelCh allocated above
            let (kch_ref, p_id) = unsafe { parse_kernel_phase_event(h) }.unwrap();
            assert_eq!(p_id, phase_id);
            assert_eq!(kch_ref.info.channel_id, 3);
            kch_ref.info.phases[p_id as usize].start_timer = 1200;
            kch_ref.info.phases[p_id as usize].stop_timer = 1800;
        } else {
            panic!("failed to get KernelPhase from ffi");
        }

        // Reclaim the KernelCh handle and verify mutated phase timestamps
        if let Some(Event::KernelCh(kch)) = unsafe { Event::from_ffi(kch_ffi) } {
            assert_eq!(kch.info.phases[1].start_timer, 1200);
            assert_eq!(kch.info.phases[1].stop_timer, 1800);
        } else {
            panic!("failed to get KernelCh from ffi");
        }
    }
}
