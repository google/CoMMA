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

use crate::daemon::AtomicHistogram;
use crate::event_ffi;
use crate::nccl_metadata;
use crate::nccl_metadata::NcclOpKey;
use crate::profiler;
use crate::profiler_shim;
use crate::slab;
use crate::step_tracker::EventStep;

use serde_json::json;

use std::sync::Arc;
use std::time::Instant;

#[derive(Debug)]
pub enum Event {
    Group(Box<Group>),
    NcclOpLite(usize),
    NcclOp(usize),
    ProxyOpLite(slab::AllocatedNode<profiler::ProxyOpLocalData>), // Lite == no step tracking
    KernelCh(slab::AllocatedNode<profiler::KernelCh>),
    ProxyOp(slab::AllocatedNode<profiler::ProxyOpLocalData>),
    Dummy(usize),
    SmallNcclOp(usize),
    ProxyStep(slab::AllocatedNode<ProxyStep>),
}

impl Event {
    pub fn new_group<E>(descr: &E, time: Instant) -> Self
    where
        E: nccl_metadata::Event,
    {
        Event::Group(Box::new(Group::from_descr(descr, time)))
    }

    pub fn new_dummyop(val: usize) -> Self {
        Event::Dummy(val)
    }
}

#[derive(Debug, Clone)]
pub struct BasicInfo {
    rank: usize,
    start_time: Instant,
    end_time: Option<Instant>,
}

impl BasicInfo {
    pub fn from_descr<E>(descr: &E, now: Instant) -> Self
    where
        E: nccl_metadata::Event,
    {
        Self {
            rank: descr.rank() as _,
            start_time: now,
            end_time: None,
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    pub fn end_time(&self) -> Option<Instant> {
        self.end_time
    }

    pub fn update_end_time(&mut self, time: Instant) {
        match self.end_time.as_mut() {
            Some(et) => {
                *et = std::cmp::max(*et, time);
            }
            None => {
                self.end_time = Some(time);
            }
        }
    }
}

pub trait ProfilerEvent {
    fn basic_info(&self) -> &BasicInfo;
    fn basic_info_mut(&mut self) -> &mut BasicInfo;

    fn trace_record<F>(&self, time_to_num: F) -> serde_json::Value
    where
        F: FnMut(Instant) -> u64;
}

#[derive(Debug)]
#[repr(align(16))]
pub struct Group {
    basic_info: BasicInfo,
}

impl Group {
    pub fn from_descr<E>(descr: &E, time: Instant) -> Self
    where
        E: nccl_metadata::Event,
    {
        Self {
            basic_info: BasicInfo::from_descr(descr, time),
        }
    }
}

// NcclOp: collective and P2p
#[derive(Debug, Clone)]
pub struct NcclOp {
    basic_info: BasicInfo,
    id: usize,
    is_p2p: bool,
    child_start_time: Option<Instant>,
    comm_hash: Option<u64>, // starting from v4 comm_hash is no longer part of the event descriptor
    descr: nccl_metadata::EventMetadata,
    proxyops: Option<Vec<ProxyOp>>,
}

#[derive(Debug, Clone)]
pub struct ProxyOpInfo {
    pub rank: usize,
    pub parent: event_ffi::ProxyParent,
    pub pid: libc::pid_t,
    pub id: u32,
    pub peer: u32,
    pub is_send: bool,
}

impl ProxyOpInfo {
    pub fn from_descr<E>(descr: &E, id: u32) -> Self
    where
        E: nccl_metadata::ProxyOp,
    {
        ProxyOpInfo {
            rank: descr.rank() as _,
            parent: event_ffi::ProxyParent::from_ffi(descr.parent_obj()),
            pid: descr.pid(),
            id,
            peer: descr.peer() as _,
            is_send: descr.is_send(),
        }
    }

    pub fn parent(&self) -> Option<usize> {
        match self.parent {
            event_ffi::ProxyParent::NcclOp(handle) => Some(handle),
            _ => None,
        }
    }

    pub fn op_key(&self, comm_hash: u64) -> NcclOpKey {
        let ctor = if self.is_send {
            NcclOpKey::NetSend
        } else {
            NcclOpKey::NetRecv
        };
        ctor(comm_hash, self.rank, self.peer as _)
    }
}

#[derive(Debug, Clone)]
pub struct ProxyOp {
    basic_info: BasicInfo,
    info: ProxyOpInfo,
    extra: Option<ProxyOpExtra>,
    tracking_time: bool,

    step_histograms: Vec<Arc<dyn AtomicHistogram<EventStep>>>,
    track_steps: bool,
    aggregate_steps: bool,
    steps: Option<Vec<EventStep>>,
}

#[derive(Debug, Clone)]
pub struct ProxyOpExtra {
    pub n_steps: libc::c_int,
    pub chunk_size: libc::c_int,
    pub channel_id: libc::c_uchar,
}

impl ProxyOpExtra {
    pub fn from_descr<E>(descr: &E) -> Self
    where
        E: nccl_metadata::ProxyOp,
    {
        Self {
            n_steps: descr.n_steps(),
            chunk_size: descr.chunk_size(),
            channel_id: descr.channel_id(),
        }
    }
}

impl ProfilerEvent for Group {
    fn basic_info(&self) -> &BasicInfo {
        &self.basic_info
    }

    fn basic_info_mut(&mut self) -> &mut BasicInfo {
        &mut self.basic_info
    }

    fn trace_record<F>(&self, mut time_to_num: F) -> serde_json::Value
    where
        F: FnMut(Instant) -> u64,
    {
        let basic_info = self.basic_info();
        let start_time = basic_info.start_time();
        let end_time = basic_info.end_time().unwrap_or(start_time);
        let is_reclaimed = basic_info.end_time().is_none();
        let mut json = json!({
            "ph": "X",
            "ts": time_to_num(start_time),
            "dur": (end_time - start_time).as_micros(),
            "cat": "GROUP",
            "name": "Group",
        });
        if is_reclaimed {
            json["reclaimed"] = json!(true);
        }
        json
    }
}

impl NcclOp {
    pub fn from_descr<E>(
        descr: &E,
        time: Instant,
        id: usize,
        comm_hash_override: Option<u64>,
    ) -> Self
    where
        E: nccl_metadata::Event,
    {
        let is_p2p = match descr.type_() as u32 {
            profiler_shim::ncclProfileColl => false,
            profiler_shim::ncclProfileP2p => true,
            _ => panic!("Unknown ncclop type"),
        };
        Self {
            basic_info: BasicInfo::from_descr(descr, time),
            is_p2p,
            id,
            child_start_time: None,
            comm_hash: comm_hash_override,
            descr: descr.clone_to_metadata(),
            proxyops: None,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn get_descr(&self) -> &nccl_metadata::EventMetadata {
        &self.descr
    }

    pub fn is_p2p(&self) -> bool {
        self.is_p2p
    }

    pub fn get_coll_descr(&self) -> Option<&dyn nccl_metadata::Coll> {
        self.descr.try_cast_to_coll()
    }

    pub fn update_child_start_time(&mut self, t: Instant) {
        match self.child_start_time.as_mut() {
            Some(t0) => {
                *t0 = std::cmp::min(*t0, t);
            }
            None => {
                self.child_start_time = Some(t);
            }
        }
    }

    pub fn child_start_time(&self) -> Option<Instant> {
        self.child_start_time
    }

    pub fn child_duration(&self) -> Option<std::time::Duration> {
        let et = self.basic_info().end_time()?;
        let st = self.child_start_time?;
        Some(et - st)
    }

    pub fn comm_hash(&self) -> u64 {
        self.comm_hash.unwrap_or_else(|| {
            if let Some(coll) = self.descr.try_cast_to_coll() {
                coll.comm_hash().unwrap_or(0)
            } else {
                let p2p = self.descr.try_cast_to_p2p().unwrap();
                p2p.comm_hash().unwrap_or(0)
            }
        })
    }

    pub fn op_key(&self) -> NcclOpKey {
        self.descr.get_op_key(self.comm_hash())
    }

    pub fn byte_count(&self) -> usize {
        self.descr.byte_count()
    }

    pub fn add_proxyop(&mut self, proxyop: ProxyOp) {
        if self.proxyops.is_none() {
            self.proxyops = Some(Vec::new());
        }
        self.proxyops.as_mut().unwrap().push(proxyop);
    }
}

impl ProfilerEvent for NcclOp {
    fn basic_info(&self) -> &BasicInfo {
        &self.basic_info
    }

    fn basic_info_mut(&mut self) -> &mut BasicInfo {
        &mut self.basic_info
    }

    fn trace_record<F>(&self, mut time_to_num: F) -> serde_json::Value
    where
        F: FnMut(Instant) -> u64,
    {
        let basic_info = self.basic_info();
        let start_time = basic_info.start_time();
        let end_time = basic_info.end_time().unwrap_or(start_time);
        let is_reclaimed = basic_info.end_time().is_none();
        let mut json = json!({
            "ph": "X",
            "ts": time_to_num(start_time),
            "dur": (end_time - start_time).as_micros(),
            "cat": if self.is_p2p { "P2P" } else { "COLL" },
            "rank": basic_info.rank(),
            "args": {
                "size": self.byte_count(),
            },
        });

        json["comm_hash"] = json!(format!("0x{:016x}", self.comm_hash()));
        if let Some(coll) = self.descr.try_cast_to_coll() {
            json["name"] = json!(coll.op_type().name());
            json["seq_num"] = json!(coll.seq_num());
            json["algo"] = json!(nccl_metadata::algo::name(coll.algo()));
            json["proto"] = json!(nccl_metadata::proto::name(coll.proto()));
            json["n_max_channel"] = json!(coll.n_max_channel());
        } else {
            let p2p = self.descr.try_cast_to_p2p().unwrap();
            json["name"] = json!(if p2p.is_send() { "send" } else { "recv" });
            json["peer"] = json!(p2p.peer());
        }

        if let Some(proxyops) = self.proxyops.as_ref() {
            json["proxyops"] = proxyops
                .iter()
                .map(|op| op.trace_record(&mut time_to_num))
                .collect();
        }

        if let Some(child_start_time) = self.child_start_time {
            json["child_start_ts"] = json!(time_to_num(child_start_time));
            if !is_reclaimed {
                json["child_dur"] = json!((end_time - child_start_time).as_micros());
            }
        }

        if is_reclaimed {
            json["reclaimed"] = json!(true);
            json["timeout_dur"] = json!(start_time.elapsed().as_micros());
        }
        json
    }
}

impl ProxyOp {
    #[allow(dead_code)]
    pub fn from_descr<E>(descr: &E, time: Instant) -> Self
    where
        E: nccl_metadata::ProxyOp,
    {
        let basic_info = BasicInfo::from_descr(descr, time);
        Self {
            basic_info,
            info: ProxyOpInfo::from_descr(descr, 0),
            extra: Some(ProxyOpExtra::from_descr(descr)),
            tracking_time: true,
            step_histograms: Vec::new(),
            track_steps: false,
            aggregate_steps: false,
            steps: None,
        }
    }

    pub fn from_info(info: &ProxyOpInfo, start_time: Instant) -> Self {
        Self {
            basic_info: BasicInfo {
                start_time,
                end_time: None,
                rank: info.rank,
            },
            info: info.clone(),
            extra: None,
            tracking_time: false,
            step_histograms: Vec::new(),
            track_steps: false,
            aggregate_steps: false,
            steps: None,
        }
    }

    pub fn add_extra_info(&mut self, extra: ProxyOpExtra) {
        self.extra = Some(extra)
    }

    pub fn try_set_start_time(&mut self, time: Instant) {
        if !self.tracking_time {
            let basic_info = self.basic_info_mut();
            basic_info.start_time = time;
            self.tracking_time = true;
        }
    }

    pub fn set_end_time(&mut self, time: Instant) {
        let basic_info = self.basic_info_mut();
        basic_info.end_time = Some(time);
    }

    pub fn info(&self) -> &ProxyOpInfo {
        &self.info
    }

    pub fn set_step_histograms(&mut self, histograms: Vec<Arc<dyn AtomicHistogram<EventStep>>>) {
        if self.steps.is_some() {
            let steps = if self.track_steps {
                self.steps.clone().unwrap()
            } else {
                self.steps.take().unwrap()
            };
            for step in steps {
                for histogram in histograms.iter() {
                    histogram.record(&step);
                }
            }
        }
        self.step_histograms = histograms;
    }

    pub fn has_step_histograms(&self) -> bool {
        !self.step_histograms.is_empty()
    }

    pub fn init_step_tracking(&mut self, track_steps: bool, aggregate_steps: bool) {
        self.track_steps = track_steps;
        self.aggregate_steps = aggregate_steps;
    }

    pub fn should_cache_step(&self) -> bool {
        self.aggregate_steps && self.step_histograms.is_empty()
    }

    pub fn add_step(&mut self, step: EventStep) {
        for histogram in self.step_histograms.iter() {
            histogram.record(&step);
        }
        if self.track_steps || self.should_cache_step() {
            if self.steps.is_none() {
                self.steps = Some(Vec::new());
            }
            self.steps.as_mut().unwrap().push(step);
        }
    }
}

impl ProfilerEvent for ProxyOp {
    fn basic_info(&self) -> &BasicInfo {
        &self.basic_info
    }

    fn basic_info_mut(&mut self) -> &mut BasicInfo {
        &mut self.basic_info
    }

    fn trace_record<F>(&self, mut time_to_num: F) -> serde_json::Value
    where
        F: FnMut(Instant) -> u64,
    {
        let basic_info = self.basic_info();
        let mut json = json!({
            "ph": "X",
            "cat": "PROXY",
            "name": "ProxyOp",
            "rank": basic_info.rank(),
            "pid": self.info.pid,
            "peer": self.info.peer,
            "parent": match &self.info.parent {
                event_ffi::ProxyParent::NcclOp(op) => {
                    format!("<ncclop {}>", op)
                }
                event_ffi::ProxyParent::Dummy(val) => {
                    format!("<dummy {}>", val)
                }
                event_ffi::ProxyParent::Null => {
                    String::from("<null>")
                }
            },
            "is_send": self.info.is_send,
        });

        if let Some(extra) = self.extra.as_ref() {
            json["n_steps"] = json!(extra.n_steps);
            json["chunk_size"] = json!(extra.chunk_size);
            json["channel_id"] = json!(extra.channel_id);
        }

        if self.tracking_time {
            let start_time = basic_info.start_time();
            let end_time = basic_info.end_time().unwrap_or(start_time);
            let is_reclaimed = basic_info.end_time().is_none();
            if is_reclaimed {
                json["reclaimed"] = json!(true);
            }
            json["ts"] = json!(time_to_num(start_time));
            json["dur"] = json!((end_time - start_time).as_micros());
        }
        if let Some(steps) = self.steps.as_ref() {
            json["steps"] = steps
                .iter()
                .map(|s| {
                    let mut r = json!({
                        "step": s.step,
                        "size": s.size,
                        "start_time": s.start_time / 1000,
                    });
                    if let Some(t_ns) = s.fifo_ready_time_ns() {
                        r["fifo_ready_time"] = json!(t_ns / 1000);
                    }
                    r["end_time"] = json!(s.end_time_ns() / 1000);
                    r
                })
                .collect();
        }
        json
    }
}

#[derive(Debug)]
#[repr(align(16))]
pub struct ProxyStep {
    pub step: i32,
    pub size: usize,
    pub start_time: Option<Instant>,
    pub fifo_ready_time: Option<Instant>,
    pub end_time: Option<Instant>,
    pub parent: *mut profiler::ProxyOpLocalData,
}

impl ProxyStep {
    pub fn new(step: i32, parent: *mut profiler::ProxyOpLocalData) -> Self {
        Self {
            step,
            size: 0,
            start_time: None,
            fifo_ready_time: None,
            end_time: None,
            parent,
        }
    }

    pub fn finalize<C>(&self, time_to_ns: C) -> EventStep
    where
        C: FnOnce(&Instant) -> u64,
    {
        let start_time = self.start_time.unwrap_or_else(Instant::now);
        let fifo_wait_dur_ns = self
            .fifo_ready_time
            .map(|t| (t - start_time).as_nanos() as u32);
        let net_start_time = self.fifo_ready_time.unwrap_or(start_time);
        EventStep {
            step: self.step,
            size: self.size,
            start_time: time_to_ns(&start_time),
            fifo_wait_dur_ns,
            dur_ns: (self.end_time.unwrap() - net_start_time).as_nanos() as _,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    #[test]
    fn update_end_time_monotonic() {
        let mut basic_info = BasicInfo {
            rank: 0,
            start_time: Instant::now(),
            end_time: None,
        };

        let t0 = Instant::now();
        let t1 = t0 + Duration::from_secs(1);

        basic_info.update_end_time(t1);
        basic_info.update_end_time(t0);
        assert_eq!(basic_info.end_time(), Some(t1));
    }
}
