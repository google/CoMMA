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

use crate::fixed_batch::{StepQueue, FIXED_QUEUE_SIZE};
use crate::nccl_metadata;
use crate::profiler::Version;
use crate::profiler_shim;

#[derive(Debug, Clone)]
pub struct EventStep {
    pub step: i32,
    pub size: usize,
    pub start_time: u64, // start time from profiler init, in nanoseconds
    pub fifo_wait_dur_ns: Option<u32>, // time spent waiting for CTS signal from receiver
    pub dur_ns: u32, // for sender, this is time spent on sending data after receiving CTS from receiver
}

impl EventStep {
    pub fn fifo_ready_time_ns(&self) -> Option<u64> {
        self.fifo_wait_dur_ns.map(|d| self.start_time + d as u64)
    }

    pub fn end_time_ns(&self) -> u64 {
        let fifo_dt = self.fifo_wait_dur_ns.unwrap_or(0);
        self.start_time + (self.dur_ns + fifo_dt) as u64
    }
}

#[derive(Debug, Copy, Clone)]
struct EventStepInProgress {
    step: i32,
    size: Option<usize>,
    start_time: u64,
    fifo_ready_time: Option<u64>,
    end_time: Option<u64>,
}

impl EventStepInProgress {
    fn new(step: i32, t: u64) -> Self {
        Self {
            step,
            size: None,
            start_time: t,
            fifo_ready_time: None,
            end_time: None,
        }
    }

    fn finalize(&self) -> EventStep {
        let fifo_wait_dur_ns = self
            .fifo_ready_time
            .map(|t| u32::try_from(t - self.start_time).unwrap_or(u32::MAX));
        let net_start_time = self.fifo_ready_time.unwrap_or(self.start_time);
        EventStep {
            step: self.step,
            size: self.size.unwrap(),
            start_time: self.start_time,
            fifo_wait_dur_ns,
            dur_ns: u32::try_from(self.end_time.unwrap() - net_start_time).unwrap_or(u32::MAX),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepTracker {
    n_steps: usize,
    is_send: bool,
    is_v1: bool,
    track_fifo_wait: bool,
    steps_in_progress: StepQueue<EventStepInProgress, FIXED_QUEUE_SIZE>,
    accumulated_size: usize,
    last_in_progress_size: usize,
}

impl StepTracker {
    pub fn new(is_send: bool, is_v1: bool, track_fifo_wait: bool) -> Self {
        Self {
            n_steps: 0,
            is_send,
            is_v1,
            track_fifo_wait,
            steps_in_progress: StepQueue::new(),
            accumulated_size: 0,
            last_in_progress_size: 0,
        }
    }

    pub fn update_step<T, S>(
        &mut self,
        state: profiler_shim::ncclProfilerEventState_v1_t,
        args: &S,
        get_time_ns: T,
    ) -> Option<EventStep>
    where
        T: FnOnce() -> u64,
        S: nccl_metadata::ProxyOpState,
    {
        let steps_in_progress = &mut self.steps_in_progress;

        // SAFETY: NCCL profiler API guarantees that this
        // update_step is only called with the proxyOp variant of the union
        let mut to_emit = None;
        match state {
            profiler_shim::proxy_event_state::v2::SEND_REM_FIFO_WAIT => {
                if self.n_steps == 0 && self.is_send {
                    self.accumulated_size = args.trans_size();
                }
                if self.is_send && self.track_fifo_wait {
                    let idx = steps_in_progress.find(|s| s.step == args.steps());
                    if idx.is_none() {
                        let step = EventStepInProgress::new(args.steps(), get_time_ns());
                        steps_in_progress.push_back(step);
                    }
                }
                None
            }
            profiler_shim::proxy_event_state::v1::SEND_TRANSMITTED => {
                let mut step_size = None;
                if S::version() == Version::V1 {
                    if let Some(last) = steps_in_progress.back_mut() {
                        let last_size = args.trans_size() - self.last_in_progress_size;
                        last.size = Some(last_size);
                        if last.end_time.is_some() {
                            // The step has already completed
                            // after filling in the size, it is safe to
                            // "finalize" it
                            let last = steps_in_progress.pop_back().unwrap();
                            to_emit = Some(last.finalize());
                        }
                    }
                    self.last_in_progress_size = args.trans_size();
                } else {
                    step_size = Some(args.trans_size() - self.accumulated_size);
                    self.accumulated_size = args.trans_size();
                }
                if self.track_fifo_wait {
                    let i = steps_in_progress.find(|s| s.step == args.steps()).unwrap();
                    let step = steps_in_progress.get_mut(i);
                    step.fifo_ready_time = Some(get_time_ns());
                    step.size = step_size;
                } else {
                    let mut step = EventStepInProgress::new(args.steps(), get_time_ns());
                    step.size = step_size;
                    steps_in_progress.push_back(step);
                };
                self.n_steps += 1;
                to_emit
            }
            profiler_shim::proxy_event_state::v1::RECV_POSTED => {
                let step = EventStepInProgress::new(args.steps(), get_time_ns());
                if self.n_steps == 0 {
                    self.accumulated_size = args.trans_size();
                }
                self.n_steps += 1;
                steps_in_progress.push_back(step);
                to_emit
            }
            profiler_shim::proxy_event_state::v1::SEND_DONE => {
                // first find the metadata of the step
                let i = steps_in_progress.find(|s| s.step == args.steps()).unwrap();
                // calculate size by using the previous step
                // here NCCL behave differently for send and recv
                let in_progress = steps_in_progress.get_mut(i);
                in_progress.end_time = Some(get_time_ns());
                if in_progress.size.is_some() {
                    to_emit = steps_in_progress.remove_and_apply(i, |s| s.finalize());
                }
                self.accumulated_size = args.trans_size();
                to_emit
            }
            profiler_shim::proxy_event_state::v1::RECV_RECEIVED => {
                // first find the metadata of the step
                let i = steps_in_progress.find(|s| s.step == args.steps()).unwrap();
                // calculate size by using the previous step
                // here NCCL behave differently for send and recv
                let in_progress = steps_in_progress.get_mut(i);
                in_progress.end_time = Some(get_time_ns());
                in_progress.size = Some(args.trans_size() - self.accumulated_size);
                to_emit = steps_in_progress.remove_and_apply(i, |s| s.finalize());
                self.accumulated_size = args.trans_size();
                to_emit
            }
            _ => None,
        }
    }

    pub fn finalize(&mut self) -> Option<EventStep> {
        if self.is_send && self.is_v1 {
            let steps_in_progress = &mut self.steps_in_progress;
            let accumulated_size = self.accumulated_size;
            let last = steps_in_progress.back_mut().unwrap();
            last.size = Some(accumulated_size - self.last_in_progress_size);
            let in_progress = steps_in_progress.pop_back().unwrap();
            Some(in_progress.finalize())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steptracker_send() {
        const SZ: usize = 65536;
        const N_STEP: usize = 1024;

        let mut tracker = StepTracker::new(true, true, false);
        let mut steps = Vec::new();
        let mut trans_size = 0;

        let t0 = std::time::Instant::now();
        let get_time = || -> u64 { t0.elapsed().as_nanos() as u64 };

        for i in 0..(N_STEP * 2) {
            let args = nccl_metadata::ProxyOpStateV1::new((i / 2) as _, trans_size);
            if i % 2 == 0 {
                if let Some(step) = tracker.update_step(
                    profiler_shim::proxy_event_state::v1::SEND_TRANSMITTED,
                    &args,
                    get_time,
                ) {
                    steps.push(step);
                }
                trans_size += SZ;
            } else if let Some(step) = tracker.update_step(
                profiler_shim::proxy_event_state::v1::SEND_DONE,
                &args,
                get_time,
            ) {
                steps.push(step);
            }
        }

        if let Some(step) = tracker.finalize() {
            steps.push(step);
        }

        assert_eq!(steps.len(), N_STEP);
        for (i, step) in steps.iter().enumerate() {
            assert_eq!(step.step, i as i32);
            assert_eq!(step.size, SZ);
        }
    }

    #[test]
    fn steptracker_recv() {
        const SZ: usize = 65536;
        const N_STEP: usize = 1024;

        let mut tracker = StepTracker::new(false, true, false);
        let mut steps = Vec::new();
        let mut trans_size = 0;

        let t0 = std::time::Instant::now();
        let get_time = || -> u64 { t0.elapsed().as_nanos() as u64 };

        for i in 0..(N_STEP * 2) {
            if i % 2 == 0 {
                let args = nccl_metadata::ProxyOpStateV1::new((i / 2) as _, trans_size);
                if let Some(step) = tracker.update_step(
                    profiler_shim::proxy_event_state::v1::RECV_POSTED,
                    &args,
                    get_time,
                ) {
                    steps.push(step);
                }
            } else {
                trans_size += SZ;
                let args = nccl_metadata::ProxyOpStateV1::new((i / 2) as _, trans_size);
                if let Some(step) = tracker.update_step(
                    profiler_shim::proxy_event_state::v1::RECV_RECEIVED,
                    &args,
                    get_time,
                ) {
                    steps.push(step);
                }
            }
        }

        if let Some(step) = tracker.finalize() {
            steps.push(step);
        }

        assert_eq!(steps.len(), N_STEP);
        for (i, step) in steps.iter().enumerate() {
            assert_eq!(step.step, i as i32);
            assert_eq!(step.size, SZ);
        }
    }

    #[test]
    fn steptracker_saturates_dur_over_u32_max() {
        const SZ: usize = 65536;
        let mut tracker = StepTracker::new(true, true, false);

        let clock = std::cell::Cell::new(0u64);
        let get_time = || -> u64 {
            let t = clock.get();
            clock.set(t + 6_000_000_000); // 6 s between events, > u32::MAX ns
            t
        };

        let args = nccl_metadata::ProxyOpStateV1::new(0, 0);
        let first = tracker.update_step(
            profiler_shim::proxy_event_state::v1::SEND_TRANSMITTED,
            &args,
            get_time,
        );
        assert!(first.is_none());

        let args = nccl_metadata::ProxyOpStateV1::new(0, SZ);
        let _ = tracker.update_step(
            profiler_shim::proxy_event_state::v1::SEND_DONE,
            &args,
            get_time,
        );

        let step = tracker.finalize().expect("expected a finalized step");
        assert_eq!(step.dur_ns, u32::MAX);
    }
}
