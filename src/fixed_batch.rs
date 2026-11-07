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

use std::collections::VecDeque;
use std::mem::MaybeUninit;

/// A fixed size vector-like type for pre-allocated buffer
///
/// This implementation also provides an in-place initialization function.
/// It allows fast initialization on pre-allocated memory regions.
///
/// With compiler optimizations this may be unnecessary as move could be
/// optimized into in-place init.
/// But having this explicit method could avoid the debug mode getting very slow.
#[derive(Debug)]
pub struct Batch<T, const N: usize> {
    elements: [MaybeUninit<T>; N],
    num_elements: usize,
}

impl<T, const N: usize> std::default::Default for Batch<T, N> {
    fn default() -> Self {
        Self {
            elements: [(); N].map(|_| MaybeUninit::uninit()),
            num_elements: 0,
        }
    }
}

impl<T, const N: usize> std::ops::Index<usize> for Batch<T, N> {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        // SAFETY: access after bound check
        if idx < self.num_elements {
            unsafe { self.elements.get_unchecked(idx).assume_init_ref() }
        } else {
            panic!("out of bound access");
        }
    }
}

impl<T, const N: usize> Batch<T, N> {
    /// # Safety
    ///
    /// ptr must be a valid pointer to Batch<T, N>
    pub unsafe fn init(ptr: *mut Self) {
        std::ptr::write(&raw mut (*ptr).num_elements, 0);
    }

    pub fn is_full(&self) -> bool {
        self.num_elements == N
    }

    pub fn push(&mut self, ele: T) {
        // SAFETY: updates on self.num_elements prevents writing to
        // already initialized data
        unsafe {
            std::ptr::write(self.elements[self.num_elements].as_mut_ptr(), ele);
            self.num_elements += 1;
        }
    }

    pub fn drain(&mut self) -> Vec<T> {
        let mut r = Vec::new();
        for i in 0..self.num_elements {
            // SAFETY: num_elements only updated with push(),
            // where the corresponding entry is also set.
            unsafe {
                r.push(self.elements[i].assume_init_read());
            }
        }
        self.num_elements = 0;
        r
    }
}

impl<T, const N: usize> std::ops::Drop for Batch<T, N> {
    fn drop(&mut self) {
        for i in 0..self.num_elements {
            // SAFETY: num_elements only updated with push(),
            // where the corresponding entry is also set.
            unsafe {
                self.elements[i].assume_init_drop();
            }
        }
    }
}

pub const FIXED_QUEUE_SIZE: usize = 4;

/// A queue (FIFO) data structure that uses a fix-sized buffer.
///
/// When the buffer is full, fallback to VecDeque.
///
/// This type is used for tracking on-going network steps.
/// The observation here is that NCCL have limited
/// parallelism of network operations (aka steps).
///
/// Therefore, we use a static array when fewer steps are on-going.
/// In the common case (fast path), the VecDeque should never be used.
#[derive(Debug)]
pub struct StepQueue<T, const N: usize = FIXED_QUEUE_SIZE> {
    fixed_arr: [MaybeUninit<T>; N],
    hd_idx: u8,
    len: usize,
    fallback: VecDeque<T>,
}

impl<T, const N: usize> StepQueue<T, N> {
    pub fn new() -> Self {
        const {
            assert!(
                N < u8::MAX as usize,
                "Queue size N must be less than 255 to fit in hd_idx (u8)"
            );
        }
        Self {
            fixed_arr: [const { MaybeUninit::uninit() }; N],
            hd_idx: 0,
            len: 0,
            fallback: VecDeque::new(),
        }
    }

    pub fn push_back(&mut self, ele: T) {
        if self.len < N as _ {
            let idx: usize = (self.hd_idx as usize + self.len) % N;
            self.fixed_arr[idx].write(ele);
        } else {
            self.fallback.push_back(ele);
        }
        self.len += 1;
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else if self.len <= N as _ {
            let idx = (self.hd_idx as usize + self.len - 1) % N;
            let ele = unsafe { self.fixed_arr[idx].assume_init_read() };
            self.len -= 1;
            Some(ele)
        } else {
            self.len -= 1;
            self.fallback.pop_back()
        }
    }

    pub fn find<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(&T) -> bool,
    {
        let mut idx: usize = 0;
        while idx < self.len as _ {
            if idx < N {
                let arr_idx = (self.hd_idx as usize + idx) % N;
                let ele = unsafe { self.fixed_arr[arr_idx].assume_init_ref() };
                if predicate(ele) {
                    return Some(idx);
                }
            } else if predicate(&self.fallback[idx - N]) {
                return Some(idx);
            }
            idx += 1;
        }
        None
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut T {
        assert!(
            idx < self.len as _,
            "index {} out of bound ({})",
            idx,
            self.len
        );
        if idx < N {
            let arr_idx = (self.hd_idx as usize + idx) % N;
            unsafe { self.fixed_arr[arr_idx].assume_init_mut() }
        } else {
            &mut self.fallback[idx - N]
        }
    }

    pub fn back_mut(&mut self) -> Option<&mut T> {
        if self.len > 0 {
            if self.len <= N {
                let idx = (self.hd_idx as usize + self.len - 1) % N;
                unsafe { Some(self.fixed_arr[idx].assume_init_mut()) }
            } else {
                self.fallback.back_mut()
            }
        } else {
            None
        }
    }

    pub fn remove_and_apply<F, R>(&mut self, idx: usize, f: F) -> Option<R>
    where
        F: FnOnce(T) -> R,
    {
        assert!(
            idx < self.len as _,
            "index {} out of bound ({})",
            idx,
            self.len
        );
        if idx < self.len {
            let ele;
            if idx < N {
                let arr_idx = (self.hd_idx as usize + idx) % N;
                ele = unsafe { f(self.fixed_arr[arr_idx].assume_init_read()) };
                let mut dst: usize = self.hd_idx as _;
                if idx == 0 {
                    self.hd_idx = ((self.hd_idx as usize + 1) % N) as u8;
                } else {
                    let mut i = idx + 1;
                    dst = arr_idx;
                    while i < std::cmp::min(N, self.len as _) {
                        let src = (self.hd_idx as usize + i) % N;
                        unsafe {
                            self.fixed_arr[dst].write(self.fixed_arr[src].assume_init_read());
                        }
                        dst = src;
                        i += 1;
                    }
                }
                if let Some(e) = self.fallback.pop_front() {
                    self.fixed_arr[dst].write(e);
                }
            } else {
                ele = f(self.fallback.remove(idx - N).unwrap());
            }
            self.len -= 1;
            Some(ele)
        } else {
            None
        }
    }

    pub fn _len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub fn remove(&mut self, idx: usize) -> Option<T> {
        self.remove_and_apply(idx, std::convert::identity)
    }
}

impl<T, const N: usize> std::ops::Drop for StepQueue<T, N> {
    fn drop(&mut self) {
        let len = std::cmp::min(N, self.len);
        let start = self.hd_idx as usize;
        for i in start..start + len {
            let idx = i % N;
            unsafe {
                self.fixed_arr[idx].assume_init_drop();
            }
        }
    }
}

impl<T, const N: usize> std::clone::Clone for StepQueue<T, N>
where
    T: std::clone::Clone,
{
    fn clone(&self) -> Self {
        let mut n = Self {
            fixed_arr: [const { MaybeUninit::uninit() }; N],
            hd_idx: self.hd_idx,
            len: self.len,
            fallback: self.fallback.clone(),
        };

        let clone_len = std::cmp::min(N, self.len);
        let start = n.hd_idx as usize;
        for i in start..start + clone_len {
            let idx = i % N;
            unsafe {
                n.fixed_arr[idx].write(self.fixed_arr[idx].assume_init_ref().clone());
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StepTestData {
        data: u64,
    }

    impl StepTestData {
        fn new(v: u64) -> Self {
            Self { data: v }
        }
    }

    #[test]
    fn stepqueue_push() {
        const N: usize = 256;

        let mut q = StepQueue::<StepTestData>::new();
        for i in 0..N {
            q.push_back(StepTestData::new(i as _));
        }

        assert_eq!(q.len, N);

        for i in 0..N {
            assert_eq!(q.find(|d| { d.data == i as u64 }), Some(i));
        }
    }

    #[test]
    fn stepqueue_remove() {
        const N: usize = 32;

        let mut q = StepQueue::<StepTestData>::new();
        for i in 0..N {
            q.push_back(StepTestData::new(i as _));
        }

        q.remove(10);
        assert!(q.find(|d| { d.data == 10 }).is_none());
    }

    #[test]
    fn stepqueue_remove_head() {
        const N: usize = 256;

        let mut q = StepQueue::<StepTestData>::new();
        for i in 0..N {
            q.push_back(StepTestData::new(i as _));
        }

        for i in 0..N {
            let head = q.remove(0).unwrap();
            assert_eq!(head.data, i as u64);
        }
        assert_eq!(q.len, 0);

        // test a mixture of push_back and remove
        let limit = FIXED_QUEUE_SIZE;
        let mut expected: u64 = 0;
        for i in 0..N {
            q.push_back(StepTestData::new(i as _));
            if q.len == limit {
                let head = q.remove(0).unwrap();
                assert_eq!(head.data, expected);
                expected += 1;

                let head = q.remove(0).unwrap();
                assert_eq!(head.data, expected);
                expected += 1;
            }
        }

        for _ in (expected as usize)..N {
            q.remove(0);
        }

        // test a mixture of push_back and remove
        // this time wait until the fallback vecdeque is used.
        let limit = FIXED_QUEUE_SIZE + 1;
        let mut expected: u64 = 0;
        for i in 0..N {
            q.push_back(StepTestData::new(i as _));
            if q.len == limit {
                let head = q.remove(0).unwrap();
                assert_eq!(head.data, expected);
                expected += 1;

                let head = q.remove(0).unwrap();
                assert_eq!(head.data, expected);
                expected += 1;
            }
        }
    }

    #[test]
    fn test_stepqueue_remove_at_all_indices() {
        for total_len in 1..=FIXED_QUEUE_SIZE + 2 {
            for start_offset in 0..FIXED_QUEUE_SIZE {
                for remove_idx in 0..total_len {
                    let mut q: StepQueue<StepTestData> = StepQueue::new();
                    for _ in 0..start_offset {
                        q.push_back(StepTestData::new(999));
                        q.remove(0);
                    }
                    for i in 0..total_len {
                        q.push_back(StepTestData::new(i as _));
                    }
                    let removed = q.remove(remove_idx).unwrap();
                    assert_eq!(removed.data, remove_idx as u64);
                    assert_eq!(q.len, total_len - 1);

                    let mut expected = Vec::new();
                    for i in 0..total_len {
                        if i != remove_idx {
                            expected.push(i as u64);
                        }
                    }
                    for exp in expected {
                        let head = q.remove(0).unwrap();
                        assert_eq!(head.data, exp);
                    }
                    assert_eq!(q.len, 0);
                }
            }
        }
    }

    #[test]
    fn batch_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        {
            let mut batch = Batch::<DropCounter, 10>::default();
            for _ in 0..5 {
                batch.push(DropCounter(counter.clone()));
            }
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }
}
