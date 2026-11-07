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

use static_assertions::const_assert;

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, Ordering};

pub const FREELIST_BATCH: usize = 1024;

#[repr(C)]
#[derive(Debug)]
pub struct FreeListNode<T> {
    data: MaybeUninit<T>,
    next: *mut FreeListNode<T>,
}

const_assert!(std::mem::offset_of!(FreeListNode<usize>, data) == 0);

impl<T> std::default::Default for FreeListNode<T> {
    fn default() -> Self {
        Self {
            data: MaybeUninit::uninit(),
            next: std::ptr::null_mut(),
        }
    }
}

#[derive(Debug)]
pub struct FreeList<T> {
    head: *mut FreeListNode<T>,
    tail: *mut FreeListNode<T>,
    cnt: usize,
}

impl<T> std::default::Default for FreeList<T> {
    fn default() -> Self {
        Self {
            head: std::ptr::null_mut(),
            tail: std::ptr::null_mut(),
            cnt: 0,
        }
    }
}

impl<T> FreeList<T> {
    pub fn new_list(sz: usize) -> Self {
        let mut next = std::ptr::null_mut();
        let mut tail = std::ptr::null_mut();
        for i in 0..sz {
            let mut node = Box::new(FreeListNode::default());
            node.next = next;
            next = Box::into_raw(node);
            if i == 0 {
                tail = next;
            }
        }

        Self {
            head: next,
            tail,
            cnt: sz,
        }
    }

    pub fn alloc<F>(
        &mut self,
        init: F,
        global: Option<&AtomicFreeList<T>>,
        alloc_new: bool,
    ) -> Option<AllocatedNode<T>>
    where
        F: FnOnce(*mut T),
    {
        if self.head.is_null() {
            if let Some(global) = global {
                self.try_fetch(global);
            }
        }

        if !self.head.is_null() {
            // SAFETY: checked that head is not null
            // linked list have the invariant that ptr->next
            // is either null or valid pointer
            unsafe {
                let ptr = self.head;
                let next = (*ptr).next;
                self.head = next;
                (*ptr).next = std::ptr::null_mut();
                init((*ptr).data.as_mut_ptr());
                self.cnt = self.cnt.saturating_sub(1);
                Some(AllocatedNode(ptr))
            }
        } else if alloc_new {
            let mut b: Box<FreeListNode<T>> = Box::default();
            init(b.data.as_mut_ptr());
            Some(AllocatedNode(Box::into_raw(b)))
        } else {
            None
        }
    }

    pub fn alloc_new(
        &mut self,
        obj: T,
        global: Option<&AtomicFreeList<T>>,
        alloc_new: bool,
    ) -> Result<AllocatedNode<T>, T> {
        let mut obj = Some(obj);
        let obj_ref = &mut obj;
        // SAFETY: allocated node is always valid. The pointer is safe to write.
        self.alloc(
            move |ptr| unsafe { std::ptr::write(ptr, obj_ref.take().unwrap()) },
            global,
            alloc_new,
        )
        .ok_or_else(|| obj.unwrap())
    }

    pub fn num_free(&self) -> usize {
        self.cnt
    }

    pub fn free(&mut self, node: AllocatedNode<T>) {
        // SAFETY: AllocatedNode is properly constructed and contains
        // valid pointer to FreeListNode
        unsafe {
            self.free_priv(node.0);
            (*node.0).data.assume_init_drop();
            std::mem::forget(node);
        }
    }

    pub fn take_and_free(&mut self, node: AllocatedNode<T>) -> T {
        // SAFETY: AllocatedNode is properly constructed and contains
        // valid pointer to FreeListNode
        unsafe {
            self.free_priv(node.0);
            let r = (*node.0).data.assume_init_read();
            std::mem::forget(node);
            r
        }
    }

    /// # Safety
    ///
    /// node has to be a valid linked list node
    unsafe fn free_priv(&mut self, node: *mut FreeListNode<T>) {
        (*node).next = self.head;
        if self.head.is_null() {
            self.tail = node;
        }
        self.head = node;
        self.cnt += 1;
    }

    fn try_fetch(&mut self, global: &AtomicFreeList<T>) {
        self.head = global.ptr.swap(std::ptr::null_mut(), Ordering::Acquire);
    }

    fn find_tail(&mut self) {
        // SAFETY: invariant of linked list make sure all pointers are
        // either null or valid
        unsafe {
            let mut ptr = self.head;
            let mut cnt = 0;
            while !ptr.is_null() {
                if (*ptr).next.is_null() {
                    self.tail = ptr;
                }
                cnt += 1;
                ptr = (*ptr).next;
            }
            self.cnt = cnt;
        }
    }

    pub fn try_publish(&mut self, global: &AtomicFreeList<T>) -> bool {
        if !self.head.is_null() {
            // SAFETY: checked that head and tail are valid before deref the pointers.
            unsafe {
                if self.tail.is_null() {
                    self.find_tail();
                }
                let list = global.ptr.load(Ordering::Relaxed);
                (*self.tail).next = list;
                if global
                    .ptr
                    .compare_exchange(list, self.head, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    self.cnt = 0;
                    self.head = std::ptr::null_mut();
                    self.tail = std::ptr::null_mut();
                    true
                } else {
                    (*self.tail).next = std::ptr::null_mut();
                    false
                }
            }
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct AtomicFreeList<T> {
    ptr: AtomicPtr<FreeListNode<T>>,
}

impl<T> std::default::Default for AtomicFreeList<T> {
    fn default() -> Self {
        Self {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AllocatedNode<T>(*mut FreeListNode<T>);

/// # Safety
///
/// The inner pointer is private and could not be copy / cloned.
/// The type is similar to a Box<T>
unsafe impl<T: Send> Send for AllocatedNode<T> {}
unsafe impl<T: Sync> Sync for AllocatedNode<T> {}

impl<T> std::ops::Drop for AllocatedNode<T> {
    fn drop(&mut self) {
        // SAFETY: data is initialized upon creation of AllocatedNode
        unsafe {
            (*self.0).data.assume_init_drop();
            let _ = Box::from_raw(self.0);
        }
    }
}

impl<T> AllocatedNode<T> {
    #[cfg(test)]
    pub fn new(inner: T) -> Self {
        let mut node = Box::new(FreeListNode::<T>::default());
        // SAFETY: node is just created via Box::new
        unsafe {
            std::ptr::write(node.data.as_mut_ptr(), inner);
        }
        Self(Box::into_raw(node))
    }

    pub fn into_raw(node: Self) -> *mut T {
        let ptr = node.0 as _;
        std::mem::forget(node);
        ptr
    }

    /// # Safety
    ///
    /// The raw pointer must be pointing to a FreeListNode,
    /// such as those returned from Self::into_raw()
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        Self(ptr as _)
    }
}

impl<T> std::ops::Deref for AllocatedNode<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        // SAFETY: data is initialized upon creation of AllocatedNode
        unsafe { (*self.0).data.assume_init_ref() }
    }
}

impl<T> std::ops::DerefMut for AllocatedNode<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: data is initialized upon creation of AllocatedNode
        unsafe { (*self.0).data.assume_init_mut() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_list_lossless() {
        const N: usize = 65536;

        let mut freelist: FreeList<usize> = FreeList::new_list(N);
        for i in 0..N {
            // we expect all of the N allocations succeed
            // SAFETY: alloc always pass valid pointer
            assert!(freelist
                .alloc(|ptr| unsafe { std::ptr::write(ptr, i) }, None, false)
                .is_some());
        }
    }

    #[test]
    fn free_list_multithread() {
        use std::sync::{Arc, Barrier};

        const N: usize = 65536;
        let (tx, rx) = std::sync::mpsc::channel::<AllocatedNode<usize>>();
        let barrier_arc = Arc::new(Barrier::new(2));
        let global_list_arc = Arc::new(AtomicFreeList::default());
        std::thread::scope(|s| {
            let barrier = barrier_arc.clone();
            let global_list = global_list_arc.clone();
            s.spawn(move || {
                let mut freelist: FreeList<usize> = FreeList::new_list(0);
                for idx in 0..N {
                    // SAFETY: alloc always pass valid pointer
                    let n = freelist.alloc(|ptr| unsafe { std::ptr::write(ptr, idx) }, None, true);
                    assert!(n.is_some());
                    let mut to_send = n.unwrap();
                    while let Err(n) = tx.send(to_send) {
                        to_send = n.0;
                    }
                }
                assert!(freelist.alloc(|_| {}, None, false).is_none());
                barrier.wait();
                for idx in 0..N {
                    // SAFETY: alloc always pass valid pointer
                    let n = freelist.alloc(
                        |ptr| unsafe { std::ptr::write(ptr, idx) },
                        Some(&global_list),
                        true,
                    );
                    assert!(n.is_some());
                }
            });

            let barrier = barrier_arc.clone();
            let global_list = global_list_arc.clone();
            s.spawn(move || {
                let mut freelist: FreeList<usize> = FreeList::new_list(0);
                for _ in 0..N {
                    let n = rx.recv().unwrap();
                    freelist.free(n);
                }
                freelist.try_publish(&global_list);
                barrier.wait();
            });
        });
    }
}
