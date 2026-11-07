// Copyright 2026 Google LLC
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

use libloading::{Library, Symbol};
use nccl_profiler::gpu;
use std::ffi::{c_int, c_uint};

// --- CUDA Driver API FFI for setting up context in test ---
type CUdevice = c_int;
type CUcontext = *mut std::ffi::c_void;
type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;

type CuInitFn = unsafe extern "C" fn(c_uint) -> CUresult;
type CuDeviceGetFn = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuCtxCreateFn = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
type CuCtxDestroyFn = unsafe extern "C" fn(CUcontext) -> CUresult;

fn main() {
    println!("==================================================");
    println!("Starting GPU active device detection test...");
    println!("==================================================");

    // 1. Initialize a dummy CUDA context to simulate a running GPU application.
    // This is required because cuCtxGetDevice queries the thread-local active context.
    let ctx = match setup_dummy_cuda_context() {
        Ok(ctx) => {
            println!(
                "Status: Successfully initialized CUDA and created a dummy context on Device 0."
            );
            Some(ctx)
        }
        Err(e) => {
            println!("Warning: Skipping CUDA context initialization: {}.", e);
            println!(
                "Note: If no active CUDA context exists, the helper will likely return an error."
            );
            None
        }
    };

    // 2. Call our helper to detect the active GPU
    println!("\nCalling gpu::get_active_gpu_metadata()...");
    match gpu::get_active_gpu_metadata() {
        Ok(metadata) => {
            println!("\nSUCCESS: Active GPU Detected!");
            println!("  UUID:         {}", metadata.uuid);
            println!("  PCIe Address: {}", metadata.pcie_address);
        }
        Err(e) => {
            println!("\nFAILURE: Failed to detect active GPU.");
            println!("  Error: {}", e);
        }
    }
    println!("==================================================");

    // 3. Cleanup context
    if let Some((lib, cu_context)) = ctx {
        unsafe {
            if let Ok(cu_ctx_destroy) = lib.get::<Symbol<CuCtxDestroyFn>>(b"cuCtxDestroy\0") {
                let _ = cu_ctx_destroy(cu_context);
                println!("Cleanup: Destroyed dummy CUDA context.");
            }
        }
    }
}

/// Dynamically loads libcuda.so and sets up a dummy context on device 0.
/// Returns the Library (to keep it alive) and the CUcontext.
fn setup_dummy_cuda_context() -> Result<(Library, CUcontext), String> {
    unsafe {
        let lib = Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|e| format!("Failed to load libcuda: {:?}", e))?;

        let cu_init: Symbol<CuInitFn> = lib
            .get(b"cuInit\0")
            .map_err(|e| format!("Failed to get cuInit symbol: {:?}", e))?;
        let cu_device_get: Symbol<CuDeviceGetFn> = lib
            .get(b"cuDeviceGet\0")
            .map_err(|e| format!("Failed to get cuDeviceGet symbol: {:?}", e))?;
        let cu_ctx_create: Symbol<CuCtxCreateFn> = lib
            .get(b"cuCtxCreate\0")
            .map_err(|e| format!("Failed to get cuCtxCreate symbol: {:?}", e))?;

        let res = cu_init(0);
        if res != CUDA_SUCCESS {
            return Err(format!("cuInit failed: {}", res));
        }

        let mut dev: CUdevice = 0;
        let res = cu_device_get(&mut dev, 0);
        if res != CUDA_SUCCESS {
            return Err(format!("cuDeviceGet failed: {}", res));
        }

        let mut ctx: CUcontext = std::ptr::null_mut();
        let res = cu_ctx_create(&mut ctx, 0, dev);
        if res != CUDA_SUCCESS {
            return Err(format!("cuCtxCreate failed: {}", res));
        }

        Ok((lib, ctx))
    }
}
