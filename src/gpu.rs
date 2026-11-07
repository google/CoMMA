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

use nvml_wrapper::Nvml;

#[derive(Debug, Clone)]
pub struct GpuDeviceMetadata {
    pub uuid: String,
    pub pcie_address: String,
}

mod cuda {
    use libloading::{Library, Symbol};
    use std::ffi::{c_char, c_int, CStr};

    // Library and Symbol Name
    const LIB_CUDA_SO_1: &str = "libcuda.so.1";
    const LIB_CUDA_SO: &str = "libcuda.so";
    const CU_CTX_GET_DEVICE: &[u8] = b"cuCtxGetDevice\0";
    const CU_DEVICE_GET_PCI_BUS_ID: &[u8] = b"cuDeviceGetPCIBusId\0";

    // CUDA FFI Types and Constants
    pub type CUdevice = c_int;
    pub type CUresult = c_int;
    pub const CUDA_SUCCESS: CUresult = 0;

    // CUDA Function Signatures
    pub type CuCtxGetDeviceFn = unsafe extern "C" fn(*mut CUdevice) -> CUresult;
    pub type CuDeviceGetPCIBusIdFn = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;

    /// Dynamically loads `libcuda.so` and queries the active GPU's PCIe Bus ID
    /// for the calling thread.
    pub fn get_active_pcie_address() -> Result<String, String> {
        unsafe {
            // Dynamically load the CUDA driver library
            let lib = Library::new(LIB_CUDA_SO_1)
                .or_else(|_| Library::new(LIB_CUDA_SO))
                .map_err(|e| format!("Failed to load libcuda: {:?}", e))?;

            // Load the required symbols using our defined constants
            let cu_ctx_get_device: Symbol<CuCtxGetDeviceFn> = lib
                .get(CU_CTX_GET_DEVICE)
                .map_err(|e| format!("Failed to get cuCtxGetDevice symbol: {:?}", e))?;

            let cu_device_get_pci_bus_id: Symbol<CuDeviceGetPCIBusIdFn> = lib
                .get(CU_DEVICE_GET_PCI_BUS_ID)
                .map_err(|e| format!("Failed to get cuDeviceGetPCIBusId symbol: {:?}", e))?;

            // 1. Get the active CUDA device handle for the current context/thread
            let mut dev: CUdevice = 0;
            let res = cu_ctx_get_device(&mut dev);
            if res != CUDA_SUCCESS {
                return Err(format!(
                    "cuCtxGetDevice failed with CUDA error code: {}",
                    res
                ));
            }

            // 2. Query the PCIe Bus ID for this device
            let mut bus_id_buf = [0 as c_char; 64];
            let res =
                cu_device_get_pci_bus_id(bus_id_buf.as_mut_ptr(), bus_id_buf.len() as c_int, dev);
            if res != CUDA_SUCCESS {
                return Err(format!(
                    "cuDeviceGetPCIBusId failed with CUDA error code: {}",
                    res
                ));
            }

            // 3. Convert the C-string to a Rust String
            let bus_id_cstr = CStr::from_ptr(bus_id_buf.as_ptr());
            let bus_id = bus_id_cstr
                .to_str()
                .map_err(|e| format!("Failed to parse PCIe Bus ID: {:?}", e))?
                .to_string();

            Ok(bus_id)
        }
    }
}

/// Attempts to query the active GPU's metadata (UUID and PCIe address) for the calling thread.
///
/// This function dynamically loads both CUDA and NVML at runtime, ensuring there are
/// no compile-time dependencies. It will return an `Err` gracefully if the driver
/// or libraries are not available.
pub fn get_active_gpu_metadata() -> Result<GpuDeviceMetadata, String> {
    // 1. Get the active GPU's physical PCIe address via our inline CUDA FFI submodule
    let active_pcie_address = cuda::get_active_pcie_address()?;

    // 2. Initialize NVML to query metadata
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {:?}", e))?;

    // 3. Look up the device in NVML using the absolute PCIe address
    let device = nvml
        .device_by_pci_bus_id(active_pcie_address.as_str())
        .map_err(|e| {
            format!(
                "Failed to get NVML device for PCIe {}: {:?}",
                active_pcie_address, e
            )
        })?;

    let uuid = device
        .uuid()
        .map_err(|e| format!("Failed to get UUID: {:?}", e))?;

    Ok(GpuDeviceMetadata {
        uuid,
        pcie_address: active_pcie_address,
    })
}
