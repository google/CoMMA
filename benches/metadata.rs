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

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nccl_profiler::nccl_metadata::{algo, datatype_name_to_nbytes, proto, NcclOpType};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::LazyLock;

static DATATYPE_MAP: LazyLock<HashMap<&'static CStr, usize>> = LazyLock::new(|| {
    [
        (c"ncclInt8", 1),
        (c"ncclInt32", 4),
        (c"ncclUint32", 4),
        (c"ncclInt64", 8),
        (c"ncclUint64", 8),
        (c"ncclFloat16", 2),
        (c"ncclFloat32", 4),
        (c"ncclFloat64", 8),
        (c"ncclBfloat16", 2),
        (c"ncclFloat8e4m3", 1),
        (c"ncclFloat8e5m2", 1),
    ]
    .into_iter()
    .collect()
});

static OP_MAP: LazyLock<HashMap<&'static CStr, NcclOpType>> = LazyLock::new(|| {
    [
        (c"Broadcast", NcclOpType::Broadcast),
        (c"Reduce", NcclOpType::Reduce),
        (c"AllGather", NcclOpType::AllGather),
        (c"ReduceScatter", NcclOpType::ReduceScatter),
        (c"AllReduce", NcclOpType::AllReduce),
        (c"Send", NcclOpType::Send),
        (c"Recv", NcclOpType::Recv),
    ]
    .into_iter()
    .collect()
});

static ALGO_MAP: LazyLock<HashMap<&'static CStr, u8>> = LazyLock::new(|| {
    [
        (c"TREE", algo::TREE),
        (c"RING", algo::RING),
        (c"COLLNET_DIRECT", algo::COLLNET_DIRECT),
        (c"COLLNET_CHAIN", algo::COLLNET_CHAIN),
        (c"NVLS", algo::NVLS),
        (c"NVLS_TREE", algo::NVLS_TREE),
        (c"PAT", algo::PAT),
        (c"Unknown", algo::UNKNOWN),
    ]
    .into_iter()
    .collect()
});

static PROTO_MAP: LazyLock<HashMap<&'static CStr, u8>> = LazyLock::new(|| {
    [
        (c"LL", proto::LL),
        (c"LL128", proto::LL128),
        (c"SIMPLE", proto::SIMPLE),
        (c"Unknown", proto::UNKNOWN),
    ]
    .into_iter()
    .collect()
});

const DATATYPES: &[&CStr] = &[
    c"ncclInt8",
    c"ncclInt32",
    c"ncclUint32",
    c"ncclInt64",
    c"ncclUint64",
    c"ncclFloat16",
    c"ncclFloat32",
    c"ncclFloat64",
    c"ncclBfloat16",
    c"ncclFloat8e4m3",
    c"ncclFloat8e5m2",
    c"Unknown",
];

const OPS: &[&CStr] = &[
    c"Broadcast",
    c"Reduce",
    c"AllGather",
    c"ReduceScatter",
    c"AllReduce",
    c"Send",
    c"Recv",
    c"AlltoAll",
];

const ALGOS: &[&CStr] = &[
    c"TREE",
    c"RING",
    c"COLLNET_DIRECT",
    c"COLLNET_CHAIN",
    c"NVLS",
    c"NVLS_TREE",
    c"PAT",
    c"Unknown",
];

const PROTOS: &[&CStr] = &[c"LL", c"LL128", c"SIMPLE", c"Unknown", c""];

fn datatype_suffix_heuristic(name: &CStr) -> usize {
    let bytes = name.to_bytes();
    let len = bytes.len();
    if len > 0 {
        let i = len - 1;
        match bytes[i] {
            b'8' => 1,
            b'6' => 2,
            b'2' => {
                if i > 0 && bytes[i - 1] == b'3' {
                    4
                } else {
                    1
                }
            }
            b'4' => 8,
            _ => DATATYPE_MAP.get(name).copied().unwrap_or(1),
        }
    } else {
        1
    }
}

fn op_prefix_suffix_heuristic(name: &CStr) -> NcclOpType {
    let bytes = name.to_bytes();
    let len = bytes.len();
    if len > 0 {
        let last_idx = len - 1;
        match bytes[0] {
            b'B' => NcclOpType::Broadcast,
            b'R' => match bytes[last_idx] {
                b'e' => NcclOpType::Reduce,
                b'r' => NcclOpType::ReduceScatter,
                b'v' => NcclOpType::Recv,
                _ => NcclOpType::Unknown,
            },
            b'A' => match bytes[last_idx] {
                b'r' => NcclOpType::AllGather,
                b'e' => NcclOpType::AllReduce,
                _ => NcclOpType::Unknown,
            },
            b'S' => NcclOpType::Send,
            _ => OP_MAP.get(name).copied().unwrap_or(NcclOpType::Unknown),
        }
    } else {
        NcclOpType::Unknown
    }
}

fn algo_prefix_suffix_heuristic(name: &CStr) -> u8 {
    let bytes = name.to_bytes();
    let len = bytes.len();
    if len > 0 {
        let last_idx = len - 1;
        match bytes[0] {
            b'T' => algo::TREE,
            b'R' => algo::RING,
            b'C' => match bytes[last_idx] {
                b'T' => algo::COLLNET_DIRECT,
                _ => algo::COLLNET_CHAIN,
            },
            b'N' => match bytes[last_idx] {
                b'E' => algo::NVLS_TREE,
                _ => algo::NVLS,
            },
            b'P' => algo::PAT,
            _ => ALGO_MAP.get(name).copied().unwrap_or(algo::UNKNOWN),
        }
    } else {
        algo::UNKNOWN
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    LazyLock::force(&DATATYPE_MAP);
    LazyLock::force(&OP_MAP);
    LazyLock::force(&ALGO_MAP);
    LazyLock::force(&PROTO_MAP);

    c.bench_function("datatype_name_to_nbytes (exact slice match)", |b| {
        b.iter(|| {
            let mut sum = 0;
            for &dt in black_box(DATATYPES) {
                sum += datatype_name_to_nbytes(black_box(dt));
            }
            sum
        })
    });

    c.bench_function("datatype_name_to_nbytes (suffix + hashmap)", |b| {
        b.iter(|| {
            let mut sum = 0;
            for &dt in black_box(DATATYPES) {
                sum += datatype_suffix_heuristic(black_box(dt));
            }
            sum
        })
    });

    c.bench_function("datatype_name_to_nbytes (hashmap)", |b| {
        b.iter(|| {
            let mut sum = 0;
            for &dt in black_box(DATATYPES) {
                sum += DATATYPE_MAP.get(black_box(dt)).copied().unwrap_or(1);
            }
            sum
        })
    });

    c.bench_function("NcclOpType::from_c_str (exact slice match)", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for &op in black_box(OPS) {
                acc += NcclOpType::from_c_str(black_box(op)) as usize;
            }
            acc
        })
    });

    c.bench_function("NcclOpType::from_c_str (prefix/suffix + hashmap)", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for &op in black_box(OPS) {
                acc += op_prefix_suffix_heuristic(black_box(op)) as usize;
            }
            acc
        })
    });

    c.bench_function("NcclOpType::from_c_str (hashmap)", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for &op in black_box(OPS) {
                acc += OP_MAP
                    .get(black_box(op))
                    .copied()
                    .unwrap_or(NcclOpType::Unknown) as usize;
            }
            acc
        })
    });

    c.bench_function("algo::from_name (exact slice match)", |b| {
        b.iter(|| {
            let mut acc = 0u8;
            for &a in black_box(ALGOS) {
                acc = acc.wrapping_add(algo::from_name(black_box(a)));
            }
            acc
        })
    });

    c.bench_function("algo::from_name (prefix/suffix + hashmap)", |b| {
        b.iter(|| {
            let mut acc = 0u8;
            for &a in black_box(ALGOS) {
                acc = acc.wrapping_add(algo_prefix_suffix_heuristic(black_box(a)));
            }
            acc
        })
    });

    c.bench_function("algo::from_name (hashmap)", |b| {
        b.iter(|| {
            let mut acc = 0u8;
            for &a in black_box(ALGOS) {
                acc =
                    acc.wrapping_add(ALGO_MAP.get(black_box(a)).copied().unwrap_or(algo::UNKNOWN));
            }
            acc
        })
    });

    c.bench_function("proto::from_name (exact slice match)", |b| {
        b.iter(|| {
            let mut acc = 0u8;
            for &p in black_box(PROTOS) {
                acc = acc.wrapping_add(proto::from_name(black_box(p)));
            }
            acc
        })
    });

    c.bench_function("proto::from_name (hashmap)", |b| {
        b.iter(|| {
            let mut acc = 0u8;
            for &p in black_box(PROTOS) {
                acc = acc.wrapping_add(
                    PROTO_MAP
                        .get(black_box(p))
                        .copied()
                        .unwrap_or(proto::UNKNOWN),
                );
            }
            acc
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
