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

use crate::gpuviz;

use log::error;

use std::str::FromStr;
use std::sync::LazyLock;
use std::time::Duration;

pub static CONFIG: LazyLock<Config> = LazyLock::new(Config::from_env);

macro_rules! field_from_env {
    ($s: expr, $field: ident) => {
        let env_name = profiler_config!(stringify!($field).to_uppercase());
        $s.$field = parse_env(&env_name);
    };
    ($s: expr, $field: ident, $d: expr) => {
        let env_name = profiler_config!(stringify!($field).to_uppercase());
        $s.$field = parse_env(&env_name).unwrap_or($d);
    };
    ($s: expr, $env_name: literal, $field: ident, $d: expr) => {
        $s.$field = parse_env($env_name).unwrap_or($d);
    };
}

macro_rules! default_from_compile_env {
    ($type: tt, $env_name: literal, $d: expr) => {
        option_env!($env_name)
            .and_then(|s| $type::parse(s).ok())
            .unwrap_or($d)
    };
}

macro_rules! profiler_config {
    ($s: expr) => {
        format!("NCCL_PROFILER_{}", $s)
    };
}

#[derive(Debug, Default, Clone)]
pub struct Config {
    // Basic enable / disable
    pub use_gpuviz: bool, // copybara:strip(gpuviz)
    // copybara:strip_begin(gpuviz)
    // the "cap" of log level we use for GPUViz logs
    // 0: warn and error mapped to info
    // 1: error maps to warn
    // 2: as-is
    pub gpuviz_log_level: usize,
    // copybara:strip_end

    // Profiling granularity
    pub track_group: bool,
    pub track_ncclop: bool,
    pub track_proxyop: bool,
    pub track_interprocess_proxyop: bool,
    pub track_steps: bool,
    pub track_recv_steps: bool,
    pub track_step_fifo_wait: bool,
    pub aggregate_steps: bool,
    pub track_kernel_ch: bool,
    pub ncclop_completion_delay: Duration,

    // Performance related
    pub fifo_batch_size: usize,
    pub ncclop_timeout: Duration,
    pub max_tracked_ncclop: usize,
    pub small_msg_threshold: usize,
    pub skip_nvls: bool,
    pub skip_small_collective: bool,
    pub skip_small_collective_steps: bool,
    pub p2p_sample_rate: f64,
    pub p2p_recv_sample_rate: f64,
    pub use_cached_clock: bool,

    // Export method & config
    pub latency_file: Option<String>,
    pub summary_file: Option<String>,
    pub summary_interval: Duration,

    // Telemetry uploading config
    pub gpuviz_lib: String, // copybara:strip(gpuviz)
    pub telemetry_mode: usize,
    pub heartbeat: bool,
    pub heartbeat_upload_interval: Duration,
}

impl Config {
    fn from_env() -> Self {
        let mut s = Config::default();
        field_from_env!(s, use_gpuviz, true); // copybara:strip(gpuviz)
        field_from_env!(s, gpuviz_log_level, 0); // copybara:strip(gpuviz)

        field_from_env!(s, track_group, false);
        field_from_env!(s, track_ncclop, true);
        field_from_env!(s, track_proxyop, false);
        field_from_env!(s, track_interprocess_proxyop, false);
        field_from_env!(s, track_steps, false);
        field_from_env!(s, track_recv_steps, false);
        field_from_env!(s, track_step_fifo_wait, true);
        field_from_env!(s, aggregate_steps, true);
        field_from_env!(s, track_kernel_ch, false);
        field_from_env!(s, ncclop_completion_delay, Duration::from_secs(2));

        field_from_env!(s, fifo_batch_size, 1024);
        field_from_env!(s, ncclop_timeout, Duration::from_secs(10));
        field_from_env!(s, max_tracked_ncclop, 65536);
        field_from_env!(s, small_msg_threshold, 65536);
        field_from_env!(s, skip_nvls, true);
        field_from_env!(s, skip_small_collective, true);
        field_from_env!(s, skip_small_collective_steps, true);

        let default_sample_rate = default_from_compile_env!(f64, "DEFAULT_P2P_SAMPLE_RATE", 1.0);
        field_from_env!(s, p2p_sample_rate, default_sample_rate);
        s.p2p_sample_rate = s.p2p_sample_rate.clamp(0.0, 1.0);
        log::info!("P2P op sample rate: {}", s.p2p_sample_rate);

        field_from_env!(s, p2p_recv_sample_rate, 0.1);
        if s.p2p_sample_rate < 1.0 && s.p2p_sample_rate > 0.0 {
            s.p2p_recv_sample_rate /= s.p2p_sample_rate;
        }
        s.p2p_recv_sample_rate = s.p2p_recv_sample_rate.clamp(0.0, 1.0);
        log::info!("Recv P2P op sample rate: {}", s.p2p_recv_sample_rate);

        field_from_env!(s, use_cached_clock, false);

        field_from_env!(s, latency_file);
        field_from_env!(s, summary_file);
        field_from_env!(s, summary_interval, Duration::from_secs(60));

        // copybara:strip_begin(gpuviz)
        field_from_env!(s, gpuviz_lib, String::from(gpuviz::GPUVIZ_LIB_NAME));
        // copybara:strip_end
        #[cfg(not(feature = "explicit-optin"))]
        field_from_env!(s, "NCCL_TELEMETRY_MODE", telemetry_mode, 3);
        field_from_env!(s, heartbeat, true);
        field_from_env!(s, heartbeat_upload_interval, Duration::from_secs(10));

        #[cfg(feature = "explicit-optin")]
        field_from_env!(s, "NCCL_TELEMETRY_MODE", telemetry_mode, 0);

        s
    }
}

trait FromConfigStr: Sized {
    type Err;
    fn parse(s: &str) -> Result<Self, Self::Err>;
}

// macro that implements FromConfigStr trait with FromStr
macro_rules! default_config_parser {
    ($t: tt) => {
        impl FromConfigStr for $t {
            type Err = <$t as FromStr>::Err;
            fn parse(s: &str) -> Result<Self, Self::Err> {
                <$t as FromStr>::from_str(s)
            }
        }
    };
}

default_config_parser!(String);
default_config_parser!(usize);
default_config_parser!(f64);

impl FromConfigStr for bool {
    type Err = String;
    fn parse(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        const TRUE: &[&str] = &["true", "y", "yes", "1"];
        const FALSE: &[&str] = &["false", "n", "no", "0"];
        if TRUE.iter().any(|i| *i == s) {
            return Ok(true);
        }
        if FALSE.iter().any(|i| *i == s) {
            return Ok(false);
        }
        Err(format!("Could not parse {} to bool", s))
    }
}

mod duration_parser {
    use nom::{
        character::complete::{alpha1, digit1},
        combinator::map_res,
        multi::fold_many1,
        sequence::pair,
        IResult, Parser as _,
    };

    use std::time::Duration;

    fn parse_u64(input: &str) -> IResult<&str, u64> {
        map_res(digit1, |s: &str| s.parse::<u64>()).parse(input)
    }

    fn parse_duration_component(input: &str) -> IResult<&str, Duration> {
        let (input, (value, unit)) = pair(parse_u64, alpha1).parse(input)?;

        // Match the unit and create the corresponding Duration.
        match unit {
            "d" => Ok((input, Duration::from_secs(value * 24 * 60 * 60))),
            "h" => Ok((input, Duration::from_secs(value * 60 * 60))),
            "m" => Ok((input, Duration::from_secs(value * 60))),
            "s" => Ok((input, Duration::from_secs(value))),
            "ms" => Ok((input, Duration::from_millis(value))),
            "us" => Ok((input, Duration::from_micros(value))),
            "ns" => Ok((input, Duration::from_nanos(value))),
            _ => Err(nom::Err::Failure(nom::error::Error::new(
                unit,
                nom::error::ErrorKind::Tag,
            ))),
        }
    }

    pub fn parse(input: &str) -> IResult<&str, Duration> {
        let (remaining, maybe_dur) = fold_many1(
            parse_duration_component,
            || Ok(Duration::default()),
            |acc: Result<_, nom::Err<_>>, item| {
                acc?.checked_add(item).ok_or_else(|| {
                    nom::Err::Failure(nom::error::Error::new(
                        input,
                        nom::error::ErrorKind::TooLarge,
                    ))
                })
            },
        )
        .parse(input)?;
        Ok((remaining, maybe_dur?))
    }
}

impl FromConfigStr for std::time::Duration {
    type Err = String;
    fn parse(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        if let Ok((remaining, dur)) = duration_parser::parse(&s) {
            if remaining.is_empty() {
                return Ok(dur);
            }
        }
        Err(format!("Could not parse {} to duration", s))
    }
}

fn parse_env<T>(name: &str) -> Option<T>
where
    T: FromConfigStr,
    <T as FromConfigStr>::Err: std::fmt::Debug,
{
    std::env::var(name).ok().and_then(|s| {
        T::parse(&s)
            .map_err(|e| {
                error!("Error parsing config {}, got error {:?}", name, e);
                e
            })
            .ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bool() {
        const TRUE_LITERALS: &[&str] = &["y", "Y", "yes", "YES", "true", "True", "1"];
        for t in TRUE_LITERALS {
            assert_eq!(bool::parse(t), Ok(true));
        }

        const FALSE_LITERALS: &[&str] = &["n", "N", "no", "NO", "false", "False", "0"];
        for f in FALSE_LITERALS {
            assert_eq!(bool::parse(f), Ok(false));
        }

        const ERR_LITERALS: &[&str] = &["not", "correct", "random"];
        for e in ERR_LITERALS {
            assert!(bool::parse(e).is_err());
        }
    }

    #[test]
    fn parse_duration() {
        use std::time::Duration;

        assert_eq!(Duration::parse("1h60s"), Ok(Duration::from_secs(3600 + 60)));
        assert_eq!(Duration::parse("22us"), Ok(Duration::from_micros(22)));
        assert_eq!(Duration::parse("10m"), Ok(Duration::from_secs(600)));
    }
}
