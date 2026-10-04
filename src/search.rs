use crate::bitmask::BitMask;
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use rayon::prelude::*;
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub const DEFAULT_BEAM_RANGE: usize = 500;

/// 基底行の生成と補集合シフトテーブルの作成
pub fn build_shift_table(primes: &[usize], cols: usize) -> Vec<Vec<BitMask>> {
    let mut shift_table = Vec::with_capacity(primes.len());

    for &p in primes {
        let mut complement_shifts = Vec::with_capacity(p);
        for k in 0..p {
            let mut mask = BitMask::new_ones(cols);
            for col in 0..cols {
                let idx = col + 1;
                if col >= k {
                    let orig_idx = idx - k;
                    if orig_idx % p == 1 {
                        mask.set(col, false);
                    }
                }
            }
            complement_shifts.push(mask);
        }
        shift_table.push(complement_shifts);
    }

    shift_table
}

#[derive(Clone)]
struct Frame {
    level: usize,
    base_mask: BitMask,
    next_idx: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum SearchMode {
    Sequential,
    Parallel,
    Beam,
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct SearchResult {
    pub max_count: usize,
    pub shifts: Vec<Vec<usize>>,
}

impl SearchResult {
    pub fn record(&mut self, count: usize, key: Vec<usize>) {
        match count.cmp(&self.max_count) {
            std::cmp::Ordering::Greater => {
                self.max_count = count;
                self.shifts.clear();
                self.shifts.push(key);
            }
            std::cmp::Ordering::Equal => self.shifts.push(key),
            std::cmp::Ordering::Less => {}
        }
    }

    pub fn merge(&mut self, other: SearchResult) {
        if other.max_count > self.max_count {
            self.max_count = other.max_count;
            self.shifts = other.shifts;
            return;
        }

        if other.max_count == self.max_count {
            self.shifts.extend(other.shifts);
        }
    }
}

#[derive(Clone)]
struct BeamState {
    key: Vec<usize>,
    mask: BitMask,
}

fn centered_window(len: usize, max_len: usize) -> std::ops::Range<usize> {
    let selected_len = len.min(max_len);
    let start = (len - selected_len) / 2;
    start..start + selected_len
}

fn starting_window(len: usize, max_len: usize) -> std::ops::Range<usize> {
    0..len.min(max_len)
}

fn ending_window(len: usize, max_len: usize) -> std::ops::Range<usize> {
    let selected_len = len.min(max_len);
    len - selected_len..len
}

fn positioned_window(len: usize, max_len: usize, position: f64) -> std::ops::Range<usize> {
    let selected_len = len.min(max_len);
    let start = ((len - selected_len) as f64 * position).floor() as usize;
    start..start + selected_len
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BeamRange {
    Center(usize),
    Start(usize),
    End(usize),
    At { position: f64, count: usize },
}

impl BeamRange {
    pub fn window(self, len: usize) -> std::ops::Range<usize> {
        match self {
            Self::Center(max_len) => centered_window(len, max_len),
            Self::Start(max_len) => starting_window(len, max_len),
            Self::End(max_len) => ending_window(len, max_len),
            Self::At { position, count } => positioned_window(len, count, position),
        }
    }
}

impl fmt::Display for BeamRange {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Center(count) => write!(formatter, "{count}"),
            Self::Start(count) => write!(formatter, "start:{count}"),
            Self::End(count) => write!(formatter, "end:{count}"),
            Self::At { position, count } => write!(formatter, "at:{position}:{count}"),
        }
    }
}

impl FromStr for BeamRange {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if let Some(value) = value.strip_prefix("at:") {
            let (position, count) = value
                .split_once(':')
                .ok_or_else(|| "beam range position must use the format at:X:COUNT".to_string())?;
            let position = position.parse::<f64>().map_err(|_| {
                format!("beam range position must be a number between 0 and 1: '{position}'")
            })?;
            if !position.is_finite() || !(0.0..=1.0).contains(&position) {
                return Err(format!(
                    "beam range position must be between 0 and 1: '{position}'"
                ));
            }
            let count = parse_beam_range_count(count)?;
            return Ok(Self::At { position, count });
        }

        let (position, count) = match value.split_once(':') {
            Some(("start", count)) => ("start", count),
            Some(("end", count)) => ("end", count),
            Some((position, _)) => {
                return Err(format!(
                    "unknown beam range position '{position}'; use a positive count, start:COUNT, end:COUNT, or at:X:COUNT"
                ));
            }
            None => ("center", value),
        };
        let count = parse_beam_range_count(count)?;

        Ok(match position {
            "start" => Self::Start(count),
            "end" => Self::End(count),
            _ => Self::Center(count),
        })
    }
}

fn parse_beam_range_count(value: &str) -> Result<usize, String> {
    let count = value
        .parse::<usize>()
        .map_err(|_| format!("beam range count must be a positive integer: '{value}'"))?;
    if count == 0 {
        return Err("beam range count must be at least 1".to_string());
    }
    Ok(count)
}

pub struct State {
    pub primes: Vec<usize>,
    pub max_depth: usize,
    pub key: Vec<usize>,
    pub zero_mask: BitMask,
    pub max_count: usize,
    pub shifts: Vec<Vec<usize>>,
    pub node_count: u64,
    pub beam_range: usize,
    shift_table: Vec<Vec<BitMask>>,
}

impl State {
    pub fn new(primes: Vec<usize>, cols: usize, shift_table: Vec<Vec<BitMask>>) -> Self {
        State {
            primes,
            max_depth: 249,
            key: Vec::new(),
            zero_mask: BitMask::new_ones(cols),
            max_count: 0,
            shifts: Vec::new(),
            node_count: 0,
            beam_range: DEFAULT_BEAM_RANGE,
            shift_table,
        }
    }

    pub fn search_beam(&mut self, depth: usize, beam_range: BeamRange) {
        self.beam_range = match beam_range {
            BeamRange::Center(range) | BeamRange::Start(range) | BeamRange::End(range) => range,
            BeamRange::At { count, .. } => count,
        };
        let pb = progress_bar();
        let mut beam = vec![BeamState {
            key: Vec::new(),
            mask: self.zero_mask.clone(),
        }];

        self.key.clear();
        let mut result = SearchResult::default();

        for level in 0..depth {
            let mut next_beam = Vec::new();
            for candidate in &beam {
                let prime = self.primes[level];
                for i in (0..prime).rev() {
                    let mut key = candidate.key.clone();
                    key.push(i);
                    let node_mask = candidate.mask.bitand(&self.shift_table[level][i]);
                    let count = node_mask.count_ones();

                    if count + (depth - level) < result.max_count {
                        continue;
                    }
                    if count < result.max_count {
                        continue;
                    }

                    if level + 1 >= depth {
                        result.record(count, key.clone());
                        if count >= result.max_count {
                            debug!("best level={} key={:?} count={}", level + 1, key, count);
                        }
                    }

                    next_beam.push(BeamState {
                        key,
                        mask: node_mask,
                    });
                }
            }

            if next_beam.is_empty() {
                break;
            }

            let window = beam_range.window(next_beam.len());
            beam = next_beam.drain(window).collect();
        }

        self.max_count = result.max_count;
        self.shifts = result.shifts;
        pb.finish_with_message("探索完了");
    }

    pub fn beam_search(&mut self, depth: usize, beam_range: BeamRange) {
        self.search_beam(depth, beam_range);
    }

    pub fn search(&mut self, depth: usize) {
        let pb = progress_bar();
        let mut stack = vec![Frame {
            level: 0,
            base_mask: self.zero_mask.clone(),
            next_idx: self.primes[0],
        }];
        let mut result = SearchResult::default();

        while let Some(frame) = stack.last_mut() {
            if frame.next_idx == 0 {
                stack.pop();
                if let Some(parent) = stack.last() {
                    self.key.pop();
                    self.zero_mask = parent.base_mask.clone();
                }
                continue;
            }

            frame.next_idx -= 1;
            let i = frame.next_idx;
            let level = frame.level;
            let base_mask = frame.base_mask.clone();
            self.key.push(i);
            self.node_count += 1;

            let node_mask = base_mask.bitand(&self.shift_table[level][i]);
            let count = node_mask.count_ones();

            if self.node_count.is_multiple_of(10_000) {
                pb.set_position(self.node_count);
                pb.set_message(format!(
                    "best: {} | depth: {}",
                    result.max_count,
                    self.key.len()
                ));
            }

            if count + (depth - level) < result.max_count {
                self.key.pop();
                continue;
            }

            if count < result.max_count {
                self.key.pop();
                continue;
            }

            if level + 1 >= depth {
                result.record(count, self.key.clone());
                debug!(
                    "best level={} key={:?} count={}",
                    level + 1,
                    self.key,
                    count
                );
                self.key.pop();
                continue;
            }

            self.zero_mask = node_mask.clone();
            stack.push(Frame {
                level: level + 1,
                base_mask: node_mask,
                next_idx: self.primes[level + 1],
            });
        }

        self.max_count = result.max_count;
        self.shifts = result.shifts;
        pb.finish_with_message("探索完了");
    }

    pub fn search_parallel(&self, depth: usize) -> SearchResult {
        let max_count = Arc::new(AtomicUsize::new(0));
        let shifts = Arc::new(Mutex::new(Vec::<Vec<usize>>::new()));
        let node_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let pb = progress_bar();

        let p0 = self.primes[0];
        (0..p0).into_par_iter().rev().for_each(|i| {
            let mut key = vec![i];
            let base_mask = self.zero_mask.bitand(&self.shift_table[0][i]);
            let count = base_mask.count_ones();
            if depth == 1 {
                let previous_max = max_count.load(Ordering::Relaxed);
                if count > previous_max {
                    max_count.store(count, Ordering::Relaxed);
                    let mut found_shifts = shifts.lock().unwrap();
                    found_shifts.clear();
                    found_shifts.push(key.clone());
                } else if count == previous_max {
                    shifts.lock().unwrap().push(key.clone());
                }
                return;
            }

            let mut stack = vec![Frame {
                level: 1,
                base_mask,
                next_idx: self.primes[1],
            }];

            while let Some(frame) = stack.last_mut() {
                if frame.next_idx == 0 {
                    stack.pop();
                    if stack.last().is_some() {
                        key.pop();
                    }
                    continue;
                }

                frame.next_idx -= 1;
                let idx = frame.next_idx;
                let level = frame.level;
                let current_base = frame.base_mask.clone();
                key.push(idx);
                let n = node_count.fetch_add(1, Ordering::Relaxed) + 1;
                let node_mask = current_base.bitand(&self.shift_table[level][idx]);
                let c_count = node_mask.count_ones();

                if n.is_multiple_of(10_000) {
                    pb.set_position(n);
                    pb.set_message(format!(
                        "best: {} | depth: {}",
                        max_count.load(Ordering::Relaxed),
                        key.len()
                    ));
                }

                if c_count + (depth - level) < max_count.load(Ordering::Relaxed) {
                    key.pop();
                    continue;
                }

                if c_count < max_count.load(Ordering::Relaxed) {
                    key.pop();
                    continue;
                }

                if level + 1 >= depth {
                    let previous_max = max_count.load(Ordering::Relaxed);
                    if c_count > previous_max {
                        max_count.store(c_count, Ordering::Relaxed);
                        let mut found_shifts = shifts.lock().unwrap();
                        found_shifts.clear();
                        found_shifts.push(key.clone());
                        debug!("best level={} key={:?} count={}", level + 1, key, c_count);
                    } else if c_count == previous_max {
                        shifts.lock().unwrap().push(key.clone());
                        // info!("best level={} key={:?} count={}", level+1, key, c_count);
                    }
                    key.pop();
                    continue;
                }

                stack.push(Frame {
                    level: level + 1,
                    base_mask: node_mask,
                    next_idx: self.primes[level + 1],
                });
            }
        });

        pb.finish_with_message("探索完了");
        let final_shifts = shifts.lock().unwrap();
        let mut result = SearchResult::default();
        result.merge(SearchResult {
            max_count: max_count.load(Ordering::Relaxed),
            shifts: final_shifts.clone(),
        });
        result
    }
}

fn progress_bar() -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] nodes: {human_pos} ({per_sec}) {msg}")
            .unwrap(),
    );
    pb
}

#[cfg(test)]
mod tests {
    use super::{
        build_shift_table, centered_window, BeamRange, SearchResult, State, DEFAULT_BEAM_RANGE,
    };

    #[test]
    fn build_shift_table_creates_expected_complement_masks() {
        let table = build_shift_table(&[2], 6);
        assert_eq!(table.len(), 1);
        assert_eq!(table[0].len(), 2);
        assert_eq!(table[0][0].count_ones(), 3);
        assert_eq!(table[0][1].count_ones(), 3);
    }

    #[test]
    fn build_shift_table_applies_each_shift_at_the_expected_columns() {
        let table = build_shift_table(&[3], 7);

        assert_eq!(table[0][0].count_ones(), 4);
        assert_eq!(table[0][1].count_ones(), 5);
        assert_eq!(table[0][2].count_ones(), 5);

        let shifted_once = table[0][1].clone();
        assert_eq!(
            shifted_once.bitand(&table[0][0]).count_ones(),
            2,
            "different shifts must exclude different residue classes"
        );
    }

    #[test]
    fn sequential_and_parallel_search_find_best_results() {
        let primes = vec![2, 3];
        let cols = 4;
        let table = build_shift_table(&primes, cols);
        let mut sequential = State::new(primes.clone(), cols, table.clone());
        sequential.max_depth = 2;
        sequential.search(2);
        let mut parallel = State::new(primes.clone(), cols, table);
        parallel.max_depth = 2;
        let result = parallel.search_parallel(2);

        assert_eq!(sequential.max_count, 2);
        assert_eq!(result.max_count, 2);
        assert!(!result.shifts.is_empty());
        for shift_path in &result.shifts {
            assert_eq!(shift_path.len(), 2);
            for (level, &shift) in shift_path.iter().enumerate() {
                assert!(shift < primes[level]);
            }
        }
    }

    #[test]
    fn beam_search_keeps_best_partial_states() {
        let primes = vec![2, 3];
        let cols = 4;
        let table = build_shift_table(&primes, cols);
        let mut beam = State::new(primes, cols, table);
        beam.max_depth = 2;

        beam.search_beam(2, BeamRange::Center(2));

        assert_eq!(beam.max_count, 2);
        assert!(!beam.shifts.is_empty());
        assert_eq!(beam.shifts[0].len(), 2);
    }

    #[test]
    fn all_search_modes_agree_when_beam_retains_every_candidate() {
        let primes = vec![2, 3];
        let cols = 8;
        let table = build_shift_table(&primes, cols);

        let mut sequential = State::new(primes.clone(), cols, table.clone());
        sequential.search(2);

        let parallel = State::new(primes.clone(), cols, table.clone()).search_parallel(2);

        let mut beam = State::new(primes, cols, table);
        beam.search_beam(2, BeamRange::Center(6));

        assert_eq!(parallel.max_count, sequential.max_count);
        assert_eq!(beam.max_count, sequential.max_count);
        assert_eq!(beam.shifts, sequential.shifts);
    }

    #[test]
    fn centered_window_limits_candidates_around_the_middle() {
        assert_eq!(centered_window(1_000, DEFAULT_BEAM_RANGE), 250..750);
        assert_eq!(centered_window(499, DEFAULT_BEAM_RANGE), 0..499);
        assert_eq!(centered_window(501, DEFAULT_BEAM_RANGE), 0..500);
    }

    #[test]
    fn end_beam_range_keeps_candidates_from_the_end() {
        assert_eq!(BeamRange::End(500).window(1_000), 500..1_000);
        assert_eq!(BeamRange::End(500).window(499), 0..499);
    }

    #[test]
    fn start_beam_range_keeps_candidates_from_the_beginning() {
        assert_eq!(BeamRange::Start(500).window(1_000), 0..500);
        assert_eq!(BeamRange::Start(500).window(499), 0..499);
    }

    #[test]
    fn positioned_beam_range_uses_the_requested_relative_start() {
        assert_eq!(
            BeamRange::At {
                position: 0.25,
                count: 500,
            }
            .window(1_000),
            125..625
        );
        assert_eq!(
            BeamRange::At {
                position: 0.5,
                count: 500,
            }
            .window(1_001),
            250..750
        );
        assert_eq!(
            BeamRange::At {
                position: 1.0,
                count: 500,
            }
            .window(1_000),
            500..1_000
        );
    }

    #[test]
    fn beam_range_parses_center_start_end_and_relative_positions() {
        assert_eq!("500".parse(), Ok(BeamRange::Center(500)));
        assert_eq!("start:500".parse(), Ok(BeamRange::Start(500)));
        assert_eq!("end:500".parse(), Ok(BeamRange::End(500)));
        assert_eq!(
            "at:0.25:500".parse(),
            Ok(BeamRange::At {
                position: 0.25,
                count: 500,
            })
        );
        assert!("end:0".parse::<BeamRange>().is_err());
        assert!("at:-0.1:500".parse::<BeamRange>().is_err());
        assert!("at:1.1:500".parse::<BeamRange>().is_err());
        assert!("at:0.5:0".parse::<BeamRange>().is_err());
        assert!("middle:500".parse::<BeamRange>().is_err());
    }

    #[test]
    fn search_result_tracks_best_count_and_ties() {
        let mut result = SearchResult::default();

        result.record(2, vec![1]);
        result.record(2, vec![0]);
        result.record(3, vec![1, 0]);
        result.record(3, vec![0, 1]);

        assert_eq!(result.max_count, 3);
        assert_eq!(result.shifts.len(), 2);
    }

    #[test]
    fn search_result_merge_replaces_lower_best_and_combines_ties() {
        let mut result = SearchResult {
            max_count: 2,
            shifts: vec![vec![0]],
        };

        result.merge(SearchResult {
            max_count: 3,
            shifts: vec![vec![1, 0]],
        });
        result.merge(SearchResult {
            max_count: 3,
            shifts: vec![vec![0, 1]],
        });
        result.merge(SearchResult {
            max_count: 1,
            shifts: vec![vec![1]],
        });

        assert_eq!(result.max_count, 3);
        assert_eq!(result.shifts, vec![vec![1, 0], vec![0, 1]]);
    }
}
