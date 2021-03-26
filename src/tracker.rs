use parking_lot::Mutex;
use serde::Serialize;
use std::{fmt::Debug, sync::RwLock};

pub struct EvalTracker {
  evaluation_stack: Mutex<EvaluationStack>,
  signal_table: RwLock<SignalTable>,
}

struct SignalTable {
  signals: Vec<SignalInfo>,
}

pub struct SignalInfo {
  pub width: u32,
}

#[derive(Serialize, Debug, Copy, Clone)]
pub struct SignalHandle {
  index: usize,
  width: u32,
}

impl SignalHandle {
  pub fn width(&self) -> u32 {
    self.width
  }
}

#[derive(Debug)]
struct EvaluationStack {
  stack: Vec<EvaluationStackEntry>,
  lazy_top: usize,
}

#[derive(Debug, Clone)]
pub struct EvaluationStackEntry {
  pub ty: EvaluationStackEntryType,
  pub loc: Option<(usize, usize)>,
}

#[derive(Debug, Copy, Clone)]
pub enum EvaluationStackEntryType {
  Expression,
  FunctionCall,
  Body,
}

pub struct EvalTrackerGuard<'a>(&'a EvalTracker);

impl EvalTracker {
  pub fn new() -> Self {
    EvalTracker {
      evaluation_stack: Mutex::new(EvaluationStack {
        stack: vec![],
        lazy_top: 0,
      }),
      signal_table: RwLock::new(SignalTable { signals: vec![] }),
    }
  }

  pub fn allocate_signal(&self, info: SignalInfo) -> SignalHandle {
    let mut table = self.signal_table.write().unwrap();
    let index = table.signals.len();
    let width = info.width;
    table.signals.push(info);
    SignalHandle { index, width }
  }

  pub fn enter(&self, entry: EvaluationStackEntry) -> EvalTrackerGuard {
    let mut stack = self.evaluation_stack.lock();
    let lazy_top = stack.lazy_top;
    stack.stack.truncate(lazy_top);
    stack.stack.push(entry);
    stack.lazy_top += 1;
    EvalTrackerGuard(self)
  }

  pub fn merge(&self, that: &Self) {
    let that_stack = that.evaluation_stack.lock();
    self
      .evaluation_stack
      .lock()
      .stack
      .extend_from_slice(&that_stack.stack);
  }

  pub fn dump(&self) -> Vec<EvaluationStackEntry> {
    self.evaluation_stack.lock().stack.clone()
  }
}

impl<'a> Drop for EvalTrackerGuard<'a> {
  fn drop(&mut self) {
    // Don't pop
    self.0.evaluation_stack.lock().lazy_top -= 1;
  }
}

impl Debug for EvalTracker {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "EvalTracker {{ evaluation_stack: {:?} }}",
      self.evaluation_stack.lock()
    )
  }
}
