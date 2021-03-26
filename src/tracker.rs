use std::fmt::Debug;
use std::sync::Mutex;

pub struct EvalTracker {
  evaluation_stack: Mutex<EvaluationStack>,
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
    }
  }

  pub fn enter(&self, entry: EvaluationStackEntry) -> EvalTrackerGuard {
    let mut stack = self.evaluation_stack.lock().unwrap();
    let lazy_top = stack.lazy_top;
    stack.stack.truncate(lazy_top);
    stack.stack.push(entry);
    stack.lazy_top += 1;
    EvalTrackerGuard(self)
  }

  pub fn merge(&self, that: &Self) {
    let that_stack = that.evaluation_stack.lock().unwrap();
    self
      .evaluation_stack
      .lock()
      .unwrap()
      .stack
      .extend_from_slice(&that_stack.stack);
  }

  pub fn dump(&self) -> Vec<EvaluationStackEntry> {
    self.evaluation_stack.lock().unwrap().stack.clone()
  }
}

impl<'a> Drop for EvalTrackerGuard<'a> {
  fn drop(&mut self) {
    // Don't pop
    self.0.evaluation_stack.lock().unwrap().lazy_top -= 1;
  }
}

impl Debug for EvalTracker {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "EvalTracker {{ evaluation_stack: {:?} }}",
      self.evaluation_stack.lock().unwrap()
    )
  }
}
