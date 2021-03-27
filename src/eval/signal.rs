use std::{collections::BTreeMap, fmt, sync::Arc};

use super::value::Value;
use parking_lot::Mutex;
use std::fmt::Debug;

#[derive(Debug)]
pub struct SignalValue {
  pub name: Option<Arc<str>>,
  pub inner_ty: Arc<Value>,
  pub ty: SignalType,
}

#[derive(Debug)]
pub enum SignalType {
  In,
  Out { assignment: AssignmentTable },
  Register { assignment: AssignmentTable },
}

pub struct AssignmentTable {
  priorities: Mutex<BTreeMap<u32, Vec<SignalAssignment>>>,
}

impl AssignmentTable {
  pub fn new() -> Self {
    Self {
      priorities: Mutex::new(BTreeMap::new()),
    }
  }
}

impl Debug for AssignmentTable {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self.priorities.try_lock() {
      Some(x) => write!(f, "AssignmentTable {{ priorities: {:?} }}", x),
      None => write!(f, "AssignmentTable {{ priorities: [locked] }}"),
    }
  }
}

#[derive(Debug)]
struct SignalAssignment {
  condition: Arc<Value>,
  value: Arc<Value>,
}
