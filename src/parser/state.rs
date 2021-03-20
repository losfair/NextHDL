use std::{collections::HashSet, sync::Arc};

use crate::util::mk_arc_str;

#[derive(Default)]
pub struct State {
  string_table: HashSet<Arc<str>>,
}

impl State {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn get_string(&mut self, s: &str) -> Arc<str> {
    if let Some(x) = self.string_table.get(s) {
      x.clone()
    } else {
      let value = mk_arc_str(s);
      self.string_table.insert(value.clone());
      value
    }
  }
}
