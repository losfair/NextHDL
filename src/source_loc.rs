use std::{
  collections::BTreeMap,
  fmt::{self, Display},
};

use crate::tracker::EvaluationStackEntry;

pub struct SourceLocResolver {
  loc2lineno: BTreeMap<u64, u64>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SourceLoc {
  pub lineno: u64,
  pub colno: u64,
}

pub struct StackDumpPrinter<'a, 'b> {
  resolver: &'a SourceLocResolver,
  stack: &'b [EvaluationStackEntry],
}

impl SourceLocResolver {
  pub fn new() -> Self {
    Self {
      loc2lineno: BTreeMap::new(),
    }
  }

  pub fn prepare(&mut self, source: &str) {
    let mut loc: u64 = 0;
    let mut lineno: u64 = 1;
    for (i, c) in source.chars().enumerate() {
      if c == '\n' {
        self.loc2lineno.insert(loc, lineno);
        loc = i as u64 + 1;
        lineno += 1;
      }
    }
  }

  pub fn resolve(&self, loc: u64) -> Option<SourceLoc> {
    self
      .loc2lineno
      .range(..=loc)
      .last()
      .map(|(start_loc, lineno)| SourceLoc {
        lineno: *lineno,
        colno: loc - *start_loc + 1,
      })
  }

  pub fn get_stack_dump_printer<'a, 'b>(
    &'a self,
    stack: &'b [EvaluationStackEntry],
  ) -> StackDumpPrinter<'a, 'b> {
    StackDumpPrinter {
      resolver: self,
      stack,
    }
  }
}

struct MaybeSourceLoc(Option<SourceLoc>);

impl Display for MaybeSourceLoc {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if let Some(x) = self.0 {
      write!(f, "{}:{}", x.lineno, x.colno)
    } else {
      write!(f, "{{unknown}}")
    }
  }
}

impl<'a, 'b> Display for StackDumpPrinter<'a, 'b> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for entry in self.stack.iter().rev() {
      let loc = entry
        .loc
        .map(|(start, end)| {
          (
            self.resolver.resolve(start as u64),
            self.resolver.resolve(end as u64),
          )
        })
        .unwrap_or_default();
      write!(
        f,
        "{:?} {}-{}\n",
        entry.ty,
        MaybeSourceLoc(loc.0),
        MaybeSourceLoc(loc.1)
      )?;
    }

    Ok(())
  }
}
