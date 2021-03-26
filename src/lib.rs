#![feature(new_uninit, maybe_uninit_extra, iterator_fold_self)]

#[macro_use]
extern crate log;

pub mod ast;
pub mod eval;
pub mod parser;
pub mod symbol;
pub mod util;
