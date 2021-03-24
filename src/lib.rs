#![feature(new_uninit, maybe_uninit_extra)]

#[macro_use]
extern crate log;

pub mod ast;
pub mod eval;
pub mod parser;
pub mod util;
pub mod symbol;
