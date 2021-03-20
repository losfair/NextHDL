use std::fs::read_to_string;

use anyhow::Result;
use nexthdl::parser::state::State;

fn main() -> Result<()> {
  let f = read_to_string(&std::env::args().nth(1).unwrap())?;
  let parser = nexthdl::parser::grammar::ModuleDefParser::new();
  let mut state = State::new();
  let ast = parser.parse(&mut state, &f).unwrap();
  println!("{:#?}", ast);
  Ok(())
}
