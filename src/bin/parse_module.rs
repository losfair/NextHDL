use std::fs::read_to_string;

use anyhow::Result;

fn main() -> Result<()> {
  let f = read_to_string(&std::env::args().nth(1).unwrap())?;
  let parser = nexthdl::parser::grammar::ModuleDefParser::new();
  let ast = parser.parse(&f).unwrap();
  println!("{:#?}", ast);
  Ok(())
}
