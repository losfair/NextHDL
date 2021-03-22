use std::fs::read_to_string;
use std::sync::Arc;

use anyhow::Result;
use nexthdl::{
  ast::{ModuleDef, ModuleItem},
  eval::{EvalContext, UnspecializedType, Value},
  parser::state::State,
};

fn main() -> Result<()> {
  let f = read_to_string(&std::env::args().nth(1).unwrap())?;
  let parser = nexthdl::parser::grammar::ModuleDefParser::new();
  let mut state = State::new();
  let ast: ModuleDef = parser.parse(&mut state, &f).unwrap();

  let mut ctx = EvalContext::default();
  for item in ast.items.iter() {
    match item {
      ModuleItem::Fn(_def) => {
        // TODO
        unimplemented!()
        /*
        ctx.names = ctx.names.insert(
          def.name.0.clone(),
          Arc::new(Value::Unspecialized(UnspecializedType::Fn(
            def.meta.clone(),
          ))),
        );
        */
      }
      ModuleItem::Struct(def) => {
        ctx.names = ctx.names.insert(
          def.name.0.clone(),
          Arc::new(Value::Unspecialized(UnspecializedType::Product(
            def.clone(),
          ))),
        );
      }
      _ => {}
    }
  }

  println!("{:#?}", ast);
  Ok(())
}
