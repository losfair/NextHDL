#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: Jemalloc = Jemalloc;

use std::fs::read_to_string;
use std::sync::Arc;

use anyhow::Result;
use arc_swap::ArcSwap;
use nexthdl::{
  ast::{ModuleDef, ModuleItem},
  eval::{
    EvalContext, SpecializedFnValue, UniqueProduct, UnspecializedFnValue, UnspecializedType, Value,
  },
  parser::state::State,
  source_loc::SourceLocResolver,
  util::mk_arc_str,
};

fn main() -> Result<()> {
  pretty_env_logger::init();

  let f = read_to_string(&std::env::args().nth(1).unwrap())?;
  let parser = nexthdl::parser::grammar::ModuleDefParser::new();
  let mut state = State::new();

  let mut source_loc_resolver = SourceLocResolver::new();
  source_loc_resolver.prepare(&f);

  let ast: ModuleDef = parser.parse(&mut state, &f).unwrap();

  println!("{:#?}", ast);

  let mut ctx = EvalContext::new();

  // Placeholder value for now
  let top_level_context = Arc::new(ArcSwap::new(Arc::new(EvalContext::new())));

  // Insert builtin types
  ctx.names.insert_mut(
    mk_arc_str("uint"),
    Arc::new(Value::Unspecialized(UnspecializedType::Uint)),
  );

  // First, insert types...
  for item in ast.items.iter() {
    match item {
      ModuleItem::Struct(def) => {
        let value = Arc::new(Value::Unspecialized(UnspecializedType::Product(Arc::new(
          UniqueProduct {
            def: def.clone(),
            context: top_level_context.clone(),
          },
        ))));
        ctx.names.insert_mut(def.name.0.clone(), value);
      }
      _ => {}
    }
  }

  // Then, insert functions.
  for item in ast.items.iter() {
    match item {
      ModuleItem::Fn(def) => {
        let value = if def.meta.tyargs.len() != 0 {
          Arc::new(Value::UnspecializedFnValue(UnspecializedFnValue {
            ty: def.meta.clone(),
            body: def.specializations.clone(),
            context: top_level_context.clone(),
          }))
        } else {
          let ty = ctx.specialize_fntype(ctx.clone(), &def.meta, &[])?;
          Arc::new(Value::SpecializedFnValue(SpecializedFnValue {
            ty,
            body: def.specializations.clone(),
            context: top_level_context.clone(),
          }))
        };
        ctx.names.insert_mut(def.name.0.clone(), value);
      }
      _ => {}
    }
  }

  // Replace top-level EvalContext.
  top_level_context.store(Arc::new(ctx.clone()));

  let entry = ctx.names.get("entry").expect("missing entry");
  let ret = EvalContext::call_function(entry.clone(), &[], None);
  match ret {
    Ok(x) => {
      println!("ret = {:?}", x);
    }
    Err(e) => {
      println!("error = {:?}", e);
      let stack = ctx.dump_stack();
      let printer = source_loc_resolver.get_stack_dump_printer(&stack);
      println!("{}", printer);
    }
  }

  Ok(())
}
