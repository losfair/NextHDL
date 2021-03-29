#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: Jemalloc = Jemalloc;

use std::fs::read_to_string;
use std::sync::Arc;

use anyhow::Result;
use arc_swap::ArcSwapWeak;
use nexthdl::{
  ast::{ModuleDef, ModuleItem},
  eval::{
    value::{
      BuiltinFnValue, SpecializedFnValue, UniqueProduct, UnspecializedFnValue, UnspecializedType,
      Value,
    },
    EvalContext,
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

  let mut ctx = EvalContext::new();

  // Placeholder value for now
  // Invalid weak reference!
  let top_level_context = Arc::new(ArcSwapWeak::new(Arc::downgrade(&Arc::new(
    EvalContext::new(),
  ))));

  // Insert builtin types
  ctx.names.insert_mut(
    mk_arc_str("uint"),
    Arc::new(Value::Unspecialized(UnspecializedType::Uint)),
  );
  ctx.names.insert_mut(
    mk_arc_str("signal"),
    Arc::new(Value::Unspecialized(UnspecializedType::Signal)),
  );
  ctx.names.insert_mut(
    mk_arc_str("mksignal"),
    Arc::new(Value::BuiltinFnValue(BuiltinFnValue::MkSignal)),
  );
  ctx.names.insert_mut(
    mk_arc_str("error"),
    Arc::new(Value::BuiltinFnValue(BuiltinFnValue::Error)),
  );
  ctx.names.insert_mut(
    mk_arc_str("undefined"),
    Arc::new(Value::BuiltinFnValue(BuiltinFnValue::Undefined)),
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
  let ctx_owner = Arc::new(ctx.clone());
  top_level_context.store(Arc::downgrade(&ctx_owner));

  let entry = ctx.names.get("entry").expect("missing entry");
  let tracker = ctx.tracker();
  let ret = EvalContext::call_function(tracker, entry.clone(), &[], None);
  match ret {
    Ok(x) => {
      println!("ret = {:#?}", x);
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
