use std::{
  cell::RefCell,
  collections::{HashMap, HashSet},
  mem::ManuallyDrop,
  rc::Rc,
};

use anyhow::Result;
use z3::{
  ast::{Int, BV},
  Config, Context, SatResult, Solver,
};

use super::{SymbolicUint, UintSymbol, UintSymbolV};
use thiserror::Error;

thread_local! {
  static CURRENT: Rc<OwnedSmtBuildContext> = Rc::new(OwnedSmtBuildContext::new());
}

#[derive(Error, Debug)]
pub enum SmtError {
  #[error("circular symbol dependency")]
  CircularSymbolDependency,
}

pub struct OwnedSmtBuildContext {
  z3_ctx: ManuallyDrop<Box<Context>>,
  build_ctx: ManuallyDrop<RefCell<SmtBuildContext<'static>>>,
}

impl OwnedSmtBuildContext {
  pub fn current() -> Rc<Self> {
    CURRENT.with(|x| x.clone())
  }

  fn new() -> Self {
    let mut config = Config::new();
    config.set_model_generation(false);
    config.set_proof_generation(false);
    let z3_ctx = Box::new(Context::new(&config));
    let build_ctx = SmtBuildContext::new(&z3_ctx);
    let build_ctx =
      unsafe { std::mem::transmute::<SmtBuildContext<'_>, SmtBuildContext<'static>>(build_ctx) };
    OwnedSmtBuildContext {
      z3_ctx: ManuallyDrop::new(z3_ctx),
      build_ctx: ManuallyDrop::new(RefCell::new(build_ctx)),
    }
  }

  pub fn solve_boolean(&self, value: &SymbolicUint) -> Result<Option<bool>> {
    self.build_ctx.borrow_mut().solve_boolean(value)
  }
}

impl Drop for OwnedSmtBuildContext {
  fn drop(&mut self) {
    unsafe {
      ManuallyDrop::drop(&mut self.build_ctx);
      ManuallyDrop::drop(&mut self.z3_ctx);
    }
  }
}

struct SmtBuildContext<'ctx> {
  z3_ctx: &'ctx Context,
  building_symbols: HashSet<[u8; 32]>,
  cache: HashMap<[u8; 32], BV<'ctx>>,

  one: BV<'ctx>,
  zero: BV<'ctx>,
}

impl<'ctx> SmtBuildContext<'ctx> {
  fn new(z3_ctx: &'ctx Context) -> Self {
    let one = BV::from_u64(z3_ctx, 1, 1);
    let zero = BV::from_u64(z3_ctx, 0, 1);
    Self {
      z3_ctx,
      building_symbols: Default::default(),
      cache: Default::default(),
      one,
      zero,
    }
  }

  fn solve_boolean(&mut self, value: &SymbolicUint) -> Result<Option<bool>> {
    let value = value.sym.build(self)?;

    let solver = Solver::new(&self.z3_ctx);

    // `not(condition)` being satisfiable means `condition` isn't always true
    solver.assert(&value.bvredor().bvule(&self.zero));
    match solver.check() {
      SatResult::Sat => {}
      SatResult::Unsat => return Ok(Some(true)),
      SatResult::Unknown => return Ok(None),
    }

    solver.reset();
    solver.assert(&value.bvredor().bvugt(&self.zero));
    match solver.check() {
      SatResult::Unsat => Ok(Some(false)),
      _ => Ok(None),
    }
  }
}

trait BuildSmtSymbol {
  fn build<'ctx>(&self, ctx: &mut SmtBuildContext<'ctx>) -> Result<BV<'ctx>>;
  fn build_uncached<'ctx>(&self, ctx: &mut SmtBuildContext<'ctx>) -> Result<BV<'ctx>>;
}

impl BuildSmtSymbol for UintSymbol {
  fn build<'ctx>(&self, ctx: &mut SmtBuildContext<'ctx>) -> Result<BV<'ctx>> {
    if let Some(x) = ctx.cache.get(&self.hash) {
      return Ok(x.clone());
    }

    if !ctx.building_symbols.insert(self.hash) {
      return Err(SmtError::CircularSymbolDependency.into());
    }

    let ret = self.build_uncached(ctx);
    ctx.building_symbols.remove(&self.hash);

    match ret {
      Ok(x) => {
        ctx.cache.insert(self.hash, x.clone());
        Ok(x)
      }
      Err(e) => Err(e),
    }
  }

  fn build_uncached<'ctx>(&self, ctx: &mut SmtBuildContext<'ctx>) -> Result<BV<'ctx>> {
    match &self.v {
      UintSymbolV::External(handle) => {
        let sym = z3::Symbol::Int(handle.index());
        let v = BV::new_const(&ctx.z3_ctx, sym, handle.width());
        Ok(v)
      }
      UintSymbolV::Const(x, bits) => {
        let v = Int::from_str(&ctx.z3_ctx, &format!("{}", x)).expect("cannot build int");
        let v = BV::from_int(&v, *bits);
        Ok(v)
      }
      UintSymbolV::Undefined(bits) => Ok(BV::fresh_const(&ctx.z3_ctx, "undefined_", *bits)),
      UintSymbolV::Add(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?;
        Ok(left + right)
      }
      UintSymbolV::Sub(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left - right)
      }
      UintSymbolV::Mul(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left * right)
      }
      UintSymbolV::Div(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left.bvudiv(&right))
      }
      UintSymbolV::Eq(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left.bvxor(&right).bvredor().bvnot())
      }
      UintSymbolV::Ne(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left.bvxor(&right).bvredor())
      }
      UintSymbolV::Gt(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left.bvugt(&right).ite(&ctx.one, &ctx.zero))
      }
      UintSymbolV::Ge(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left.bvuge(&right).ite(&ctx.one, &ctx.zero))
      }
      UintSymbolV::Lt(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left.bvult(&right).ite(&ctx.one, &ctx.zero))
      }
      UintSymbolV::Le(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?.align_width_u(&left);
        Ok(left.bvule(&right).ite(&ctx.one, &ctx.zero))
      }
      UintSymbolV::LogicAnd(left, right) => {
        let left = left.build(ctx)?;

        // No alignment needed
        let right = right.build(ctx)?;
        Ok(left.bvredor().bvand(&right.bvredor()))
      }
      UintSymbolV::LogicOr(left, right) => {
        let left = left.build(ctx)?;

        // No alignment needed
        let right = right.build(ctx)?;
        Ok(left.bvredor().bvor(&right.bvredor()))
      }
      UintSymbolV::Resize {
        from,
        target_bits,
        signed,
      } => {
        let from = from.build(ctx)?;
        Ok(from.align_width_raw(*target_bits, *signed))
      }
      UintSymbolV::Select {
        predicate,
        on_true,
        on_false,
      } => {
        let on_true = on_true.build(ctx)?;
        let on_false = on_false.build(ctx)?;
        let predicate = predicate.build(ctx)?.bvredor().bvugt(&ctx.zero);
        Ok(predicate.ite(&on_true, &on_false))
      }
      UintSymbolV::Concat(left, right) => {
        let left = left.build(ctx)?;
        let right = right.build(ctx)?;

        // TODO: is the order correct?
        Ok(left.concat(&right))
      }
      UintSymbolV::Slice { base, high, low } => {
        let base = base.build(ctx)?;
        Ok(base.extract(*high, *low))
      }
    }
  }
}

trait BVHelper<'ctx> {
  fn align_width_u(self, to: &BV) -> BV<'ctx>;
  fn align_width_raw(self, to_size: u32, signed: bool) -> BV<'ctx>;
}

impl<'ctx> BVHelper<'ctx> for BV<'ctx> {
  fn align_width_u(self, to: &BV) -> BV<'ctx> {
    self.align_width_raw(to.get_size(), false)
  }

  fn align_width_raw(self, to_size: u32, signed: bool) -> BV<'ctx> {
    let size = self.get_size();
    if size == to_size {
      self
    } else if size < to_size {
      if signed {
        self.sign_ext(to_size - size)
      } else {
        self.zero_ext(to_size - size)
      }
    } else {
      assert!(to_size > 0);
      self.extract(to_size - 1, 0)
    }
  }
}
