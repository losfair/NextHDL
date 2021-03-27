mod smt;

use anyhow::Result;
use num_bigint::BigUint;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::convert::TryFrom;
use std::{
  ops::{Add, Div, Mul, Sub},
  sync::Arc,
};
use thiserror::Error;

use crate::tracker::SignalHandle;

use self::smt::OwnedSmtBuildContext;

#[derive(Error, Debug)]
pub enum SymbolicError {
  #[error("cannot reduce symbolic value to constant")]
  CannotReduceToConstant,
}

#[derive(Clone, Debug)]
pub struct SymbolicUint {
  sym: Arc<UintSymbol>,
}

impl SymbolicUint {
  pub fn new_const(value: BigUint, bits: u32) -> Self {
    Self::new_v(UintSymbolV::Const(value, bits))
  }

  pub fn new_external(x: SignalHandle) -> Self {
    Self::new_v(UintSymbolV::External(x))
  }

  pub fn smt_solve_boolean(&self) -> Result<Option<bool>> {
    info!("Running SMT solver on SymbolicUint: {:?}", self);
    OwnedSmtBuildContext::current().solve_boolean(self)
  }

  fn new_v(v: UintSymbolV) -> Self {
    Self {
      sym: UintSymbol::new(v),
    }
  }

  pub fn sym_select(self, on_true: SymbolicUint, on_false: SymbolicUint) -> SymbolicUint {
    Self::new_v(UintSymbolV::Select {
      predicate: self.sym,
      on_true: on_true.sym,
      on_false: on_false.sym,
    })
  }

  pub fn sym_eq(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Eq(self.sym, rhs.sym))
  }

  pub fn sym_ne(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Ne(self.sym, rhs.sym))
  }

  pub fn sym_gt(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Gt(self.sym, rhs.sym))
  }

  pub fn sym_ge(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Ge(self.sym, rhs.sym))
  }

  pub fn sym_lt(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Lt(self.sym, rhs.sym))
  }

  pub fn sym_le(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Le(self.sym, rhs.sym))
  }

  pub fn sym_logic_and(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::LogicAnd(self.sym, rhs.sym))
  }

  pub fn sym_logic_or(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::LogicOr(self.sym, rhs.sym))
  }

  pub fn sym_add(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Add(self.sym, rhs.sym))
  }

  pub fn sym_sub(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Sub(self.sym, rhs.sym))
  }

  pub fn sym_mul(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Mul(self.sym, rhs.sym))
  }

  pub fn sym_div(self, rhs: Self) -> SymbolicUint {
    Self::new_v(UintSymbolV::Div(self.sym, rhs.sym))
  }

  pub fn sym_resize(self, target_bits: u32, signed: bool) -> SymbolicUint {
    Self::new_v(UintSymbolV::Resize {
      from: self.sym,
      target_bits,
      signed,
    })
  }

  pub fn bits(&self) -> u32 {
    self.sym.bits
  }

  pub fn as_const(&self) -> Result<&BigUint> {
    match self.sym.v {
      UintSymbolV::Const(ref x, _) => Ok(x),
      _ => Err(SymbolicError::CannotReduceToConstant.into()),
    }
  }

  pub fn const_truthy(&self) -> Option<bool> {
    match self.sym.v {
      UintSymbolV::Const(ref x, _) => Some(u32::try_from(x) != Ok(0)),
      _ => None,
    }
  }

  pub fn is_const(&self) -> bool {
    match self.sym.v {
      UintSymbolV::Const(_, _) => true,
      _ => false,
    }
  }

  pub fn get_hash(&self) -> &[u8; 32] {
    &self.sym.hash
  }
}

impl Add for SymbolicUint {
  type Output = Self;
  fn add(self, rhs: Self) -> Self::Output {
    Self::new_v(UintSymbolV::Add(self.sym, rhs.sym))
  }
}

impl Sub for SymbolicUint {
  type Output = Self;
  fn sub(self, rhs: Self) -> Self::Output {
    Self::new_v(UintSymbolV::Sub(self.sym, rhs.sym))
  }
}

impl Mul for SymbolicUint {
  type Output = Self;
  fn mul(self, rhs: Self) -> Self::Output {
    Self::new_v(UintSymbolV::Mul(self.sym, rhs.sym))
  }
}

impl Div for SymbolicUint {
  type Output = Self;
  fn div(self, rhs: Self) -> Self::Output {
    Self::new_v(UintSymbolV::Div(self.sym, rhs.sym))
  }
}

impl From<bool> for SymbolicUint {
  fn from(x: bool) -> Self {
    match x {
      true => Self::new_const(1u32.into(), 1),
      false => Self::new_const(0u32.into(), 1),
    }
  }
}

/// A symbolic value of `uint`.
#[derive(Serialize, Debug)]
pub(self) struct UintSymbol {
  #[serde(skip)]
  pub v: UintSymbolV,

  #[serde(skip)]
  pub bits: u32,

  /// The hash calculated from `v`.
  pub hash: [u8; 32],
}

impl PartialEq for UintSymbol {
  fn eq(&self, other: &Self) -> bool {
    self.hash == other.hash
  }
}

impl Eq for UintSymbol {}

/// A variant of `UintSymbol`.
#[derive(Serialize, Debug)]
pub(self) enum UintSymbolV {
  /// An external signal.
  External(SignalHandle),

  /// A constant.
  Const(BigUint, u32),

  /// The result of adding two `UintSymbol`s.
  Add(Arc<UintSymbol>, Arc<UintSymbol>),
  Sub(Arc<UintSymbol>, Arc<UintSymbol>),
  Mul(Arc<UintSymbol>, Arc<UintSymbol>),
  Div(Arc<UintSymbol>, Arc<UintSymbol>),

  /// The result of comparing two `UintSymbol`s for equality.
  Eq(Arc<UintSymbol>, Arc<UintSymbol>),

  /// The result of comparing two `UintSymbol`s for inequality.
  Ne(Arc<UintSymbol>, Arc<UintSymbol>),

  /// The result of comparing two `UintSymbol`s for greater-than.
  Gt(Arc<UintSymbol>, Arc<UintSymbol>),

  /// The result of comparing two `UintSymbol`s for greater-than-or-equal.
  Ge(Arc<UintSymbol>, Arc<UintSymbol>),

  /// The result of comparing two `UintSymbol`s for less-than.
  Lt(Arc<UintSymbol>, Arc<UintSymbol>),

  /// The result of comparing two `UintSymbol`s for less-than-or-equal.
  Le(Arc<UintSymbol>, Arc<UintSymbol>),

  LogicAnd(Arc<UintSymbol>, Arc<UintSymbol>),
  LogicOr(Arc<UintSymbol>, Arc<UintSymbol>),

  Resize {
    from: Arc<UintSymbol>,
    target_bits: u32,
    signed: bool,
  },

  /// The result of selecting from two `UintSymbol`s based on a predicate.
  Select {
    predicate: Arc<UintSymbol>,
    on_true: Arc<UintSymbol>,
    on_false: Arc<UintSymbol>,
  },
}

impl UintSymbol {
  fn new(v: UintSymbolV) -> Arc<Self> {
    if let Some(x) = v.try_reduce() {
      return x;
    }

    let bits = match &v {
      UintSymbolV::External(x) => x.width(),
      UintSymbolV::Const(_, bits) => *bits,
      UintSymbolV::Add(left, _) => {
        // Right value will be truncated to match left
        left.bits
      }
      UintSymbolV::Sub(left, _) => {
        // Right value will be truncated to match left
        left.bits
      }
      UintSymbolV::Mul(left, _) => {
        // Right value will be truncated to match left
        left.bits
      }
      UintSymbolV::Div(left, _) => {
        // Right value will be truncated to match left
        left.bits
      }
      UintSymbolV::Eq(_, _)
      | UintSymbolV::Ne(_, _)
      | UintSymbolV::Lt(_, _)
      | UintSymbolV::Le(_, _)
      | UintSymbolV::Gt(_, _)
      | UintSymbolV::Ge(_, _) => 1,
      UintSymbolV::LogicAnd(_, _) | UintSymbolV::LogicOr(_, _) => 1,
      UintSymbolV::Resize { target_bits, .. } => *target_bits,
      UintSymbolV::Select {
        on_true, on_false, ..
      } => {
        assert_eq!(on_true.bits, on_false.bits);
        on_true.bits
      }
    };
    let serialized = bincode::serialize(&v).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(&serialized);
    let hash = hasher.finalize();
    Arc::new(Self {
      v,
      bits,
      hash: hash.into(),
    })
  }
}

impl UintSymbolV {
  fn try_reduce(&self) -> Option<Arc<UintSymbol>> {
    match self {
      UintSymbolV::External { .. } => None,
      UintSymbolV::Const(_, _) => None,
      UintSymbolV::Add(left, right) => reduce_const_binop(left, right, |a, b| a + b),
      UintSymbolV::Sub(left, right) => reduce_const_binop(left, right, |a, b| a - b),
      UintSymbolV::Mul(left, right) => reduce_const_binop(left, right, |a, b| a * b),
      UintSymbolV::Div(left, right) => {
        // FIXME: Division by zero
        reduce_const_binop(left, right, |a, b| a / b)
      }
      UintSymbolV::Eq(left, right) => {
        if left.hash == right.hash {
          Some(UintSymbol::new(UintSymbolV::Const(BigUint::from(1u32), 1)))
        } else {
          reduce_const_relop(left, right, |a, b| a == b)
        }
      }
      UintSymbolV::Ne(left, right) => {
        if left.hash == right.hash {
          Some(UintSymbol::new(UintSymbolV::Const(BigUint::from(0u32), 1)))
        } else {
          reduce_const_relop(left, right, |a, b| a != b)
        }
      }
      UintSymbolV::Gt(left, right) => {
        if left.hash == right.hash {
          Some(UintSymbol::new(UintSymbolV::Const(BigUint::from(0u32), 1)))
        } else {
          reduce_const_relop(left, right, |a, b| a > b)
        }
      }
      UintSymbolV::Ge(left, right) => {
        if left.hash == right.hash {
          Some(UintSymbol::new(UintSymbolV::Const(BigUint::from(1u32), 1)))
        } else {
          reduce_const_relop(left, right, |a, b| a >= b)
        }
      }
      UintSymbolV::Lt(left, right) => {
        if left.hash == right.hash {
          Some(UintSymbol::new(UintSymbolV::Const(BigUint::from(0u32), 1)))
        } else {
          reduce_const_relop(left, right, |a, b| a < b)
        }
      }
      UintSymbolV::Le(left, right) => {
        if left.hash == right.hash {
          Some(UintSymbol::new(UintSymbolV::Const(BigUint::from(1u32), 1)))
        } else {
          reduce_const_relop(left, right, |a, b| a <= b)
        }
      }
      UintSymbolV::LogicAnd(left, right) => {
        if left.hash == right.hash {
          Some(UintSymbol::new(UintSymbolV::Resize {
            from: left.clone(),
            target_bits: 1,
            signed: false,
          }))
        } else {
          match (&left.v, &right.v) {
            (UintSymbolV::Const(v, _), _) if u32::try_from(v) == Ok(0) => {
              Some(UintSymbol::new(UintSymbolV::Const(0u32.into(), 1)))
            }
            (_, UintSymbolV::Const(v, _)) if u32::try_from(v) == Ok(0) => {
              // TODO: short-circuiting?
              Some(UintSymbol::new(UintSymbolV::Const(0u32.into(), 1)))
            }
            (UintSymbolV::Const(ll, _), UintSymbolV::Const(rr, _))
              if u32::try_from(ll) != Ok(0) && u32::try_from(rr) != Ok(0) =>
            {
              Some(UintSymbol::new(UintSymbolV::Const(1u32.into(), 1)))
            }
            _ => None,
          }
        }
      }
      UintSymbolV::LogicOr(left, right) => {
        if left.hash == right.hash {
          Some(UintSymbol::new(UintSymbolV::Resize {
            from: left.clone(),
            target_bits: 1,
            signed: false,
          }))
        } else {
          match (&left.v, &right.v) {
            (UintSymbolV::Const(v, _), _) if u32::try_from(v) == Ok(1) => {
              Some(UintSymbol::new(UintSymbolV::Const(1u32.into(), 1)))
            }
            (_, UintSymbolV::Const(v, _)) if u32::try_from(v) == Ok(1) => {
              // TODO: short-circuiting?
              Some(UintSymbol::new(UintSymbolV::Const(1u32.into(), 1)))
            }
            (UintSymbolV::Const(ll, _), UintSymbolV::Const(rr, _))
              if u32::try_from(ll) == Ok(0) && u32::try_from(rr) == Ok(0) =>
            {
              Some(UintSymbol::new(UintSymbolV::Const(0u32.into(), 1)))
            }
            _ => None,
          }
        }
      }
      UintSymbolV::Resize {
        from,
        target_bits,
        signed,
      } => {
        if from.bits == *target_bits {
          Some(from.clone())
        } else {
          match &from.v {
            UintSymbolV::Const(v, bits) => {
              if *bits == 0 {
                Some(UintSymbol::new(UintSymbolV::Const(
                  0u32.into(),
                  *target_bits,
                )))
              } else if bits < target_bits {
                let mut v = v.clone();

                // Sign extension
                if *signed && v.bit((*bits - 1) as u64) {
                  for index in (*bits..*target_bits).rev() {
                    v.set_bit(index as u64, true);
                  }
                }

                Some(UintSymbol::new(UintSymbolV::Const(v, *target_bits)))
              } else {
                assert!(bits > target_bits);
                let mut v = v.clone();
                truncate_biguint(&mut v, *target_bits);
                Some(UintSymbol::new(UintSymbolV::Const(v, *target_bits)))
              }
            }
            _ => None,
          }
        }
      }
      UintSymbolV::Select {
        predicate,
        on_true,
        on_false,
      } => match &predicate.v {
        UintSymbolV::Const(v, _) => {
          if *v != BigUint::from(0u32) {
            Some(on_true.clone())
          } else {
            Some(on_false.clone())
          }
        }
        _ => None,
      },
    }
  }
}

fn truncate_biguint(x: &mut BigUint, target_bits: u32) {
  let value_bits = x.bits();
  for i in (target_bits as u64)..value_bits {
    x.set_bit(i, false);
  }
  assert!(x.bits() <= target_bits as u64);
}

fn reduce_const_binop(
  left: &Arc<UintSymbol>,
  right: &Arc<UintSymbol>,
  op: fn(_: &BigUint, _: &BigUint) -> BigUint,
) -> Option<Arc<UintSymbol>> {
  match (&left.v, &right.v) {
    (UintSymbolV::Const(lvalue, lbits), UintSymbolV::Const(rvalue, _)) => {
      let mut value = op(lvalue, rvalue);
      let lbits = *lbits;
      truncate_biguint(&mut value, lbits);
      Some(UintSymbol::new(UintSymbolV::Const(value, lbits)))
    }
    _ => None,
  }
}

fn reduce_const_relop(
  left: &Arc<UintSymbol>,
  right: &Arc<UintSymbol>,
  op: fn(_: &BigUint, _: &BigUint) -> bool,
) -> Option<Arc<UintSymbol>> {
  match (&left.v, &right.v) {
    (UintSymbolV::Const(lvalue, _), UintSymbolV::Const(rvalue, _)) => {
      let value = op(lvalue, rvalue);
      let value = if value { 1u32 } else { 0u32 };
      Some(UintSymbol::new(UintSymbolV::Const(value.into(), 1)))
    }
    _ => None,
  }
}
