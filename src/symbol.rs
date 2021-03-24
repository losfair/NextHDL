use std::{ops::Add, sync::Arc};
use serde::Serialize;
use num_bigint::BigUint;
use sha2::{Sha256, Digest};

#[derive(Clone, Debug)]
pub struct SymbolicUint {
  sym: Arc<UintSymbol>,
}

impl SymbolicUint {
  pub fn new_const(value: BigUint, bits: u32) -> Self {
    Self::new_v(UintSymbolV::Const(value, bits))
  }

  pub fn new_external(name: Arc<str>, bits: u32) -> Self {
    Self::new_v(UintSymbolV::External { name, bits })
  }

  fn new_v(v: UintSymbolV) -> Self {
    Self {
      sym: UintSymbol::new(v),
    }
  }

  pub fn select(self, on_true: SymbolicUint, on_false: SymbolicUint) -> SymbolicUint {
    Self::new_v(UintSymbolV::Select { predicate: self.sym, on_true: on_true.sym, on_false: on_false.sym })
  }
}

impl Add for SymbolicUint {
  type Output = Self;
  fn add(self, rhs: Self) -> Self::Output {
      Self::new_v(UintSymbolV::Add(self.sym, rhs.sym))
  }
}

/// A symbolic value of `uint`.
#[derive(Serialize, Debug)]
struct UintSymbol {
  #[serde(skip)]
  v: UintSymbolV,

  #[serde(skip)]
  bits: u32,

  /// The hash calculated from `v`.
  hash: [u8; 32],
}

impl PartialEq for UintSymbol {
  fn eq(&self, other: &Self) -> bool {
    self.hash == other.hash
  }
}

impl Eq for UintSymbol {}

/// A variant of `UintNode`.
#[derive(Serialize, Debug)]
enum UintSymbolV {
  /// An external signal.
  External { name: Arc<str>, bits: u32 },

  /// A constant.
  Const(BigUint, u32),

  /// The result of adding two `UintNode`s.
  Add(Arc<UintSymbol>, Arc<UintSymbol>),

  /// The result of selecting from two `UintNode`s based on a predicate.
  Select { predicate: Arc<UintSymbol>, on_true: Arc<UintSymbol>, on_false: Arc<UintSymbol> },
}

impl UintSymbol {
  fn new(v: UintSymbolV) -> Arc<Self> {
    if let Some(x) = v.try_reduce() {
      return x;
    }

    let bits = match &v {
      UintSymbolV::External { bits, .. } => *bits,
      UintSymbolV::Const(_, bits) => *bits,
      UintSymbolV::Add(left, _) => {
        // Right value will be truncated to match left
        left.bits
      },
      UintSymbolV::Select { on_true, on_false, .. } => {
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
      UintSymbolV::Add(left, right) => {
        match (&left.v, &right.v) {
          (UintSymbolV::Const(lvalue, lbits), UintSymbolV::Const(rvalue, _)) => {
            let mut value = lvalue + rvalue;
            let lbits = *lbits;
            truncate_biguint(&mut value, lbits);
            Some(UintSymbol::new(UintSymbolV::Const(value, lbits)))
          }
          _ => None,
        }
      }
      UintSymbolV::Select { predicate, on_true, on_false } => {
        match &predicate.v {
          UintSymbolV::Const(v, _) => {
            if *v != BigUint::from(0u32) {
              Some(on_true.clone())
            } else {
              Some(on_false.clone())
            }
          }
          _ => None
        }
      }
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
