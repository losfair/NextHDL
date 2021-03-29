use std::{collections::BTreeMap, sync::Arc};

use arc_swap::ArcSwapWeak;

use crate::{
  ast::{FnMeta, FnSpecialization, Identifier, StructDef},
  symbol::SymbolicUint,
};

use super::{error::EvalError, signal::SignalValue, EvalContext};
use anyhow::Result;
use std::fmt::Debug;

/// An unspecialized type. Contains zero or more unspecified type variables.
#[derive(Debug)]
pub enum UnspecializedType {
  Product(Arc<UniqueProduct>),
  Uint,
  Signal,
}

#[derive(Debug)]
pub struct UniqueProduct {
  pub def: StructDef,
  pub context: Arc<ArcSwapWeak<EvalContext>>,
}

/// The output of some computation. Evaluating an `Expr` produces a `Value`.
#[derive(Debug)]
pub enum Value {
  /// A rank-0 `uint` value.
  UintValue(SymbolicUint),

  /// A rank-0 signal value.
  SignalValue(SignalValue),

  /// A rank-0 unit value.
  Unit,

  /// A rank-0 concrete `string` value.
  StringValue(Arc<str>),

  /// A rank-0 concrete product (struct) value.
  ProductValue(ProductValue),

  /// A rank-0 function value that is not yet specialized.
  UnspecializedFnValue(UnspecializedFnValue),

  /// A rank-0 function value that is already specialized.
  SpecializedFnValue(SpecializedFnValue),

  /// A built-in function.
  BuiltinFnValue(BuiltinFnValue),

  /// A rank-1 specialized function type.
  FnType(SpecializedFnType),

  /// A rank-1 product (struct) type.
  ProductType(ProductType),

  /// A rank-1 builtin type.
  BuiltinType(BuiltinType),

  /// A rank-2 unspecialized type.
  Unspecialized(UnspecializedType),
}

#[derive(Debug)]
pub enum BuiltinFnValue {
  MkSignal,
  Error,
  Undefined,
}

#[derive(Copy, Clone, Debug)]
pub enum ValueOrdering {
  Lt,
  Gt,
  Le,
  Ge,
}

impl ValueOrdering {
  pub fn get_comparator(&self) -> fn(_: SymbolicUint, _: SymbolicUint) -> SymbolicUint {
    match self {
      ValueOrdering::Lt => SymbolicUint::sym_lt,
      ValueOrdering::Le => SymbolicUint::sym_le,
      ValueOrdering::Gt => SymbolicUint::sym_gt,
      ValueOrdering::Ge => SymbolicUint::sym_ge,
    }
  }
}

impl PartialEq for Value {
  fn eq(&self, other: &Self) -> bool {
    self.compare_eq(other).const_truthy().unwrap_or(false)
  }
}

// NOT really Eq though. Only PartialEq.~
impl Eq for Value {}

impl Value {
  pub fn compare_eq(&self, other: &Value) -> SymbolicUint {
    match (self, other) {
      (Value::UintValue(ll), Value::UintValue(rr)) => ll.clone().sym_eq(rr.clone()),
      (Value::StringValue(ll), Value::StringValue(rr)) => (ll == rr).into(),
      (Value::ProductValue(ll), Value::ProductValue(rr)) => ll.compare_eq(rr),
      (Value::FnType(ll), Value::FnType(rr)) => (ll == rr).into(),
      (Value::ProductType(ll), Value::ProductType(rr)) => (ll == rr).into(),
      (Value::BuiltinType(ll), Value::BuiltinType(rr)) => (ll == rr).into(),
      (Value::Unspecialized(ll), Value::Unspecialized(rr)) => match (ll, rr) {
        (UnspecializedType::Product(ll), UnspecializedType::Product(rr)) => {
          Arc::ptr_eq(ll, rr).into()
        }
        (UnspecializedType::Signal, UnspecializedType::Signal) => true.into(),
        (UnspecializedType::Uint, UnspecializedType::Uint) => true.into(),
        _ => false.into(),
      },
      _ => false.into(),
    }
  }

  pub fn compare_ord(&self, other: &Value, ord: ValueOrdering) -> Option<SymbolicUint> {
    match (self, other) {
      (Value::UintValue(ll), Value::UintValue(rr)) => {
        Some((ord.get_comparator())(ll.clone(), rr.clone()))
      }
      _ => None,
    }
  }

  pub fn const_truthy(&self) -> Option<bool> {
    match self {
      Value::UintValue(x) => x.const_truthy(),
      _ => None,
    }
  }

  pub fn smt_truthy(&self) -> Result<Option<bool>> {
    match self {
      Value::UintValue(x) => x.smt_solve_boolean(),
      _ => Ok(None),
    }
  }

  pub fn is_const(&self) -> bool {
    match self {
      Value::UintValue(x) => x.is_const(),
      Value::ProductValue(x) => x
        .fields
        .iter()
        .map(|(_, v)| v.is_const())
        .reduce(|a, b| a && b)
        .unwrap_or(true),
      _ => true,
    }
  }

  pub fn select(&self, on_true: &Arc<Value>, on_false: &Arc<Value>) -> Result<Arc<Value>> {
    let predicate = match self {
      Value::UintValue(x) => x,
      _ => return Err(EvalError::ValueCannotBeUsedForSelection.into()),
    };

    if !on_true
      .get_type()?
      .compare_eq(&*on_false.get_type()?)
      .const_truthy()
      .unwrap_or(false)
    {
      return Err(
        EvalError::SelectTypeMismatch {
          left: on_true.clone(),
          right: on_false.clone(),
        }
        .into(),
      );
    }

    // Allow higher-ranked values if the predicate is constant...
    if let Some(truthy) = predicate.const_truthy() {
      return Ok(if truthy {
        on_true.clone()
      } else {
        on_false.clone()
      });
    }

    match (&**on_true, &**on_false) {
      (Value::UintValue(ll), Value::UintValue(rr)) => Ok(Arc::new(Value::UintValue(
        predicate.clone().sym_select(ll.clone(), rr.clone()),
      ))),
      (Value::ProductValue(ll), Value::ProductValue(rr)) => {
        assert!(Arc::ptr_eq(&ll.unique, &rr.unique));
        let fields = ll
          .fields
          .iter()
          .map(|(k, v_ll)| {
            let v_rr = rr
              .fields
              .get(k)
              .expect("select: ProductValue: field mismatch");
            self.select(v_ll, v_rr).map(|x| (k.clone(), x))
          })
          .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Arc::new(Value::ProductValue(ProductValue {
          fields,
          unique: ll.unique.clone(),
        })))
      }
      _ => Err(
        EvalError::BadSelectionArms {
          left: on_true.clone(),
          right: on_false.clone(),
        }
        .into(),
      ),
    }
  }

  pub fn to_string(&self) -> Option<Arc<str>> {
    if let Value::StringValue(x) = self {
      Some(x.clone())
    } else {
      None
    }
  }

  pub fn try_to_string(self: &Arc<Self>) -> Result<Arc<str>> {
    self
      .to_string()
      .ok_or_else(|| EvalError::CannotConvertToString(self.clone()).into())
  }

  pub fn rank1_width(self: &Arc<Self>) -> Result<u32> {
    match &**self {
      Value::ProductType(ty) => {
        ty.fields
          .iter()
          .map(|(_, v)| v.rank1_width())
          .try_fold(0, |acc, x| match x {
            Ok(x) => Ok(acc + x),
            Err(e) => Err(e),
          })
      }
      Value::BuiltinType(BuiltinType::Uint { bits }) => Ok(*bits),
      _ => Err(EvalError::CannotDetermineWidth(self.clone()).into()),
    }
  }

  pub fn pack(self: &Arc<Self>) -> Result<Option<SymbolicUint>> {
    match &**self {
      Value::UintValue(x) => Ok(Some(x.clone())),
      Value::ProductValue(v) => v
        .fields
        .iter()
        .filter_map(|(_, v)| v.pack().transpose())
        .try_fold(None, |acc: Option<SymbolicUint>, x| match x {
          Ok(x) => match acc {
            Some(acc) => Ok(Some(acc.sym_concat(x))),
            None => Ok(Some(x)),
          },
          Err(e) => Err(e),
        }),
      _ => Err(EvalError::CannotDetermineWidth(self.clone()).into()),
    }
  }

  pub fn unpack(that: SymbolicUint, ty: &Arc<Value>) -> Result<Self> {
    let ty_width = ty.rank1_width()?;
    if that.bits() != ty_width {
      return Err(
        EvalError::UnpackWidthMismatch {
          expected: ty_width,
          actual: that.bits(),
        }
        .into(),
      );
    }

    Self::do_unpack(&mut Some(that), ty)
  }

  fn do_unpack(source: &mut Option<SymbolicUint>, ty: &Arc<Value>) -> Result<Self> {
    match &**ty {
      Value::BuiltinType(BuiltinType::Uint { bits }) => {
        // already checked in unpack()
        let that = source.as_mut().expect("do_unpack: empty source");

        let result = that.clone().sym_slice(*bits - 1, 0)?;
        let that_width = that.bits();

        if that_width == *bits {
          *source = None;
        } else {
          *that = that.clone().sym_slice(that_width - 1, *bits)?;
        }

        Ok(Value::UintValue(result))
      }
      Value::ProductType(ty) => {
        let value_map = ty
          .fields
          .iter()
          .map(|(k, v)| Self::do_unpack(source, v).map(|x| (k.clone(), Arc::new(x))))
          .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Value::ProductValue(ProductValue {
          fields: value_map,
          unique: ty.unique.clone(),
        }))
      }
      _ => {
        // already checked in unpack()
        unreachable!()
      }
    }
  }
}

impl ProductValue {
  fn compare_eq(&self, other: &ProductValue) -> SymbolicUint {
    if !Arc::ptr_eq(&self.unique, &other.unique) {
      return false.into();
    }
    self
      .fields
      .iter()
      .map(|(k, v)| {
        (
          v,
          other
            .fields
            .get(k)
            .expect("ProductValue: Type eq but fields names are different"),
        )
      })
      .map(|(l, r)| l.compare_eq(r))
      .reduce(|prev, this| prev.sym_logic_and(this))
      .unwrap_or_else(|| {
        // With zero fields...
        SymbolicUint::new_const(1u32.into(), 1)
      })
  }
}

impl Value {
  /// Computes the type of a rank-0 or rank-1 `Value`.
  pub fn get_type(&self) -> Result<Arc<Value>> {
    Ok(match self {
      Value::UintValue(x) => Arc::new(Value::BuiltinType(BuiltinType::Uint { bits: x.bits() })),
      Value::StringValue(_) => Arc::new(Value::BuiltinType(BuiltinType::String)),
      Value::ProductValue(value) => {
        let mut fields = BTreeMap::new();
        for (k, v) in &value.fields {
          fields.insert(k.clone(), v.get_type()?);
        }
        Arc::new(Value::ProductType(ProductType {
          fields,
          unique: value.unique.clone(),
        }))
      }
      Value::SpecializedFnValue(value) => Arc::new(Value::FnType(value.ty.clone())),
      Value::SignalValue(v) => Arc::new(Value::BuiltinType(BuiltinType::Signal {
        inner: v.inner_ty.clone(),
      })),
      Value::ProductType(ty) => Arc::new(Value::Unspecialized(UnspecializedType::Product(
        ty.unique.clone(),
      ))),
      Value::BuiltinType(BuiltinType::Uint { .. }) => {
        Arc::new(Value::Unspecialized(UnspecializedType::Uint))
      }
      Value::BuiltinType(BuiltinType::Signal { .. }) => {
        Arc::new(Value::Unspecialized(UnspecializedType::Signal))
      }
      Value::Unit => Arc::new(Value::Unit),
      _ => {
        warn!("unknown type for value: {:?}", self);
        return Err(EvalError::GetTypeForValueOfUnknownType.into());
      }
    })
  }

  /// Casts a value to this type.
  pub fn cast_to_this_type(&self, value: &Arc<Value>) -> Result<Arc<Value>> {
    match self {
      Value::BuiltinType(BuiltinType::Uint { bits }) => {
        // Truncate
        let value = match &**value {
          Value::UintValue(x) => x.clone().sym_resize(*bits, false),
          _ => return Err(EvalError::BadCast.into()),
        };
        Ok(Arc::new(Value::UintValue(value)))
      }
      Value::BuiltinType(BuiltinType::String) => {
        let value = match &**value {
          Value::StringValue(x) => x.clone(),
          _ => return Err(EvalError::BadCast.into()),
        };
        Ok(Arc::new(Value::StringValue(value)))
      }
      _ => Err(EvalError::BadCast.into()),
    }
  }
}

#[derive(Clone, Debug)]
pub struct SpecializedFnType {
  /// Value of type arguments.
  pub tyargs: BTreeMap<Identifier, Arc<Value>>,
  pub args: Vec<SpecializedFnArg>,
  pub ret: Option<Arc<Value>>,
}

impl PartialEq for SpecializedFnType {
  fn eq(&self, other: &SpecializedFnType) -> bool {
    if self.ret != other.ret {
      return false;
    }

    if self.args.len() != other.args.len() {
      return false;
    }

    for (ll, rr) in self.args.iter().zip(other.args.iter()) {
      if ll.ty != rr.ty {
        return false;
      }
    }

    return true;
  }
}

impl Eq for SpecializedFnType {}

#[derive(Debug)]
pub struct UnspecializedFnValue {
  pub ty: FnMeta,
  pub body: Arc<[FnSpecialization]>,
  pub context: Arc<ArcSwapWeak<EvalContext>>,
}

#[derive(Clone, Debug)]
pub struct SpecializedFnValue {
  pub ty: SpecializedFnType,
  pub body: Arc<[FnSpecialization]>,
  pub context: Arc<ArcSwapWeak<EvalContext>>,
}

#[derive(Clone, Debug)]
pub struct SpecializedFnArg {
  pub name: Identifier,
  pub ty: Arc<Value>,
  pub default_value: Option<Arc<Value>>,
}

#[derive(Clone, Debug)]
pub struct ProductValue {
  pub fields: BTreeMap<Arc<str>, Arc<Value>>,
  pub unique: Arc<UniqueProduct>,
}

#[derive(Clone, Debug)]
pub struct ProductType {
  pub fields: BTreeMap<Arc<str>, Arc<Value>>,
  pub unique: Arc<UniqueProduct>,
}

impl PartialEq for ProductType {
  fn eq(&self, other: &ProductType) -> bool {
    // Same unique root & same field specialization
    Arc::ptr_eq(&self.unique, &other.unique) && self.fields == other.fields
  }
}

impl Eq for ProductType {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuiltinType {
  Uint { bits: u32 },
  String,
  Signal { inner: Arc<Value> },
}
