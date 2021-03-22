use anyhow::Result;
use num_bigint::BigUint;
use num_traits::ops::checked::CheckedDiv;
use rpds::RedBlackTreeMapSync;
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;

use crate::ast::{Expr, ExprV, FnMeta, Identifier, LiteralV, StructDef, TypeAssign};
use std::convert::TryFrom;

#[derive(Error, Debug)]
pub enum EvalError {
  #[error("identifier not found: {0}")]
  IdentifierNotFound(Arc<str>),

  #[error("dot operator used on a non-product value")]
  DotOnNonProductValue,

  #[error("field not found on product value")]
  FieldNotFound,

  #[error("specializing a non-unspecialized value")]
  SpecializeNonUnspecializedValue,

  #[error("bad specialization")]
  BadSpecialization,

  #[error("non-def function type cannot have default values in arguments")]
  NonDefFnTypeArgDefaultValue,

  #[error("attempting to get type for value of unknown type")]
  GetTypeForValueOfUnknownType,

  #[error("type mismatch")]
  TypeMismatch,

  #[error("missing type")]
  MissingType,

  #[error("bad type assignment")]
  BadTypeAssign,

  #[error("missing type assignment")]
  MissingTypeAssign,

  #[error("unknown builtin call: {0}")]
  UnknownBuiltinCall(Arc<str>),

  #[error("missing argument")]
  MissingArgument,

  #[error("division by zero")]
  DivByZero,

  #[error("not implemented: {0}")]
  NotImplemented(&'static str),

  #[error("expression not implemented: {0:?}")]
  ExprNotImplemented(Expr),
}

#[derive(Clone, Default)]
pub struct EvalContext {
  pub names: RedBlackTreeMapSync<Arc<str>, Arc<Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum UnspecializedType {
  Product(StructDef),
  Uint,
  Signal,
  Fn(FnMeta),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IdentPath(pub Arc<[Identifier]>);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Value {
  UintValue(UintValue),
  Unspecialized(UnspecializedType),
  FnType(SpecializedFnType),
  ProductValue(ProductValue),
  ProductType(ProductType),
  BuiltinType(BuiltinType),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct UintValue {
  pub value: BigUint,
  pub bits: Option<u32>,
}

impl Value {
  pub fn get_type(&self) -> Result<Arc<Value>> {
    Ok(match self {
      Value::UintValue(UintValue { bits, .. }) => {
        Arc::new(Value::BuiltinType(BuiltinType::Uint { bits: *bits }))
      }
      Value::ProductValue(value) => {
        let mut fields = BTreeMap::new();
        for (k, v) in &value.fields {
          fields.insert(k.clone(), v.get_type()?);
        }
        Arc::new(Value::ProductType(ProductType { fields }))
      }
      _ => return Err(EvalError::GetTypeForValueOfUnknownType.into()),
    })
  }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SpecializedFnType {
  pub args: Vec<SpecializedFnArg>,
  pub ret: Option<Arc<Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SpecializedFnArg {
  pub name: Identifier,
  pub ty: Arc<Value>,
  pub default_value: Option<Arc<Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProductValue {
  pub fields: BTreeMap<Arc<str>, Arc<Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProductType {
  pub fields: BTreeMap<Arc<str>, Arc<Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BuiltinType {
  Uint { bits: Option<u32> },
  Signal { inner: Arc<Value> },
}

impl EvalContext {
  fn lookup_name(&self, x: &Identifier) -> Result<Arc<Value>> {
    match self.names.get(&x.0) {
      Some(x) => Ok(x.clone()),
      None => Err(EvalError::IdentifierNotFound(x.0.clone()).into()),
    }
  }

  /// Specializes the type of a `fn`.
  ///
  /// Used in two places:
  ///
  /// - Call: `some_fn<TypeA, TypeB = C>(...)`
  /// - Expression: `struct A<TypeA> { some_field: fn<TypeA>(v: TypeA) }`
  ///
  /// In the Call case, `tyassigns` should be the types passed for specialization.
  ///
  /// In the Expression case, `tyassigns` should be empty, and the "context" will be used.
  pub fn specialize_fntype(
    &self,
    meta: &FnMeta,
    tyassigns: &[TypeAssign],
    is_def: bool,
  ) -> Result<SpecializedFnType> {
    let mut this = self.clone();

    let named_tyassigns = tyassigns
      .iter()
      .filter_map(|x| x.ty.as_ref().map(|name| (name, &x.e)))
      .collect::<BTreeMap<_, _>>();

    // Compute concrete values of type arguments.
    for (i, tyarg) in meta.tyargs.iter().enumerate() {
      // Only allow default values in function definition place.
      if !is_def && tyarg.default_value.is_some() {
        return Err(EvalError::NonDefFnTypeArgDefaultValue.into());
      }

      // Get the concrete type.
      let tyassign = tyassigns
        .get(i)
        .filter(|x| x.ty.is_none())
        .map(|x| this.eval_expr(&x.e))
        .or_else(|| {
          named_tyassigns
            .get(&&tyarg.name)
            .map(|x| this.eval_expr(*x))
        })
        .or_else(|| tyarg.default_value.as_ref().map(|x| this.eval_expr(x)))
        .transpose()?
        .ok_or_else(|| EvalError::MissingTypeAssign)?;

      // Get the expected kind of the type.
      let expected_kind = if let Some(kind) = &tyarg.kind {
        Some(this.eval_expr(kind)?)
      } else {
        None
      };

      // Compute the actual kind of the type.
      let actual_kind = tyassign.get_type()?;

      // Do they match?
      if let Some(expected_kind) = expected_kind {
        if expected_kind != actual_kind {
          return Err(EvalError::TypeMismatch.into());
        }
      }

      // Ok let's insert it
      this.names = this.names.insert(tyarg.name.0.clone(), tyassign);
    }

    // Compute concrete types of arguments.
    let mut args = Vec::new();
    for arg in meta.args.iter() {
      let ty = this.eval_expr(&arg.ty)?;
      args.push(SpecializedFnArg {
        name: arg.name.clone(),
        ty,

        // TODO: default value
        default_value: None,
      });
    }

    // Compute concrete type of the return value.
    let ret = meta
      .ret
      .as_ref()
      .map(|x| this.eval_expr(&**x))
      .transpose()?;

    return Ok(SpecializedFnType { args, ret });
  }

  pub fn eval_expr(&self, e: &Expr) -> Result<Arc<Value>> {
    let value = match &e.v {
      ExprV::Lit(x) => match x.v {
        LiteralV::Uint(ref value) => Value::UintValue(UintValue {
          bits: None,
          value: value.clone(),
        }),
      },
      ExprV::Ident(x) => {
        return Ok(self.lookup_name(&x)?);
      }
      ExprV::Dot { base, id } => {
        let base = self.eval_expr(&base)?;
        match &*base {
          Value::ProductValue(ref product_value) => {
            let field = product_value
              .fields
              .get(&id.0)
              .ok_or_else(|| EvalError::FieldNotFound)?;
            return Ok(field.clone());
          }
          _ => {
            return Err(EvalError::DotOnNonProductValue.into());
          }
        }
      }
      ExprV::Specialize { base, tyassigns } => {
        let base = self.eval_expr(&base)?;
        match &*base {
          Value::Unspecialized(ref ty) => match ty {
            UnspecializedType::Product(ref def) => {
              let mut fields: BTreeMap<Arc<str>, Arc<Value>> = BTreeMap::new();
              for (k, v) in def.fields.iter() {
                let ty = self.eval_expr(v)?;
                fields.insert(k.clone(), ty);
              }
              Value::ProductType(ProductType { fields })
            }
            UnspecializedType::Uint => {
              if tyassigns.len() == 0 {
                Value::BuiltinType(BuiltinType::Uint { bits: None })
              } else if tyassigns.len() == 1 && tyassigns[0].ty.is_none() {
                let width = self.eval_expr(&tyassigns[0].e)?;
                match &*width {
                  Value::UintValue(UintValue { ref value, .. }) => {
                    let width = match u32::try_from(value) {
                      Ok(x) => x,
                      Err(_) => return Err(EvalError::BadSpecialization.into()),
                    };
                    Value::BuiltinType(BuiltinType::Uint { bits: Some(width) })
                  }
                  _ => return Err(EvalError::BadSpecialization.into()),
                }
              } else {
                return Err(EvalError::BadSpecialization.into());
              }
            }
            UnspecializedType::Signal => {
              if tyassigns.len() == 1 && tyassigns[0].ty.is_none() {
                let inner = self.eval_expr(&tyassigns[0].e)?;
                Value::BuiltinType(BuiltinType::Signal { inner })
              } else {
                return Err(EvalError::BadSpecialization.into());
              }
            }
            UnspecializedType::Fn(meta) => {
              Value::FnType(self.specialize_fntype(meta, tyassigns, false)?)
            }
          },
          _ => {
            return Err(EvalError::SpecializeNonUnspecializedValue.into());
          }
        }
      }
      ExprV::Call { base, args } => {
        if let ExprV::Dot { base, id } = &base.v {
          let base = self.eval_expr(&base)?;
          match &*id.0 {
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
              let right = args.get(0).ok_or_else(|| EvalError::MissingArgument)?;
              let right = self.eval_expr(right)?;

              let value = match &*id.0 {
                "eq" => base == right,
                "ne" => base != right,
                _ => {
                  let base_ty = base.get_type()?;
                  let right_ty = right.get_type()?;
                  if base_ty != right_ty {
                    return Err(EvalError::TypeMismatch.into());
                  }

                  match &*id.0 {
                    "lt" => base < right,
                    "le" => base <= right,
                    "gt" => base > right,
                    "ge" => base >= right,
                    _ => unreachable!(),
                  }
                }
              };
              let value = if value { 1u32 } else { 0u32 };
              Value::UintValue(UintValue {
                value: BigUint::from(value),
                bits: Some(1),
              })
            }
            "add" | "sub" | "mul" | "div" => {
              let right = args.get(0).ok_or_else(|| EvalError::MissingArgument)?;
              let right = self.eval_expr(right)?;
              match (&*base, &*right) {
                (Value::UintValue(ll), Value::UintValue(rr)) => {
                  let mut value = match &*id.0 {
                    "add" => &ll.value + &rr.value,
                    "sub" => &ll.value - &rr.value,
                    "mul" => &ll.value * &rr.value,
                    "div" => ll
                      .value
                      .checked_div(&rr.value)
                      .ok_or_else(|| EvalError::DivByZero)?,
                    _ => unreachable!(),
                  };

                  // Truncate
                  if let Some(target_bits) = ll.bits {
                    let value_bits = value.bits();
                    for i in (target_bits as u64)..value_bits {
                      value.set_bit(i, false);
                    }
                    assert!(value.bits() <= target_bits as u64);
                  }

                  Value::UintValue(UintValue {
                    value,
                    bits: ll.bits,
                  })
                }
                _ => return Err(EvalError::TypeMismatch.into()),
              }
            }
            "select" => {
              let on_true = args.get(0).ok_or_else(|| EvalError::MissingArgument)?;
              let on_false = args.get(1).ok_or_else(|| EvalError::MissingArgument)?;
              let predicate = match &*base {
                Value::UintValue(UintValue { value, .. }) => {
                  if u32::try_from(value).unwrap_or(1) == 0 {
                    false
                  } else {
                    true
                  }
                }
                _ => return Err(EvalError::TypeMismatch.into()),
              };
              if predicate {
                return Ok(self.eval_expr(on_true)?);
              } else {
                return Ok(self.eval_expr(on_false)?);
              }
            }
            _ => return Err(EvalError::UnknownBuiltinCall(id.0.clone()).into()),
          }
        } else {
          return Err(EvalError::ExprNotImplemented(e.clone()).into());
        }
      }
      _ => {
        return Err(EvalError::ExprNotImplemented(e.clone()).into());
      }
    };
    Ok(Arc::new(value))
  }
}
