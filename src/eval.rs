use anyhow::Result;
use num_bigint::BigUint;
use std::{
  collections::{BTreeMap, HashMap},
  sync::Arc,
};
use thiserror::Error;

use crate::ast::{Expr, ExprV, Identifier, LiteralV, StructDef, Type};
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

  #[error("not implemented: {0}")]
  NotImplemented(&'static str),
}

pub struct EvalContext {
  names: HashMap<Arc<str>, Arc<Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum UnspecializedType {
  Product(StructDef),
  Uint,
  Signal,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IdentPath(pub Arc<[Identifier]>);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SpecializedFnType {
  pub args: Vec<SpecializedFnArg>,
  pub ret: Option<Arc<Value>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SpecializedFnArg {
  pub name: Identifier,
  pub ty: Arc<Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Value {
  UintValue { bits: Option<u32>, value: BigUint },
  Unspecialized(UnspecializedType),
  ProductValue(ProductValue),
  ProductType(ProductType),
  BuiltinType(BuiltinType),
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

  fn specialize_type(&self, _ty: &Type) -> Result<Arc<Value>> {
    Err(EvalError::NotImplemented("specialize_type").into())
  }

  pub fn eval_expr(&self, e: &Expr) -> Result<Arc<Value>> {
    let value = match &e.v {
      ExprV::Lit(x) => match x.v {
        LiteralV::Uint(ref value) => Value::UintValue {
          bits: None,
          value: value.clone(),
        },
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
                let ty = self.specialize_type(v)?;
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
                  Value::UintValue { ref value, .. } => {
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
          },
          _ => {
            return Err(EvalError::SpecializeNonUnspecializedValue.into());
          }
        }
      }
      _ => {
        return Err(EvalError::NotImplemented("").into());
      }
    };
    Ok(Arc::new(value))
  }
}
