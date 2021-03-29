use std::sync::Arc;

use thiserror::Error;

use crate::ast::Expr;

use super::value::Value;

#[derive(Error, Debug)]
pub enum EvalError {
  #[error("identifier not found: \"{0}\"")]
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

  #[error("argument type mismatch: expected {expected:?}, got {actual:?}")]
  ArgumentTypeMismatch {
    expected: Arc<Value>,
    actual: Arc<Value>,
  },

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

  #[error("argument count mismatch")]
  ArgumentCountMismatch,

  #[error("call to non-callable value")]
  CallingNonCallable,

  #[error("return type mismatch: expected {expected:?}, got {actual:?}")]
  ReturnTypeMismatch {
    expected: Arc<Value>,
    actual: Arc<Value>,
  },

  #[error("no specialization selected")]
  NoSpecializationSelected,

  #[error("value cannot be used for selection")]
  ValueCannotBeUsedForSelection,

  #[error("select type mismatch: left_value {left:?}, right_value {right:?}")]
  SelectTypeMismatch { left: Arc<Value>, right: Arc<Value> },

  #[error("bad cast")]
  BadCast,

  #[error("kind mismatch: expected {expected_kind:?}, got {actual_kind:?}")]
  KindMismatch {
    expected_kind: Arc<Value>,
    actual_kind: Arc<Value>,
  },

  #[error("uncomparable types: op {op}, left {left:?}, right {right:?}")]
  UncomparableTypes {
    op: Arc<str>,
    left: Arc<Value>,
    right: Arc<Value>,
  },

  #[error("bad selection arms: left {left:?}, right {right:?}")]
  BadSelectionArms { left: Arc<Value>, right: Arc<Value> },

  #[error("where arm cannot be statically evaluated: {condition:?}")]
  NonStaticWhereArm { condition: Arc<Value> },

  #[error("static if cannot be statically evaluated: {condition:?}")]
  NonStaticStaticIf { condition: Arc<Value> },

  #[error("type has no true value: {0:?}")]
  TypeHasNoTrueValue(Arc<Value>),

  #[error("operation '{optype}' not supported on value {value:?}")]
  OperationNotSupported { optype: Arc<str>, value: Arc<Value> },

  #[error("symbolic value in type-level computation: {0:?}")]
  SymbolicValueInTypeLevel(Arc<Value>),

  #[error("bad signal type: {0}")]
  BadSignalType(Arc<str>),

  #[error("cannot convert value to string: {0:?}")]
  CannotConvertToString(Arc<Value>),

  #[error("requested read on an 'out' signal")]
  ReadOnOutSignal,

  #[error("requested write on an 'in' signal")]
  WriteOnInSignal,

  #[error("zero-sized uint")]
  ZeroSizedUint,

  #[error("cannot determine width of value: {0:?}")]
  CannotDetermineWidth(Arc<Value>),

  #[error("unpack width mismatch: expected {expected} bits, got {actual} bits")]
  UnpackWidthMismatch { expected: u32, actual: u32 },

  #[error("empty product value not allowed here")]
  EmptyProductValueNotAllowed,

  #[error("missing local variable declaration")]
  MissingLocalDecl,

  #[error("invalid left side of assign statement")]
  InvalidAssignLeft,

  #[error("cannot resolve identifier path segment")]
  IdentifierPathSegmentResolveFail,

  #[error("slicing on non-uint value")]
  SliceOnNonUint,

  #[error("slice bounds must be uint")]
  SliceBoundsMustBeUint,

  #[error("user error: {0}")]
  UserError(Arc<str>),

  #[error("not implemented: {0}")]
  NotImplemented(&'static str),

  #[error("expression not implemented: {0:?}")]
  ExprNotImplemented(Expr),
}
