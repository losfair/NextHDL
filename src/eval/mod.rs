pub mod error;
pub mod signal;
pub mod value;

use error::EvalError;
use value::*;

use anyhow::Result;
use arc_swap::ArcSwapWeak;
use rpds::RedBlackTreeMapSync;
use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

use crate::{
  ast::Body,
  tracker::{EvalTracker, EvaluationStackEntry, EvaluationStackEntryType},
  util::mk_arc_str,
};
use crate::{
  ast::{
    Expr, ExprV, FnMeta, FnSpecialization, Identifier, LiteralV, Stmt, StmtV, TyArg, TypeAssign,
  },
  symbol::SymbolicUint,
};
use std::convert::TryFrom;

use self::signal::{AssignmentTable, SignalType, SignalValue};

#[derive(Clone, Debug)]
pub struct EvalContext {
  /// All names in the current context. A persistent red-black tree is used for efficient
  /// scope nesting.
  pub names: RedBlackTreeMapSync<Arc<str>, Arc<Value>>,

  pub tracker: Arc<EvalTracker>,
}

#[derive(Default)]
pub struct BlockContext {
  /// A map from names of local variables to their types.
  pub local_variable_types: BTreeMap<Identifier, Arc<Value>>,

  /// Updated variables in this block.
  pub updated_variables: BTreeMap<Identifier, Arc<Value>>,

  /// The result of the last expression.
  pub last_result: Option<Arc<Value>>,
}

impl EvalContext {
  pub fn new() -> Self {
    Self {
      names: Default::default(),
      tracker: Arc::new(EvalTracker::new()),
    }
  }

  pub fn dump_stack(&self) -> Vec<EvaluationStackEntry> {
    self.tracker.dump()
  }

  pub fn tracker(&self) -> &Arc<EvalTracker> {
    &self.tracker
  }

  fn lookup_name(&self, x: &Identifier) -> Result<Arc<Value>> {
    match self.names.get(&x.0) {
      Some(x) => Ok(x.clone()),
      None => {
        debug!(
          "identifier not found. all identifiers: {:?}",
          self.names.iter().map(|x| x.0).collect::<Vec<_>>()
        );
        Err(EvalError::IdentifierNotFound(x.0.clone()).into())
      }
    }
  }

  fn compute_tyassigns(
    &self,
    native_context: &mut EvalContext,
    tyassigns: &[TypeAssign],
    tyargs: &[TyArg],
  ) -> Result<BTreeMap<Identifier, Arc<Value>>> {
    // Make a copy of our current context.
    let mut this = self.clone();

    let named_tyassigns = tyassigns
      .iter()
      .filter_map(|x| x.ty.as_ref().map(|name| (name, &x.e)))
      .collect::<BTreeMap<_, _>>();

    let mut tyarg_values = BTreeMap::new();

    // Compute concrete values of type arguments.
    for (i, tyarg) in tyargs.iter().enumerate() {
      // Get the concrete type.
      // This should be evaluated in the current context.
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

      // Must not be a symbolic value
      if !tyassign.is_const() {
        return Err(EvalError::SymbolicValueInTypeLevel(tyassign).into());
      }

      // Get the expected kind of the type.
      // In the native context.
      let expected_kind = if let Some(kind) = &tyarg.kind {
        Some(native_context.eval_expr(kind)?)
      } else {
        None
      };

      // Compute the actual kind of the type.
      let actual_kind = tyassign.get_type()?;

      // Do they match?
      if let Some(expected_kind) = expected_kind {
        if expected_kind != actual_kind {
          return Err(
            EvalError::KindMismatch {
              expected_kind,
              actual_kind,
            }
            .into(),
          );
        }
      }

      // Ok let's insert it
      this
        .names
        .insert_mut(tyarg.name.0.clone(), tyassign.clone());
      native_context
        .names
        .insert_mut(tyarg.name.0.clone(), tyassign.clone());
      tyarg_values.insert(tyarg.name.clone(), tyassign);
    }

    Ok(tyarg_values)
  }

  /// Specializes a rank-2 `UnspecializedType::Fn` into a rank-1 `SpecializedFnType`.
  ///
  /// Takes a function signature like `fn <TypeA, TypeB: uint>(a: TypeA, b: uint<TypeB>) -> uint<TypeB.add(1)>`
  /// and an array of assignments to type variables, and eliminates all type variables and produces a
  /// concrete function type like `fn(a: signal<uint<1>>, b: uint<8>) -> uint<9>`.
  pub fn specialize_fntype(
    &self,
    mut native_context: EvalContext,
    meta: &FnMeta,
    tyassigns: &[TypeAssign],
  ) -> Result<SpecializedFnType> {
    let tyarg_values = self.compute_tyassigns(&mut native_context, tyassigns, &meta.tyargs)?;

    // Compute concrete types of arguments.
    let mut args = Vec::new();
    for arg in meta.args.iter() {
      let ty = native_context.eval_expr(&arg.ty)?;
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
      .map(|x| native_context.eval_expr(&**x))
      .transpose()?;

    return Ok(SpecializedFnType {
      tyargs: tyarg_values,
      args,
      ret,
    });
  }

  /// Specializes an rank-2 `UnspecializedType` into a rank-1 type.
  fn specialize_type(
    &self,
    ty: &UnspecializedType,
    tyassigns: &[TypeAssign],
  ) -> Result<Arc<Value>> {
    Ok(Arc::new(match ty {
      UnspecializedType::Product(unique) => {
        // Specialize in its own context...
        let mut specialization_context: EvalContext = (*unique
          .context
          .load()
          .upgrade()
          .expect("cannot upgrade context"))
        .clone();

        self.compute_tyassigns(&mut specialization_context, tyassigns, &unique.def.tyargs)?;

        let mut fields: BTreeMap<Arc<str>, Arc<Value>> = BTreeMap::new();
        for (k, v) in unique.def.fields.iter() {
          let ty = specialization_context.eval_expr(v)?;
          fields.insert(k.clone(), ty);
        }
        Value::ProductType(ProductType {
          fields,
          unique: unique.clone(),
        })
      }
      UnspecializedType::Uint => {
        if tyassigns.len() == 0 {
          Value::BuiltinType(BuiltinType::Uint {
            bits: std::u32::MAX,
          })
        } else if tyassigns.len() == 1 && tyassigns[0].ty.is_none() {
          let width = self.eval_expr(&tyassigns[0].e)?;
          match &*width {
            Value::UintValue(x) => {
              let width = match u32::try_from(x.as_const()?) {
                Ok(x) => x,
                Err(_) => return Err(EvalError::BadSpecialization.into()),
              };
              if width == 0 {
                return Err(EvalError::ZeroSizedUint.into());
              }
              Value::BuiltinType(BuiltinType::Uint { bits: width })
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
    }))
  }

  fn eval_builtin_call(
    &self,
    base: &Arc<Value>,
    id: &Identifier,
    args: &[Expr],
  ) -> Result<Option<Arc<Value>>> {
    let value = match &*id.0 {
      "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
        let right = args.get(0).ok_or_else(|| EvalError::MissingArgument)?;
        let right = self.eval_expr(right)?;

        let value = match &*id.0 {
          "eq" => base.compare_eq(&right),
          "ne" => base.compare_eq(&right).sym_eq(false.into()),
          _ => {
            let ord = match &*id.0 {
              "lt" => ValueOrdering::Lt,
              "le" => ValueOrdering::Le,
              "gt" => ValueOrdering::Gt,
              "ge" => ValueOrdering::Ge,
              _ => unreachable!(),
            };
            match base.compare_ord(&right, ord) {
              Some(x) => x,
              None => {
                return Err(
                  EvalError::UncomparableTypes {
                    op: id.0.clone(),
                    left: base.clone(),
                    right,
                  }
                  .into(),
                )
              }
            }
          }
        };
        Value::UintValue(value)
      }
      "logicand" | "logicor" => {
        let right = args.get(0).ok_or_else(|| EvalError::MissingArgument)?;

        // TODO: short-circuiting semantics?
        let right = self.eval_expr(right)?;

        let result = match (&**base, &*right) {
          (Value::UintValue(ll), Value::UintValue(rr)) => match &*id.0 {
            "logicand" => ll.clone().sym_logic_and(rr.clone()),
            "logicor" => ll.clone().sym_logic_or(rr.clone()),
            _ => unreachable!(),
          },
          _ => return Err(EvalError::TypeMismatch.into()),
        };
        Value::UintValue(result)
      }
      "add" | "sub" | "mul" | "div" => {
        let right = args.get(0).ok_or_else(|| EvalError::MissingArgument)?;
        let right = self.eval_expr(right)?;
        let base_ty = base.get_type()?;
        let right = base_ty.cast_to_this_type(&right)?;
        match (&**base, &*right) {
          (Value::StringValue(ll), Value::StringValue(rr)) => match &*id.0 {
            "add" => Value::StringValue(mk_arc_str(&format!("{}{}", ll, rr))),
            _ => {
              return Err(
                EvalError::OperationNotSupported {
                  optype: id.0.clone(),
                  value: base.clone(),
                }
                .into(),
              )
            }
          },
          (Value::UintValue(ll), Value::UintValue(rr)) => {
            let value = match &*id.0 {
              "add" => ll.clone().sym_add(rr.clone()),
              "sub" => ll.clone().sym_sub(rr.clone()),
              "mul" => ll.clone().sym_mul(rr.clone()),
              "div" => ll.clone().sym_div(rr.clone()),
              _ => unreachable!(),
            };

            Value::UintValue(value)
          }
          _ => return Err(EvalError::TypeMismatch.into()),
        }
      }
      "cast" => {
        let target_type = args.get(0).ok_or_else(|| EvalError::MissingArgument)?;
        let target_type = self.eval_expr(target_type)?;
        let output = target_type.cast_to_this_type(&base)?;
        return Ok(Some(output));
      }
      "read" => {
        let base = match &**base {
          Value::SignalValue(v) => v,
          _ => return Err(EvalError::TypeMismatch.into()),
        };
        match base.ty {
          SignalType::In => {}
          _ => return Err(EvalError::ReadOnOutSignal.into()),
        }
        let value = SymbolicUint::new_external(base.handle);
        Value::unpack(value, &base.inner_ty)?
      }
      _ => return Ok(None),
    };
    Ok(Some(Arc::new(value)))
  }

  /// Evaluates a `Call` expression.
  ///
  /// Operator overloading is implicitly supported.
  fn eval_call(&self, base: &Expr, args: &[Expr], loc: (usize, usize)) -> Result<Arc<Value>> {
    let base = if let ExprV::Dot { base, id } = &base.v {
      let base = self.eval_expr(&base)?;
      match self.eval_dot(&base, id) {
        Ok(x) => x,
        Err(direct_eval_error) => {
          // Field not found. This can only be a builtin call.
          match self.eval_builtin_call(&base, id, args)? {
            Some(x) => return Ok(x),
            None => return Err(direct_eval_error),
          }
        }
      }
    } else {
      self.eval_expr(&base)?
    };
    let mut arg_values = Vec::with_capacity(args.len());
    for e in args.iter() {
      arg_values.push(self.eval_expr(e)?);
    }
    return Ok(Self::call_function(
      &self.tracker,
      base,
      &arg_values,
      Some(loc),
    )?);
  }

  fn eval_dot(&self, base: &Arc<Value>, id: &Identifier) -> Result<Arc<Value>> {
    match &**base {
      Value::ProductValue(ref product_value) => {
        let field = product_value
          .fields
          .get(&id.0)
          .ok_or_else(|| EvalError::FieldNotFound)?;
        Ok(field.clone())
      }
      _ => Err(EvalError::DotOnNonProductValue.into()),
    }
  }

  /// Evaluates an `Expr` and produces a `Value`.
  pub fn eval_expr(&self, e: &Expr) -> Result<Arc<Value>> {
    let _tracker = self.tracker.enter(EvaluationStackEntry {
      ty: EvaluationStackEntryType::Expression,
      loc: Some((e.loc_start, e.loc_end)),
    });
    let value = match &e.v {
      ExprV::Lit(x) => match x.v {
        LiteralV::Uint(ref value) => {
          Value::UintValue(SymbolicUint::new_const(value.clone(), std::u32::MAX))
        }
        LiteralV::String(ref value) => Value::StringValue(value.clone()),
      },
      ExprV::Ident(x) => {
        return Ok(self.lookup_name(&x)?);
      }
      ExprV::Dot { base, id } => {
        let base = self.eval_expr(&base)?;
        return self.eval_dot(&base, id);
      }
      ExprV::Specialize { base, tyassigns } => {
        let base = self.eval_expr(&base)?;
        match &*base {
          // A rank-2 unspecialized type.
          Value::Unspecialized(ty) => {
            let specialized = self.specialize_type(ty, &*tyassigns)?;
            debug!("specialized {:?} -> {:?}", ty, specialized);
            return Ok(specialized);
          }

          // A rank-0 function value with unspecialized signature.
          // The only place where this is allowed is a top-level function -
          // TODO: Check and error otherwise.
          Value::UnspecializedFnValue(value) => {
            let native_context = value.context.load();
            let ty = self.specialize_fntype(
              (*native_context.upgrade().expect("cannot upgrade context")).clone(),
              &value.ty,
              &*tyassigns,
            )?;
            Value::SpecializedFnValue(SpecializedFnValue {
              ty,
              body: value.body.clone(),
              context: value.context.clone(),
            })
          }

          _ => {
            return Err(EvalError::SpecializeNonUnspecializedValue.into());
          }
        }
      }
      ExprV::Call { base, args } => {
        return Ok(self.eval_call(&*base, &*args, (e.loc_start, e.loc_end))?);
      }
      ExprV::Fn(meta) => {
        let ty = self.specialize_fntype(self.clone(), meta, &[])?;
        Value::FnType(ty)
      }
      ExprV::Block(block) => {
        let meta = FnMeta {
          tyargs: Arc::new([]),
          args: block.args.clone(),
          // TODO: support returning value from block
          ret: block.ret.clone(),
        };
        let ty = self.specialize_fntype(self.clone(), &meta, &[])?;
        let ctx = Arc::downgrade(&self.tracker.allocate_context(self.clone()));
        Value::SpecializedFnValue(SpecializedFnValue {
          ty,
          body: Arc::new([FnSpecialization {
            where_expr: None,
            body: Body {
              body: block.body.clone(),
              loc_start: e.loc_start,
              loc_end: e.loc_end,
            },
          }]),
          context: Arc::new(ArcSwapWeak::new(ctx)),
        })
      }
    };
    Ok(Arc::new(value))
  }

  pub fn eval_stmt(&mut self, stmt: &Stmt, ctx: &mut BlockContext) -> Result<()> {
    match &stmt.v {
      StmtV::Let { def } => {
        let init_value = def
          .init_value
          .as_ref()
          .map(|x| self.eval_expr(x))
          .transpose()?;
        let actual_ty = init_value.as_ref().map(|x| x.get_type()).transpose()?;
        let ty = if let Some(expected_ty) = &def.ty {
          let expected_ty = self.eval_expr(expected_ty)?;
          if let Some(actual_ty) = actual_ty {
            if expected_ty != actual_ty {
              return Err(EvalError::TypeMismatch.into());
            }
          }
          expected_ty
        } else {
          actual_ty.ok_or_else(|| EvalError::MissingType)?
        };
        ctx.local_variable_types.insert(def.name.clone(), ty);

        if let Some(x) = init_value {
          self.names.insert_mut(def.name.0.clone(), x);
        }
      }
      StmtV::Assign { left, right } => {
        let right = self.eval_expr(right)?;

        // Get actual type
        let right_ty = right.get_type()?;

        // Get expected type
        let expected_ty = if let Some(x) = ctx.local_variable_types.get(left) {
          Some(x.clone())
        } else if let Some(x) = self.names.get(&left.0) {
          Some(x.get_type()?)
        } else {
          None
        };

        // Typeck
        if let Some(expected_ty) = expected_ty {
          if expected_ty != right_ty {
            return Err(EvalError::TypeMismatch.into());
          }
        }

        // Update state.
        self.names.insert_mut(left.0.clone(), right.clone());
        if ctx.local_variable_types.get(left).is_none() {
          ctx.updated_variables.insert(left.clone(), right);
        }
      }
      StmtV::IfElse {
        condition,
        if_body,
        else_body,
      } => {
        let condition = self.eval_expr(condition)?;
        if let Some(x) = condition.const_truthy() {
          if x {
            self.eval_body(if_body, Some(ctx))?;
          } else {
            if let Some(else_body) = else_body {
              self.eval_body(else_body, Some(ctx))?;
            }
          }
        } else {
          // Evaluate both & select
          unimplemented!()
        }
      }
      StmtV::Expr { e } => {
        ctx.last_result = Some(self.eval_expr(e)?);
      }
    }

    Ok(())
  }

  pub fn eval_body(
    &mut self,
    body: &Body,
    mut parent_ctx: Option<&mut BlockContext>,
  ) -> Result<Option<Arc<Value>>> {
    let tracker = self.tracker.clone();
    let _tracker_guard = tracker.enter(EvaluationStackEntry {
      ty: EvaluationStackEntryType::Body,
      loc: Some((body.loc_start, body.loc_end)),
    });
    let mut ctx = BlockContext::default();
    let mut this = self.clone();
    for stmt in body.body.iter() {
      this.eval_stmt(stmt, &mut ctx)?;
    }
    for (name, value) in ctx.updated_variables {
      self.names.insert_mut(name.0.clone(), value.clone());

      // Propagate update to parent context.
      if let Some(parent) = parent_ctx {
        if parent.local_variable_types.get(&name).is_none() {
          parent.updated_variables.insert(name, value);
        }
        parent_ctx = Some(parent);
      }
    }

    if let Some(parent) = parent_ctx {
      if let Some(last_result) = &ctx.last_result {
        parent.last_result = Some(last_result.clone());
      }
    }

    Ok(ctx.last_result)
  }

  fn call_builtin_function(
    tracker: &EvalTracker,
    target: &BuiltinFnValue,
    args: &[Arc<Value>],
  ) -> Result<Arc<Value>> {
    match target {
      BuiltinFnValue::MkSignal => {
        let signal_ty = args
          .get(0)
          .ok_or_else(|| EvalError::MissingArgument)?
          .try_to_string()?;
        let inner_ty = args
          .get(1)
          .ok_or_else(|| EvalError::MissingArgument)?
          .clone();
        let name = args.get(2).map(|x| x.try_to_string()).transpose()?;
        let signal_ty = match &*signal_ty {
          "in" => SignalType::In,
          "out" => SignalType::Out {
            assignment: AssignmentTable::new(),
          },
          "register" => SignalType::Register {
            assignment: AssignmentTable::new(),
          },
          _ => return Err(EvalError::BadSignalType(signal_ty.clone()).into()),
        };
        let value = SignalValue::generate(tracker, signal_ty, inner_ty, name)?;
        Ok(Arc::new(Value::SignalValue(value)))
      }
      BuiltinFnValue::Error => {
        let msg = args
          .get(0)
          .ok_or_else(|| EvalError::MissingArgument)?
          .try_to_string()?;
        Err(EvalError::UserError(msg).into())
      }
    }
  }

  pub fn call_function(
    tracker: &Arc<EvalTracker>,
    target: Arc<Value>,
    args: &[Arc<Value>],
    loc: Option<(usize, usize)>,
  ) -> Result<Arc<Value>> {
    let target = match &*target {
      Value::SpecializedFnValue(value) => value,
      Value::BuiltinFnValue(x) => return Self::call_builtin_function(&**tracker, x, args),
      _ => return Err(EvalError::CallingNonCallable.into()),
    };
    if target.ty.args.len() != args.len() {
      return Err(EvalError::ArgumentCountMismatch.into());
    }

    // Typeck
    for (value, arg_info) in args.iter().zip(target.ty.args.iter()) {
      let ty = value.get_type()?;
      if ty != arg_info.ty {
        return Err(
          EvalError::ArgumentTypeMismatch {
            expected: arg_info.ty.clone(),
            actual: ty,
          }
          .into(),
        );
      }
    }

    // Derive a callee context.
    let mut callee_ctx = (*target
      .context
      .load()
      .upgrade()
      .expect("cannot upgrade context"))
    .clone();
    let tracker = callee_ctx.tracker.clone();
    let _tracker_guard = tracker.enter(EvaluationStackEntry {
      ty: EvaluationStackEntryType::FunctionCall,
      loc,
    });

    // First, insert type parameters...
    for (k, v) in target.ty.tyargs.iter() {
      callee_ctx.names.insert_mut(k.0.clone(), v.clone());
    }

    // Then, insert runtime parameters.
    for (i, arg) in target.ty.args.iter().enumerate() {
      callee_ctx
        .names
        .insert_mut(arg.name.0.clone(), args[i].clone());
    }

    debug!(
      "call_function with context: {:?}",
      callee_ctx.names.iter().map(|x| x.0).collect::<Vec<_>>()
    );

    // Select body.
    // TODO: Should this be done before runtime parameter evaluation?
    let mut selected_spec: Option<&FnSpecialization> = None;
    for spec in target.body.iter() {
      if let Some(condition) = &spec.where_expr {
        let condition = callee_ctx.eval_expr(condition)?;
        let truthy = condition
          .const_truthy()
          .map(Ok)
          .or_else(|| condition.smt_truthy().transpose())
          .transpose()?;
        match truthy {
          Some(true) => {
            selected_spec = Some(spec);
            break;
          }
          Some(false) => {}
          None => return Err(EvalError::NonStaticWhereArm { condition }.into()),
        }
      } else {
        // Default
        selected_spec = Some(spec);
        break;
      }
    }

    let selected_spec = selected_spec.ok_or_else(|| EvalError::NoSpecializationSelected)?;

    // Run it!
    let retval = callee_ctx
      .eval_body(&selected_spec.body, None)?
      .unwrap_or_else(|| Arc::new(Value::UndefinedValue));
    let actual_ret_type = retval.get_type()?;

    // Implicitly ignore the return type if nothing is expected
    if let Some(ref expected_ret_type) = target.ty.ret {
      if *expected_ret_type != actual_ret_type {
        return Err(
          EvalError::ReturnTypeMismatch {
            expected: expected_ret_type.clone(),
            actual: actual_ret_type,
          }
          .into(),
        );
      }
      return Ok(retval);
    } else {
      // The "unit type".
      return Ok(Arc::new(Value::UndefinedValue));
    }
  }
}
