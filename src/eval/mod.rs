pub mod error;
pub mod signal;
pub mod value;

use error::EvalError;
use indexmap::IndexMap;
use value::*;

use anyhow::Result;
use arc_swap::ArcSwapWeak;
use rpds::RedBlackTreeMapSync;
use std::{
  collections::{BTreeMap, BTreeSet},
  fmt::Debug,
  sync::Arc,
};

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

  constraints: Option<Arc<ConstraintStackEntry>>,
}

#[derive(Debug)]
struct ConstraintStackEntry {
  constraint: SymbolicUint,
  link: Option<Arc<ConstraintStackEntry>>,
}

#[derive(Default)]
pub struct BlockContext {
  /// A map from names of local variables to their types.
  pub local_variable_types: RedBlackTreeMapSync<Identifier, Arc<Value>>,

  /// Non-propagating variables.
  pub non_propagating_variables: BTreeSet<Identifier>,

  /// Updated variables in this block.
  pub updated_variables: BTreeMap<Identifier, Arc<Value>>,

  /// The result of the last expression.
  pub last_result: Option<Arc<Value>>,
}

impl BlockContext {
  /// Creates an "observer" BlockContext that absorbs all variable updates.
  fn new_child(&self) -> Self {
    Self {
      local_variable_types: self.local_variable_types.clone(),
      ..Default::default()
    }
  }
}

impl EvalContext {
  pub fn new() -> Self {
    Self {
      names: Default::default(),
      tracker: Arc::new(EvalTracker::new()),
      constraints: None,
    }
  }

  pub fn unshare_evaluation_stack(&self) -> Self {
    let mut this = self.clone();
    this.tracker = Arc::new(this.tracker.unshare_evaluation_stack());
    this
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

        let mut fields: IndexMap<Arc<str>, Arc<Value>> = IndexMap::new();
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
          Value::BuiltinType(BuiltinType::Uint { bits: 32 })
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
      "concat" => {
        let right = args.get(0).ok_or_else(|| EvalError::MissingArgument)?;
        let right = self.eval_expr(right)?;

        match (&**base, &*right) {
          (Value::UintValue(ll), Value::UintValue(rr)) => {
            Value::UintValue(ll.clone().sym_concat(rr.clone()))
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
      "len" => {
        let len = match &**base {
          Value::UintValue(v) => v.bits(),
          Value::StringValue(x) => x.len() as u32,
          _ => return Err(EvalError::TypeMismatch.into()),
        };
        Value::UintValue(SymbolicUint::new_const(len.into(), 32))
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
        LiteralV::Uint(ref value) => Value::UintValue(SymbolicUint::new_const(
          value.clone(),
          value.bits().max(32) as u32,
        )),
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
      ExprV::Slice { base, from, to } => {
        let base = self.eval_expr(base)?;
        let from = self.eval_expr(from)?;
        let to = self.eval_expr(to)?;
        let (from, to) = match (&*from, &*to) {
          (Value::UintValue(from), Value::UintValue(to)) => {
            let from = u32::try_from(from.as_const()?)?;
            let to = u32::try_from(to.as_const()?)?;
            (from, to)
          }
          _ => return Err(EvalError::SliceBoundsMustBeUint.into()),
        };
        match &*base {
          Value::UintValue(base) => Value::UintValue(base.clone().sym_slice(from, to)?),
          _ => return Err(EvalError::SliceOnNonUint.into()),
        }
      }
    };
    Ok(Arc::new(value))
  }

  fn rebuild_product_value(
    ident_path: &mut Vec<&Identifier>,
    v: &ProductValue,
    right: Arc<Value>,
  ) -> Result<Arc<Value>> {
    let mut new_value = v.clone();
    let id = ident_path
      .pop()
      .ok_or_else(|| EvalError::IdentifierPathSegmentResolveFail)?;
    if let Some(x) = new_value.fields.get(&id.0) {
      let x = match &**x {
        Value::ProductValue(x) => Self::rebuild_product_value(ident_path, x, right)?,
        _ => {
          let prev_ty = x.get_type()?;
          let right_ty = right.get_type()?;
          if prev_ty != right_ty {
            return Err(EvalError::TypeMismatch.into());
          }
          right
        }
      };
      new_value.fields.insert(id.0.clone(), x);
    } else {
      return Err(EvalError::FieldNotFound.into());
    }
    Ok(Arc::new(Value::ProductValue(new_value)))
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
        ctx.local_variable_types.insert_mut(def.name.clone(), ty);
        ctx.non_propagating_variables.insert(def.name.clone());

        // Update value table
        if let Some(x) = init_value {
          self.names.insert_mut(def.name.0.clone(), x);
        } else {
          self.names.remove_mut(&def.name.0);
        }
      }
      StmtV::Assign { left, right } => {
        let right = self.eval_expr(right)?;

        // Resolve identifier
        let mut left = left;
        let mut ident_path = vec![];
        loop {
          match &left.v {
            ExprV::Dot { base, id } => {
              ident_path.push(id);
              left = &**base;
            }
            ExprV::Ident(x) => {
              ident_path.push(x);
              break;
            }
            _ => return Err(EvalError::InvalidAssignLeft.into()),
          }
        }

        let top = ident_path.pop().unwrap();
        let current_value = self.names.get(&top.0);

        let new_value = if let Some(x) = current_value {
          if let Value::ProductValue(v) = &**x {
            Self::rebuild_product_value(&mut ident_path, v, right)?
          } else {
            right
          }
        } else {
          right
        };

        // Ensure nothing is left
        if !ident_path.is_empty() {
          return Err(EvalError::IdentifierPathSegmentResolveFail.into());
        }

        // Get expected type
        let expected_ty = ctx
          .local_variable_types
          .get(top)
          .ok_or_else(|| EvalError::MissingLocalDecl)?;

        let new_ty = new_value.get_type()?;

        // Typeck
        if *expected_ty != new_ty {
          return Err(EvalError::TypeMismatch.into());
        }

        self.names.insert_mut(top.0.clone(), new_value.clone());
        if !ctx.non_propagating_variables.contains(top) {
          ctx.updated_variables.insert(top.clone(), new_value);
        }
      }
      StmtV::IfElse {
        condition,
        if_body,
        else_body,
        is_static,
      } => {
        let condition = self.eval_expr(condition)?;
        let mut truthy = condition.const_truthy();
        if truthy.is_none() && *is_static {
          truthy = Some(
            condition
              .smt_truthy()?
              .ok_or_else(|| EvalError::NonStaticStaticIf {
                condition: condition.clone(),
              })?,
          );
        }
        if let Some(x) = truthy {
          if x {
            self.eval_body(if_body, Some(ctx))?;
          } else {
            if let Some(else_body) = else_body {
              self.eval_body(else_body, Some(ctx))?;
            }
          }
        } else {
          // Evaluate both & select
          let condition = condition
            .pack()?
            .ok_or_else(|| EvalError::EmptyProductValueNotAllowed)?;

          // "observer" contexts
          let mut observer_on_true = ctx.new_child();
          let mut observer_on_false = ctx.new_child();

          let mut res_if = None;
          let mut res_else = None;

          // Unshare since we are going to send them to different threads
          let mut this_if = self.unshare_evaluation_stack();
          let mut this_else = self.unshare_evaluation_stack();

          // Push constraints
          this_if.constraints = Some(Arc::new(ConstraintStackEntry {
            constraint: condition.clone(),
            link: this_if.constraints.clone(),
          }));
          this_else.constraints = Some(Arc::new(ConstraintStackEntry {
            constraint: condition.clone().sym_logic_not(),
            link: this_else.constraints.clone(),
          }));

          info!("exploring both branches on condition: {:?}", condition);

          rayon::scope(|s| {
            s.spawn(|_| {
              res_if = Some(
                this_if
                  .eval_body(if_body, Some(&mut observer_on_true))
                  .map(|_| ()),
              );
            });

            s.spawn(|_| {
              if let Some(else_body) = else_body.as_ref() {
                res_else = Some(
                  this_else
                    .eval_body(else_body, Some(&mut observer_on_false))
                    .map(|_| ()),
                );
              } else {
                res_else = Some(Ok(()));
              }
            });
          });

          // Check errors
          if let Err(e) = res_if.unwrap() {
            self.tracker.merge(&this_if.tracker);
            return Err(e);
          }

          if let Err(e) = res_else.unwrap() {
            self.tracker.merge(&this_else.tracker);
            return Err(e);
          }

          let keys = observer_on_true
            .updated_variables
            .iter()
            .map(|x| x.0.clone())
            .chain(
              observer_on_false
                .updated_variables
                .iter()
                .map(|x| x.0.clone()),
            )
            .collect::<BTreeSet<_>>();

          for k in keys {
            let candidate_true = observer_on_true.updated_variables.get(&k);
            let candidate_false = observer_on_false.updated_variables.get(&k);

            let result = if let (Some(unpacked_tt), Some(unpacked_ff)) =
              (candidate_true, candidate_false)
            {
              // If updated in both cases...

              let ty = unpacked_tt.get_type()?;
              assert_eq!(unpacked_ff.get_type()?, ty);
              let tt = unpacked_tt.pack()?;
              let ff = unpacked_ff.pack()?;
              assert!(tt.is_some() == ff.is_some());

              if tt.is_some() {
                let tt = tt.unwrap();
                let ff = ff.unwrap();
                let result = condition.clone().sym_select(tt, ff);
                Arc::new(Value::unpack(result, &ty)?)
              } else {
                unpacked_tt.clone()
              }
            } else {
              // Otherwise only updated in one case.
              let (condition, value, ty) = if let Some(unpacked_tt) = candidate_true {
                (condition.clone(), unpacked_tt, unpacked_tt.get_type()?)
              } else if let Some(unpacked_ff) = candidate_false {
                (
                  condition.clone().sym_logic_not(),
                  unpacked_ff,
                  unpacked_ff.get_type()?,
                )
              } else {
                unreachable!()
              };

              match value.pack()? {
                Some(value) => {
                  // Compare with our current value.
                  let already_here = self.names.get(&k.0)
                    .map(|x| x.pack()
                      .map(|x|
                        x.expect("internal inconsistency: got non-zero-sized value but we currently have a zero-sized value")
                      )
                    ).transpose()?
                    .unwrap_or_else(|| SymbolicUint::new_undefined(value.bits()));

                  assert_eq!(already_here.bits(), value.bits());

                  let target_value = condition.sym_select(value, already_here);
                  Arc::new(Value::unpack(target_value, &ty)?)
                }
                None => {
                  // Zero-sized value.
                  // Already typeck-ed. Just insert and return.
                  value.clone()
                }
              }
            };

            self.names.insert_mut(k.0, result);
          }
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
    let mut ctx = match parent_ctx.as_mut() {
      Some(x) => x.new_child(),
      None => BlockContext::default(),
    };
    let mut this = self.clone();
    for stmt in body.body.iter() {
      this.eval_stmt(stmt, &mut ctx)?;
    }
    for (name, value) in ctx.updated_variables {
      self.names.insert_mut(name.0.clone(), value.clone());

      // Propagate update to parent context.
      if let Some(parent) = parent_ctx {
        if !parent.non_propagating_variables.contains(&name) {
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
      BuiltinFnValue::Undefined => {
        let ty = args
          .get(0)
          .ok_or_else(|| EvalError::MissingArgument)?
          .clone();
        let signal = SymbolicUint::new_undefined(ty.rank1_width()?);
        let value = Arc::new(Value::unpack(signal, &ty)?);
        Ok(value)
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
      .unwrap_or_else(|| Arc::new(Value::Unit));
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
      return Ok(Arc::new(Value::Unit));
    }
  }
}
