use std::{collections::BTreeMap, sync::Arc};

use num_bigint::BigUint;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ModuleDef {
  pub items: Arc<[ModuleItem]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Comment {
  pub content: Arc<str>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ModuleItem {
  Import(ImportItem),
  Fn(FnDef),
  Struct(StructDef),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct StructDef {
  pub name: Identifier,
  pub fields: Arc<BTreeMap<Arc<str>, Expr>>,
  pub tyargs: Arc<[TyArg]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImportItem {
  pub path: Arc<str>,
  pub as_name: Identifier,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct GenericDef {
  pub name: Identifier,
  pub ty: Option<Expr>,
  pub init_value: Option<Expr>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnDef {
  pub name: Identifier,
  pub meta: FnMeta,
  pub specializations: Arc<[FnSpecialization]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnSpecialization {
  pub where_expr: Option<Expr>,
  pub body: Body,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Body {
  pub body: Arc<[Stmt]>,
  pub loc_start: usize,
  pub loc_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnMeta {
  pub tyargs: Arc<[TyArg]>,
  pub args: Arc<[FnArg]>,
  pub ret: Option<Arc<Expr>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TyArg {
  pub name: Identifier,
  pub kind: Option<Expr>,
  pub default_value: Option<Expr>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnArg {
  pub name: Identifier,
  pub ty: Expr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Stmt {
  pub v: StmtV,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum StmtV {
  Let {
    def: GenericDef,
  },
  IfElse {
    condition: Expr,
    if_body: Body,
    else_body: Option<Body>,
  },
  Assign {
    left: Identifier,
    right: Expr,
  },
  Expr {
    e: Expr,
  },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Block {
  pub args: Arc<[FnArg]>,
  pub ret: Option<Arc<Expr>>,
  pub body: Arc<[Stmt]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Expr {
  pub v: ExprV,
  pub loc_start: usize,
  pub loc_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ExprV {
  Lit(Literal),
  Ident(Identifier),
  Fn(FnMeta),
  Dot {
    base: Arc<Expr>,
    id: Identifier,
  },
  Specialize {
    base: Arc<Expr>,
    tyassigns: Arc<[TypeAssign]>,
  },
  Call {
    base: Arc<Expr>,
    args: Arc<[Expr]>,
  },
  Block(Arc<Block>),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TypeAssign {
  pub ty: Option<Identifier>,
  pub e: Expr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Literal {
  pub v: LiteralV,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum LiteralV {
  Uint(BigUint),
  String(Arc<str>),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Identifier(pub Arc<str>);
