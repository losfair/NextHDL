use std::sync::Arc;

use indexmap::IndexMap;
use num_bigint::BigUint;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleDef {
  pub items: Arc<[ModuleItem]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Comment {
  pub content: Arc<str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModuleItem {
  Import(ImportItem),
  Fn(FnDef),
  Struct(StructDef),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructDef {
  pub name: Identifier,
  pub fields: Arc<IndexMap<Arc<str>, Expr>>,
  pub tyargs: Arc<[TyArg]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportItem {
  pub path: Arc<str>,
  pub as_name: Identifier,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericDef {
  pub name: Identifier,
  pub ty: Option<Expr>,
  pub init_value: Option<Expr>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FnDef {
  pub name: Identifier,
  pub meta: FnMeta,
  pub specializations: Arc<[FnSpecialization]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FnSpecialization {
  pub where_expr: Option<Expr>,
  pub body: Body,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Body {
  pub body: Arc<[Stmt]>,
  pub loc_start: usize,
  pub loc_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FnMeta {
  pub tyargs: Arc<[TyArg]>,
  pub args: Arc<[FnArg]>,
  pub ret: Option<Arc<Expr>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TyArg {
  pub name: Identifier,
  pub kind: Option<Expr>,
  pub default_value: Option<Expr>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FnArg {
  pub name: Identifier,
  pub ty: Expr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stmt {
  pub v: StmtV,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StmtV {
  Let {
    def: GenericDef,
  },
  IfElse {
    condition: Expr,
    if_body: Body,
    else_body: Option<Body>,
    is_static: bool,
  },
  Assign {
    left: Expr,
    right: Expr,
  },
  Expr {
    e: Expr,
  },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block {
  pub args: Arc<[FnArg]>,
  pub ret: Option<Arc<Expr>>,
  pub body: Arc<[Stmt]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Expr {
  pub v: ExprV,
  pub loc_start: usize,
  pub loc_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
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
  Slice {
    base: Arc<Expr>,
    from: Arc<Expr>,
    to: Arc<Expr>,
  },
  NewStruct {
    ty: Arc<Expr>,
    init_fields: Arc<IndexMap<Arc<str>, Expr>>,
  },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeAssign {
  pub ty: Option<Identifier>,
  pub e: Expr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Literal {
  pub v: LiteralV,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LiteralV {
  Uint(BigUint),
  String(Arc<str>),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Identifier(pub Arc<str>);
