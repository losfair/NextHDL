use std::sync::Arc;

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
  ExternSignal(GenericDef),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImportItem {
  pub path: Arc<str>,
  pub as_name: Identifier,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct GenericDef {
  pub name: Identifier,
  pub ty: Option<Type>,
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
  pub body: FnBody,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnBody {
  pub body: Arc<[Stmt]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnMeta {
  pub tyargs: Arc<[FnTyArg]>,
  pub args: Arc<[FnArg]>,
  pub ret: Option<Arc<Type>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Type {
  pub v: TypeV,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TypeV {
  Fn { meta: FnMeta },
  Dynamic { tyexp: Expr },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnTyArg {
  pub name: Identifier,
  pub kind: Type,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnArg {
  pub name: Identifier,
  pub ty: Type,
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
  Signal {
    def: GenericDef,
  },
  IfElse {
    condition: Expr,
    if_body: Arc<[Stmt]>,
    else_body: Option<Arc<[Stmt]>>,
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
  pub body: Arc<[Stmt]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Expr {
  pub v: ExprV,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ExprV {
  Lit(Literal),
  Ident(Identifier),
  Dot {
    base: Arc<Expr>,
    id: Identifier,
  },
  Specialize {
    base: Arc<Expr>,
    tyargs: Arc<[Expr]>,
  },
  Call {
    base: Arc<Expr>,
    args: Arc<[Expr]>,
  },
  Block(Arc<Block>),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Literal {
  pub v: LiteralV,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum LiteralV {
  Uint(BigUint),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Identifier(pub Arc<str>);
