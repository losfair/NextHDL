use num_bigint::BigUint;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ModuleDef {
  pub items: Vec<ModuleItem>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ModuleItem {
  Fn(FnDef),
  ExternSignal(GenericDef),
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
  pub specializations: Vec<FnSpecialization>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnSpecialization {
  pub where_expr: Option<Expr>,
  pub body: FnBody,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnBody {
  pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FnMeta {
  pub tyargs: Vec<FnTyArg>,
  pub args: Vec<FnArg>,
  pub ret: Option<Box<Type>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Type {
  pub v: TypeV,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TypeV {
  Fn { meta: FnMeta },
  Named { name: Identifier, tyargs: Vec<Expr> },
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
  Let { def: GenericDef },
  Signal { def: GenericDef },
  IfElse { condition: Expr, if_body: Vec<Stmt>, else_body: Option<Vec<Stmt>> },
  Assign { left: Identifier, right: Expr },
  Expr { e: Expr },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Block {
  pub args: Vec<FnArg>,
  pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Expr {
  pub v: ExprV,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ExprV {
  Lit(Literal),
  Ident(Identifier),
  Dot { base: Box<Expr>, id: Identifier },
  Specialize { base: Box<Expr>, tyargs: Vec<Expr> },
  Call { base: Box<Expr>, args: Vec<Expr> },
  Block(Box<Block>),
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
pub struct Identifier(pub String);
