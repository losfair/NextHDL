use super::error::LocalParseError;
use crate::{
  ast::{self, Expr, ExprV},
  util::{mk_arc_slice, mk_arc_str},
};
use lalrpop_util::ParseError;
use num_bigint::BigUint;
use std::sync::Arc;

pub fn parse_radix_prefixed_str<L, T>(
  s: &str,
  prefix: &str,
  radix: u32,
) -> Result<ast::Literal, ParseError<L, T, LocalParseError>> {
  BigUint::parse_bytes(s.strip_prefix(prefix).unwrap().as_bytes(), radix)
    .map(|x| ast::Literal {
      v: ast::LiteralV::Uint(x),
    })
    .ok_or_else(|| ParseError::User {
      error: LocalParseError::InvalidLiteral,
    })
}

pub fn gen_binop_call(name: &str, left: Expr, right: Expr) -> ExprV {
  ast::ExprV::Call {
    base: Arc::new(ast::Expr {
      v: ast::ExprV::Dot {
        base: Arc::new(left),
        id: ast::Identifier(mk_arc_str(name)),
      },
    }),
    args: mk_arc_slice(std::iter::once(right)),
  }
}
