use super::{error::LocalParseError, state::State};
use crate::{
  ast::{self, Expr, ExprV},
  util::mk_arc_slice,
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

pub fn gen_binop_call(state: &mut State, name: &str, left: Expr, right: Expr) -> ExprV {
  let loc_start = left.loc_start;
  let loc_end = right.loc_end;
  ast::ExprV::Call {
    base: Arc::new(ast::Expr {
      v: ast::ExprV::Dot {
        base: Arc::new(left),
        id: ast::Identifier(state.get_string(name)),
      },
      loc_start,
      loc_end,
    }),
    args: mk_arc_slice(std::iter::once(right)),
  }
}
