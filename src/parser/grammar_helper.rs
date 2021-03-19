use super::error::LocalParseError;
use crate::ast;
use lalrpop_util::ParseError;
use num_bigint::BigUint;

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
