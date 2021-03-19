use thiserror::Error;

#[derive(Error, Debug)]
pub enum LocalParseError {
  #[error("invalid literal")]
  InvalidLiteral,
}
