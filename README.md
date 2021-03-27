# NextHDL

A hardware description language with an advanced type system.

Idea: abstract execution *is* circuit synthesis.

Progress:

- [x] Parser
- [x] Basic type checker
- [x] Symbolic execution engine
- [ ] SMT solver-based dependent type checking
- [ ] Module system
- [ ] Netlist backends: Verilog, etc.
- [ ] Automatic scheduling

**This is an early work in progress and is not usable yet.**

## What does it look like?

Most likely this isn't the final shape yet but here's what an LEB128 decoder module *should* look like:

```
fn leb128_decode_statem<N: uint, M: uint>(
  next_byte: fn (_: fn (_: uint<8>)),
  completion: fn (_: uint<32>, ok: uint<1>),
  buf: uint<M>,
) where N == 0 {
  fork {
    completion(undefined, 0);
  }
} default {
  next_byte {
    (byte: uint<8>) in

    let buf = byte[6..0].append(buf);
    if byte[7] {
      leb128_decode_statem<N - 1, M + 7>(next_byte, completion, buf);
    } else {
      fork {
        completion(buf.slice[31..0], 1);
      }
    }
  }
}
```
