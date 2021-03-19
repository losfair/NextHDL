# NextHDL

A dependently typed hardware description language featuring automatic scheduling.

**This is an early work in progress and is not usable yet.**

## Example

LEB128 decoder:

```
fn leb128_decode_statem<N: Int, M: Int>(
  next_byte: fn (_: fn (_: uint<8>)),
  completion: fn (_: uint<32>, ok: uint<1>),
  buf: uint<M>,
) where N.eq(0) {
  fork {
    completion(undefined, 0);
  }
} default {
  next_byte {
    (byte: uint<8>) in

    let buf = byte.slice<6, 0>().append(buf);
    if byte.get(7) {
      leb128_decode_statem<N.sub(1), M.add(7)>(next_byte, completion, buf);
    } else {
      fork {
        completion(buf.slice<31, 0>(), 1);
      }
    }
  }
}
```