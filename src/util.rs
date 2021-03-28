use std::{
  mem::{ManuallyDrop, MaybeUninit},
  sync::Arc,
};

use num_bigint::BigUint;

pub fn mk_arc_str(from: &str) -> Arc<str> {
  let s = mk_arc_slice(from.as_bytes().iter().copied());
  unsafe { std::mem::transmute::<Arc<[u8]>, Arc<str>>(s) }
}

pub fn mk_arc_slice<T, I: Iterator<Item = T>>(from: I) -> Arc<[T]> {
  // Allocate memory for elements.
  let size_hint = from.size_hint();
  if Some(size_hint.0) != size_hint.1 {
    panic!("mk_arc_slice: invalid size");
  }
  let len = size_hint.0;

  let mut raw: Arc<[MaybeUninit<T>]> = Arc::new_uninit_slice(len);
  let raw_mut = Arc::get_mut(&mut raw).unwrap();

  let mut count: usize = 0;

  for (i, elem) in from.enumerate() {
    raw_mut[i].write(elem);
    count += 1;
  }

  // Check that all elements are actually written to before assume_init.
  if count != len {
    panic!("mk_arc_size: count mismatch. memory leaked.");
  }

  unsafe { raw.assume_init() }
}

pub trait ArcSliceExt {
  type Item;

  fn take_all(&mut self) -> ArcSliceTakeAllIterator<Self::Item>;
}

impl<T: Clone> ArcSliceExt for Arc<[T]> {
  type Item = T;
  fn take_all(&mut self) -> ArcSliceTakeAllIterator<Self::Item> {
    // Ensure unique ownership
    if Arc::strong_count(self) != 1 {
      *self = mk_arc_slice(self.iter().cloned());
      assert!(Arc::strong_count(self) == 1);
    }

    // Take out
    let me = std::mem::replace(self, Arc::new([]));

    ArcSliceTakeAllIterator {
      me: unsafe { std::mem::transmute::<Arc<[T]>, Arc<[ManuallyDrop<T>]>>(me) },
      index: 0,
    }
  }
}

pub struct ArcSliceTakeAllIterator<T> {
  /// Unique ownership guaranteed
  me: Arc<[ManuallyDrop<T>]>,
  index: usize,
}

impl<T> Drop for ArcSliceTakeAllIterator<T> {
  fn drop(&mut self) {
    let me = Arc::get_mut(&mut self.me).unwrap();

    // Drop remaining elements
    for i in self.index..me.len() {
      unsafe {
        ManuallyDrop::drop(&mut me[i]);
      }
    }
  }
}

impl<T> Iterator for ArcSliceTakeAllIterator<T> {
  type Item = T;

  fn size_hint(&self) -> (usize, Option<usize>) {
    let n = self.me.len() - self.index;
    (n, Some(n))
  }

  fn next(&mut self) -> Option<Self::Item> {
    let me = Arc::get_mut(&mut self.me).unwrap();

    if self.index == me.len() {
      None
    } else {
      let value = unsafe { ManuallyDrop::take(&mut me[self.index]) };
      self.index += 1;
      Some(value)
    }
  }
}

pub fn truncate_biguint(x: &mut BigUint, target_bits: u32) {
  let value_bits = x.bits();
  for i in (target_bits as u64)..value_bits {
    x.set_bit(i, false);
  }
  assert!(x.bits() <= target_bits as u64);
}
