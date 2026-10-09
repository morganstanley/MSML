#![allow(dead_code)]
use std::{
    iter::FusedIterator,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use derive_more::{Debug, IntoIterator};

#[derive(IntoIterator, Debug)]
#[debug("{inner:?}")]
pub(crate) struct TypedVec<I, T> {
    #[into_iterator(owned, ref, ref_mut)]
    inner: Vec<T>,
    _phantom: PhantomData<I>,
}

impl<I, T: Clone> Clone for TypedVec<I, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner);
    }
}

impl<I, T: PartialEq> PartialEq for TypedVec<I, T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<I, T: Eq> Eq for TypedVec<I, T> {}

impl<I: TypedVecIndex, T> TypedVec<I, T> {
    pub fn with_capacity(capacity: I) -> Self {
        Self {
            inner: Vec::with_capacity(capacity.into_usize()),
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, val: T) -> I {
        let len = self.len();
        self.inner.push(val);
        len
    }

    #[inline(always)]
    pub fn get(&self, id: I) -> Option<&T> {
        let id = id.into_usize();
        self.inner.get(id)
    }

    #[inline(always)]
    pub fn swap(&mut self, u: I, v: I) {
        self.inner.swap(u.into_usize(), v.into_usize());
    }

    #[inline(always)]
    pub fn len(&self) -> I {
        I::from_usize(self.inner.len())
    }

    pub fn enumerate(
        &self,
    ) -> impl DoubleEndedIterator<Item = (I, &T)> + ExactSizeIterator + FusedIterator {
        self.inner
            .iter()
            .enumerate()
            .map(|(i, t)| (I::from_usize(i), t))
    }

    pub fn enumerate_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (I, &mut T)> + ExactSizeIterator + FusedIterator {
        self.inner
            .iter_mut()
            .enumerate()
            .map(|(i, t)| (I::from_usize(i), t))
    }

    pub fn keys(
        &self,
    ) -> impl DoubleEndedIterator<Item = I> + ExactSizeIterator + FusedIterator + 'static {
        (0..self.len().into_usize()).map(I::from_usize)
    }
}

impl<I, T> TypedVec<I, T> {
    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_ref()
    }

    #[inline(always)]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator + FusedIterator {
        self.inner.iter()
    }

    #[inline(always)]
    pub fn iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut T> + ExactSizeIterator + FusedIterator {
        self.inner.iter_mut()
    }

    #[inline(always)]
    pub fn last(&self) -> Option<&T> {
        self.inner.last()
    }
}

impl<I, T> AsRef<[T]> for TypedVec<I, T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        &self.inner
    }
}

impl<I: TypedVecIndex, T: Clone> TypedVec<I, T> {
    pub fn new_with(element: T, capacity: I) -> Self {
        vec![element; capacity.into_usize()].into()
    }

    pub fn enumerate_cloned(
        &self,
    ) -> impl DoubleEndedIterator<Item = (I, T)> + ExactSizeIterator + FusedIterator {
        self.inner
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, t)| (I::from_usize(i), t))
    }
}

impl<I: TypedVecIndex, T: Copy> TypedVec<I, T> {
    pub fn enumerate_copied(
        &self,
    ) -> impl DoubleEndedIterator<Item = (I, T)> + ExactSizeIterator + FusedIterator {
        self.inner
            .iter()
            .copied()
            .enumerate()
            .map(|(i, t)| (I::from_usize(i), t))
    }
}

impl<I: TypedVecIndex, T> Index<I> for TypedVec<I, T> {
    type Output = <Vec<T> as Index<usize>>::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        self.inner.index(index.into_usize())
    }
}

impl<I: TypedVecIndex, T> IndexMut<I> for TypedVec<I, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.inner.index_mut(index.into_usize())
    }
}

impl<I, T, C: Into<Vec<T>>> From<C> for TypedVec<I, T> {
    fn from(inner: C) -> Self {
        Self {
            inner: inner.into(),
            _phantom: PhantomData,
        }
    }
}

impl<I, T> FromIterator<T> for TypedVec<I, T> {
    fn from_iter<Iter: IntoIterator<Item = T>>(iter: Iter) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl<I, T> Extend<T> for TypedVec<I, T> {
    fn extend<C: IntoIterator<Item = T>>(&mut self, iter: C) {
        self.inner.extend(iter);
    }
}

pub(crate) trait TypedVecIndex: 'static + PartialEq + PartialOrd {
    fn into_usize(self) -> usize;
    fn from_usize(val: usize) -> Self;
}

pub fn vec_with_head<T, I: TypedVecIndex>(head: T, cap: I) -> Vec<T> {
    let mut stack = TypedVec::with_capacity(cap);
    stack.push(head);
    stack.inner
}

macro_rules! from_usize {
    ($name:ident, usize) => {};
    ($name:ident, $inner:tt) => {
        impl From<usize> for $name {
            #[inline(always)]
            fn from(val: usize) -> Self {
                Self(val as $inner)
            }
        }
    };
}

macro_rules! typed_vec_index {
    ($vis:vis $name:ident, $inner:tt $(,[$($derive:path),+])?) => {
        #[derive(
            Copy,
            Clone,
            Default,
            Hash,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            ::derive_more::Debug,
            ::derive_more::Display,
            ::derive_more::Into,
            ::derive_more::From,
            ::derive_more::Not,
            ::derive_more::Add,
            ::derive_more::Sub,
        )]
        $(#[derive($($derive),+)])?
        #[debug("{}", _0)]
        $vis struct $name($inner);

        #[allow(dead_code)]
        impl $name {
            pub const ZERO: Self = Self($inner::MIN);
            pub const MAX: Self = Self($inner::MAX);

            #[inline(always)]
            pub const fn new(inner: $inner) -> Self {
                Self(inner)
            }

            #[inline(always)]
            pub const fn next(self) -> Self {
                Self(self.0 + 1)
            }

            #[inline(always)]
            pub const fn prev(self) -> Self {
                Self(self.0.saturating_sub(1))
            }

            #[inline(always)]
            pub const fn inner(self) -> $inner {
                self.0
            }

            #[inline(always)]
            pub const fn inner_ref(&self) -> &$inner {
                &self.0
            }

            #[inline(always)]
            pub const fn inner_mut(&mut self) -> &mut $inner {
                &mut self.0
            }

            #[inline(always)]
            pub const fn as_usize(self) -> usize {
                self.inner() as usize
            }
        }

        impl $crate::typed_vec::TypedVecIndex for $name {
            #[inline(always)]
            fn into_usize(self) -> usize {
                self.as_usize()
            }

            #[inline(always)]
            fn from_usize(val: usize) -> Self {
                Self(val.min(<$inner>::MAX as usize) as $inner)
            }
        }

        $crate::typed_vec::from_usize!($name, $inner);

        impl ::core::borrow::Borrow<$inner> for $name {
            #[inline(always)]
            fn borrow(&self) -> &$inner {
                &self.0
            }
        }
    };
}

pub(crate) use {from_usize, typed_vec_index};

#[cfg(test)]
mod tests {
    typed_vec_index!(U8, u8);
    typed_vec_index!(U16, u16);
    typed_vec_index!(U32, u32);
    typed_vec_index!(U64, u64);
    typed_vec_index!(Usize, usize);
}
