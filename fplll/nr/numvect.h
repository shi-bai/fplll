/* Copyright (C) 2011 Xavier Pujol.

   This file is part of fplll. fplll is free software: you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation,
   either version 2.1 of the License, or (at your option) any later version.

   fplll is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with fplll. If not, see <http://www.gnu.org/licenses/>. */

#ifndef FPLLL_NUMVECT_H
#define FPLLL_NUMVECT_H

#include "nr.h"
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#pragma message("# [warn]: AVX2 + FMA optimizations ENABLED")
#endif

FPLLL_BEGIN_NAMESPACE

/** Extends the size of the given vector. */
template <class T> void extend_vect(vector<T> &v, int size)
{
  if (static_cast<int>(v.size()) < size)
  {
    v.resize(size);
  }
}

/** Reverses a vector by consecutive swaps. */
template <class T> void reverse_by_swap(vector<T> &v, int first, int last)
{
  for (; first < last; first++, last--)
    v[first].swap(v[last]);
}

/** Rotates a vector by consecutive swaps. */
template <class T> void rotate_by_swap(vector<T> &v, int first, int middle, int last)
{
  // Algorithm from STL code
  reverse_by_swap(v, first, middle - 1);
  reverse_by_swap(v, middle, last);
  for (; first < middle && middle <= last; first++, last--)
  {
    v[first].swap(v[last]);
  }
  if (first == middle)
    reverse_by_swap(v, middle, last);
  else
    reverse_by_swap(v, first, middle - 1);
}

/** Rotates a vector left-wise by consecutive swaps. */
template <class T> void rotate_left_by_swap(vector<T> &v, int first, int last)
{
  FPLLL_DEBUG_CHECK(0 <= first && first <= last && last < static_cast<int>(v.size()));
  for (int i = first; i < last; i++)
  {
    v[i].swap(v[i + 1]);
  }
}

/** Rotates a vector right-wise by consecutive swaps. */
template <class T> void rotate_right_by_swap(vector<T> &v, int first, int last)
{
  FPLLL_DEBUG_CHECK(0 <= first && first <= last && last < static_cast<int>(v.size()));
  for (int i = last - 1; i >= first; i--)
  {
    v[i].swap(v[i + 1]);
  }
}

/** Print a vector on stream os. */
template <class T> ostream &operator<<(ostream &os, const vector<T> &v)
{
  os << "[";
  int n = v.size();
  for (int i = 0; i < n; i++)
  {
    if (i > 0)
      os << " ";
    os << v[i];
  }
  os << "]";
  return os;
}

/** Reads a vector from stream is. */
template <class T> istream &operator>>(istream &is, vector<T> &v)
{
  char c;
  v.clear();
  if (!(is >> c))
    return is;
  if (c != '[')
  {
    is.setstate(ios::failbit);
    return is;
  }
  while (is >> c && c != ']')
  {
    is.putback(c);
    v.resize(v.size() + 1);
    if (!(is >> v.back()))
    {
      v.pop_back();
      return is;
    }
  }
  return is;
}

/** Generate a zero vector. */
template <class T> void gen_zero_vect(vector<T> &v, int n)
{
  v.resize(n);
  fill(v.begin(), v.end(), 0);
}

template <class T> class NumVect;

template <class T> ostream &operator<<(ostream &os, const NumVect<T> &v);

template <class T> istream &operator>>(istream &is, NumVect<T> &v);

template <class T> class NumVect
{
public:
  typedef typename vector<T>::iterator iterator;
  /** Creates an empty NumVect (0). */
  NumVect() {}
  /** Initializes NumVect with the elements of the given NumVect. */
  NumVect(const NumVect &v) : data(v.data) {}
  /** Initializes NumVect with the elements of a given vector. */
  NumVect(const vector<T> &v) : data(v) {}
  /** Initializes NumVect of specific size, The initial content is
      undefined. */
  NumVect(int size) : data(size) {}
  NumVect(int size, const T &t) : data(size, t) {}
  /** Sets the NumVect to the elements of the given NumVect. */
  void operator=(const NumVect &v)
  {
    if (this != &v)
      data = v.data;
  }
  /** Swaps the data between the NumVect and the given NumVect. */
  void swap(NumVect &v) { data.swap(v.data); }
  /** Returns an iterator to the beginning of NumVect.*/
  const iterator begin() { return data.begin(); }
  /** Returns an iterator to the end of NumVect. */
  iterator end() { return data.end(); }
  /** Returns the number of elements in NumVect. */
  int size() const { return data.size(); }
  /** Checks whether NumVect is empty. */
  bool empty() const { return data.empty(); }
  /** Sets the size of NumVect. */
  void resize(int size) { data.resize(size); }
  /** Sets the size of NumVect and all its elemnts to t.*/
  void resize(int size, const T &t) { data.resize(size, t); }
  /** Sets the size of NumVect and all its elements to zero. */
  void gen_zero(int size)
  {
    data.resize(size);
    fill(0);
  }

  /**Compares two NumVects by comparing the underlying vectors. Returns true if equivalent & false
   * otherwise.
   * Note that this only works if the two NumVects have the same template type parameter. **/
  bool operator==(NumVect<T> &other) { return other.data == data; }

  /** Inserts an element in the back of NumVect. */
  void push_back(const T &t) { data.push_back(t); }
  /** Removes the back element of NumVect. */
  void pop_back() { data.pop_back(); }
  /** Returns a reference to the front element of NumVect. */
  T &front() { return data.front(); }
  /** Returns a const reference to the front element of NumVect,
      on constant object. */
  const T &front() const { return data.front(); }
  /** Returns a reference to the back element of NumVect. */
  T &back() { return data.back(); }
  /** Returns a const reference to the back element of NumVect,
      on constant object. */
  const T &back() const { return data.back(); }
  /** Extends the size of NumVect, only if it is needed. */
  void extend(int maxSize)
  {
    if (size() < maxSize)
      data.resize(maxSize);
  }
  void clear() { data.clear(); }
  /** Returns a reference to the i-th element of NumVect. */
  T &operator[](int i) { return data[i]; }
  /** Returns a const reference to the i-th element of NumVect on constant
      object. */
  const T &operator[](int i) const { return data[i]; }
  /** Addition of two NumVector objects, till index n. */
  void add(const NumVect<T> &v, int n);
  /** Addition of two NumVector objects. */
  void add(const NumVect<T> &v) { add(v, size()); }
  /** Subtraction of two NumVector objects, till index n. */
  void sub(const NumVect<T> &v, int n);
  /** Subtraction of two NumVector objects. */
  void sub(const NumVect<T> &v) { sub(v, size()); }
  /** Multiplication of NumVector and a number c, from index b till index n. */
  void mul(const NumVect<T> &v, int b, int n, T c);
  /** Multiplication of NumVector and a number c, till index n. */
  void mul(const NumVect<T> &v, int n, T c);
  /** Multiplication of NumVector and a number c. */
  void mul(const NumVect<T> &v, T c); // { mul(v, size(), c); }
  /** Division of NumVector and a number c, from index b till index n. */
  void div(const NumVect<T> &v, int b, int n, T c);
  /** Division of NumVector and a number c, till index n. */
  void div(const NumVect<T> &v, int n, T c);
  /** Division of NumVector and a number c. */
  void div(const NumVect<T> &v, T c) { div(v, size(), c); }
  /** Incremeanting each coefficient of NumVector by its product with
      number c, from beg to index n - 1. */
  void addmul(const NumVect<T> &v, T x, int beg, int n);
  /** Incremeanting each coefficient of NumVector by its product with
      number c, till index n. */
  void addmul(const NumVect<T> &v, T x, int n);
  /** Incremeanting each coefficient of NumVector by its product with
      number c. */
  void addmul(const NumVect<T> &v, T x) { addmul(v, x, size()); }
  void addmul_2exp(const NumVect<T> &v, const T &x, long expo, T &tmp)
  {
    addmul_2exp(v, x, expo, size(), tmp);
  }
  void addmul_2exp(const NumVect<T> &v, const T &x, long expo, int n, T &tmp);
  void addmul_si(const NumVect<T> &v, long x) { addmul_si(v, x, size()); }
  void addmul_si(const NumVect<T> &v, long x, int n);
  void addmul_si_2exp(const NumVect<T> &v, long x, long expo, T &tmp)
  {
    addmul_si_2exp(v, x, expo, size(), tmp);
  }
  void addmul_si_2exp(const NumVect<T> &v, long x, long expo, int n, T &tmp);

  /** (v[first],...,v[last]) becomes (v[first+1],...,v[last],v[first]) */
  void rotate_left(int first, int last) { rotate_left_by_swap(data, first, last); }

  /** (v[first],...,v[last]) becomes (v[last],v[first],...,v[last-1]) */
  void rotate_right(int first, int last) { rotate_right_by_swap(data, first, last); }

  /** Returns expo >= 0 such that all elements are < 2^expo. */
  long get_max_exponent();

  /** Fills NumVect with the value given. */
  void fill(long value);

  /** Checks if NumVect has zero elements from fromCol. */
  bool is_zero(int fromCol = 0) const;

  /** Returns last non-zero index of NumVector. */
  int size_nz() const;

  friend ostream &operator<< <T>(ostream &os, const NumVect<T> &v);
  friend istream &operator>> <T>(istream &is, NumVect<T> &v);

private:
  vector<T> data;
};

/** Generic add **/
template <class T> void NumVect<T>::add(const NumVect<T> &v, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  for (int i = n - 1; i >= 0; i--)
    data[i].add(data[i], v[i]);
}

/** AVX2 add for FP_NR<double> **/
#if defined(__AVX2__)
template <> inline void NumVect<FP_NR<double>>::add(const NumVect<FP_NR<double>> &v, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  double* dpt = reinterpret_cast<double*>(&data[0]);
  const double* vpt = reinterpret_cast<const double*>(&v.data[0]);
  int i = 0;
  if (n >= 4) {
    for (; i <= n - 4; i += 4) {
      __m256d t1 = _mm256_loadu_pd(&dpt[i]);
      __m256d t2 = _mm256_loadu_pd(&vpt[i]);
      __m256d result = _mm256_add_pd(t1, t2);
      _mm256_storeu_pd(&dpt[i], result);
    }
  }
  for (; i < n; i++) {
    dpt[i] += vpt[i];
  }
}
#endif

/** AVX2 add for Z_NR<long> **/
#if defined(__AVX2__)
template <> inline void NumVect<Z_NR<long>>::add(const NumVect<Z_NR<long>> &v, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  long* dpt = reinterpret_cast<long*>(&data[0]);
  const long* vpt = reinterpret_cast<const long*>(&v.data[0]);
  int i = 0;
  if (n >= 4) {
    for (; i <= n - 4; i += 4) {
      __m256i t1 = _mm256_loadu_si256((const __m256i*)&dpt[i]);
      __m256i t2 = _mm256_loadu_si256((const __m256i*)&vpt[i]);
      __m256i result = _mm256_add_epi64(t1, t2);
      _mm256_storeu_si256((__m256i*)&dpt[i], result);
    }
  }
  for (; i < n; i++) {
    dpt[i] += vpt[i];
  }
}
#endif

/** Generic sub **/
template <class T> void NumVect<T>::sub(const NumVect<T> &v, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  for (int i = n - 1; i >= 0; i--)
    data[i].sub(data[i], v[i]);
}

/** AVX2 sub for FP_NR<double> **/
#if defined(__AVX2__)
template <> inline void NumVect<FP_NR<double>>::sub(const NumVect<FP_NR<double>> &v, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  double* dpt = reinterpret_cast<double*>(&data[0]);
  const double* vpt = reinterpret_cast<const double*>(&v.data[0]);
  int i = 0;
  if (n >= 4) {
    for (; i <= n - 4; i += 4) {
      __m256d t1 = _mm256_loadu_pd(&dpt[i]);
      __m256d t2 = _mm256_loadu_pd(&vpt[i]);
      __m256d result = _mm256_sub_pd(t1, t2);
      _mm256_storeu_pd(&dpt[i], result);
    }
  }
  for (; i < n; i++) {
    dpt[i] -= vpt[i];
  }
}
#endif

/** AVX2 sub for Z_NR<long> **/
#if defined(__AVX2__)
template <> inline void NumVect<Z_NR<long>>::sub(const NumVect<Z_NR<long>> &v, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  long* dpt = reinterpret_cast<long*>(&data[0]);
  const long* vpt = reinterpret_cast<const long*>(&v.data[0]);
  int i = 0;
  if (n >= 4) {
    for (; i <= n - 4; i += 4) {
      __m256i t1 = _mm256_loadu_si256((const __m256i*)&dpt[i]);
      __m256i t2 = _mm256_loadu_si256((const __m256i*)&vpt[i]);
      __m256i result = _mm256_sub_epi64(t1, t2);
      _mm256_storeu_si256((__m256i*)&dpt[i], result);
    }
  }
  for (; i < n; i++) {
    dpt[i] -= vpt[i];
  }
}
#endif

/** Generic scalar mul **/
template <class T> inline void NumVect<T>::mul(const NumVect<T> &v, int b, int n, T c)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size());
  for (int i = b; i < n; i++)
    data[i].mul(v[i], c);
}

template <class T> inline void NumVect<T>::mul(const NumVect<T> &v, int n, T c)
{
  mul(v, 0, n, c);
}

template <class T> inline void NumVect<T>::mul(const NumVect<T> &v, T c)
{
  mul(v, 0, size(), c);
}

/** AVX2 mul for FP_NR<double> **/
#if defined(__AVX2__)
template <> inline void NumVect<FP_NR<double>>::mul(const NumVect<FP_NR<double>> &v, int b, int n, FP_NR<double> c)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size());
  double c_raw = c.get_d();
  double* dpt = reinterpret_cast<double*>(&data[0]);
  const double* vpt = reinterpret_cast<const double*>(&v.data[0]);
  int i = b;
  if ((n - b) >= 4) {
    __m256d t_c = _mm256_set1_pd(c_raw);
    for (; i <= n - 4; i += 4) {
      __m256d t_v = _mm256_loadu_pd(vpt + i);
      __m256d res = _mm256_mul_pd(t_v, t_c);
      _mm256_storeu_pd(dpt + i, res);
    }
  }
  for (; i < n; i++) {
    dpt[i] = vpt[i] * c_raw;
  }
}
#endif

/** AVX2 mul for Z_NR<long> **/
#if defined(__AVX2__)
template <> inline void NumVect<Z_NR<long>>::mul(const NumVect<Z_NR<long>> &v, int b, int n, Z_NR<long> c)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size());
  long c_raw = c.get_si();
  long* dpt = reinterpret_cast<long*>(&data[0]);
  const long* vpt = reinterpret_cast<const long*>(&v.data[0]);
  int i = b;
  if ((n - b) >= 4) {
    __m256i t_c = _mm256_set1_epi64x(c_raw);
    for (; i <= n - 4; i += 4) {
      __m256i t_v = _mm256_loadu_si256((const __m256i*)(vpt + i));
      __m256i low_bits = _mm256_mul_epu32(t_v, t_c);
      __m256i v_high   = _mm256_srli_epi64(t_v, 32);
      __m256i c_high   = _mm256_srli_epi64(t_c, 32);
      __m256i high_bits = _mm256_add_epi64(_mm256_mul_epu32(v_high, t_c),
                                           _mm256_mul_epu32(t_v, c_high));
      __m256i res = _mm256_add_epi64(low_bits, _mm256_slli_epi64(high_bits, 32));
      _mm256_storeu_si256((__m256i*)(dpt + i), res);
    }
  }
  for (; i < n; i++) {
    dpt[i] = vpt[i] * c_raw;
  }
}
#endif

/** Generic div **/
template <class T> void NumVect<T>::div(const NumVect<T> &v, int b, int n, T c)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  for (int i = n - 1; i >= b; i--)
    data[i].div(v[i], c);
}

template <class T> void NumVect<T>::div(const NumVect<T> &v, int n, T c) { div(v, 0, n, c); }

/** AVX2 div for FP_NR<double> **/
template <> inline void NumVect<FP_NR<double>>::div(const NumVect<FP_NR<double>> &v, int b, int n, FP_NR<double> c)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  double c_val = c.get_d();
  double inv_c = 1.0 / c_val;
  FP_NR<double> invw(inv_c);
  this->mul(v, b, n, invw);
}

/** Generic addmul (data = data + v*x) **/
template <class T> void NumVect<T>::addmul(const NumVect<T> &v, T x, int beg, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  for (int i = n - 1; i >= beg; i--)
    data[i].addmul(v[i], x);
}

template <class T> void NumVect<T>::addmul(const NumVect<T> &v, T x, int n)
{
  this->addmul(v, x, 0, n);
}

/** AVX2 addmul for FP_NR<double> **/
#if defined(__AVX2__) && defined(__FMA__)
template <> inline void NumVect<FP_NR<double>>::addmul(const NumVect<FP_NR<double>> &v, FP_NR<double> x, int beg, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  int count = n - beg;
  if (count <= 0)
    return;
  double factor = x.get_d();
  double* p_dst = reinterpret_cast<double*>(&data[0]);
  const double* p_src = reinterpret_cast<const double*>(&v.data[0]);
  int i = beg;
  if (count >= 4) {
    __m256d v_factor = _mm256_set1_pd(factor);
    for (; i <= n - 4; i += 4) {
      __m256d vd = _mm256_loadu_pd(p_dst + i);
      __m256d vs = _mm256_loadu_pd(p_src + i);
      _mm256_storeu_pd(p_dst + i, _mm256_fmadd_pd(vs, v_factor, vd));
    }
  }
  for (; i < n; i++) {
    p_dst[i] += p_src[i] * factor;
  }
}
#endif

/** AVX2 addmul for Z_NR<long> **/
#if defined(__AVX2__)
template <> inline void NumVect<Z_NR<long>>::addmul(const NumVect<Z_NR<long>> &v, Z_NR<long> x, int beg, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  int count = n - beg;
  if (count <= 0) return;
  long x_raw = x.get_si();
  long* p_dst = reinterpret_cast<long*>(&data[0]);
  const long* p_src = reinterpret_cast<const long*>(&v.data[0]);
  int i = beg;
  if (count >= 4) {
    __m256i t_x = _mm256_set1_epi64x(x_raw);
    __m256i x_high = _mm256_srli_epi64(t_x, 32);
    for (; i <= n - 4; i += 4) {
      __m256i t_v = _mm256_loadu_si256((const __m256i*)(p_src + i));
      __m256i t_d = _mm256_loadu_si256((const __m256i*)(p_dst + i));
      __m256i v_high = _mm256_srli_epi64(t_v, 32);
      __m256i low_bits = _mm256_mul_epu32(t_v, t_x);
      __m256i mid_bits = _mm256_add_epi64( _mm256_mul_epu32(v_high, t_x),
                                           _mm256_mul_epu32(t_v, x_high) );
      __m256i product = _mm256_add_epi64(low_bits, _mm256_slli_epi64(mid_bits, 32));
      __m256i res = _mm256_add_epi64(t_d, product);
      _mm256_storeu_si256((__m256i*)(p_dst + i), res);
    }
  }
  for (; i < n; i++) {
    p_dst[i] += p_src[i] * x_raw;
  }
}
#endif

#ifdef FPLLL_WITH_QD
/** Specialized addmul for dd_real (data = data + v * x) **/
template <>
inline void NumVect<FP_NR<dd_real>>::addmul(const NumVect<FP_NR<dd_real>> &v, 
                                            FP_NR<dd_real> x, int beg, int n)
{
  int count = n - beg;
  if (count <= 0) return;

  dd_real factor = x.get_data();
  // Use __restrict to tell the compiler these pointers don't alias
  dd_real* __restrict p_dst = reinterpret_cast<dd_real*>(&data[beg]);
  const dd_real* __restrict p_src = reinterpret_cast<const dd_real*>(&v.data[beg]);

  int i = 0;
  // 4-way unrolling often beats 2-way for addmul on modern CPUs
  // because it hides the latency of the dd_real multiplication (which is ~20+ cycles)
  for (; i <= count - 4; i += 4) {
    p_dst[i]   += p_src[i]   * factor;
    p_dst[i+1] += p_src[i+1] * factor;
    p_dst[i+2] += p_src[i+2] * factor;
    p_dst[i+3] += p_src[i+3] * factor;
  }

  for (; i < count; i++) {
    p_dst[i] += p_src[i] * factor;
  }    
}
#endif

template <class T>
void NumVect<T>::addmul_2exp(const NumVect<T> &v, const T &x, long expo, int n, T &tmp)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  for (int i = n - 1; i >= 0; i--)
  {
    tmp.mul(v[i], x);
    tmp.mul_2si(tmp, expo);
    data[i].add(data[i], tmp);
  }
}

/** Generic addmul_si  **/
template <class T> void NumVect<T>::addmul_si(const NumVect<T> &v, long x, int n)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size() && v.is_zero(n));
  for (int i = n - 1; i >= 0; i--)
    data[i].addmul_si(v[i], x);
}

/** AVX2 addmul_si for Z_NR<long>, used in update_gso_row() **/
#if defined(__AVX2__)
template <> inline void NumVect<Z_NR<long>>::addmul_si(const NumVect<Z_NR<long>> &v, long x, int n)
{
  if (n <= 0 || x == 0)
    return;
  long* dpt = reinterpret_cast<long*>(&data[0]);
  const long* vpt = reinterpret_cast<const long*>(&v.data[0]);
  int i = 0;
  if (n >= 4) {
    if (x == 1) { /* easy case */
      for (; i <= n - 4; i += 4) {
        __m256i t_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vpt + i));
        __m256i t_d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dpt + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dpt + i), _mm256_add_epi64(t_d, t_v));
      }
    }
    else if (x == -1) { /* easy case */
      for (; i <= n - 4; i += 4) {
        __m256i t_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vpt + i));
        __m256i t_d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dpt + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dpt + i), _mm256_sub_epi64(t_d, t_v));
      }
    }
    else {
      for (; i <= n - 4; i += 4) {
        dpt[i] += vpt[i] * x;
        dpt[i+1] += vpt[i+1] * x;
        dpt[i+2] += vpt[i+2] * x;
        dpt[i+3] += vpt[i+3] * x;
      }
    }
  }
  for (; i < n; i++) {
    dpt[i] += vpt[i] * x;
  }
}
#endif

/** Generic addmul_si_2exp  **/
template <class T> inline void NumVect<T>::addmul_si_2exp(const NumVect<T> &v, long x, long expo, int n, T &tmp)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size());
  for (int i = 0; i < n; i++)
  {
    tmp.mul_si(v[i], x);
    tmp.mul_2si(tmp, expo);
    data[i].add(data[i], tmp);
  }
}

/** AVX2 + FMA addmul_si_2exp for FP_NR<double> **/
#if defined(__AVX2__) && defined(__FMA__)
template <>
inline void NumVect<FP_NR<double>>::addmul_si_2exp(const NumVect<FP_NR<double>> &v, long x, long expo, int n, FP_NR<double> &tmp)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size());
  if (n <= 0)
    return;
  double multiplier = static_cast<double>(x) * std::ldexp(1.0, expo);
  double* dpt = reinterpret_cast<double*>(&data[0]);
  const double* vpt = reinterpret_cast<const double*>(&v.data[0]);
  int i = 0;
  if (n >= 4) {
    __m256d v_mult = _mm256_set1_pd(multiplier);
    for (; i <= n - 4; i += 4) {
      __m256d vd = _mm256_loadu_pd(dpt + i);
      __m256d vs = _mm256_loadu_pd(vpt + i);
      _mm256_storeu_pd(dpt + i, _mm256_fmadd_pd(vs, v_mult, vd));
    }
  }
  for (; i < n; i++) {
    dpt[i] += vpt[i] * multiplier;
  }
}
#endif

/** AVX2 addmul_si_2exp for Z_NR<long> **/
#if defined(__AVX2__)
template <>
inline void NumVect<Z_NR<long>>::addmul_si_2exp(const NumVect<Z_NR<long>> &v, long x, long expo, int n, Z_NR<long> &tmp)
{
  FPLLL_DEBUG_CHECK(n <= size() && size() == v.size());
  if (n <= 0) return;
  long* dpt = reinterpret_cast<long*>(&data[0]);
  const long* vpt = reinterpret_cast<const long*>(&v.data[0]);
  int i = 0;
  if (n >= 4) {
    if (expo >= 0 && expo < 63) {
      long shifted_x = x << expo;
      __m256i v_x = _mm256_set1_epi64x(shifted_x);
      __m256i x_hi = _mm256_srli_epi64(v_x, 32);
      for (; i <= n - 4; i += 4) {
        __m256i vs = _mm256_loadu_si256((const __m256i*)(vpt + i));
        __m256i vd = _mm256_loadu_si256((__m256i*)(dpt + i));
        __m256i v_hi = _mm256_srli_epi64(vs, 32);
        __m256i low = _mm256_mul_epu32(vs, v_x);
        __m256i mid = _mm256_add_epi64(_mm256_mul_epu32(v_hi, v_x), _mm256_mul_epu32(vs, x_hi));
        __m256i prod = _mm256_add_epi64(low, _mm256_slli_epi64(mid, 32));
        _mm256_storeu_si256((__m256i*)(dpt + i), _mm256_add_epi64(vd, prod));
      }
    }
  }
  for (; i < n; i++) {
    tmp.mul_si(vpt[i], x);
    tmp.mul_2si(tmp, expo);
    dpt[i] += tmp.get_si();
  }
}
#endif

template <class T> long NumVect<T>::get_max_exponent()
{
  long max_expo = 0;
  for (int i = 0; i < size(); i++)
  {
    max_expo = max(max_expo, data[i].exponent());
  }
  return max_expo;
}

template <class T> void NumVect<T>::fill(long value)
{
  for (int i = 0; i < size(); i++)
  {
    data[i] = value;
  }
}

template <class T> bool NumVect<T>::is_zero(int fromCol) const
{
  for (int i = fromCol; i < size(); i++)
  {
    if (!data[i].is_zero())
      return false;
  }
  return true;
}

template <class T> int NumVect<T>::size_nz() const
{
  int i;
  for (i = data.size(); i > 0; i--)
  {
    if (data[i - 1] != 0)
      break;
  }
  return i;
}

/** Compute the truncated dot product between tow Numvect using coefficients [beg, n).
 * Constraint: n > beg.
 */
template <class T>
inline void dot_product(T &result, const NumVect<T> &v1, const NumVect<T> &v2, int beg, int n)
{
  FPLLL_DEBUG_CHECK(beg >= 0 && n > beg && n <= v1.size() && n <= v2.size());
  //(v1.is_zero(n) || v2.is_zero(n))); tested previously
  result.mul(v1[beg], v2[beg]);
  for (int i = beg + 1; i < n; i++)
  {
    result.addmul(v1[i], v2[i]);
  }
}

template <class T>
inline void dot_product(T &result, const NumVect<T> &v1, const NumVect<T> &v2, int n)
{
  FPLLL_DEBUG_CHECK(n <= v1.size() && v1.size() == v2.size() && (v1.is_zero(n) || v2.is_zero(n)));
  dot_product(result, v1, v2, 0, n);
}

template <class T> inline void dot_product(T &result, const NumVect<T> &v1, const NumVect<T> &v2)
{
  dot_product(result, v1, v2, v1.size());
}

/** AVX2 dot_product for FP_NR<double> **/
#if defined(__AVX2__) && defined(__FMA__)
template <>
inline void dot_product<FP_NR<double>>(FP_NR<double> &result,
                                       const NumVect<FP_NR<double>> &v1,
                                       const NumVect<FP_NR<double>> &v2,
                                       int beg, int n)
{
  int count = n - beg;
  if (count <= 0)
  {
    result = 0.0;
    return;
  }
  const double* __restrict p1 = reinterpret_cast<const double*>(&v1[beg]);
  const double* __restrict p2 = reinterpret_cast<const double*>(&v2[beg]);
  double res_scalar = 0.0;
  int i = 0;
  __m256d acc = _mm256_setzero_pd();
  if (count >= 8) {
    __m256d acc2 = _mm256_setzero_pd();
    for (; i <= count - 8; i += 8) {
      acc  = _mm256_fmadd_pd(_mm256_loadu_pd(p1 + i),     _mm256_loadu_pd(p2 + i),     acc);
      acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(p1 + i + 4), _mm256_loadu_pd(p2 + i + 4), acc2);
    }
    acc = _mm256_add_pd(acc, acc2);
  }
  for (; i <= count - 4; i += 4) {
    acc = _mm256_fmadd_pd(_mm256_loadu_pd(p1 + i), _mm256_loadu_pd(p2 + i), acc);
  }
  if (i > 0) {
    __m128d low = _mm256_castpd256_pd128(acc);
    __m128d high = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(low, high);
    sum = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));
    res_scalar = _mm_cvtsd_f64(sum);
  }
  for (; i < count; i++) {
    res_scalar += p1[i] * p2[i];
  }
  result = res_scalar;
}

template <>
inline void dot_product<fplll::FP_NR<double>>(fplll::FP_NR<double> &result,
                                              const fplll::NumVect<fplll::FP_NR<double>> &v1,
                                              const fplll::NumVect<fplll::FP_NR<double>> &v2,
                                              int n)
{
  dot_product<fplll::FP_NR<double>>(result, v1, v2, 0, n);
}

template <>
inline void dot_product<fplll::FP_NR<double>>(fplll::FP_NR<double> &result,
                                              const fplll::NumVect<fplll::FP_NR<double>> &v1,
                                              const fplll::NumVect<fplll::FP_NR<double>> &v2)
{
  dot_product<fplll::FP_NR<double>>(result, v1, v2, 0, v1.size());
}
#endif



#ifdef FPLLL_WITH_QD

// --- Core Implementation (AVX2 or Scalar) ---
template <>
inline void dot_product<FP_NR<dd_real>>(FP_NR<dd_real> &result,
                                        const NumVect<FP_NR<dd_real>> &v1,
                                        const NumVect<FP_NR<dd_real>> &v2,
                                        int beg, int n)
{
  const int count = n - beg;
  if (count <= 0) { result = 0.0; return; }

  const dd_real* __restrict__ p1 = reinterpret_cast<const dd_real*>(&v1[beg]);
  const dd_real* __restrict__ p2 = reinterpret_cast<const dd_real*>(&v2[beg]);
  int i = 0;
  dd_real s_total = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
  // --- AVX2 PATH ---
  __m256d acc_h = _mm256_setzero_pd();
  __m256d acc_l = _mm256_setzero_pd();

  for (; i <= count - 4; i += 4) {
    __m256d a1 = _mm256_loadu_pd((double*)&p1[i]);
    __m256d a2 = _mm256_loadu_pd((double*)&p1[i + 2]);
    __m256d v_ah = _mm256_unpacklo_pd(a1, a2); 
    __m256d v_al = _mm256_unpackhi_pd(a1, a2);

    __m256d b1 = _mm256_loadu_pd((double*)&p2[i]);
    __m256d b2 = _mm256_loadu_pd((double*)&p2[i + 2]);
    __m256d v_bh = _mm256_unpacklo_pd(b1, b2);
    __m256d v_bl = _mm256_unpackhi_pd(b1, b2);

    // two_prod
    __m256d p1_v = _mm256_mul_pd(v_ah, v_bh);
    __m256d p2_v = _mm256_fmsub_pd(v_ah, v_bh, p1_v); 
    // cross terms
    p2_v = _mm256_fmadd_pd(v_ah, v_bl, _mm256_fmadd_pd(v_al, v_bh, p2_v));

    // acc += p1_v + p2_v
    __m256d s = _mm256_add_pd(acc_h, p1_v);
    __m256d tmp = _mm256_sub_pd(s, acc_h);
    __m256d e = _mm256_add_pd(_mm256_sub_pd(acc_h, _mm256_sub_pd(s, tmp)), _mm256_sub_pd(p1_v, tmp));
    e = _mm256_add_pd(e, _mm256_add_pd(acc_l, p2_v));
        
    acc_h = _mm256_add_pd(s, e);
    acc_l = _mm256_sub_pd(e, _mm256_sub_pd(acc_h, s));
  }

  double res_h[4], res_l[4];
  _mm256_storeu_pd(res_h, acc_h);
  _mm256_storeu_pd(res_l, acc_l);
  for (int j = 0; j < 4; ++j) s_total += dd_real(res_h[j], res_l[j]);

#else
  // --- SCALAR FALLBACK (Optimized with 4 accumulators) ---
  dd_real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  for (; i <= count - 4; i += 4) {
    s0 += p1[i]   * p2[i];
    s1 += p1[i+1] * p2[i+1];
    s2 += p1[i+2] * p2[i+2];
    s3 += p1[i+3] * p2[i+3];
  }
  s_total = (s0 + s1) + (s2 + s3);
#endif

  // --- REMAINDER LOOP (Shared) ---
  for (; i < count; ++i) {
    s_total += p1[i] * p2[i];
  }
  result.get_data() = s_total;
}

// --- Wrappers ---
template <>
inline void dot_product<FP_NR<dd_real>>(FP_NR<dd_real> &result,
                                        const NumVect<FP_NR<dd_real>> &v1,
                                        const NumVect<FP_NR<dd_real>> &v2,
                                        int n)
{
  dot_product<FP_NR<dd_real>>(result, v1, v2, 0, n);
}

template <>
inline void dot_product<FP_NR<dd_real>>(FP_NR<dd_real> &result,
                                        const NumVect<FP_NR<dd_real>> &v1,
                                        const NumVect<FP_NR<dd_real>> &v2)
{
  dot_product<FP_NR<dd_real>>(result, v1, v2, 0, v1.size());
}

#endif

template <class T> inline void squared_norm(T &result, const NumVect<T> &v)
{
  dot_product(result, v, v);
}

/** Prints a NumVect on stream os. */
template <class T> ostream &operator<<(ostream &os, const NumVect<T> &v) { return os << v.data; }

/** Reads a NumVect from stream is. */
template <class T> istream &operator>>(istream &is, NumVect<T> &v) { return is >> v.data; }

FPLLL_END_NAMESPACE

#endif
