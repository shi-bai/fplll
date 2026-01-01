// test_dd.cpp
// Standalone benchmark: double vs QD dd_real for scalar add/mul/fma-like and dot-product.
//
// Build:
//    g++ -O3 -march=native -std=c++17 -fno-omit-frame-pointer -DNDEBUG test_dd.cpp -lqd -o test_dd

/**
sizeof(double)  = 8
sizeof(dd_real) = 16 (expect 16)

Scalar ops (Dependency chain on sink prevents full ILP)
  double add                          3.00 ns/iter
  dd_real add                         6.18 ns/iter
  dd_real (a*b+c)                     6.13 ns/iter

Dot product sweep (N sweep)

N=100 (reps=10000000)
  dot double (unroll, but no avx2)       19.87 ns/iter
  dot dd_real (serial)              296.09 ns/iter
  dot dd_real (unrolled)            151.88 ns/iter
  Ratio Serial/Double:     14.90x
  Ratio Unrolled/Double:   7.65x

N=200 (reps=2500000)
  dot double (unroll, but no avx2)       38.42 ns/iter
  dot dd_real (serial)              601.58 ns/iter
  dot dd_real (unrolled)            298.03 ns/iter
  Ratio Serial/Double:     15.66x
  Ratio Unrolled/Double:   7.76x

N=400 (reps=800000)
  dot double (unroll, but no avx2)       76.04 ns/iter
  dot dd_real (serial)             1210.75 ns/iter
  dot dd_real (unrolled)            591.15 ns/iter
  Ratio Serial/Double:     15.92x
  Ratio Unrolled/Double:   7.77x

Precision check (1.0 + 1e-30 - 1.0): 1.00e-30
 */


#include <qd/dd_real.h>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using clk = std::chrono::high_resolution_clock;

// -------------------- robust anti-optimizer helpers --------------------
#if defined(__GNUC__) || defined(__clang__)
template <class T>
__attribute__((always_inline)) inline void DoNotOptimize(const T& value) {
  asm volatile("" : : "g"(value) : "memory");
}
__attribute__((always_inline)) inline void DoNotOptimizeMem(const void* p) {
  asm volatile("" : : "r"(p) : "memory");
}
__attribute__((always_inline)) inline void ClobberMemory() {
  asm volatile("" : : : "memory");
}
#else
template <class T> inline void DoNotOptimize(const T&) {}
inline void DoNotOptimizeMem(const void*) {}
inline void ClobberMemory() {}
#endif

static double sink_d = 0.0;
static dd_real sink_dd = 0.0;

// -------------------- timing utility --------------------
template <class F>
double time_ns_per_iter(const char* name, int R, F&& fn) {
  for (int i = 0; i < 10000; ++i) fn(i); // Warm up
  ClobberMemory();

  auto t0 = clk::now();
  for (int i = 0; i < R; ++i) fn(i);
  auto t1 = clk::now();

  double sec = std::chrono::duration<double>(t1 - t0).count();
  double ns = (sec * 1e9) / R;

  std::cout << "  " << std::left << std::setw(28) << name
            << std::right << std::setw(12) << std::fixed << std::setprecision(2)
            << ns << " ns/iter\n";
  return ns;
}

// -------------------- dot implementations --------------------

// Optimized double dot product with __restrict__ and 4-way ILP
// This allows the compiler to use FMA and AVX instructions efficiently.
static inline double dot_double(const double* __restrict__ a, const double* __restrict__ b, int n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  int i = 0;
  for (; i <= n - 4; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i+1] * b[i+1];
    s2 += a[i+2] * b[i+2];
    s3 += a[i+3] * b[i+3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

// Standard serial implementation (bottlenecked by dependency chain)
static inline dd_real dot_dd(const dd_real* __restrict__ a, const dd_real* __restrict__ b, int n) {
  dd_real s = 0.0;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// Optimized implementation using 4 accumulators to hide latency (ILP)
static inline dd_real dot_dd_optimized(const dd_real* __restrict__ a, const dd_real* __restrict__ b, int n) {
  dd_real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  int i = 0;
  for (; i <= n - 4; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i+1] * b[i+1];
    s2 += a[i+2] * b[i+2];
    s3 += a[i+3] * b[i+3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

int main() {
  std::cout << "sizeof(double)  = " << sizeof(double) << "\n";
  std::cout << "sizeof(dd_real) = " << sizeof(dd_real) << " (expect 16)\n\n";

  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  constexpr int T = 1024; 
  std::vector<double> td(T), te(T), tf(T);
  std::vector<dd_real> tdd(T), tde(T), tdf(T);

  for (int i = 0; i < T; ++i) {
    double a = dist(rng), b = dist(rng), c = dist(rng);
    td[i] = a; te[i] = b; tf[i] = c;
    double alo = ((i & 1) ? 1.0 : -1.0) * 1e-30;
    double blo = ((i & 2) ? 1.0 : -1.0) * 1e-30;
    double clo = ((i & 4) ? 1.0 : -1.0) * 1e-30;
    tdd[i] = dd_real(a, alo);
    tde[i] = dd_real(b, blo);
    tdf[i] = dd_real(c, clo);
  }

  const int R = 20'000'000;
  std::cout << "Scalar ops (Dependency chain on sink prevents full ILP)\n";

  time_ns_per_iter("double add", R, [&](int i) {
    int idx = i & (T - 1);
    sink_d += td[idx] + te[idx];
    DoNotOptimize(sink_d);
  });

  time_ns_per_iter("dd_real add", R, [&](int i) {
    int idx = i & (T - 1);
    sink_dd += tdd[idx] + tde[idx];
    DoNotOptimizeMem(&sink_dd);
  });

  time_ns_per_iter("dd_real (a*b+c)", R, [&](int i) {
    int idx = i & (T - 1);
    sink_dd += tdd[idx] * tde[idx] + tdf[idx];
    DoNotOptimizeMem(&sink_dd);
  });

  std::cout << "\nDot product sweep (N sweep)\n";
  for (int N : {100, 200, 400}) {
    int Rd = (N == 100) ? 10000000 : (N == 200) ? 2500000 : 800000;
    std::vector<double> a(N), b(N);
    std::vector<dd_real> add_vec(N), bdd_vec(N);

    for (int i = 0; i < N; ++i) {
      a[i] = dist(rng); b[i] = dist(rng);
      add_vec[i] = dd_real(a[i], 1e-30);
      bdd_vec[i] = dd_real(b[i], 1e-30);
    }

    std::cout << "\nN=" << N << " (reps=" << Rd << ")\n";
    double ns_dot_d = time_ns_per_iter("dot double (unroll, but no avx2)", Rd, [&](int i) {
      double s = dot_double(a.data(), b.data(), N);
      sink_d += s;
      DoNotOptimize(sink_d);
    });

    double ns_dot_dd = time_ns_per_iter("dot dd_real (serial)", Rd, [&](int i) {
      dd_real s = dot_dd(add_vec.data(), bdd_vec.data(), N);
      sink_dd += s;
      DoNotOptimizeMem(&sink_dd);
    });

    double ns_dot_dd_opt = time_ns_per_iter("dot dd_real (unrolled)", Rd, [&](int i) {
      dd_real s = dot_dd_optimized(add_vec.data(), bdd_vec.data(), N);
      sink_dd += s;
      DoNotOptimizeMem(&sink_dd);
    });

    std::cout << "  Ratio Serial/Double:     " << (ns_dot_dd / ns_dot_d) << "x\n";
    std::cout << "  Ratio Unrolled/Double:   " << (ns_dot_dd_opt / ns_dot_d) << "x\n";
  }

  // Final verification
  dd_real v_a("1.0");
  dd_real v_b("1e-30");
  dd_real v_c = (v_a + v_b) - v_a;
  std::cout << "\nPrecision check (1.0 + 1e-30 - 1.0): " << std::scientific << v_c << std::endl;

  return 0;
}
