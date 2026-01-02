// compare different vector types -- conclusion: no slow down Numvect wrapper
// Build with: g++ -O3 -march=native test_vector.cpp -lqd -lfplll -o test_vector
/**
Optimized Benchmarking (AVX2/FMA allowed)
sizeof(dd_real) = 16

N=100 (reps=2000000)
  double (Ptr + restrict)              24.17 ns/iter
  fplll double (avx2)                   5.98 ns/iter
  fplll dd_real (unroll)              150.51 ns/iter
  dd_real (Ptr Serial)                295.84 ns/iter
  dd_real (Unrolled ILP x4)           150.29 ns/iter
-------------------------------------------
N=200 (reps=500000)
  double (Ptr + restrict)              54.71 ns/iter
  fplll double (avx2)                  11.72 ns/iter
  fplll dd_real (unroll)              295.36 ns/iter
  dd_real (Ptr Serial)                600.36 ns/iter
  dd_real (Unrolled ILP x4)           295.26 ns/iter
-------------------------------------------
N=400 (reps=200000)
  double (Ptr + restrict)             129.13 ns/iter
  fplll double (avx2)                  23.21 ns/iter
  fplll dd_real (unroll)              585.89 ns/iter
  dd_real (Ptr Serial)               1210.41 ns/iter
  dd_real (Unrolled ILP x4)           586.26 ns/iter
-------------------------------------------
   
*/
   

#include "../fplll/fplll.h" 
#include <qd/dd_real.h>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using clk = std::chrono::high_resolution_clock;

// -------------------- Optimization Helpers --------------------
// We keep these only for the final result of each benchmark to prevent
// the compiler from deleting the entire loop, but we remove them from
// the inner logic to allow full pipeline utilization.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline)) inline void FinalSink(const void* p) {
  asm volatile("" : : "r"(p) : "memory");
}
#else
inline void FinalSink(const void*) {}
#endif

static double sink_d = 0.0;
static dd_real sink_dd = 0.0;

// -------------------- Double Implementations --------------------

// Standard double dot product - Added __restrict__ to allow auto-vectorization
static inline double dot_double_ptr(const double* __restrict__ a, const double* __restrict__ b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline double dot_fplll_double(const fplll::NumVect<fplll::FP_NR<double>>& a,
                               const fplll::NumVect<fplll::FP_NR<double>>& b,
                               int n) 
{
  fplll::FP_NR<double> r;
  fplll::dot_product<fplll::FP_NR<double>>(r, a, b, 0, n);    
  return r.get_d(); 
}

// -------------------- dd_real Implementations --------------------

static inline dd_real dot_dd_ptr(const dd_real* __restrict__ a, const dd_real* __restrict__ b, int n) {
  dd_real s = 0.0;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

static inline dd_real dot_dd_vector_optimized(const std::vector<dd_real>& a, const std::vector<dd_real>& b) {
  dd_real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  size_t n = a.size();
  const dd_real* __restrict__ pa = a.data();
  const dd_real* __restrict__ pb = b.data();
  size_t i = 0;
  // 4-way ILP to hide dd_real addition latency
  for (; i <= n - 4; i += 4) {
    s0 += pa[i] * pb[i];
    s1 += pa[i+1] * pb[i+1];
    s2 += pa[i+2] * pb[i+2];
    s3 += pa[i+3] * pb[i+3];
  }
  for (; i < n; ++i) s0 += pa[i] * pb[i];
  return (s0 + s1) + (s2 + s3);
}

inline dd_real dot_fplll_dd(const fplll::NumVect<fplll::FP_NR<dd_real>>& a,
                            const fplll::NumVect<fplll::FP_NR<dd_real>>& b,
                            int n) 
{
  fplll::FP_NR<dd_real> r;
  fplll::dot_product<fplll::FP_NR<dd_real>>(r, a, b, 0, n);    
  return r.get_data(); 
}

// -------------------- Timing Utility --------------------
template <class F>
double time_ns_per_iter(const char* name, int R, F&& fn) {
  for (int i = 0; i < 10000; ++i) fn(); // Warm up
    
  auto t0 = clk::now();
  for (int i = 0; i < R; ++i) fn();
  auto t1 = clk::now();

  double ns = (std::chrono::duration<double>(t1 - t0).count() * 1e9) / R;
  std::cout << "  " << std::left << std::setw(30) << name
            << std::right << std::setw(12) << std::fixed << std::setprecision(2)
            << ns << " ns/iter\n";
  return ns;
}

int main() {
  std::cout << "Optimized Benchmarking (AVX2/FMA allowed)\n";
  std::cout << "sizeof(dd_real) = " << sizeof(dd_real) << "\n\n";

  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int N : {100, 200, 400}) {
    int Rd = (N == 100) ? 2000000 : (N == 200) ? 500000 : 200000;
        
    std::vector<dd_real> va_dd(N), vb_dd(N);
    fplll::NumVect<fplll::FP_NR<dd_real>> fva_dd(N), fvb_dd(N);
    std::vector<double> va_d(N), vb_d(N);
    fplll::NumVect<fplll::FP_NR<double>> fva_d(N), fvb_d(N);

    for (int i = 0; i < N; ++i) {
      double d1 = dist(rng), d2 = dist(rng);
      va_d[i] = d1; vb_d[i] = d2;
      fva_d[i] = d1; fvb_d[i] = d2;
      dd_real dd1(d1, 1e-30), dd2(d2, 1e-30);
      va_dd[i] = dd1; vb_dd[i] = dd2;
      fva_dd[i].get_data() = dd1; fvb_dd[i].get_data() = dd2;
    }

    std::cout << "N=" << N << " (reps=" << Rd << ")\n";

    // Double Benchmarks
    time_ns_per_iter("double (Ptr + restrict)", Rd, [&]() {
      sink_d += dot_double_ptr(va_d.data(), vb_d.data(), N);
      FinalSink(&sink_d);
    });

    time_ns_per_iter("fplll double (avx2)", Rd, [&]() {
      sink_d += dot_fplll_double(fva_d, fvb_d, N);
      FinalSink(&sink_d);
    });

    // dd_real Benchmarks
    time_ns_per_iter("fplll dd_real (unroll)", Rd, [&]() {
      sink_dd += dot_fplll_dd(fva_dd, fvb_dd, N);
      FinalSink(&sink_dd);
    });

    time_ns_per_iter("dd_real (Ptr Serial)", Rd, [&]() {
      sink_dd += dot_dd_ptr(va_dd.data(), vb_dd.data(), N);
      FinalSink(&sink_dd);
    });

    time_ns_per_iter("dd_real (Unrolled ILP x4)", Rd, [&]() {
      sink_dd += dot_dd_vector_optimized(va_dd, vb_dd);
      FinalSink(&sink_dd);
    });
    std::cout << "-------------------------------------------\n";
  }
  return 0;
}
