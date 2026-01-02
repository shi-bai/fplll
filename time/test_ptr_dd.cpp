// compare dd/double array type dot product
//
// Build:
//   g++ -O3 -march=native -DNDEBUG test_ptr_dd.cpp -lqd -o test_ptr_dd
/**
Scalar ops (Dependency chain via DoNotOptimize)
  double add                            2.85 ns/iter
  dd_real add                           5.94 ns/iter

Dot Product (N=400)
  double [unrolled]                    83.49 ns/iter
  double [avx2]                        50.98 ns/iter
  dd_real [serial]                   1143.67 ns/iter
  dd_real [unrolled]                  555.46 ns/iter
*/
   
#include <qd/dd_real.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <immintrin.h>

using clk = std::chrono::high_resolution_clock;

// -------------------- anti-optimizer helpers --------------------
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

// -------------------- dot implementations --------------------

static inline double dot_double_unrolled(const double* __restrict__ a, const double* __restrict__ b, int n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        s0 += a[i] * b[i]; s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2]; s3 += a[i+3] * b[i+3];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

static inline double dot_double_avx2(const double* __restrict__ a, const double* __restrict__ b, int n) {
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        acc = _mm256_fmadd_pd(va, vb, acc);
    }
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d sum = _mm_add_pd(hi, lo);
    __m128d final_v = _mm_hadd_pd(sum, sum);
    double s = _mm_cvtsd_f64(final_v);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

static inline dd_real dot_dd_serial(const dd_real* __restrict__ a, const dd_real* __restrict__ b, int n) {
    dd_real s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static inline dd_real dot_dd_unrolled(const dd_real* __restrict__ a, const dd_real* __restrict__ b, int n) {
    dd_real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        s0 += a[i] * b[i]; s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2]; s3 += a[i+3] * b[i+3];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

// -------------------- timing utility --------------------
template <class F>
double time_ns_per_iter(const char* name, int R, F&& fn) {
    for (int i = 0; i < 10000; ++i) fn(i);
    ClobberMemory();
    auto t0 = clk::now();
    for (int i = 0; i < R; ++i) fn(i);
    auto t1 = clk::now();
    double ns = (std::chrono::duration<double>(t1 - t0).count() * 1e9) / R;
    std::cout << "  " << std::left << std::setw(30) << name
              << std::right << std::setw(12) << std::fixed << std::setprecision(2)
              << ns << " ns/iter\n";
    return ns;
}

int main() {
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int T = 1024;
    std::vector<double> td(T), te(T), tf(T);
    std::vector<dd_real> tdd(T), tde(T), tdf(T);
    for (int i = 0; i < T; ++i) {
        td[i] = dist(rng); te[i] = dist(rng); tf[i] = dist(rng);
        tdd[i] = dd_real(td[i], 1e-30); tde[i] = dd_real(te[i], 1e-30); tdf[i] = dd_real(tf[i], 1e-30);
    }

    std::cout << "Scalar ops (Dependency chain via DoNotOptimize)\n";
    const int R_scalar = 10'000'000;
    time_ns_per_iter("double add", R_scalar, [&](int i) {
        sink_d += td[i & (T-1)] + te[i & (T-1)];
        DoNotOptimize(sink_d);
    });
    time_ns_per_iter("dd_real add", R_scalar, [&](int i) {
        sink_dd += tdd[i & (T-1)] + tde[i & (T-1)];
        DoNotOptimizeMem(&sink_dd);
    });

    for (int N : {400}) {
        int Rd = 1000000;
        std::vector<double> a(N), b(N);
        std::vector<dd_real> av(N), bv(N);
        for (int i = 0; i < N; ++i) {
            a[i] = dist(rng); b[i] = dist(rng);
            av[i] = dd_real(a[i], 1e-30); bv[i] = dd_real(b[i], 1e-30);
        }
        std::cout << "\nDot Product (N=" << N << ")\n";
        time_ns_per_iter("double [unrolled]", Rd, [&](int i) {
            double s = dot_double_unrolled(a.data(), b.data(), N);
            sink_d += s; DoNotOptimize(sink_d);
        });
        time_ns_per_iter("double [avx2]", Rd, [&](int i) {
            double s = dot_double_avx2(a.data(), b.data(), N);
            sink_d += s; DoNotOptimize(sink_d);
        });
        time_ns_per_iter("dd_real [serial]", Rd, [&](int i) {
            dd_real s = dot_dd_serial(av.data(), bv.data(), N);
            sink_dd += s; DoNotOptimizeMem(&sink_dd);
        });
        time_ns_per_iter("dd_real [unrolled]", Rd, [&](int i) {
            dd_real s = dot_dd_unrolled(av.data(), bv.data(), N);
            sink_dd += s; DoNotOptimizeMem(&sink_dd);
        });
    }
    return 0;
}
