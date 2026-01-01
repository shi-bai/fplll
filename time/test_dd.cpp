// test_dd.cpp
// Standalone benchmark: double vs QD dd_real for scalar add/mul/fma-like and dot-product.
// Includes manual loop unrolling for dd_real to provide a fairer comparison.
//
// Build (Strict & Correct):
//    g++ -O3 -march=native -std=c++17 -fno-omit-frame-pointer -DNDEBUG test_dd.cpp -lqd -o bench
//
// Run:
//    export OMP_NUM_THREADS=1
//    taskset -c 2 ./bench

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
static inline double dot_double(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// Standard serial implementation (bottlenecked by dependency chain)
static inline dd_real dot_dd(const dd_real* a, const dd_real* b, int n) {
    dd_real s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// Optimized implementation using 4 accumulators to hide latency (ILP)
static inline dd_real dot_dd_optimized(const dd_real* a, const dd_real* b, int n) {
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
    std::cout << "Scalar ops (Dependency chain on sink_dd prevents full ILP here)\n";

    double ns_add_d = time_ns_per_iter("double add", R, [&](int i) {
        int idx = i & (T - 1);
        double x = td[idx] + te[idx];
        sink_d += x;
        DoNotOptimize(sink_d);
    });

    double ns_add_dd = time_ns_per_iter("dd_real add", R, [&](int i) {
        int idx = i & (T - 1);
        dd_real x = tdd[idx] + tde[idx];
        sink_dd += x;
        DoNotOptimizeMem(&sink_dd);
    });

    double ns_fma_dd = time_ns_per_iter("dd_real (a*b+c)", R, [&](int i) {
        int idx = i & (T - 1);
        dd_real x = tdd[idx] * tde[idx] + tdf[idx];
        sink_dd += x;
        DoNotOptimizeMem(&sink_dd);
    });

    std::cout << "\nDot product sweep (N sweep)\n";
    for (int N : {512, 2048, 8192}) {
        int Rd = (N == 512) ? 80000 : (N == 2048) ? 20000 : 5000;
        std::vector<double> a(N), b(N);
        std::vector<dd_real> add(N), bdd(N);

        for (int i = 0; i < N; ++i) {
            a[i] = dist(rng); b[i] = dist(rng);
            add[i] = dd_real(a[i], 1e-30);
            bdd[i] = dd_real(b[i], 1e-30);
        }

        std::cout << "\nN=" << N << " (reps=" << Rd << ")\n";
        double ns_dot_d = time_ns_per_iter("dot double", Rd, [&](int i) {
            double s = dot_double(a.data(), b.data(), N);
            sink_d += s;
            DoNotOptimize(sink_d);
        });

        double ns_dot_dd = time_ns_per_iter("dot dd_real (serial)", Rd, [&](int i) {
            dd_real s = dot_dd(add.data(), bdd.data(), N);
            sink_dd += s;
            DoNotOptimizeMem(&sink_dd);
        });

        double ns_dot_dd_opt = time_ns_per_iter("dot dd_real (unrolled)", Rd, [&](int i) {
            dd_real s = dot_dd_optimized(add.data(), bdd.data(), N);
            sink_dd += s;
            DoNotOptimizeMem(&sink_dd);
        });

        std::cout << "  Ratio Serial/Double:   " << (ns_dot_dd / ns_dot_d) << "x\n";
        std::cout << "  Ratio Unrolled/Double: " << (ns_dot_dd_opt / ns_dot_d) << "x\n";
    }

    DoNotOptimize(sink_d);
    DoNotOptimizeMem(&sink_dd);


// Final verification: 1.0 + 1e-30 - 1.0 should NOT be 0.
dd_real a("1.0");
dd_real b("1e-30");
dd_real c = (a + b) - a;
std::cout << "Precision check: " << c << std::endl;    
    return 0;
}
