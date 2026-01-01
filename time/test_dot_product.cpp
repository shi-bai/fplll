// Build (GCC / g++):
// g++ -O3 -march=native -ffast-math test_dot_product.cpp ../fplll/.libs/libfplll.a -lopenblas -lquadmath -lqd -o bench && ./bench
//
// Notes:
// - __float128 requires <quadmath.h> and linking with -lquadmath.
// - OpenBLAS has no long double / __float128 / dd_real dot; so those are scalar only.
// - dd_real is from the QD library (double-double). We explicitly add tiny "low parts"
//   to force true dd precision to matter, and we use DoNotOptimize/ClobberMemory so the
//   compiler cannot eliminate/rewrite the benchmarked work.

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <limits>
#include <iomanip>
#include <sstream>

#include <qd/dd_real.h>
#include <fplll/fplll.h>

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
#endif

#include <cblas.h>
#include <quadmath.h>

using namespace fplll;
using clk = std::chrono::high_resolution_clock;

// -------------------- "do-not-eliminate" helpers --------------------
#if defined(__GNUC__) || defined(__clang__)
template <class T>
__attribute__((always_inline)) inline void DoNotOptimize(const T& value) {
    asm volatile("" : : "g"(value) : "memory");
}
__attribute__((always_inline)) inline void ClobberMemory() {
    asm volatile("" : : : "memory");
}
#else
template <class T> inline void DoNotOptimize(const T&) {}
inline void ClobberMemory() {}
#endif

inline void DoNotOptimizeDD(const dd_real& x) {
    asm volatile("" : : "m"(x) : "memory"); // force memory reference to full object
}

// Keep sinks non-volatile and use DoNotOptimize barriers instead.
// (volatile works for builtin FP, but breaks for dd_real methods.)
static double sink = 0.0;
static long double sink_ld = 0.0L;
static __float128 sink_q = 0;
static dd_real sink_dd = 0.0;

// ============================================================
// double dot products
// ============================================================

inline double dot_plain_scalar(const double* a, const double* b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline, hot, target("avx2,fma"), optimize("O3")))
#endif
inline double dot_plain_avx2(const double* a, const double* b, int n)
{
#if !(defined(__AVX2__) && defined(__FMA__))
    return dot_plain_scalar(a, b, n);
#else
    int i = 0;
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    for (; i <= n - 8; i += 8) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i),
                               _mm256_loadu_pd(b + i),
                               acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4),
                               _mm256_loadu_pd(b + i + 4),
                               acc1);
    }
    __m256d acc = _mm256_add_pd(acc0, acc1);

    // horizontal sum
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));
    double res = _mm_cvtsd_f64(sum);

    for (; i < n; ++i) res += a[i] * b[i];
    return res;
#endif
}

inline double dot_openblas(const double* a, const double* b, int n)
{
    return cblas_ddot(n, a, 1, b, 1);
}

// fplll FP_NR<double>
inline double dot_fplll_double(const NumVect<FP_NR<double>>& a,
                               const NumVect<FP_NR<double>>& b,
                               int n)
{
    FP_NR<double> r;
    dot_product(r, a, b, 0, n);
    return r.get_d();
}

// ============================================================
// long double (plain, plus ILP4 version)
// ============================================================

inline long double dot_plain_scalar_ld(const long double* a, const long double* b, int n)
{
    long double s = 0.0L;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline long double dot_plain_scalar_ld_ilp4(const long double* a, const long double* b, int n)
{
    long double s0 = 0.0L, s1 = 0.0L, s2 = 0.0L, s3 = 0.0L;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

// ============================================================
// dd_real (plain, plus ILP4 version)
// ============================================================

inline dd_real dot_plain_scalar_dd(const dd_real* a, const dd_real* b, int n)
{
    dd_real s = 0.0;
    for (int i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

inline dd_real dot_plain_scalar_dd_ilp4(const dd_real* a, const dd_real* b, int n)
{
    dd_real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        s0 += a[i+0] * b[i+0];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    for (; i < n; ++i)
        s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

// ============================================================
// __float128 (plain, plus ILP4 version)
// ============================================================

inline __float128 dot_plain_scalar_q(const __float128* a, const __float128* b, int n)
{
    __float128 s = 0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline __float128 dot_plain_scalar_q_ilp4(const __float128* a, const __float128* b, int n)
{
    __float128 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

// ============================================================
// benchmarking helpers (with DoNotOptimize barriers)
// ============================================================

template <class F>
double bench_double(const char* name, F&& fn, int R)
{
    for (int i = 0; i < 1000; ++i) {
        double x = fn();
        DoNotOptimize(x);
        sink += x;
        DoNotOptimize(sink);
    }
    ClobberMemory();

    auto t0 = clk::now();
    double acc = 0.0;

    for (int r = 0; r < R; ++r) {
        double x = fn();
        DoNotOptimize(x);
        acc += x;
    }

    sink += acc;
    DoNotOptimize(sink);
    ClobberMemory();

    auto t1 = clk::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ns = (sec * 1e9) / R;
    std::cout << "  " << name << " : " << ns << " ns/call\n";
    return ns;
}

template <class F>
double bench_long_double(const char* name, F&& fn, int R)
{
    for (int i = 0; i < 1000; ++i) {
        long double x = fn();
        DoNotOptimize(x);
        sink_ld += x;
        DoNotOptimize(sink_ld);
    }
    ClobberMemory();

    auto t0 = clk::now();
    long double acc = 0.0L;

    for (int r = 0; r < R; ++r) {
        long double x = fn();
        DoNotOptimize(x);
        acc += x;
    }

    sink_ld += acc;
    DoNotOptimize(sink_ld);
    ClobberMemory();

    auto t1 = clk::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ns = (sec * 1e9) / R;
    std::cout << "  " << name << " : " << ns << " ns/call\n";
    return ns;
}

template <class F>
double bench_float128(const char* name, F&& fn, int R)
{
    for (int i = 0; i < 200; ++i) {
        __float128 x = fn();
        DoNotOptimize(x);
        sink_q += x;
        DoNotOptimize(sink_q);
    }
    ClobberMemory();

    auto t0 = clk::now();
    __float128 acc = 0;

    for (int r = 0; r < R; ++r) {
        __float128 x = fn();
        DoNotOptimize(x);
        acc += x;
    }

    sink_q += acc;
    DoNotOptimize(sink_q);
    ClobberMemory();

    auto t1 = clk::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ns = (sec * 1e9) / R;
    std::cout << "  " << name << " : " << ns << " ns/call\n";
    return ns;
}

template <class F>
double bench_dd_real(const char* name, F&& fn, int R)
{
    for (int i = 0; i < 200; ++i) {
        dd_real x = fn();
        DoNotOptimize(x);
        sink_dd += x;
        DoNotOptimize(sink_dd);
    }
    ClobberMemory();

    auto t0 = clk::now();
    dd_real acc = 0.0;

    for (int r = 0; r < R; ++r) {
        dd_real x = fn();
        DoNotOptimize(x);
        acc += x;
    }

    sink_dd += acc;
    DoNotOptimize(sink_dd);
    ClobberMemory();

    auto t1 = clk::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ns = (sec * 1e9) / R;
    std::cout << "  " << name << " : " << ns << " ns/call\n";
    return ns;
}

// helper to print __float128
static void print_q(const char* label, __float128 x)
{
    char buf[160];
    quadmath_snprintf(buf, sizeof(buf), "%.36Qg", x);
    std::cout << label << buf << "\n";
}

// ============================================================
// main
// ============================================================

// Helper: run a scoped formatter (restores flags/precision)
struct IosFormatGuard {
    std::ios& os;
    std::ios::fmtflags f;
    std::streamsize p;
    explicit IosFormatGuard(std::ios& os_) : os(os_), f(os_.flags()), p(os_.precision()) {}
    ~IosFormatGuard() { os.flags(f); os.precision(p); }
};

int main()
{
#if defined(__AVX2__) && defined(__FMA__)
    std::cout << "AVX2+FMA enabled\n";
#else
    std::cout << "AVX2+FMA NOT enabled\n";
#endif

    const int N = 500;
    const int R = 500000;

    // -------------------- generate base data in double --------------------
    std::vector<double> a(N), b(N);
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < N; ++i) { a[i] = dist(rng); b[i] = dist(rng); }

    // fplll vectors (double)
    NumVect<FP_NR<double>> fa(N), fb(N);
    for (int i = 0; i < N; ++i) { fa[i] = a[i]; fb[i] = b[i]; }

    // long double mirrors
    std::vector<long double> a_ld(N), b_ld(N);
    for (int i = 0; i < N; ++i) { a_ld[i] = (long double)a[i]; b_ld[i] = (long double)b[i]; }

    // __float128 mirrors
    std::vector<__float128> a_q(N), b_q(N);
    for (int i = 0; i < N; ++i) { a_q[i] = (__float128)a[i]; b_q[i] = (__float128)b[i]; }

    // dd_real mirrors with explicit low parts to force true dd precision
    std::vector<dd_real> a_dd(N), b_dd(N);
    for (int i = 0; i < N; ++i) {
        double ahi = a[i];
        double bhi = b[i];
        double alo = ((i & 1) ? 1.0 : -1.0) * 1e-30;
        double blo = ((i & 2) ? 1.0 : -1.0) * 1e-30;
        a_dd[i] = dd_real(ahi, alo);
        b_dd[i] = dd_real(bhi, blo);
    }

    // -------------------- correctness check --------------------
    {
        IosFormatGuard g(std::cout);
        std::cout << "\nResults check:\n";
        std::cout << std::fixed << std::setprecision(6);

        double ref_d_scalar = dot_plain_scalar(a.data(), b.data(), N);
        std::cout << "  scalar(double)             = " << ref_d_scalar << "\n";
        std::cout << "  fplll(double)              = " << dot_fplll_double(fa, fb, N) << "\n";
        std::cout << "  openblas(double)           = " << dot_openblas(a.data(), b.data(), N) << "\n";
#if defined(__AVX2__) && defined(__FMA__)
        std::cout << "  avx2(double)               = " << dot_plain_avx2(a.data(), b.data(), N) << "\n";
#endif

        long double ref_ld  = dot_plain_scalar_ld(a_ld.data(), b_ld.data(), N);
        long double ref_ld4 = dot_plain_scalar_ld_ilp4(a_ld.data(), b_ld.data(), N);
        std::cout << "  scalar(long double)        = " << (double)ref_ld  << " (printed as double)\n";
        std::cout << "  scalar(long double ilp4)   = " << (double)ref_ld4 << " (printed as double)\n";

        // __float128: keep your existing print_q (high precision)
        __float128 ref_q  = dot_plain_scalar_q(a_q.data(), b_q.data(), N);
        __float128 ref_q4 = dot_plain_scalar_q_ilp4(a_q.data(), b_q.data(), N);
        print_q("  scalar(__float128)         = ", ref_q);
        print_q("  scalar(__float128 ilp4)    = ", ref_q4);

        // dd_real: show full precision + prove dd differs from double
        dd_real ref_dd  = dot_plain_scalar_dd(a_dd.data(), b_dd.data(), N);
        dd_real ref_dd4 = dot_plain_scalar_dd_ilp4(a_dd.data(), b_dd.data(), N);
        dd_real dd_from_double = dd_real(ref_d_scalar);
        dd_real dd_diff = ref_dd - dd_from_double;

        std::cout.setf(std::ios::scientific);
        std::cout << std::setprecision(30);
        std::cout << "  scalar(dd_real)            = " << ref_dd  << "\n";
        std::cout << "  scalar(dd_real ilp4)       = " << ref_dd4 << "\n";
        std::cout << "  dd_real - double(dot)      = " << dd_diff << "  (should be non-zero)\n";
    }

    // -------------------- timings --------------------
    std::cout << "\nTiming (N=" << N << ", R=" << R << ")\n";

    double ns_scalar = bench_double("plain scalar (double)", [&](){
        return dot_plain_scalar(a.data(), b.data(), N);
    }, R);

#if defined(__AVX2__) && defined(__FMA__)
    double ns_avx2 = bench_double("plain avx2+fma (double)", [&](){
        return dot_plain_avx2(a.data(), b.data(), N);
    }, R);
#endif

    double ns_blas = bench_double("openblas cblas_ddot (double)", [&](){
        return dot_openblas(a.data(), b.data(), N);
    }, R);

    double ns_fplll = bench_double("fplll FP_NR<double>", [&](){
        return dot_fplll_double(fa, fb, N);
    }, R);

    double ns_ld = bench_long_double("plain scalar (long double)", [&](){
        return dot_plain_scalar_ld(a_ld.data(), b_ld.data(), N);
    }, R);

    double ns_ld4 = bench_long_double("plain scalar ILP4 (long double)", [&](){
        return dot_plain_scalar_ld_ilp4(a_ld.data(), b_ld.data(), N);
    }, R);

    const int Rq = R;
    double ns_q = bench_float128("plain scalar (__float128)", [&](){
        return dot_plain_scalar_q(a_q.data(), b_q.data(), N);
    }, Rq);

    double ns_q4 = bench_float128("plain scalar ILP4 (__float128)", [&](){
        return dot_plain_scalar_q_ilp4(a_q.data(), b_q.data(), N);
    }, Rq);

    double ns_dd = bench_dd_real("plain scalar (dd_real)", [&](){
    DoNotOptimize(a_dd[0]);   // <-- here
    DoNotOptimize(b_dd[0]);   // <-- here
      
        return dot_plain_scalar_dd(a_dd.data(), b_dd.data(), N);
    }, R);

    double ns_dd4 = bench_dd_real("plain scalar ILP4 (dd_real)", [&](){
    DoNotOptimize(a_dd[0]);   // <-- here
    DoNotOptimize(b_dd[0]);   // <-- here
      
        return dot_plain_scalar_dd_ilp4(a_dd.data(), b_dd.data(), N);
    }, R);

    // -------------------- ratios (polished formatting) --------------------
    {
        IosFormatGuard g(std::cout);
        std::cout << "\nSummary (lower is faster)\n";
        std::cout << std::fixed << std::setprecision(2);

        auto line = [&](const char* label, double ns, double base) {
            double ratio = ns / base;
            std::cout << "  " << std::left  << std::setw(28) << label
                      << std::right << std::setw(10) << ns << " ns/call"
                      << "   (" << std::setw(6) << ratio << "x)\n";
        };

        const double base = ns_scalar;
        line("plain scalar (double)", ns_scalar, base);
#if defined(__AVX2__) && defined(__FMA__)
        line("plain avx2+fma (double)", ns_avx2, base);
#endif
        line("openblas cblas_ddot (double)", ns_blas, base);
        line("fplll FP_NR<double>", ns_fplll, base);
        line("plain scalar (long double)", ns_ld, base);
        line("plain scalar ILP4 (long double)", ns_ld4, base);
        line("plain scalar (__float128)", ns_q, base);
        line("plain scalar ILP4 (__float128)", ns_q4, base);
        line("plain scalar (dd_real)", ns_dd, base);
        line("plain scalar ILP4 (dd_real)", ns_dd4, base);
    }

    return 0;
}
