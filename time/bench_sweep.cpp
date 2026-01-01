// g++ -O3 -march=native -ffast-math bench_sweep.cpp ../fplll/.libs/libfplll.a -lopenblas -o bench_sweep OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bench_sweep


#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

#include <fplll/fplll.h>
#include <cblas.h>

using namespace fplll;
using clk = std::chrono::high_resolution_clock;

static volatile double sink = 0.0;

inline double dot_openblas(const double* a, const double* b, int n)
{
    return cblas_ddot(n, a, 1, b, 1);
}

inline double dot_fplll(const NumVect<FP_NR<double>>& a,
                        const NumVect<FP_NR<double>>& b,
                        int n)
{
    FP_NR<double> r;
    dot_product(r, a, b, 0, n);
    return r.get_d();
}

template <class F>
double time_ns_per_call(F&& fn, int R)
{
    // warm-up
    for (int i = 0; i < 2000; ++i) sink += fn();

    auto t0 = clk::now();
    double acc = 0.0;
    for (int i = 0; i < R; ++i) acc += fn();
    auto t1 = clk::now();
    sink += acc;

    double sec = std::chrono::duration<double>(t1 - t0).count();
    return (sec * 1e9) / R;
}

static int choose_R_for_N(int N)
{
    // Aim for stable timings without taking forever.
    // Increase R for small N, decrease for large N.
    if (N <= 64)   return 20'000'000;
    if (N <= 128)  return 10'000'000;
    if (N <= 256)  return 5'000'000;
    if (N <= 512)  return 2'000'000;
    if (N <= 1024) return 1'000'000;
    if (N <= 2048) return 500'000;
    if (N <= 4096) return 250'000;
    return 100'000;
}

int main()
{
    // Single random dataset, re-used for all N (prefixes).
    const int Nmax = 16384;

    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> a(Nmax), b(Nmax);
    for (int i = 0; i < Nmax; ++i) { a[i] = dist(rng); b[i] = dist(rng); }

    NumVect<FP_NR<double>> fa(Nmax), fb(Nmax);
    for (int i = 0; i < Nmax; ++i) { fa[i] = a[i]; fb[i] = b[i]; }

    std::cout << "Note: for fair BLAS benchmarking, run with OPENBLAS_NUM_THREADS=1 (and OMP_NUM_THREADS=1)\n\n";

    std::cout << std::left
              << std::setw(8)  << "N"
              << std::setw(14) << "fplll ns"
              << std::setw(16) << "openblas ns"
              << std::setw(12) << "ratio(B/F)"
              << "\n";

    std::cout << std::string(8+14+16+12, '-') << "\n";

    // Sweep sizes
    const int sizes[] = {16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};

    for (int N : sizes)
    {
        int R = choose_R_for_N(N);

        // Correctness check once per N (cheap)
        double ref_blas = dot_openblas(a.data(), b.data(), N);
        double ref_fpl  = dot_fplll(fa, fb, N);
        if (std::abs(ref_blas - ref_fpl) > 1e-9 * std::max(1.0, std::abs(ref_blas))) {
            std::cerr << "Mismatch at N=" << N << ": blas=" << ref_blas << " fplll=" << ref_fpl << "\n";
            return 1;
        }

        double ns_fplll = time_ns_per_call([&](){ return dot_fplll(fa, fb, N); }, R);
        double ns_blas  = time_ns_per_call([&](){ return dot_openblas(a.data(), b.data(), N); }, R);

        std::cout << std::left
                  << std::setw(8)  << N
                  << std::setw(14) << std::fixed << std::setprecision(3) << ns_fplll
                  << std::setw(16) << std::fixed << std::setprecision(3) << ns_blas
                  << std::setw(12) << std::fixed << std::setprecision(3) << (ns_blas / ns_fplll)
                  << "\n";
    }

    return 0;
}
