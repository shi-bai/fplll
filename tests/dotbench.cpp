// dotbench_dd_compare.cpp
//
// Compare dot-product cost for:
//   1) fplll::FP_NR<dd_real> inside fplll::NumVect  (uses FP_NR ops: mul/addmul)
//   2) raw std::vector<dd_real> (uses dd_real ops directly)
// Also prints the FP_NR<double> AVX2 case + BLAS ddot for reference.
//
// IMPORTANT: include your source-tree numvect.h FIRST so FP_NR<double> AVX2 specialization is used.

#include "../fplll/nr/numvect.h"
#include <fplll/fplll.h>

#include <qd/dd_real.h>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

extern "C" {
#include <cblas.h>
}

using namespace std;
using namespace fplll;

static volatile double g_sink = 0.0;

inline void consume_double(double x) { g_sink += x * 1e-300; }

template <class FT>
inline void consume_fp(const FT& x)
{
  g_sink += x.get_d() * 1e-300;
}

// -------- helpers ----------
template <class FT>
NumVect<FT> make_numvect_from_double(const vector<double>& src)
{
  NumVect<FT> v((int)src.size());
  for (int i = 0; i < (int)src.size(); i++)
    v[i] = src[(size_t)i];
  return v;
}

static vector<dd_real> make_ddvec_from_double(const vector<double>& src)
{
  vector<dd_real> v(src.size());
  for (size_t i = 0; i < src.size(); i++)
    v[i] = src[i];
  return v;
}

// -------- benchmarks ----------
template <class FT>
long long bench_fplll_dot(const NumVect<FT>& a, const NumVect<FT>& b, size_t rounds, const char* name)
{
  using clock_type = chrono::high_resolution_clock;
  FT tmp;

  auto t1 = clock_type::now();
  for (size_t r = 0; r < rounds; r++)
  {
    fplll::dot_product(tmp, a, b, 0, a.size());
    consume_fp(tmp);
  }
  auto t2 = clock_type::now();

  const auto ms = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
  cerr << left << setw(45) << name << ": " << ms << " ms\n";
  return ms;
}

static long long bench_blas_ddot(const vector<double>& a, const vector<double>& b, size_t rounds, const char* name)
{
  using clock_type = chrono::high_resolution_clock;

  auto t1 = clock_type::now();
  for (size_t r = 0; r < rounds; r++)
  {
    const double tmp = cblas_ddot((int)a.size(), a.data(), 1, b.data(), 1);
    consume_double(tmp);
  }
  auto t2 = clock_type::now();

  const auto ms = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
  cerr << left << setw(45) << name << ": " << ms << " ms\n";
  return ms;
}

// Raw dd_real dot (no FP_NR wrapper)
static inline dd_real ddot_raw_dd(const dd_real* __restrict x, const dd_real* __restrict y, int n)
{
  dd_real sum = 0.0;
  for (int i = 0; i < n; i++)
    sum += x[i] * y[i];
  return sum;
}

static long long bench_raw_ddot_dd(const vector<dd_real>& a, const vector<dd_real>& b, size_t rounds, const char* name)
{
  using clock_type = chrono::high_resolution_clock;

  auto t1 = clock_type::now();
  for (size_t r = 0; r < rounds; r++)
  {
    const dd_real tmp = ddot_raw_dd(a.data(), b.data(), (int)a.size());
    // Convert to double just to sink it
    g_sink += to_double(tmp) * 1e-300;
  }
  auto t2 = clock_type::now();

  const auto ms = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
  cerr << left << setw(45) << name << ": " << ms << " ms\n";
  return ms;
}

static void print_ratio(const char* label, long long num_ms, long long den_ms)
{
  const double ratio = (den_ms == 0) ? 0.0 : (double)num_ms / (double)den_ms;
  cerr << "  " << left << setw(34) << label << ": " << ratio << "x\n";
}

int main()
{
  const int dim = 8192;
  const size_t rounds = 20000;

  mt19937_64 rng(12345);
  uniform_real_distribution<double> dist(-1.0, 1.0);

  vector<double> a0((size_t)dim), b0((size_t)dim);
  for (int i = 0; i < dim; i++)
  {
    a0[(size_t)i] = dist(rng);
    b0[(size_t)i] = dist(rng);
  }

  // Reference: fplll FP_NR<double> (AVX2 specialization) + BLAS double*
  const auto a_d = make_numvect_from_double<FP_NR<double>>(a0);
  const auto b_d = make_numvect_from_double<FP_NR<double>>(b0);

  // dd_real as FP_NR wrapper (fplll path)
  const auto a_dd_fp = make_numvect_from_double<FP_NR<dd_real>>(a0);
  const auto b_dd_fp = make_numvect_from_double<FP_NR<dd_real>>(b0);

  // dd_real raw vectors
  const auto a_dd_raw = make_ddvec_from_double(a0);
  const auto b_dd_raw = make_ddvec_from_double(b0);

  cerr << "dim=" << dim << " rounds=" << rounds << "\n\n";

  const auto t_fplll_double = bench_fplll_dot(a_d, b_d, rounds, "fplll dot_product FP_NR<double> (AVX2)");
  const auto t_blas         = bench_blas_ddot(a0, b0, rounds, "OpenBLAS cblas_ddot (double*)");

  const auto t_fplll_dd = bench_fplll_dot(a_dd_fp, b_dd_fp, rounds, "fplll dot_product FP_NR<dd_real>");
  const auto t_raw_dd   = bench_raw_ddot_dd(a_dd_raw, b_dd_raw, rounds, "raw std::vector<dd_real> dot");

  cerr << "\nRatios:\n";
  print_ratio("fplll(double)/BLAS", t_fplll_double, t_blas);
  print_ratio("fplll(dd_real)/fplll(double)", t_fplll_dd, t_fplll_double);
  print_ratio("raw dd_real / fplll(dd_real)", t_raw_dd, t_fplll_dd);
  print_ratio("raw dd_real / fplll(double)", t_raw_dd, t_fplll_double);

  cerr << "\n(sink=" << g_sink << ")\n";
  return 0;
}
