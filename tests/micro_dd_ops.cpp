// micro_dd_ops_same_iters.cpp
#include <qd/dd_real.h>
#include <immintrin.h>

#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;

static volatile double sink = 0.0;

inline void consume(double x) { sink += x * 1e-300; }
inline void consume(const dd_real& x) { sink += to_double(x) * 1e-300; }

template <class F>
long long time_ms(F&& f)
{
  using clk = chrono::high_resolution_clock;
  auto t1 = clk::now();
  f();
  auto t2 = clk::now();
  return chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
}

int main()
{
  constexpr size_t iters = 50'000'000; // SAME for all tests

  cerr << "iters = " << iters << "\n\n";

  // --------------------------------------------------
  // scalar double
  // --------------------------------------------------
  {
    double a = 1.0000001, b = 0.9999999, c = 0.0;
    auto ms = time_ms([&] {
      for (size_t i = 0; i < iters; ++i)
      {
        c += a * b;
        a += 1e-16;
        b -= 1e-16;
      }
    });
    consume(c);
    cerr << left << setw(32) << "double: c += a*b (scalar)"
         << ": " << ms << " ms\n";
  }

  // --------------------------------------------------
  // AVX2 double (explicit FMA)
  // --------------------------------------------------
#if defined(__AVX2__) && defined(__FMA__)
  {
    __m256d a = _mm256_set1_pd(1.0000001);
    __m256d b = _mm256_set1_pd(0.9999999);
    __m256d c = _mm256_setzero_pd();

    auto ms = time_ms([&] {
      for (size_t i = 0; i < iters; i += 4)
        c = _mm256_fmadd_pd(a, b, c);
    });

    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, c);
    consume(tmp[0] + tmp[1] + tmp[2] + tmp[3]);

    cerr << left << setw(32) << "double: c += a*b (AVX2)"
         << ": " << ms << " ms\n";
  }
#endif

  // --------------------------------------------------
  // dd_real: c = a*b
  // --------------------------------------------------
  {
    dd_real a = 1.0000001, b = 0.9999999, c = 0.0;
    auto ms = time_ms([&] {
      for (size_t i = 0; i < iters; ++i)
      {
        c = a * b;
        a += dd_real(1e-30);
        b -= dd_real(1e-30);
      }
    });
    consume(c);
    cerr << left << setw(32) << "dd_real: c = a*b"
         << ": " << ms << " ms\n";
  }

  // --------------------------------------------------
  // dd_real: c += a*b
  // --------------------------------------------------
  {
    dd_real a = 1.0000001, b = 0.9999999, c = 0.0;
    auto ms = time_ms([&] {
      for (size_t i = 0; i < iters; ++i)
      {
        c += a * b;
        a += dd_real(1e-30);
        b -= dd_real(1e-30);
      }
    });
    consume(c);
    cerr << left << setw(32) << "dd_real: c += a*b"
         << ": " << ms << " ms\n";
  }

  cerr << "\n(sink=" << sink << ")\n";
  return 0;
}
