#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <gmp.h>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;
using namespace std::chrono;

// Boost Fixed-Precision Types (Stack allocated)
typedef number<cpp_int_backend<128, 128, signed_magnitude, unchecked, void>> int128_f;
typedef number<cpp_int_backend<256, 256, signed_magnitude, unchecked, void>> int256_f;
typedef number<cpp_int_backend<384, 384, signed_magnitude, unchecked, void>> int384_f;

// Forces the compiler to actually write the value to a register/memory
template <typename T>
inline void memory_barrier(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

const int ITERATIONS = 100000000; // 10 Million operations per test

// Runtime value to prevent constant folding
long get_runtime_mu() {
    return (time(NULL) % 100) + 1234567; 
}

template<typename T>
void bench_boost(const char* name, const char* val_str) {
    T a(val_str);
    T b("987654321098765432109876543210");
    long mu = get_runtime_mu();
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        // Core LLL operation: a = a + b * mu
        a += b * mu;
        
        // Bitwise mix to force carry-chain evaluation across all limbs
        a ^= (b >> (i % 64)); 
        
        // Barrier: Compiler cannot skip this
        memory_barrier(a);
    }
    auto end = high_resolution_clock::now();
    
    std::cout << std::left << std::setw(15) << name 
              << ": " << duration_cast<milliseconds>(end - start).count() << " ms" 
              << " | Final (bits): " << msb(a) << std::endl;
}

void bench_gmp(const char* name, const char* val_str) {
    mpz_t a, b, temp;
    mpz_init_set_str(a, val_str, 10);
    mpz_init_set_str(b, "987654321098765432109876543210", 10);
    mpz_init(temp);
    unsigned long mu = (unsigned long)get_runtime_mu();
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        // Core LLL operation: a = a + b * mu
        mpz_addmul_ui(a, b, mu);
        
        // Bitwise mix equivalent
        mpz_tdiv_q_2exp(temp, b, (i % 64));
        mpz_xor(a, a, temp);
        
        // Barrier: Compiler cannot skip this
        memory_barrier(a);
    }
    auto end = high_resolution_clock::now();
    
    std::cout << std::left << std::setw(15) << name 
              << ": " << duration_cast<milliseconds>(end - start).count() << " ms" 
              << " | Final (bits): " << mpz_sizeinbase(a, 2) << std::endl;
    
    mpz_clears(a, b, temp, NULL);
}

int main() {
    const char* v128 = "170141183460469231731687303715884105727";
    const char* v256 = "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    const char* v384 = "39402006196394479212279040100143613805079739270465446667948293404245721771497210611414266254884915640806627990306815";

    std::cout << "--- HARDENED BENCHMARK (10M Iterations) ---\n";
    bench_gmp("GMP 128", v128);
    bench_boost<int128_f>("Boost 128", v128);
    std::cout << "-------------------------------------------\n";
    bench_gmp("GMP 256", v256);
    bench_boost<int256_f>("Boost 256", v256);
    std::cout << "-------------------------------------------\n";
    bench_gmp("GMP 384", v384);
    bench_boost<int384_f>("Boost 384", v384);
    
    return 0;
}
