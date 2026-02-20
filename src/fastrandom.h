#ifndef FASTRANDOM_H
#define FASTRANDOM_H

#include <cstdint>
#include <limits>

/**
 * @brief Fast PRNG based on xorshift128+ algorithm.
 *
 * Satisfies the C++ UniformRandomBitGenerator concept, so it can be used
 * as a drop-in replacement for std::mt19937 in standard distributions.
 * Significantly faster than std::mt19937 for performance-critical paths
 * like population mutation.
 */
class FastRandom {
public:
  using result_type = std::uint64_t;

  explicit FastRandom(result_type seed = 0) : state{seed, 0} {}

  result_type operator()() {
    result_type x = state[0];
    result_type const y = state[1];
    state[0] = y;
    x ^= x << 23;                             // a
    state[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
    return state[1] + y;
  }

  static constexpr result_type min() {
    return std::numeric_limits<result_type>::min();
  }
  static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }

private:
  result_type state[2];
};

#endif // FASTRANDOM_H
