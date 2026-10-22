/* Copyright (c) 2026. Licensed under the MIT License. */
#ifndef LGBMNPU_SIMT_FIXED_POINT_V2_H_
#define LGBMNPU_SIMT_FIXED_POINT_V2_H_

// Deterministic fixed-point accumulation helpers for the AI Core SIMT fused
// histogram.
//
// Why: SIMT on Ascend 950 (dav-3510) has no FP64 arithmetic and no FP64
// atomics, so a parallel histogram cannot reproduce the reference "sequential
// FP64 adds" bit for bit. FP32 accumulation is not acceptable either: the
// probe fixtures contain values up to FLT_MAX, and summing a few dozen of them
// overflows FP32.
//
// Strategy: quantize every FP32 input to an int64 grid of a per-call power of
// two step, accumulate with INT64 atomics (integer addition is associative and
// commutative, so the result is deterministic and independent of thread
// scheduling), then convert the exact int64 partial sum back to an IEEE-754
// binary64 value with round-to-nearest-even.
//
// Scale selection
//   Let E = floor(log2(max|x|)) and C = ceil(log2(count)).
//   shift = 61 - E - C, so |q(x)| <= 2^(61-C) for every input (|x| <= max|x|)
//   and sum|q| <= 2^C * 2^(61-C) = 2^61, which leaves two bits of head room in
//   int64. Per-element quantization error is at most 2^-shift/2 (round to
//   nearest), i.e. relative to max|x| it is count / 2^61. For a 1024-row leaf
//   that is ~2^-51, for a 100k-row leaf ~2^-44. Both are far below FP32.
//
// Range
//   The shift is an exponent, not a float multiplier, so it is applied by
//   shifting the 24-bit float significand. Subnormal inputs and inputs up to
//   FLT_MAX are handled without overflowing the int64 accumulator.
//
// Limitations
//   * Inf/NaN inputs are skipped (quantized to 0). The LightGBM histogram
//     contract does not feed them.
//   * The final int64 -> double conversion is exact for a 53-bit significand
//     and rounds once (RNE) otherwise; the double subnormal range is
//     unreachable because the smallest non-zero result is 2^-129.
//
// All helpers are header-only, free of std:: calls and usable from both the
// .asc device/host translation unit and a plain host C++ test.

#include <cstdint>

#if defined(__NPU_ARCH__) || defined(__DAV_VEC__) || defined(__CCE__)
#define LGBM_SIMT_HD __host__ __aicore__
#else
#define LGBM_SIMT_HD
#endif

namespace lgbm_simt {

union FloatUint {
  float f;
  uint32_t u;
};

union DoubleUint {
  double d;
  uint64_t u;
};

LGBM_SIMT_HD inline uint32_t FloatToBits(float x) {
  FloatUint c;
  c.f = x;
  return c.u;
}

LGBM_SIMT_HD inline float BitsToFloat(uint32_t u) {
  FloatUint c;
  c.u = u;
  return c.f;
}

LGBM_SIMT_HD inline int32_t CountLeadingZeros32(uint32_t x) {
  return __builtin_clz(x);
}

// Composed from the 32-bit builtin: bisheng 15.0.5 crashes its backend
// ("HiTPE Stall Cycle Refactor") on __builtin_clzll inside a SIMT kernel.
// The 64-bit count is identical for both the host and the device.
LGBM_SIMT_HD inline int32_t CountLeadingZeros64(uint64_t x) {
  const uint32_t high = static_cast<uint32_t>(x >> 32);
  if (high != 0u) return CountLeadingZeros32(high);
  return 32 + CountLeadingZeros32(static_cast<uint32_t>(x));
}

// ceil(log2(count)) for count <= 1 returns 0.
LGBM_SIMT_HD inline int32_t CeilLog2Count(int64_t count) {
  if (count <= 1) return 0;
  return 64 - CountLeadingZeros64(static_cast<uint64_t>(count - 1));
}

// floor(log2(|x|)) given the raw (possibly negative) bit pattern of x.
// Returns 0 for zero and 128 for inf/NaN (caller clamps).
LGBM_SIMT_HD inline int32_t FloorLog2AbsBits(uint32_t raw_bits) {
  const uint32_t u = raw_bits & 0x7FFFFFFFu;
  if (u == 0u) return 0;
  const int32_t exponent = static_cast<int32_t>((u >> 23) & 0xFFu);
  if (exponent == 0) {
    const uint32_t mantissa = u & 0x7FFFFFu;
    return -149 + (31 - CountLeadingZeros32(mantissa));
  }
  if (exponent == 0xFF) return 128;
  return exponent - 127;
}

// Quantization exponent. q(x) = round(x * 2^shift); the value is reconstructed
// as q * 2^(-shift), i.e. ScaledInt64ToDoubleBits(q, -shift).
LGBM_SIMT_HD inline int32_t HistogramShift(uint32_t max_abs_bits,
                                           int64_t count) {
  if ((max_abs_bits & 0x7FFFFFFFu) == 0u) return 0;
  int32_t exponent = FloorLog2AbsBits(max_abs_bits);
  if (exponent > 127) exponent = 127;  // inf/NaN: dropped by QuantizeFloat
  return 61 - exponent - CeilLog2Count(count);
}

// q = round_to_nearest(x * 2^shift), saturating; inf/NaN map to 0.
LGBM_SIMT_HD inline int64_t QuantizeFloat(float x, int32_t shift) {
  const uint32_t bits = FloatToBits(x);
  const uint32_t sign = (bits >> 31) & 1u;
  const uint32_t exponent = (bits >> 23) & 0xFFu;
  const uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent == 0xFFu) return 0;  // inf / NaN
  uint64_t magnitude;
  int32_t base_exponent;
  if (exponent == 0u) {
    if (mantissa == 0u) return 0;
    magnitude = static_cast<uint64_t>(mantissa);
    base_exponent = -149;
  } else {
    magnitude = static_cast<uint64_t>(mantissa | 0x800000u);
    base_exponent = static_cast<int32_t>(exponent) - 127 - 23;
  }
  const int32_t scale = base_exponent + shift;
  int64_t q;
  if (scale >= 0) {
    constexpr int64_t kInt64Max = 0x7FFFFFFFFFFFFFFFLL;
    if (scale >= 40) {
      const uint64_t limit =
          static_cast<uint64_t>(kInt64Max) >> scale;
      q = (magnitude > limit) ? kInt64Max
                              : static_cast<int64_t>(magnitude << scale);
    } else {
      q = static_cast<int64_t>(magnitude << scale);
    }
  } else {
    const int32_t right = -scale;
    if (right >= 63) {
      q = 0;
    } else {
      const uint64_t half = static_cast<uint64_t>(1) << (right - 1);
      q = static_cast<int64_t>((magnitude + half) >> right);
    }
  }
  return sign ? -q : q;
}

// Exact int64 partial sum -> IEEE-754 binary64 bits (value = q * 2^shift),
// round-to-nearest-even.
LGBM_SIMT_HD inline uint64_t ScaledInt64ToDoubleBits(int64_t q, int32_t shift) {
  const uint64_t sign_bit = (q < 0) ? (static_cast<uint64_t>(1) << 63) : 0u;
  uint64_t magnitude;
  if (q < 0) {
    magnitude = static_cast<uint64_t>(-(q + 1)) + 1u;
  } else {
    magnitude = static_cast<uint64_t>(q);
  }
  if (magnitude == 0u) return 0u;  // +0.0

  const int32_t nbits = 64 - CountLeadingZeros64(magnitude);
  int64_t exponent = static_cast<int64_t>(nbits - 1) + shift;

  uint64_t significand;
  const int32_t drop = nbits - 53;
  if (drop <= 0) {
    significand = magnitude << (-drop);
  } else {
    const uint64_t remainder = magnitude & ((static_cast<uint64_t>(1) << drop) - 1u);
    significand = magnitude >> drop;
    const uint64_t half = static_cast<uint64_t>(1) << (drop - 1);
    if (remainder > half || (remainder == half && (significand & 1u))) {
      ++significand;
      if (significand == (static_cast<uint64_t>(1) << 53)) {
        significand >>= 1;
        ++exponent;
      }
    }
  }

  const int64_t biased = exponent + 1023;
  if (biased >= 2047) {
    return sign_bit | 0x7FF0000000000000ull;  // +-inf (unreachable)
  }
  if (biased >= 1) {
    const uint64_t fraction = significand & ((static_cast<uint64_t>(1) << 52) - 1u);
    return sign_bit | (static_cast<uint64_t>(biased) << 52) | fraction;
  }
  // A non-zero result is at least 2^-129, well above the double subnormal
  // range, so this branch only guards malformed input.
  return sign_bit;
}

// ---------------------------------------------------------------------------
// v2: multiple magnitude windows.
//
// A single int64 grid spans ~61 bits (61 - ceil(log2(count)) of dynamic range
// above the bin maximum). A bin that mixes FLT_MAX with 1/3 needs ~129 bits of
// range, so v2 keeps up to three windows, each `gap` bits finer than the last:
//
//   shift1 = base                                      (coarsest)
//   shift2 = shift1 + gap
//   shift3 = shift2 + gap                              (finest)
//
// An element that quantizes to zero in window k is re-quantized in window k+1.
// Window 1 bounds |q| < 2^(62-C) per element, so its sum stays below 2^62;
// windows 2 and 3 only see elements that vanished one level up (|x| <
// 2^-shift_k), which bounds |q_k| < 2^(gap-1) per element and the sum by
// 2^(C+gap-1) <= 2^60 when gap <= 61 - C. int64 (2^63) therefore always keeps
// head room.
//
//   value = q1 * 2^shift1 + q2 * 2^shift2 + q3 * 2^shift3
//         = T * 2^shift3,   T = q1 * 2^(2*gap) + q2 * 2^gap + q3
//
// T is an exact signed integer of at most 62 + 2*61 = 184 bits, held here as a
// 5-word (320-bit) two's complement so the sign extension is always inside the
// container. The result is a single round-to-nearest-even of T * 2^shift3.
//
// __int128 is not usable on this target (bisheng 15.0.5 segfaults in its
// backend), which is why the wide value is built and rounded word by word.
//
// The kernel calls ScaledWindowsToDoubleBits, which takes the plain int64 path
// whenever q2 and q3 are zero; that keeps every ordinary bin bitwise identical
// to the single-window (v1) conversion.
// ---------------------------------------------------------------------------

// 320-bit two's complement accumulator. Fields rather than an array so that
// every access is statically indexed (no local-memory spilling on device).
struct Wide320 {
  uint64_t w0;
  uint64_t w1;
  uint64_t w2;
  uint64_t w3;
  uint64_t w4;
};

LGBM_SIMT_HD inline uint32_t Wide320Bit(const Wide320& value, int32_t pos) {
  if (pos < 0) return 0u;
  const int32_t word = pos >> 6;
  const int32_t bit = pos & 63;
  uint64_t x;
  if (word == 0) {
    x = value.w0;
  } else if (word == 1) {
    x = value.w1;
  } else if (word == 2) {
    x = value.w2;
  } else if (word == 3) {
    x = value.w3;
  } else if (word == 4) {
    x = value.w4;
  } else {
    return 0u;
  }
  return static_cast<uint32_t>((x >> bit) & 1u);
}

// accumulator += q * 2^shift, with q * 2^shift sign-extended over all 320 bits.
LGBM_SIMT_HD inline void Wide320AddShifted(Wide320* accumulator, int64_t q,
                                           int32_t shift) {
  if (q == 0) return;
  const int32_t word = shift >> 6;
  const int32_t bit = shift & 63;
  const uint64_t sign_ext =
      (q < 0) ? ~static_cast<uint64_t>(0) : static_cast<uint64_t>(0);
  uint64_t lo;
  uint64_t hi;
  if (bit == 0) {
    lo = static_cast<uint64_t>(q);
    hi = sign_ext;
  } else {
    lo = static_cast<uint64_t>(q) << bit;
    // Arithmetic shift of a negative int64 fills ones, which is exactly the
    // sign extension we need in the next word.
    hi = static_cast<uint64_t>(q >> (64 - bit));
  }
  uint64_t a0 = 0u;
  uint64_t a1 = 0u;
  uint64_t a2 = 0u;
  uint64_t a3 = 0u;
  const uint64_t a4 = sign_ext;
  if (word == 0) {
    a0 = lo;
    a1 = hi;
    a2 = sign_ext;
    a3 = sign_ext;
  } else if (word == 1) {
    a1 = lo;
    a2 = hi;
    a3 = sign_ext;
  } else {
    a2 = lo;
    a3 = hi;
  }

  uint64_t carry = 0u;
  uint64_t s;
  uint64_t c;
  s = accumulator->w0 + a0;
  c = (s < accumulator->w0) ? 1u : 0u;
  accumulator->w0 = s;
  carry = c;
  s = accumulator->w1 + a1;
  c = (s < accumulator->w1) ? 1u : 0u;
  accumulator->w1 = s + carry;
  carry = (c != 0u) | ((accumulator->w1 < s) ? 1u : 0u);
  s = accumulator->w2 + a2;
  c = (s < accumulator->w2) ? 1u : 0u;
  accumulator->w2 = s + carry;
  carry = (c != 0u) | ((accumulator->w2 < s) ? 1u : 0u);
  s = accumulator->w3 + a3;
  c = (s < accumulator->w3) ? 1u : 0u;
  accumulator->w3 = s + carry;
  carry = (c != 0u) | ((accumulator->w3 < s) ? 1u : 0u);
  accumulator->w4 += a4 + carry;
}

// value = q1 * 2^(shift3 + 2*gap) + q2 * 2^(shift3 + gap) + q3 * 2^shift3,
// rounded once (round-to-nearest-even) to binary64 bits.
LGBM_SIMT_HD inline uint64_t ScaledThreeWindowToDoubleBits(
    int64_t q1, int64_t q2, int64_t q3, int32_t shift3, int32_t gap) {
  if (gap < 0 || gap > 61) {
    // Malformed request: fall back to the well-covered coarse window.
    const int64_t pick = (q1 != 0) ? q1 : ((q2 != 0) ? q2 : q3);
    const int32_t pick_shift = (q1 != 0) ? shift3 + 2 * gap
                                         : ((q2 != 0) ? shift3 + gap : shift3);
    return ScaledInt64ToDoubleBits(pick, pick_shift);
  }

  Wide320 total;
  total.w0 = 0u;
  total.w1 = 0u;
  total.w2 = 0u;
  total.w3 = 0u;
  total.w4 = 0u;
  Wide320AddShifted(&total, q3, 0);
  Wide320AddShifted(&total, q2, gap);
  Wide320AddShifted(&total, q1, 2 * gap);

  if (total.w0 == 0u && total.w1 == 0u && total.w2 == 0u && total.w3 == 0u &&
      total.w4 == 0u) {
    return 0u;  // +0.0
  }
  const bool negative = (total.w4 >> 63) != 0u;
  if (negative) {
    total.w0 = ~total.w0;
    total.w1 = ~total.w1;
    total.w2 = ~total.w2;
    total.w3 = ~total.w3;
    total.w4 = ~total.w4;
    total.w0 += 1u;
    if (total.w0 == 0u) {
      total.w1 += 1u;
      if (total.w1 == 0u) {
        total.w2 += 1u;
        if (total.w2 == 0u) {
          total.w3 += 1u;
          if (total.w3 == 0u) total.w4 += 1u;
        }
      }
    }
  }

  int32_t nbits;
  if (total.w4 != 0u) {
    nbits = 320 - CountLeadingZeros64(total.w4);
  } else if (total.w3 != 0u) {
    nbits = 256 - CountLeadingZeros64(total.w3);
  } else if (total.w2 != 0u) {
    nbits = 192 - CountLeadingZeros64(total.w2);
  } else if (total.w1 != 0u) {
    nbits = 128 - CountLeadingZeros64(total.w1);
  } else {
    nbits = 64 - CountLeadingZeros64(total.w0);
  }
  if (nbits <= 0) return 0u;

  int64_t exponent = static_cast<int64_t>(nbits - 1) + shift3;
  const int32_t drop = nbits - 53;

  // Top 53 bits, one bit at a time (this path runs once per bin and only for
  // bins that actually needed a finer window, so clarity beats speed here).
  uint64_t significand = 0u;
  for (int32_t i = 0; i < 53; ++i) {
    significand = (significand << 1) |
                  static_cast<uint64_t>(Wide320Bit(total, nbits - 1 - i));
  }
  bool round_up = false;
  if (drop > 0) {
    const bool half = Wide320Bit(total, drop - 1) != 0u;
    if (half) {
      bool sticky = false;
      for (int32_t i = 0; i < drop - 1; ++i) {
        if (Wide320Bit(total, i) != 0u) {
          sticky = true;
          break;
        }
      }
      round_up = sticky || ((significand & 1u) != 0u);
    }
  }
  if (round_up) {
    ++significand;
    if (significand == (static_cast<uint64_t>(1) << 53)) {
      significand >>= 1;
      ++exponent;
    }
  }

  const uint64_t sign_bit =
      negative ? (static_cast<uint64_t>(1) << 63) : static_cast<uint64_t>(0);
  const int64_t biased = exponent + 1023;
  if (biased >= 2047) {
    return sign_bit | 0x7FF0000000000000ull;  // +-inf (unreachable)
  }
  if (biased >= 1) {
    const uint64_t fraction =
        significand & ((static_cast<uint64_t>(1) << 52) - 1u);
    return sign_bit | (static_cast<uint64_t>(biased) << 52) | fraction;
  }
  return sign_bit;
}

// value = q1 * 2^shift1 + q2 * 2^(shift1 + gap) + q3 * 2^(shift1 + 2*gap).
// Takes the plain int64 path whenever the finer windows are empty, which is
// what keeps ordinary bins bitwise identical to the single-window result.
LGBM_SIMT_HD inline uint64_t ScaledWindowsToDoubleBits(
    int64_t q1, int64_t q2, int64_t q3, int32_t shift1, int32_t gap) {
  if (q2 == 0 && q3 == 0) return ScaledInt64ToDoubleBits(q1, shift1);
  return ScaledThreeWindowToDoubleBits(q1, q2, q3, shift1 - 2 * gap, gap);
}

LGBM_SIMT_HD inline double ScaledInt64ToDouble(int64_t q, int32_t shift) {
  DoubleUint c;
  c.u = ScaledInt64ToDoubleBits(q, shift);
  return c.d;
}

LGBM_SIMT_HD inline void StoreDoubleBits(double* destination, uint64_t bits) {
  DoubleUint c;
  c.u = bits;
  *destination = c.d;
}

}  // namespace lgbm_simt

#endif  // LGBMNPU_SIMT_FIXED_POINT_V2_H_
