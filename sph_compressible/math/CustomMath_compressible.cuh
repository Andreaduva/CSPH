
// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Math utilities for the Chrono::FSI::compressible module.
// These functions can be invoked either on the CPU (host) or on the GPU (device)
// Defines many overloaded functions and operators to work with Cuda vector types like float2 etc etc in addition to ones defined in cudaruntime.h
// =============================================================================

#ifndef CHFSI_CUSTOM_MATH_COMPRESSIBLE_H
#define CHFSI_CUSTOM_MATH_COMPRESSIBLE_H

#include <cuda_runtime.h>
#include <cmath>

#include <thrust/device_ptr.h>   
#include <thrust/detail/raw_pointer_cast.h>

#include "chrono_fsi/sph/ChFsiDataTypesSPH.h"
#include "chrono_fsi/sph/math/CustomMath.cuh"

namespace chrono {
namespace fsi {
namespace sph {

/// Struct of 5 reals.

struct alignas(16) Real5 {
    Real x, y, z, w, t;
};

// Insertion of a Real5 to output stream.
inline std::ostream& operator<<(std::ostream& out, const Real5& v) {
    out << v.x << "  " << v.y << "  " << v.z << "  " << v.w << "   " << v.t;
    return out;
}

// Print a Real5 struct.
inline void printStruct(struct Real5& s) {
    std::cout << "x = " << s.x << ", ";
    std::cout << "y = " << s.y << ", ";
    std::cout << "z = " << s.z << ", ";
    std::cout << "w = " << s.w << ", ";
    std::cout << "t = " << s.t << ", " << std::endl;
}

__host__ __device__ inline Real3 make_Real3(Real5 a) {
    Real3 result;
    result.x = a.x;
    result.y = a.y;
    result.z = a.z;
    return result;
}

__host__ __device__ inline Real5 make_Real5(Real a, Real b, Real c, Real d, Real e)  ///
{
    Real5 f;
    f.x = a;
    f.y = b;
    f.z = c;
    f.w = d;
    f.t = e;
    return f;
}

__host__ __device__ inline Real5 make_Real5(Real s) {
    return make_Real5(s, s, s, s, s);
}

__host__ __device__ inline Real5 make_Real5(Real3 a) {
    return make_Real5(a.x, a.y, a.z, 0.0, 0.0);
}

__host__ __device__ inline Real5 make_Real5(Real3 a, Real w, Real z) {
    return make_Real5(a.x, a.y, a.z, w, z);
}

__host__ __device__ inline Real5 make_Real5(Real4 a) {
    return make_Real5(a.x, a.y, a.z, a.w, 0.0);
}

__host__ __device__ inline Real5 make_Real5(int4 a) {
    return make_Real5(Real(a.x), Real(a.y), Real(a.z), Real(a.w), 0.0);
}

__host__ __device__ inline Real5 make_Real5(uint4 a) {
    return make_Real5(Real(a.x), Real(a.y), Real(a.z), Real(a.w), 0.0);
}

__host__ __device__ inline Real5 operator-(Real5& a) {
    return make_Real5(-a.x, -a.y, -a.z, -a.w, -a.t);
}

__host__ __device__ inline Real5 operator+(Real5 a, Real5 b) {
    return make_Real5(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w, a.t + b.t);
}

__host__ __device__ inline void operator+=(Real5& a, Real5 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    a.t += b.t;
}

__host__ __device__ inline Real5 operator+(Real5 a, Real b) {
    return make_Real5(a.x + b, a.y + b, a.z + b, a.w + b, a.t + b);
}

__host__ __device__ inline Real5 operator+(Real b, Real5 a) {
    return make_Real5(a.x + b, a.y + b, a.z + b, a.w + b, a.t + b);
}

__host__ __device__ inline void operator+=(Real5& a, Real b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    a.t += b;
}

__host__ __device__ inline Real5 operator-(Real5 a, Real5 b) {
    return make_Real5(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w, a.t - b.t);
}

__host__ __device__ inline void operator-=(Real5& a, Real5 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    a.t -= b.t;
}

__host__ __device__ inline Real5 operator-(Real5 a, Real b) {
    return make_Real5(a.x - b, a.y - b, a.z - b, a.w - b, a.t - b);
}

__host__ __device__ inline void operator-=(Real5& a, Real b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
    a.t -= b;
}

__host__ __device__ inline Real5 operator*(Real5 a, Real5 b) {
    return make_Real5(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w, a.t * b.t);
}

__host__ __device__ inline void operator*=(Real5& a, Real5 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    a.t *= b.t;
}

__host__ __device__ inline Real5 operator*(Real5 a, Real b) {
    return make_Real5(a.x * b, a.y * b, a.z * b, a.w * b, a.t * b);
}

__host__ __device__ inline Real5 operator*(Real b, Real5 a) {
    return make_Real5(b * a.x, b * a.y, b * a.z, b * a.w, b * a.t);
}

__host__ __device__ inline void operator*=(Real5& a, Real b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    a.t *= b;
}

__host__ __device__ inline Real5 operator/(Real5 a, Real5 b) {
    return make_Real5(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w, a.t / b.t);
}

__host__ __device__ inline void operator/=(Real5& a, Real5 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
    a.t /= b.t;
}

__host__ __device__ inline Real5 operator/(Real5 a, Real b) {
    return make_Real5(a.x / b, a.y / b, a.z / b, a.w / b, a.t / b);
}

__host__ __device__ inline void operator/=(Real5& a, Real b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
    a.t /= b;
}

__host__ __device__ inline Real5 operator/(Real b, Real5 a) {
    return make_Real5(b / a.x, b / a.y, b / a.z, b / a.w, b / a.t);
}

__host__ __device__ inline Real5 rminr(Real5 a, Real5 b) {
    Real5 result;
    result.x = sph::rminr(a.x, b.x);
    result.y = sph::rminr(a.y, b.y);
    result.z = sph::rminr(a.z, b.z);
    result.w = sph::rminr(a.w, b.w);
    result.t = sph::rminr(a.t, b.t);
    return result;
}

__host__ __device__ inline Real5 rmaxr(Real5 a, Real5 b) {
    Real5 result;
    result.x = sph::rmaxr(a.x, b.x);
    result.y = sph::rmaxr(a.y, b.y);
    result.z = sph::rmaxr(a.z, b.z);
    result.w = sph::rmaxr(a.w, b.w);
    result.t = sph::rmaxr(a.t, b.t);
    return result;
}

__host__ __device__ inline Real5 lerp(Real5 a, Real5 b, Real t) {
    return a + t * (b - a);
}

__host__ __device__ inline Real dot(Real5 a, Real5 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w + a.t * b.t;
}

__host__ __device__ inline Real length(Real5 v) {
    return sqrt(dot(v, v));
}

__host__ __device__ inline Real5 get_normalized(Real5 v) {
    Real invLen = rsqrtr(dot(v, v));
    return v * invLen;
}

__host__ __device__ inline void normalize(Real5& v) {
    Real invLen = rsqrtr(dot(v, v));
    v *= invLen;
}

__host__ __device__ inline bool IsFinite(Real5 v) {
#ifdef __CUDA_ARCH__
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(v.w) && isfinite(v.t);
#else
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w) && std::isfinite(v.t);
#endif
}

#define mR5 make_Real5
#define mR5CAST(x) (Real5*)thrust::raw_pointer_cast(&x[0])

}  // end namespace sph
}  // namespace fsi
}  // namespace chrono

#endif
