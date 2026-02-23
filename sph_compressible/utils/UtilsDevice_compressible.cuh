// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Copy here the function ComputeGridSize to avoid linking error being the original
// one in UtilsDevice.cuh and .cu not actually exported
// =============================================================================

#ifndef CH_UTILS_DEVICE_COMPRESSIBLE_H
#define CH_UTILS_DEVICE_COMPRESSIBLE_H

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "chrono/core/ChTypes.h"

#include "chrono_fsi/sph/math/CustomMath.cuh"

// #include "chrono_fsi/sph_compressible/math/CustomMath_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {





// Compute number of blocks and threads for calculation on GPU.
// This function calculates the number of blocks and threads for a given number of elements based on the blockSize.
inline void computeCudaGridSize(uint n,           //< total number of elements
                            uint blockSize,   //< block size (threads per block)
                            uint& numBlocks,  //< number of blocks [output]
                            uint& numThreads){//< number of threads [output]
    uint n2 = (n == 0) ? 1 : n;
    numThreads = min(blockSize, n2);
    numBlocks = (n2 % numThreads != 0) ? (n2 / numThreads + 1) : (n2 / numThreads);
};


} // end namespace compressible
} // end namespace chrono::fsi::sph

#endif