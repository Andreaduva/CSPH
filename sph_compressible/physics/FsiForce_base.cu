// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Base class for processing sph force in compressible fsi system.
// =============================================================================

#include <thrust/extrema.h>
#include <thrust/sort.h>

#include "chrono_fsi/sph/physics/FsiForce.cuh"
#include "chrono_fsi/sph/utils/UtilsDevice.cuh"
#include "chrono_fsi/sph/physics/SphGeneral.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiForce_base.cuh"
#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {



FsiForce_base::FsiForce_base(FsiDataManager_csph& data_mgr, BceManager_csph& bce_mgr, bool verbose)
    : m_data_mgr(data_mgr), m_bce_mgr(bce_mgr), m_verbose(verbose) {
    cudaMallocErrorFlag(m_errflagD);
}



FsiForce_base::~FsiForce_base() {
    cudaFreeErrorFlag(m_errflagD);
}



void FsiForce_base::Initialize() {
    cudaMemcpyToSymbolAsync(paramsD_csph, m_data_mgr.paramsH.get(), sizeof(ChFsiParamsSPH_csph));
    cudaMemcpyToSymbolAsync(countersD_csph, m_data_mgr.countersH.get(), sizeof(Counters_csph));
}


// Use invasive to avoid one extra copy.
// However, keep in mind that sorted is changed.
void FsiForce_base::CopySortedToOriginal_Invasive_R1(thrust::device_vector<Real>& original,
                                                     thrust::device_vector<Real>& sorted,
                                                     const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
    thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(),
                        sorted.begin());  // sorts the indexes in dummyGridMarkerIndex, so that we get back to order by
                                          // index, not by hash. Values in "sorted" vector are ordered accordingly.
    dummyMarkerIndex.clear();             // removes all elements from the vector leaving it with a size of 0
    thrust::copy(sorted.begin(), sorted.end(), original.begin());  // Only one copy operation
}

void FsiForce_base::CopySortedToOriginal_NonInvasive_R1(thrust::device_vector<Real>& original,
                                                        const thrust::device_vector<Real>& sorted,
                                                        const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<Real> dummySorted = sorted;
    CopySortedToOriginal_Invasive_R1(original, dummySorted, gridMarkerIndex);
    // A bit less efficient since we have to create and copy a new dummy sorted array that will be modified but without
    // consecuences
}




// Use invasive to avoid one extra copy.
// However, keep in mind that sorted is changed.
void FsiForce_base::CopySortedToOriginal_Invasive_R3(thrust::device_vector<Real3>& original,
                                                     thrust::device_vector<Real3>& sorted,
                                                     const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
    thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(),
                        sorted.begin());  // sorts the indexes in dummyGridMarkerIndex, so that we get back to order by
                                          // index, not by hash. Values in "sorted" vector are ordered accordingly.
    dummyMarkerIndex.clear();             // removes all elements from the vector leaving it with a size of 0
    thrust::copy(sorted.begin(), sorted.end(), original.begin());  // Only one copy operation
}



void FsiForce_base::CopySortedToOriginal_NonInvasive_R3(thrust::device_vector<Real3>& original,
                                                        const thrust::device_vector<Real3>& sorted,
                                                        const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<Real3> dummySorted = sorted;
    CopySortedToOriginal_Invasive_R3(original, dummySorted, gridMarkerIndex);
    // A bit less efficient since we have to create and copy a new dummy sorted array that will be modified but without
    // consecuences
}


// Use invasive to avoid one extra copy.
// However, keep in mind that sorted is changed.
void FsiForce_base::CopySortedToOriginal_Invasive_R4(thrust::device_vector<Real4>& original,
                                                     thrust::device_vector<Real4>& sorted,
                                                     const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
    thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(), sorted.begin());
    dummyMarkerIndex.clear();
    thrust::copy(sorted.begin(), sorted.end(), original.begin());
}


void FsiForce_base::CopySortedToOriginal_NonInvasive_R4(thrust::device_vector<Real4>& original,
                                                        thrust::device_vector<Real4>& sorted,
                                                        const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<Real4> dummySorted = sorted;
    CopySortedToOriginal_Invasive_R4(original, dummySorted, gridMarkerIndex);
}



// Use invasive to avoid one extra copy.
// However, keep in mind that sorted is changed.
void FsiForce_base::CopySortedToOriginal_Invasive_R2(thrust::device_vector<Real2>& original,
                                                     thrust::device_vector<Real2>& sorted,
                                                     const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
    thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(), sorted.begin());
    dummyMarkerIndex.clear();
    thrust::copy(sorted.begin(), sorted.end(), original.begin());
}



void FsiForce_base::CopySortedToOriginal_NonInvasive_R2(thrust::device_vector<Real2>& original,
                                                        thrust::device_vector<Real2>& sorted,
                                                        const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<Real2> dummySorted = sorted;
    CopySortedToOriginal_Invasive_R2(original, dummySorted, gridMarkerIndex);
}



// Use invasive to avoid one extra copy.
// However, keep in mind that sorted is changed.
void FsiForce_base::CopySortedToOriginal_Invasive_R5(thrust::device_vector<Real5>& original,
                                                     thrust::device_vector<Real5>& sorted,
                                                     const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
    thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(), sorted.begin());
    dummyMarkerIndex.clear();
    thrust::copy(sorted.begin(), sorted.end(), original.begin());
}

void FsiForce_base::CopySortedToOriginal_NonInvasive_R5(thrust::device_vector<Real5>& original,
                                                        thrust::device_vector<Real5>& sorted,
                                                        const thrust::device_vector<uint>& gridMarkerIndex) {
    thrust::device_vector<Real5> dummySorted = sorted;
    CopySortedToOriginal_Invasive_R5(original, dummySorted, gridMarkerIndex);
}

}  // end namespace compressible
}  // end namespace chrono::fsi::sph
