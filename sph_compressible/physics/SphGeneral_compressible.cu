// =============================================================================
// Author: Andrea D'Uva - 2025
//
//
// =============================================================================

#include "chrono_fsi/sph/physics/FsiDataManager.cuh"
#include "chrono_fsi/sph/physics/SphGeneral.cuh"

#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {

// copy the Params struct defined on the host, to the const static variable ParamsD, defined on the device
void CopyParametersToDevice_csph(std::shared_ptr<ChFsiParamsSPH_csph> paramsH, std::shared_ptr<Counters_csph> countersH) {
    cudaMemcpyToSymbolAsync(paramsD_csph, paramsH.get(), sizeof(ChFsiParamsSPH_csph));
    cudaCheckError();
    cudaMemcpyToSymbolAsync(countersD_csph, countersH.get(), sizeof(Counters_csph));
    cudaCheckError();

}

}  // namespace compressible
}  // namespace chrono::fsi::sph
