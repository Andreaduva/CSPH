// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Modification of original class FsiParticleRelocator to adapt for compressibility
//
// =============================================================================

#ifndef FSI_PARTICLE_RELOCATOR_COMPRESSIBLE_CUH
#define FSI_PARTICLE_RELOCATOR_COMPRESSIBLE_CUH

#include "chrono_fsi/sph/physics/FsiDataManager.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"

namespace chrono::fsi::sph{
namespace compressible {

class FsiParticleRelocator_csph {
  public:
    struct DefaultProperties_csph {
        Real rho0;  // density
        Real p0;    // pressure
        Real e0;    // energy
    };

    FsiParticleRelocator_csph(FsiDataManager_csph& data_mgr, const DefaultProperties_csph& props);
    ~FsiParticleRelocator_csph() {}

    // Shift all particles of specified type by the given vector.
    // Properties (density, pressure, energy) of relocated particles are overwritten with the specified values.
    void Shift(MarkerType type, const Real3& shift);

    // Move all particles of specified type that are currently inside the source AABB to the given AABB.
    // The destination AABB is assumed to be given in integer grid coordinates. Properties (density, pressure, energy) of
    // relocated particles are overwritten with the specified values.
    void MoveAABB2AABB(MarkerType type, const RealAABB& aabb_src, const IntAABB& aabb_dest, Real spacing);

  private:
    FsiDataManager_csph& m_data_mgr;  //< FSI data manager
    DefaultProperties_csph m_props;   //< particle density, pressure and energy after relocation
};



}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif
