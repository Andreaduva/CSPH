// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Custom FSI interface for coupling the SPH-based fluid system with a Chrono MBS
// Modified to account for compressible data structures
// =============================================================================

#ifndef CH_FSI_INTERFACE_SPH_COMPRESSIBLE_H
#define CH_FSI_INTERFACE_SPH_COMPRESSIBLE_H

#include "chrono_fsi/ChApiFsi.h"
#include "chrono_fsi/ChFsiInterface.h"

#include "chrono_fsi/sph/ChFsiFluidSystemSPH.h"

#include "chrono_fsi/sph_compressible/ChFsiFluidSystemSPH_compressible.h"

namespace chrono::fsi::sph {
namespace compressible {

struct FsiDataManager_csph; 


// Custom FSI interface between a Chrono multibody system and the SPH-based fluid system.
// This custom FSI interface is paired with ChFsiFluidSystemSPH_csph and provides a more efficient coupling with a Chrono
// MBS that a generic FSI interface does, because it works directly with the data structures of ChFsiFluidSystemSPH_csph.
class ChFsiInterfaceSPH_csph : public ChFsiInterface {
  public:
    ChFsiInterfaceSPH_csph(ChSystem& sysMBS,
                      ChFsiFluidSystemSPH_csph& sysSPH);  // constructor given a MBS system and a FluidSystemSPH system
    ~ChFsiInterfaceSPH_csph();

    // Exchange solid phase state information between the MBS and fluid system.
    // Extract FSI body states from the MBS, copy to the SPH data manager on the host, and then
    // transfer to GPU memory. Directly access SPH data manager.
    virtual void ExchangeSolidStates() override;

    // Exchange solid phase force information between the multibody and fluid systems.
    // Transfer rigid forces from the SPH data manager on the GPU to the host, then apply fluid forces and
    // torques as external loads in the MBS. Directly access SPH data manager.
    virtual void ExchangeSolidForces() override;

    // Both Exchange functions call FsiDataManager_csph to exchange the data.

  private:
    FsiDataManager_csph* m_data_mgr;  //< FSI data manager

    // Also protected member from ChFsiInterface base class like:
    // std::vector<std::shared_ptr<FsiBody>> m_fsi_bodies;
    // ChSystem& m_sysMBS;
    // ChFsiFluidSystem& m_sysCFD;   but actually points to the derived ChFsiFluidSystemSPH_csph

    // FsiBodies are added by the base class method AddFsiBody(), which also calls the FLuidSystem method that
    // adds the FsiBody also to the fluid system and data manager.
};


}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif