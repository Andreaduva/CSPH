// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Extension of the base class for processing boundary condition enforcing (BCE) particles forces
// in FSI system, considering the compressible case
// All methods work with rigid bce markers only, numRigidMarkers. Bounday/wall bce markers excluded
// =============================================================================

#ifndef CH_BCE_MANAGER_COMPRESSIBLE_H
#define CH_BCE_MANAGER_COMPRESSIBLE_H

#include "chrono_fsi/ChApiFsi.h"
#include "chrono_fsi/sph/physics/FsiDataManager.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"


namespace chrono::fsi::sph {
namespace compressible {

// Manager for processing boundary condition enforcing (BCE) particle forces in an FSI system.
// This class handles the Fluid-Solid Interaction by enforcing:
// - forces from the compressible fluid dynamics system to the MBD system
// - displacements from the MBD system to the fluid dynamics system
// Works with rigid bce markers, not with boundary bce ones.
class BceManager_csph {
  public:
    BceManager_csph(FsiDataManager_csph& data_mgr,  //< FSI data
                    bool verbose,                   //< verbose terminal output
                    bool check_errors);             //< check CUDA errors
    

    ~BceManager_csph();

    // Update position and velocity of BCE markers on rigid solids
    void UpdateBodyMarkerState();
    void UpdateBodyMarkerStateInitial();

    // Calculate fluid forces on rigid bodies.
    // results stored in m_data_mgr.rigidFSIforces_D and .rigid_FSI_torques_D
    void Rigid_Forces_Torques();

    // public method, calling CalcRigidBceAcceleration() under the hood to populate m_data_mgr.bceAcc()
    void updateBCEAcc();

    // Complete construction of the BCE at the initial configuration of the system.
    // m_data_mgr should already be correctly initialized
    // It populates device vectors based on host vectors already present in m_data_mgr
    void Initialize(std::vector<int> fsiBodyBceNum);

  private:
    /// Set up block sizes for rigid body force accumulation.
    void SetForceAccumulationBlocks(std::vector<int> fsiBodyBceNum);

    // Calculate accelerations of solid BCE markers based on the information of the ChSystem.
    // Works on data already present in m_data_mgr; accelerations are stored in the bcaAcc vector variable of that
    // FsiDataManager struct
    void CalcRigidBceAcceleration();

  private:
    FsiDataManager_csph& m_data_mgr;   //< FSI data manager

    thrust::device_vector<Real3> m_totalForceRigid;   //< Total forces from fluid to bodies. One vector for each solid body.
    thrust::device_vector<Real3> m_totalTorqueRigid;  //< Total torques from fluid to bodies

    uint m_rigid_block_size;                                  //< Block size for the rigid force accumulator kernel
    uint m_rigid_grid_size;                                   //< Grid size for the rigid force accumulator kernel
    thrust::device_vector<uint> m_rigid_valid_threads;        //< numbers of valid (non-padding) threads in the block
    thrust::device_vector<uint> m_rigid_accumulated_threads;  //< accumulated numbers of padded threads before a block

    bool m_verbose;
    bool m_check_errors;
};


}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif
