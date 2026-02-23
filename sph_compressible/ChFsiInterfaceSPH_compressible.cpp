// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Custom FSI interface for coupling the SPH-based fluid system with a Chrono MBS
// Modified to use compressible SPH data structures
// =============================================================================

#include "chrono/utils/ChUtils.h"
#include "chrono_fsi/sph/utils/UtilsPrintSph.cuh"

#include "chrono_fsi/sph/ChFsiInterfaceSPH.h"
#include "chrono_fsi/sph/physics/FsiDataManager.cuh"
#include "chrono_fsi/sph/physics/FsiInterface.cuh"
#include "chrono_fsi/sph/utils/UtilsDevice.cuh"
#include "chrono_fsi/sph/utils/UtilsTypeConvert.cuh"

#include "chrono_fsi/sph_compressible/utils/UtilsPrintSph_compressible.cuh"

#include "chrono_fsi/sph_compressible/ChFsiInterfaceSPH_compressible.h"
#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"
#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {

// Derived class constructor (now public), initializes the members with a call to FsiDataManager constructor and with a
// call to the base class ChFsiInterface constructor. Recall that this base class holds references to a ChSystem and to
// a ChFsiFluidSystem (not SPH), and its constructor is actually protected. Here we pass to such base constructor a
// ChFsiFluidSystemSPH& (derived class) but it's not a problem: polymorphism works also with references.
ChFsiInterfaceSPH_csph::ChFsiInterfaceSPH_csph(ChSystem& sysMBS, ChFsiFluidSystemSPH_csph& sysSPH)
    : ChFsiInterface(sysMBS, sysSPH), m_data_mgr(sysSPH.m_data_mgr.get()) {}

ChFsiInterfaceSPH_csph::~ChFsiInterfaceSPH_csph() {}

// Copies MBS solid state from the host, and apply them into the SPH system into device memory. FSIDataManager does the
// transfer.
void ChFsiInterfaceSPH_csph::ExchangeSolidStates() {
    {
        // Load from rigid bodies on host (vector of FsiBodies m_fsi_bodies stored in base class ChFsiInterface)
        // first to host vector in FsiDataManager, and then to its device vectors
        int index = 0;
        for (const auto& fsi_body : m_fsi_bodies) { /* m_fsi_bodies is vector of FsiBody added to FSI system*/
            m_data_mgr->fsiBodyState_H->pos[index] = ToReal3(fsi_body->body->GetPos());
            m_data_mgr->fsiBodyState_H->lin_vel[index] = ToReal3(fsi_body->body->GetPosDt());
            m_data_mgr->fsiBodyState_H->lin_acc[index] = ToReal3(fsi_body->body->GetPosDt2());
            m_data_mgr->fsiBodyState_H->rot[index] = ToReal4(fsi_body->body->GetRot());
            m_data_mgr->fsiBodyState_H->ang_vel[index] = ToReal3(fsi_body->body->GetAngVelLocal());
            m_data_mgr->fsiBodyState_H->ang_acc[index] = ToReal3(fsi_body->body->GetAngAccLocal());
            index++;
        }

        // Transfer to device
        m_data_mgr->fsiBodyState_D->CopyFromH(*m_data_mgr->fsiBodyState_H);
    }

}


// Transfers rigid forces from the data manager on the GPU to the host, then applies fluid forces and
// torques as external loads in the MBS.
void ChFsiInterfaceSPH_csph::ExchangeSolidForces() {
    {  // Rigid bodies
        // Transfer to host
        auto forcesH = m_data_mgr->GetRigidForces();  // returns a std::vector<Real3> which is the copy of rigid_FSI_ForcesD,
                                                      // vector of the fluid forces acting on each rigid body.
        auto torquesH = m_data_mgr->GetRigidTorques();

        // Apply the loads to rigid bodies
        int index = 0;
        for (const auto& fsi_body : m_fsi_bodies) {  // All FsiBodies in the vector.
            m_fsi_bodies[index]->fsi_force =  ToChVector(forcesH[index]);  // fsi_force is a ChVector3d for the fluid force acting on the body
            m_fsi_bodies[index]->fsi_torque = ToChVector(torquesH[index]);
            fsi_body->body->EmptyAccumulator(fsi_body->fsi_accumulator);  // body is a pointer to a ChBody obj
            fsi_body->body->AccumulateForce(fsi_body->fsi_accumulator, m_fsi_bodies[index]->fsi_force,
                                            fsi_body->body->GetPos(), false);
            fsi_body->body->AccumulateTorque(fsi_body->fsi_accumulator, m_fsi_bodies[index]->fsi_torque, false);
            index++;
        }
    }


}


}  // end namespace compressible
}  // end namespace chrono::fsi::sph
