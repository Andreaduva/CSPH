// =============================================================================
// Author: Andrea D'Uva - 2025
// 
// Implementation of class derived from base FsiForce_csph, to actually compute the 
// inter-particle force in compressible Euler sph.
// =============================================================================

#ifndef CH_FSI_FORCE_EXPLICITSPH_COMPRESSIBLE_H_
#define CH_FSI_FORCE_EXPLICITSPH_COMPRESSIBLE_H_

#include "chrono_fsi/ChApiFsi.h"
#include "chrono_fsi/sph/physics/FsiForce.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiForce_base.cuh"

namespace chrono::fsi::sph {
namespace compressible {



// Inter-particle force calculation for explicit compressible SPH schemes.
class FsiForce_csph : public FsiForce_base {
  public:

    // Force class implemented using WCSPH with an explicit integrator.
    // Supports for both fluid and granular material dynamics.
    //  Recall FsiDataManager holds pointers and vectors of particles and associated quantities. BceManager objects also
    //  have inside them a DataManager pointer. BceManager serves as link between fluid and solid. It computes fluid
    //  forces and torques acting on bodies, and displacements/velocities/acceleration of bce markers due to the bodies
    //  movement. This is the constructor. The input FsiDataManager also contains the simulation parameters, so optional
    //  methods like particle shiftings are applied based on what the data manager contains.
    FsiForce_csph(FsiDataManager_csph& data_mgr,   //< FSI data manager
                  BceManager_csph& bce_mgr,        //< BCE manager
                  bool verbose,                    //< verbose output
                  bool check_errors
    );

    ~FsiForce_csph();

    void Initialize() override;

  private:
    // Function to calculate forces on SPH particles. Overwrites the pure virtual method of base class.
    // It just calls the other private methods in the order: density_reinitialization, ApplyBcs, CalcRhs, Shifting Method
    // (If specified) Notice 1: the step parameter is not used in here, but present here because the base virtual
    // method declares it. Notice 2: this method is declared private. However it's a virtual method, designed to be
    // called by pointers to the base class FsiForce. This is allowed: polymorphism would still work. Imagine: FsiForce*
    // ptr = new FsiForce_csph(); ptr->ForceSPH will result in still calling the proper derived method even though it's
    // private. Following would not work and cause compilation error: FsiForce_csph* ptr = new FsiForce_csph();
    // ptr->ForceSPH; will cause compilation error. Notice derivative of density computed with original velocities, not
    // XSPH corrected ones.
    void ForceSPH(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD, Real time, Real step) override;


    // Computes density at current step with summation equation:
    void CfdCalcRhoSum(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);

    // Perform density re-initialization (as needed).
    void density_reinitialization(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);

    // CFD (fluid): computes right hand-side of the differential equations
    void CfdApplyBC(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);
    void CfdCalcRHS(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);

    // CFD (fluid): updates the kernel radius h of particles.
    // evolves the kernel radius of each particle according to the ADKE method
    void CfdCalcRadADKE(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);
    // Computes the derivative of kernel radius, proportional to d(rho)/dt:
    void CfdCalcRadRHS(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);
    // Function to calculate the shifting of the particles.
    // Can use XSPH for now.
    // Result is computation of the deltaV to be applied for each particle. Sum of velocity and shifting velocity done
    // externally
    void CalculateShifting(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);

    
    int density_initialization;  // used as counter

    // CUDA execution configuration grid
    uint numActive;   //< total number of threads
    uint numBlocks;   //< number of blocks
    uint numThreads;  //< number of threads per block

    bool m_check_errors;
};



}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif
