// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Class for performing time integration in fluid system.
//
// =============================================================================

#ifndef CH_FLUIDDYNAMICS_COMPRESSIBLE_H_
#define CH_FLUIDDYNAMICS_COMPRESSIBLE_H_

#include "chrono_fsi/sph/physics/FsiDataManager.cuh"
#include "chrono_fsi/sph/physics/FsiForce.cuh"
#include "chrono_fsi/sph/physics/CollisionSystem.cuh"
#include "chrono_fsi/sph/utils/UtilsDevice.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/FsiForce_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/CollisionSystem_compressible.cuh"
#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {


// Class to represent the fluid/granular dynamics system.

// This class is used to represent a fluid system and take care of the time integration of the fluid/granular
// dynamics. This is a class designed for base SPH simulation. The class holds pointer to data, which is hold somewhere
// else. It also include a forceSystem, which takes care of the computation of force between particles.

class FluidDynamics_csph {
  public:
    // Constructor of the fluid dynamics class.
    // - Instantiate FsiForce, i.e. force system;
    // - Copy the pointer to SPH particle data, parameters,
    //   and number of objects to member variables.
    //  Notice we pass references to FsiDataManager and BceManager (which is not an attribute of this class, but of FsiForce_base)
    //  But we don't pass neither a collisionSystem nor a FsiForce pointer/object.
    //  These two pointers are created by calling their constructors, which need references to FsiDataManager and
    //  BceManager.
    FluidDynamics_csph(FsiDataManager_csph& data_mgr,  ///< FSI data manager
                       BceManager_csph& bce_mgr,       ///< BCE manager
                       bool verbose,              ///< verbose output
                       bool check_errors          ///< check CUDA errors
    );

    // Destructor of the fluid/granular dynamics class.
    ~FluidDynamics_csph();

    // Overload of the following method to pass it two members of m_data_mgr as default arguments. 
    void ProximitySearch();

    // Perform proximity search.
    // Sort particles (broad-phase) and create neighbor lists (narrow-phase) by calling methods of CollisionSystem.
    // Grid update performed on input original markers. sortedSphMarkers is destination of sorting procedure.
    // Neighbor list and related variables are stored in underlying FsiDataManager_csph object.
    void ProximitySearch(const std::shared_ptr<SphMarkerDataD_csph>& originalSphMarkersD,
                         const std::shared_ptr<SphMarkerDataD_csph>& sortedSphMarkersD);

    //// TODO: make private
    // copies markers posRadD, velMas, RhoPreEnD vectors from a pointer to another.
    void CopySortedMarkers(const std::shared_ptr<SphMarkerDataD_csph>& in, std::shared_ptr<SphMarkerDataD_csph>& out);

    // Advance the dynamics of the SPH fluid system.
    // In a explicit scheme, the force system calculates the forces between the particles which are then used to update
    // the particles position, velocity, and density; The density is then used, through the equation of state, to
    // update pressure.
    // For explicit schemes, under the hood will call a proper combination of EulerStep, FsiForce, ApplyBCs methods.
    void DoStepDynamics(std::shared_ptr<SphMarkerDataD_csph> y,  //< marker state (in/out)
                        Real t,                                  //< current simulation time
                        Real h,                                  //< simulation stepsize
                        IntegrationScheme_csph scheme);          //< integration scheme
    

    // Copy markers in the specified group from sorted arrays to original-order arrays.
    // Recall group defined as {FLUID, SOLID, BOUNDARY, NON_FLUID, NON_SOLID, NON_BOUNDARY, ALL}
    void CopySortedToOriginal(MarkerGroup group,
                              std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD2,    // the 2 because in CHFluidSystem called with sortedMarkers2_D
                              std::shared_ptr<SphMarkerDataD_csph> sphMarkersD);

    // Copy markers in the specified group from original order arrays to sorted order. Does not order
    void CopyOriginalToSorted(MarkerGroup group, std::shared_ptr<SphMarkerDataD_csph> originalSphMarkersD,
                              std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);

    // Initialize the force and colision systems.
    // Synchronize data between device and host (parameters and counters).
    void Initialize();

    // Return the FsiForce type used in the simulation.
    std::shared_ptr<FsiForce_base> GetForceSystem() { return forceSystem; }

    // Update activity of SPH particles.
    // SPH particles which are in an active domain (e.g., close to a solid) are set as active particles.
    // Works with NumAllMarkers by default.
    // It's the actual method that computes the ActivityIdentifier and extendedActivityIdentifier vectors.
    void UpdateActivity(std::shared_ptr<SphMarkerDataD_csph> sphMarkersD);

    // Check if arrays must be resized due to change in particle activity.
    // Notice returns a bool: true is numExtendedParticle < numAllMarkers, 0 otherwise
    // Also perform an exclusive scan on the activity identifiers, treating -1 flags as 0.
    bool CheckActivityArrayResize();
 
    // Compute the time step used in temporal integration depending on the current CFL condition
    // Returns a double3 struct composed as: (min_delta_t, min_delta_t_courant, min_delta_t_force)
    double3 computeTimeStep() const;

  private:
    FsiDataManager_csph& m_data_mgr;  //< FSI data manager
    std::shared_ptr<FsiForce_base> forceSystem;  //< force system object; calculates the force between particles. This is a pointer to the base
                                                 //< class, but once constructed will actually point
                                                 //   to a derived object --> takes advantage of polymorphism.
    std::shared_ptr<CollisionSystem_csph> collisionSystem;  //< collision system for building neighbors list

    bool m_verbose;
    bool m_check_errors;

    // Actual methods that perform computations, used in different ways by DoStepDynamics().

    // Advance the state of the fluid system using an explicit Euler step.
    void EulerStep(std::shared_ptr<SphMarkerDataD_csph> sortedMarkers, Real dT);

    // Advance the state of the fluid system using a mid-point step.
    void MidpointStep(std::shared_ptr<SphMarkerDataD_csph> sortedMarkers, Real dT);

    // Apply boundary conditions on the sides of the computational domain. (Either periodic or inlet conditions)
    void ApplyBoundaryConditions(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);

    // Update grid parameters and perform again neighbor search at mid-point step.
    void MidpointNeighSearch(const std::shared_ptr<SphMarkerDataD_csph>& sortedInitialMarkers,
                             const std::shared_ptr<SphMarkerDataD_csph>& sortedMidpointMarkers);

    
};




}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif