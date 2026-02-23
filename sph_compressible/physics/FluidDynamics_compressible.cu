// =============================================================================
// Author: Andrea D'Uva
// =============================================================================
//
// Class for performing time integration in fluid system.
// =============================================================================

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>

#include "chrono/utils/ChConstants.h"
#include "chrono_fsi/sph/physics/FluidDynamics.cuh"
#include "chrono_fsi/sph/physics/FsiForceWCSPH.cuh"
#include "chrono_fsi/sph/physics/FsiForceISPH.cuh"
#include "chrono_fsi/sph/physics/SphGeneral.cuh"

#include "chrono_fsi/sph_compressible/physics/FluidDynamics_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/FsiForce_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"


using std::cout;
using std::endl;

namespace chrono::fsi::sph {
namespace compressible {

// Constructor: interesting stuff is BceManager is not a class attribute, and as such is not initialized, but used to
// set collisionSystem and forceSystem pointers. Notice collisionSystem and forceSystem not present in the initializer
// list. This is allowed because they actually are shared pointers, they use a default initializer and if unset later will point to random stuff. Not allowed with
// references, which must always be initialized. So that's why the auto keyword is missing in collisionsystem and
// forceSystem: they have been correctly initialized and just need to be set.


FluidDynamics_csph::FluidDynamics_csph(FsiDataManager_csph& data_mgr, BceManager_csph& bce_mgr, bool verbose, bool check_errors)
    : m_data_mgr(data_mgr), m_verbose(verbose), m_check_errors(check_errors) {

    // collisionSystem has no derived classes, its contructor requires only FsiDataManager_csph&
    collisionSystem = chrono_types::make_shared<CollisionSystem_csph>(data_mgr);

    // forceSystem can have derived classes. We istantiate the available FsiForce_csph.
    // 
    forceSystem = chrono_types::make_shared<FsiForce_csph>(data_mgr, bce_mgr, verbose, m_check_errors);
}

FluidDynamics_csph::~FluidDynamics_csph() {}

// -----------------------------------------------------------------------------
// Initialize device vectors and both collisionSystem and forceSystem
void FluidDynamics_csph::Initialize() {
    cudaMemcpyToSymbolAsync(paramsD_csph, m_data_mgr.paramsH.get(), sizeof(ChFsiParamsSPH_csph));
    cudaMemcpyToSymbolAsync(countersD_csph, m_data_mgr.countersH.get(), sizeof(Counters_csph));
    cudaMemcpyFromSymbol(m_data_mgr.paramsH.get(), paramsD_csph, sizeof(ChFsiParamsSPH_csph));

    forceSystem->Initialize();
    collisionSystem->Initialize();
}


// -----------------------------------------------------------------------------
// Overloaded function that provides default arguments to the following one
void FluidDynamics_csph::ProximitySearch() {
    ProximitySearch(m_data_mgr.sphMarkers_D, m_data_mgr.sortedSphMarkers2_D);
}
 

// populates the proximity search data by calling both broad- and narrow-phase search through collisionSystem
// Uses data present in FsiDataManager and updates its neighbor lists.
void FluidDynamics_csph::ProximitySearch(const std::shared_ptr<SphMarkerDataD_csph>& originalSphMarkersD,   // defaults to m_data_mgr.sphmarkers_D
                                         const std::shared_ptr<SphMarkerDataD_csph>& sortedSphMarkersD) {   // defaults to m_data_mgr,sortedSPhMarkers2_D

    // if h is allowed to vary, need also to adapt the grid parameters:
    if (m_data_mgr.paramsH->h_variation) {
        collisionSystem->UpdateGridParams(originalSphMarkersD);
    }
    collisionSystem->ArrangeData(originalSphMarkersD, sortedSphMarkersD);
    collisionSystem->NeighborSearch(sortedSphMarkersD);
  

}

// -----------------------------------------------------------------------------

void FluidDynamics_csph::CopySortedMarkers(const std::shared_ptr<SphMarkerDataD_csph>& in, std::shared_ptr<SphMarkerDataD_csph>& out) {
    thrust::copy(in->posRadD.begin(), in->posRadD.end(), out->posRadD.begin());
    thrust::copy(in->velD.begin(), in->velD.end(), out->velD.begin());
    thrust::copy(in->rhoPresEnD.begin(), in->rhoPresEnD.end(), out->rhoPresEnD.begin());
    thrust::copy(in->soundD.begin(), in->soundD.end(), out->soundD.begin());
}




double3 FluidDynamics_csph::computeTimeStep() const {
    size_t valid_entries = m_data_mgr.countersH->numExtendedParticles;
    double min_courant_time_step = 
        static_cast<double>( thrust::reduce(thrust::device, m_data_mgr.dT_velD.begin(), 
                             m_data_mgr.dT_velD.begin() + valid_entries, 
                             std::numeric_limits<Real>::max(), thrust::minimum<Real>()) );
    double min_force_time_step = 
        static_cast<double>( thrust::reduce(thrust::device, m_data_mgr.dT_forceD.begin(),
                             m_data_mgr.dT_forceD.begin() + valid_entries, 
                             std::numeric_limits<Real>::max(), thrust::minimum<Real>()) );

    double CFL_time_step = m_data_mgr.paramsH->C_cfl * std::min(min_courant_time_step, m_data_mgr.paramsH->C_force * min_force_time_step);
    std::cout << "Variable_time_step = " << CFL_time_step << std::endl;
    
    return make_double3(CFL_time_step, 
                        min_courant_time_step, 
                        m_data_mgr.paramsH->C_force*min_force_time_step);
}
    //------------------------------------------------------
// time integration of fluid particles
// -----------------------------------------------------


// Input: pointer to markerdata (also output), current simulation time t, stepsize h, integrationScheme
// Depending on the integration scheme specified for the problem we select the proper integrator.
// For all explicit integrator the forceSystem will point to FsiForce_csph object.
// Notice that this way the XSPH correction is only applied in EulerStep to the position update of particles, density
// update uses original velocities.


// Each call to forceSystem->ForceSPH populates the vectors of derivatives depending on the actual algorithm chosen.
// If density computed by summation, pressure already updated.
// Calls to EulerStep or MidpointStep do update properties like pressure, temperature and speed of sound immediately after
// step integration has completed.

// Recall that values of derivatives for each particle are computed by forceSPH method of forceSystem. It takes as input 
// a pointer to a struct of arrays that stores the particle properties.
// So y and y_temp store different properties.
// Derivatives computed by forceSph instead are stored uniquely in the DataManager class, accessible by all other classes.
// Each call to forceSph then overwrites these values, that are implicitely availabe when calling EulerStep or MidpointStep.


// The input y is given by ChFsiFluidSystem::DoStepDynamics as data_manager.sortedSphMarkers2_D

void FluidDynamics_csph::DoStepDynamics(std::shared_ptr<SphMarkerDataD_csph> y, Real t, Real h, IntegrationScheme_csph scheme) {
    switch (scheme) {
        case IntegrationScheme_csph::EULER: {
            Real dummy = 0;  // force calculation for compressible sph does not need the step size, nor the actual time explicitely

            forceSystem->ForceSPH(y, t, dummy);    // f(t_n, y_n). Actually calls the method in FsiForce_csph, where stepSize is useless.

            EulerStep(y, h);                       // y <==  y_{n+1} = y_n + h * f(t_n, y_n)
            ApplyBoundaryConditions(y);
            break;
        }

        case IntegrationScheme_csph::RK2: {
            Real dummy = 0;  // force calculation for compressible sph does not need the step size

            auto& y_tmp = m_data_mgr.sortedSphMarkers1_D;    // initialize helper variable
            CopySortedMarkers(y, y_tmp);                     // y_tmp <- y_n. Sets y_tmp to current value of properties.

            forceSystem->ForceSPH(y, t, dummy);              // computes derivatives as f(t_n, y_n). Stored in the data manager struct.

            EulerStep(y_tmp, h / 2);                         // y_tmp <==  K1 = y_n + (h/2) * f(t_n, y_n). 
            ApplyBoundaryConditions(y_tmp);

            if (m_data_mgr.paramsH->h_variation && m_data_mgr.paramsH->midpoint_neigh_search) {   // if both true, update grid at midpoint
                MidpointNeighSearch(y, y_tmp);   // now both y, y_tmp sorted with new grid computed at t_(n+1/2)
            }

            forceSystem->ForceSPH(y_tmp, t + h / 2, dummy);  // f(t_n + h/2, K1).  Computes derivatives with the state being at step (n+1)/2 and overwrites previous values.
            EulerStep(y, h);                                 // y <== y_{n+1} = y_n + h * f(t_n + h/2, K1). 
            ApplyBoundaryConditions(y);


             /*         
            // DEBUG ------------------------------------------------
            thrust::host_vector<Real4> debugRhoH = m_data_mgr.sortedSphMarkers2_D->rhoPresEnD;
            thrust::host_vector<Real5> debugDerivH = m_data_mgr.derivVelRhoEnD;
            std::cout << "INSIDE FluidDynamics_csph.DoStepDynamics() AFTER forceSystem.ForceSPH()"
                << std::endl;
                */

            // std::cout << "MarkerTypes and index position:" << std::endl;
            // for (int i = 0; i < debugRhoH.size(); i++)
            //     std::cout << debugRhoH[i].w << ", ";
            // std::cout << " Size = " << debugRhoH.size() << std::endl;
            /*
            std::cout << "X component of DerivVelRhoEn > 100:" << std::endl;
            for (int i = 0; i < debugDerivH.size(); i++)
                if (debugDerivH[i].x > 100)
                    std::cout << debugDerivH[i].x << "(" << i << ")"
                              << ", ";
            std::cout << " Size = " << debugDerivH.size() << std::endl;
            */
            /*
            std::cout << "Y component of DerivVelRhoEn:" << std::endl;
            for (int i = 0; i < debugDerivH.size(); i++)
                std::cout << debugDerivH[i].y << ", ";
            std::cout << std::endl;
            std::cout << "Z component of DerivVelRhoEn:" << std::endl;
            for (int i = 0; i < debugDerivH.size(); i++)
                std::cout << debugDerivH[i].z << ", ";
            std::cout << std::endl;
            std::cout << "RHO component of DerivVelRhoEn:" << std::endl;
            for (int i = 0; i < debugDerivH.size(); i++)
                std::cout << debugDerivH[i].w << ", ";
            std::cout << std::endl;
            std::cout << "En component of DerivVelRhoEn:" << std::endl;
            for (int i = 0; i < debugDerivH.size(); i++)
                std::cout << debugDerivH[i].t << ", ";
            std::cout << std::endl;
            */

            
           // -------------------------------------------------------------


            break;
        }

        case IntegrationScheme_csph::SYMPLECTIC: {
            Real dummy = 0;  // force calculation for compressible sph does not need the step size

            auto& y_tmp = m_data_mgr.sortedSphMarkers1_D;
            CopySortedMarkers(y, y_tmp);                     // y_tmp <- y_n

            forceSystem->ForceSPH(y, t, dummy);              // f(t_n, y_n). calls the method in FsiForce_csph, where stepSize is useless.

            EulerStep(y_tmp, h / 2);                         // y_tmp <== y_{n+1/2} = y_n + (h/2) * f(t_n, y_n)
            ApplyBoundaryConditions(y_tmp);

            if (m_data_mgr.paramsH->h_variation &&
                m_data_mgr.paramsH->midpoint_neigh_search) {  // if both true, update grid at midpoint
                MidpointNeighSearch(y, y_tmp);  // now both y, y_tmp sorted with new grid computed at t_(n+1/2)
            }

            forceSystem->ForceSPH(y_tmp, t + h / 2, dummy);  // f_{n+1/2} = f(t_n + h/2, y_{n+1/2})
            MidpointStep(y, h);                              // y_{n+1} = y_n + h * f_{n+1/2}

            break;
        }

    }
}


// -----------------------------------------------------------------------------
// UPDATE GRID PROPERTIES DURING MIDPOINT STEP WHEN H VARIES 
// -----------------------------------------------------------------------------


// given the sorted data obtained at the midpoint step, it updates the grid parameters and neighbor search
// vectors.
// For sorting it works with input vectors, neighbors list vectors manipulated are ones in FsiDataManager
void FluidDynamics_csph::MidpointNeighSearch(const std::shared_ptr<SphMarkerDataD_csph>& sortedInitialMarkersD,      // y or sortedSphMarkers2_D
                                             const std::shared_ptr<SphMarkerDataD_csph>& sortedMidpointMarkersD) {   // y_tmp or sortedSphMarkers1_D

    // copy sorted markers at both t_n and t_(n+1/2) back to original order
    // need a temporary struct
    std::shared_ptr<SphMarkerDataD_csph> dummyInitialMarkersD = chrono_types::make_shared<SphMarkerDataD_csph>();
    dummyInitialMarkersD->resize(sortedInitialMarkersD->posRadD.size());
    CopySortedToOriginal(MarkerGroup::ALL, sortedInitialMarkersD, dummyInitialMarkersD);           // dummyInitMark = original order, state at t_n
    CopySortedToOriginal(MarkerGroup::ALL, sortedMidpointMarkersD, m_data_mgr.sphMarkers_D);      // sphMarkers_D = original order, state at t_(n+1/2)
    
    // update grid parameters and neighbor list based on state at t_(n+1/2)
    ProximitySearch(m_data_mgr.sphMarkers_D,m_data_mgr.sortedSphMarkers1_D);                      // sortedSphMarkers1_D (y_tmp) = sorted on grid at t_(n+1/2), state at t_(n+1/2)
    // now sort the data at state t_n into hash order corresponding to midpoint step:
    CopyOriginalToSorted(MarkerGroup::ALL, dummyInitialMarkersD, m_data_mgr.sortedSphMarkers2_D);  // sortedSphMarkers2_D (y) = sorted on grid at t_(n+1/2), state at t_n
    
}

// -----------------------------------------------------------------------------
// ACTIVITY UPDATE SECTION
// -----------------------------------------------------------------------------

// Actual Cuda kernel that performs the activity update: sph particles in active domain are marked as active.
// Works with numAllMArkers by default. Function that populates with 1,0,-1 the activityIdentifierD and
// extendedActivityIdD vectors, which have length of numAllMarkers.
// Markerd as 1 if active wrt at least one FSI body, 0 if inactive wrt all solid bodies, -1 if outside the whole fluid computational domain.
// Vectors in original order, not sorted by hash.
// Looks that this kernel consideres activity of a particle only with respect to interaction wrt a solid body.
// It should not mean that its fluid dynamic evolution is suspended entirely. CHECK WITH SUBSEQUENT KERNELS.
// In fact, in the FSIFORCE class, the kernels that update the RHS are called with numExtendedMarkers.
__global__ void UpdateActivityD(const Real4* posRadD,
                                Real3* velD,
                                const Real3* pos_bodies_D,
                                int32_t* activityIdentifierD,
                                int32_t* extendedActivityIdD,
                                const Real4* rhoPresEnD) {      // contains also particle type.
                                            

    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // particle index
    if (index >= countersD_csph.numAllMarkers) {
        return;
    }

    // Set the particle as an active particle
    activityIdentifierD[index] = 1;
    extendedActivityIdD[index] = 1;
    Real3 domainDims = paramsD_csph.boxDims;
    Real3 domainOrigin = paramsD_csph.worldOrigin;
    bool x_periodic = paramsD_csph.x_periodic;
    bool y_periodic = paramsD_csph.y_periodic;
    bool z_periodic = paramsD_csph.z_periodic;

    Real3 posA = mR3(posRadD[index]);
    Real h_eff = rmaxr(paramsD_csph.h, posRadD[index].w);     // conservative approach: use maximum value of h to define the buffer zone.
    
    size_t numFsiBodies = countersD_csph.numFsiBodies;
    size_t numTotal = numFsiBodies;

    // Check the activity of this particle
    uint isNotActive = 0;  // these are counters for each particle. They count how many bodies the particle can be
                           // considered inactive towards.
    uint isNotExtended = 0;
    Real3 Acdomain = paramsD_csph.bodyActiveDomain;  // parameter defined as domain influenced by an FsiBody. Appears to
                                                     // be unique even with multiple bodies in the system.
        
    // REVISIT. h_multiplier*h = kernel radius. depends on actual kernel function.
    // CORRECT TO USE INITIAL VALUE OF H HERE?
    Real3 ExAcdomain = paramsD_csph.bodyActiveDomain +
                           mR3(2 * paramsD_csph.h_multiplier * h_eff);  // active domain of bodies + buffer region

    // check activity of particle wrt bodies in the system based on their distance.
    // First rigid bodies.
    for (uint num = 0; num < numFsiBodies; num++) {
        Real3 detPos = posA - pos_bodies_D[num];        // distance particle-body
        if (abs(detPos.x) > Acdomain.x || abs(detPos.y) > Acdomain.y || abs(detPos.z) > Acdomain.z)  // particle outside body interaction domain
            isNotActive = isNotActive + 1;
        if (abs(detPos.x) > ExAcdomain.x || abs(detPos.y) > ExAcdomain.y || abs(detPos.z) > ExAcdomain.z)          // particle also outside buffered domain
            isNotExtended = isNotExtended + 1;
    }
        
    // If particle is inactive wrt all fsi bodies in the system then marked as inactive and its velocity set to
    // zero.
    if (isNotActive == numTotal && numTotal > 0) {
        activityIdentifierD[index] = 0;
        velD[index] = mR3(0.0);
    }
    if (isNotExtended == numTotal && numTotal > 0)          // If particle is inactive wrt all bodies even considering the buffered domain
        extendedActivityIdD[index] = 0;

    // Check if the particle is outside the zombie domain.
    // zombie domain meaning outside computational domain. If such the particle is marked as "zombie" (identifier = -1)
    // but is not cancelled from memory. So active particles = 1, inactive particles (wrt solids) in domain = 0,
    // particles in the zombie domain = -1.
    if (IsFluidParticle(rhoPresEnD[index].w)) {  // for fluid particles only
        bool outside_domain = false;

        // the domainOrigin is defined as one of the corners of the computational domain.
        // Check X boundaries - only inactivate if not periodic
        if (!x_periodic && (posA.x < domainOrigin.x || posA.x > domainOrigin.x + domainDims.x)) {
            outside_domain = true;
        }

        // Check Y boundaries - only inactivate if not periodic
        if (!y_periodic && (posA.y < domainOrigin.y || posA.y > domainOrigin.y + domainDims.y)) {
            outside_domain = true;
        }

        // Check Z boundaries - only inactivate if not periodic
        if (!z_periodic && (posA.z < domainOrigin.z || posA.z > domainOrigin.z + domainDims.z)) {
            outside_domain = true;
        }

        if (outside_domain) {
            activityIdentifierD[index] = -1;  // If particle outside a domain direction that is not periodic, marked as zombie particle with -1
            extendedActivityIdD[index] = -1;
            velD[index] = mR3(0.0);           // The particle is "frozen".
        }
    }

    return;
}


// Method that recomputes the vectors of activity identifier for each particle. Works with vectors not in hash order.
void FluidDynamics_csph::UpdateActivity(std::shared_ptr<SphMarkerDataD_csph> sphMarkersD) {
    uint numBlocks, numThreads;
    computeCudaGridSize((uint)m_data_mgr.countersH->numAllMarkers, 1024, numBlocks, numThreads);

    UpdateActivityD<<<numBlocks, numThreads>>>(
        mR4CAST(sphMarkersD->posRadD), mR3CAST(sphMarkersD->velD), mR3CAST(m_data_mgr.fsiBodyState_D->pos),
        INT_32CAST(m_data_mgr.activityIdentifierOriginalD), INT_32CAST(m_data_mgr.extendedActivityIdentifierOriginalD),
        mR4CAST(sphMarkersD->rhoPresEnD));

}



// ------------------------------------------------------------
// CHECK ACTIVITY FLAGS AND RESIZE ARRAYS
// ------------------------------------------------------------


// Resize data based on the active particles
// Custom functor for exclusive scan that treats -1 (zombie particles) the same as 0 (sleep particles)
struct ActivityScanOp {
    __host__ __device__ int operator()(const int& a, const int& b) const {
        // Treat -1 the same as 0 (only add positive values)
        int b_value = (b <= 0) ? 0 : b;
        return a + b_value;
    }
};


// methods that checks if arrays must be resized due to change in particle activity. Does not call custom cuda kernels
// like updateActivityD
bool FluidDynamics_csph::CheckActivityArrayResize() {
    auto& countersH = m_data_mgr.countersH;

    // Exclusive scan for extended activity identifier using custom functor to handle -1 values as if they were 0 values
    thrust::exclusive_scan(thrust::device,                                          // execution policy
                           m_data_mgr.extendedActivityIdentifierOriginalD.begin(),  // in start
                           m_data_mgr.extendedActivityIdentifierOriginalD.end(),    // in end
                           m_data_mgr.prefixSumExtendedActivityIdD.begin(),         // out start
                           0,                                                       // initial value
                           ActivityScanOp());

    // Copy the last element of prefixSumD to host and since we used exclusive scan, need to add the last flag (the last
    // element flag has not been added to the prefix sum)
    // Need cudaMemcpy or thrust::copy to copy a device value into host variable.
    uint lastPrefixVal;
    cudaMemcpy(&lastPrefixVal,
               thrust::raw_pointer_cast(&m_data_mgr.prefixSumExtendedActivityIdD[countersH->numAllMarkers - 1]),
               sizeof(uint), cudaMemcpyDeviceToHost);
    
    int32_t lastFlagInt32;                                                      // host variable here.
    cudaMemcpy(&lastFlagInt32,
               thrust::raw_pointer_cast(&m_data_mgr.extendedActivityIdentifierOriginalD[countersH->numAllMarkers - 1]),
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    uint lastFlag = (lastFlagInt32 > 0) ? 1 : 0;  // Only need to add last value if different from 0

    countersH->numExtendedParticles = lastPrefixVal + lastFlag;  // add the last flag. They can be safely summed as they are both unsigned int.
                                                                 

    return countersH->numExtendedParticles < countersH->numAllMarkers;
}


// ---------------------------------------------------------------------------------
// Definition of some helper functions executed on the device, used later in private
// methods.
// They compute the Euler and Midpoint steps for position, velocity and density.
// They take references as input, so they modify the input values.
// ----------------------------------------------------------------------------------



// Position Euler Step is: pos{n+1} = pos{n} + dt*vel{n}
// Updates the position alone, preserving the kernel radius value
// In case of XSPH the input velocity will be the vel + vel_XSPH
__device__ void PositionEulerStep(Real dT, const Real3& vel, Real4& pos) {
    Real3 p = mR3(pos);
    p += dT * vel;
    pos = mR4(p, pos.w);
}


// velocity Euler step is: vel{n+1} = vel{n} + dt*acc{n}
// So here assume acc is computed at step n.
__device__ void VelocityEulerStep(Real dT, const Real3& acc, Real3& vel) {
    vel += dT * acc;
}


// Density Euler step is: rho{n+1} = rho{n} + dt*deriv{n}
// So here assume deriv computed at step n
// It updates the density after computing density. Correct placement?
__device__ void DensityEulerStep(Real dT, const Real& deriv, Real4& rhoPresEn) {
    rhoPresEn.x += dT * deriv;        // update density (not with XSPH corrected velocities)
}


// Energy Euler step is: en{n+1} = en{n} + dt*deriv{n}
// So here assumed deriv computed at step n
__device__ void EnergyEulerStep(Real dT, const Real& deriv, Real4& rhoPresEn) {
    rhoPresEn.z += dT * deriv;    // update energy
}

// Kernel radius Euler step is: h{n+1} = h{n} + dt*deriv{n}
__device__ void RadEulerStep(Real dT, const Real& deriv, Real4& posRad) {
    posRad.w += dT * deriv; // update kernel radius
}



// The mid-point updates for position and velocity are:
//    v_{n+1} = v_n + h * F_{n+1/2}
//    r_{n+1} = r_n + h * (v_{n+1} + v_n) / 2
// These are implemented in reverse order (because the velocity update would overwrite v_n) as:
//    r_{n+1} = r_n + h * v_n + 0.5 * h^2 * F_{n+1/2}
//    v_{n+1} = v_n + h * F_{n+1/2}
// So Position Midpoint step is: pos{n+1} = pos{n} + dt*vel{n} + 0.5*dt^2*acc{n+1/2}
// Thus we assume here that input pos computed at n, input vel computed at n, input acc computed at n+1/2
__device__ void PositionMidpointStep(Real dT, const Real3& vel, const Real3& acc, Real4& pos) {
    Real3 p = mR3(pos);
    p += dT * vel + 0.5 * dT * dT * acc;
    pos = mR4(p, pos.w);
}



// Kernel to update the fluid properities of a particle, using an explicit Euler step.
// First, update the particle position and velocity. Next,
// - For a CFD problem, advance density (if not from summation), energy, kernel radius (if from derivative)
// - calculate pressure, temperature, speed of sound from the Equation of State;
//
// Important note: the derivVelRhoD calculated by ChForceExplicitSPH is the negative of actual time
// derivative. That is important to keep the derivVelRhoD to be the force/mass for fsi forces.
// - calculate the force, that is f=m dv/dt
// - derivVelRhoD[index] *= paramsD_csph.markerMass;
// Here derivatives assumed computed at time-step n
__global__ void EulerStep_D(
    Real4* posRadD,       // input position at timestep n, will be rewritten
    Real3* velD,          // input velocity at timestep n, will be rewritten
    Real4* rhoPresEnD,    // input rho, P, at timestep n will be rewritten
    Real* soundD,          // input speed of sound, updated after integration
    const Real3* vel_XSPH_D,      // delta_vel for XSPH. Applied only to position update. Computed at timestep n.
    const Real5* derivVelRhoEnD,  // Derivatives DvDt and DrhoDt, computed by external calls to forceSystem. Assumed
                                  // computed at timestep n
    const Real*  derivRadD,       // Derivative of kernel radius, used only if evolved differentially.
    const uint* freeSurfaceIdD,
    const int32_t* activityIdentifierSortedD,
    const uint numActive,
    Real dT) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    // Only update active SPH particles, not extended active particles or bce markers. IsBceMarker is true for code >= 0
    if (IsBceMarker(rhoPresEnD[index].w) || activityIdentifierSortedD[index] <= 0)
        return;

    // Euler step for position. (pos_new = pos_old + vel(state_old,t) * dt)
    // Notice position is updated with the XSPH corrected velocity.
    PositionEulerStep(dT, velD[index] + vel_XSPH_D[index], posRadD[index]);

    // Euler step for velocity. (vel_new = vel_old + derivVel(state_old,t) * dt)
    VelocityEulerStep(dT, mR3(derivVelRhoEnD[index]), velD[index]);

    // Euler step for density.
    if (paramsD_csph.rho_evolution == Rho_evolution_csph::DIFFERENTIAL) {
        DensityEulerStep(dT, derivVelRhoEnD[index].w, rhoPresEnD[index]);
    }
    
    // Euler step for energy
    EnergyEulerStep(dT, derivVelRhoEnD[index].t, rhoPresEnD[index]);

    // Euler step for kernel radius h
    if (paramsD_csph.h_evolution == H_evolution_csph::DIFFERENTIAL) {
        RadEulerStep(dT, derivRadD[index], posRadD[index]);
    }

    // Update pressure, speed of sound
    rhoPresEnD[index].y = Eos_csph(rhoPresEnD[index].x, rhoPresEnD[index].z, EosType_csph::IDEAL_RHOEN);
    soundD[index] = SoundFromPresRho(rhoPresEnD[index].y, rhoPresEnD[index].x);

}



// Kernel to update the fluid properities of a particle, using an mid-point step.
// Note: the derivatives (provided in input vectors) are assumed to have been calculated at the mid-point!
// The mid-point updates for position and velocity are:
//    v_{n+1} = v_n + h * F_{n+1/2}                 --> equivalent to Euler step with accelerations computed at step n+1/2
//    r_{n+1} = r_n + h * (v_{n+1} + v_n) / 2
// These are implemented in reverse order (because the velocity update would overwrite v_n) as:
//    r_{n+1} = r_n + h * v_n + 0.5 * h^2 * F_{n+1/2}
//    v_{n+1} = v_n + h * F_{n+1/2}
// After the position and velocity updates, the mid-point update for density and energy (CFD) are equivalent to the
// velocity update above (i.e., an Euler step but with the acceleration computed at timestep n+1/2).
__global__ void MidpointStep_D(
    Real4* posRadD,           // input pos at timestep n, will be rewritten
    Real3* velD,              // input vel at timestep n, will be rewritten
    Real4* rhoPresEnD,        // input rho at timestep n, will be rewritten
    Real*  soundD,
    const Real3* vel_XSPH_D,      // delta_vel for XSPH, applied to position only, computed at timestep n
    const Real5* derivVelRhoEnD,  // input derivative of velocity, density and energy assumed computed at timestep n+1/2
    const Real*  derivRadD,
    const uint* freeSurfaceIdD,
    const int32_t* activityIdentifierSortedD,
    const uint numActive,
    Real dT) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    // Only update active SPH particles, not extended active particles
    if (IsBceMarker(rhoPresEnD[index].w) || activityIdentifierSortedD[index] <= 0)
        return;

    // Advance position using midpoint equation. input pos at timestep n, input corrected velocity at timestep n,
    // derivative of velocity at timestep n+1/2
    PositionMidpointStep(dT, velD[index] + vel_XSPH_D[index], mR3(derivVelRhoEnD[index]), posRadD[index]);

    // Advance velocity with simple Euler step, just the acc/derivative is computed at timestep n+1/2
    VelocityEulerStep(dT, mR3(derivVelRhoEnD[index]), velD[index]);

    // Euler step for density
    if (paramsD_csph.rho_evolution == Rho_evolution_csph::DIFFERENTIAL) {
        // Euler step for tau and pressure update
        DensityEulerStep(dT, derivVelRhoEnD[index].w, rhoPresEnD[index]);
    }
    
    // Euler step for energy
    EnergyEulerStep(dT, derivVelRhoEnD[index].t, rhoPresEnD[index]);

    // Euler step for kernel radius h
    if (paramsD_csph.h_evolution == H_evolution_csph::DIFFERENTIAL) {
        RadEulerStep(dT, derivRadD[index], posRadD[index]);
    }
    
    // Update pressure, temperature, speed of sound
    rhoPresEnD[index].y = Eos_csph(rhoPresEnD[index].x, rhoPresEnD[index].z, EosType_csph::IDEAL_RHOEN);
    soundD[index] = SoundFromPresRho(rhoPresEnD[index].y, rhoPresEnD[index].x);
    
}


template <typename T>
struct check_infinite {
    __host__ __device__ bool operator()(const T& v) { return !IsFinite(v); }
};



//-------------------------------------------------------------------------
// Actual class methods (these are private ones, used by DoStepDynamics)
// that use the Cuda kernels defined above
//--------------------------------------------------------------------------


// simple Euler step for position, velocity and pressure.
// Derivatives in derivVelRhoD computed at timestep n
void FluidDynamics_csph::EulerStep(std::shared_ptr<SphMarkerDataD_csph> sortedMarkers, Real dT) {
    uint numActive = (uint)m_data_mgr.countersH->numExtendedParticles;
    uint numBlocks, numThreads;
    computeCudaGridSize(numActive, 256, numBlocks, numThreads);

    EulerStep_D<<<numBlocks, numThreads>>>(
        mR4CAST(sortedMarkers->posRadD), mR3CAST(sortedMarkers->velD), mR4CAST(sortedMarkers->rhoPresEnD),
        R1CAST(sortedMarkers->soundD), mR3CAST(m_data_mgr.vel_XSPH_D),
        mR5CAST(m_data_mgr.derivVelRhoEnD), R1CAST(m_data_mgr.derivRadD),
        U1CAST(m_data_mgr.freeSurfaceIdD), INT_32CAST(m_data_mgr.activityIdentifierSortedD), numActive, dT);


    if (m_check_errors) {
        cudaCheckError();
        if (thrust::any_of(sortedMarkers->posRadD.begin(), sortedMarkers->posRadD.end(), check_infinite<Real4>()))
            cudaThrowError("A particle position is NaN");
        if (thrust::any_of(sortedMarkers->rhoPresEnD.begin(), sortedMarkers->rhoPresEnD.end(), check_infinite<Real4>()))
            cudaThrowError("A particle density is NaN");
    }
}

// Midpoint step update. Position computed with midpoint formula first, then simple Euler steps for velocity and
// pressure. Derivatives in derivVelRhoD assumed computed externally at timestep n+1/2
void FluidDynamics_csph::MidpointStep(std::shared_ptr<SphMarkerDataD_csph> sortedMarkers, Real dT) {
    uint numActive = (uint)m_data_mgr.countersH->numExtendedParticles;
    uint numBlocks, numThreads;
    computeCudaGridSize(numActive, 256, numBlocks, numThreads);

    MidpointStep_D<<<numBlocks, numThreads>>>(
        mR4CAST(sortedMarkers->posRadD), mR3CAST(sortedMarkers->velD), mR4CAST(sortedMarkers->rhoPresEnD),
        R1CAST(sortedMarkers->soundD), mR3CAST(m_data_mgr.vel_XSPH_D),
        mR5CAST(m_data_mgr.derivVelRhoEnD), R1CAST(m_data_mgr.derivRadD),
        U1CAST(m_data_mgr.freeSurfaceIdD), INT_32CAST(m_data_mgr.activityIdentifierSortedD), numActive, dT);
    

    if (m_check_errors) {
        cudaCheckError();
        if (thrust::any_of(sortedMarkers->posRadD.begin(), sortedMarkers->posRadD.end(), check_infinite<Real4>()))
            cudaThrowError("A particle position is NaN");
        if (thrust::any_of(sortedMarkers->rhoPresEnD.begin(), sortedMarkers->rhoPresEnD.end(), check_infinite<Real4>()))
            cudaThrowError("A particle density is NaN");
    }
}

// -----------------------------------------------------------------------------------------
// Copy sorted data (marker states and derivatives in DataManager) back into original order
// -----------------------------------------------------------------------------------------

// Kernel to copy sorted data back to original order (CSPH)
__global__ void CopySortedToOriginal_csphD(MarkerGroup group,
                                            const Real4* sortedPosRad,
                                            const Real3* sortedVel,
                                            const Real4* sortedRhoPresEn,
                                            const Real*  sortedSound,
                                            const Real5* derivVelRhoEn,
                                            const Real*  derivRad,
                                            const uint numActive,
                                            Real4* posRadOriginal,
                                            Real3* velOriginal,
                                            Real4* rhoPresEnOriginal,
                                            Real*  soundOriginal,
                                            Real5* derivVelRhoEnOriginal,
                                            Real*  derivRadOriginal,
                                            uint* gridMarkerIndex) {

    uint id = blockIdx.x * blockDim.x + threadIdx.x;  // thread index
    if (id >= numActive)
        return;

    Real type = sortedRhoPresEn[id].w;
    if (!IsInMarkerGroup(group, type))  // If particle type doesn't match requested group
        return;

    uint index = gridMarkerIndex[id];          // get mapping sorted to original, using thread index as sorted particle index
    posRadOriginal[index] = sortedPosRad[id];  // copy data. "index" is the original index corresponding to the sorted index "id"
    velOriginal[index] = sortedVel[id];
    rhoPresEnOriginal[index] = sortedRhoPresEn[id];
    soundOriginal[index] = sortedSound[id];
    derivVelRhoEnOriginal[index] = derivVelRhoEn[id];
    if (paramsD_csph.h_evolution == H_evolution_csph::DIFFERENTIAL) {
        derivRadOriginal[index] = derivRad[id];
    }
    
}


// member function that actually uses the Cuda kernel above
void FluidDynamics_csph::CopySortedToOriginal(MarkerGroup group,
                                         std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD,
                                         std::shared_ptr<SphMarkerDataD_csph> sphMarkersD) {

    uint numActive = (uint)m_data_mgr.countersH->numExtendedParticles;
    uint numBlocks, numThreads;
    computeCudaGridSize(numActive, 1024, numBlocks, numThreads);

    CopySortedToOriginal_csphD<<<numBlocks, numThreads>>>(
            group, mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velD),
            mR4CAST(sortedSphMarkersD->rhoPresEnD), R1CAST(sortedSphMarkersD->soundD),
            mR5CAST(m_data_mgr.derivVelRhoEnD), R1CAST(m_data_mgr.derivRadD), numActive,
            mR4CAST(sphMarkersD->posRadD), mR3CAST(sphMarkersD->velD), mR4CAST(sphMarkersD->rhoPresEnD),
            R1CAST(sphMarkersD->soundD), mR5CAST(m_data_mgr.derivVelRhoEnOriginalD), 
            R1CAST(m_data_mgr.derivRadOriginalD), U1CAST(m_data_mgr.markersProximity_D->gridMarkerIndexD));
    

    if (m_check_errors) {
        cudaCheckError();
    }


    
    
}

// -----------------------------------------------------------------------------------
// Copy original order data and derivatives into sorted order 
// -----------------------------------------------------------------------------------


// kernel to copy original order arrays into sorted order:
__global__ void OriginalToSorted_csphD(MarkerGroup group,
                                       const Real4* __restrict__ posRad,
                                       const Real3* __restrict__ vel,
                                       const Real4* __restrict__ rhoPresEn,
                                       const Real* __restrict__  sound,
                                       const Real5* __restrict__ derivVelRhoEn,
                                       const Real* __restrict__  derivRad,
                                       const uint numActive,
                                       Real4* __restrict__ sortedPosRad,
                                       Real3* __restrict__ sortedVel,
                                       Real4* __restrict__ sortedRhoPresEn,
                                       Real* __restrict__  sortedSound,
                                       Real5* __restrict__ sortedDerivVelRhoEn,
                                       Real*  __restrict__ sortedDerivRad,
                                       const uint* __restrict__ gridMarkerIndex) {

    uint id = threadIdx.x + blockIdx.x * blockDim.x;  // thread index
    if (id >= numActive)
        return;

    uint originalIndex = gridMarkerIndex[id];   // original index position corresponding to sorted index being thread id
    Real type = rhoPresEn[originalIndex].w;
    if (!IsInMarkerGroup(group, type))
        return;

    // set values of sorted arrays:
    sortedPosRad[id] = posRad[originalIndex];
    sortedRhoPresEn[id] = rhoPresEn[originalIndex];
    sortedVel[id] = vel[originalIndex];
    sortedSound[id] = sound[originalIndex];
    sortedDerivVelRhoEn[id] = derivVelRhoEn[originalIndex];
    if (paramsD_csph.h_evolution == H_evolution_csph::DIFFERENTIAL)
        sortedDerivRad[id] = derivRad[originalIndex];

}

// actual method
void FluidDynamics_csph::CopyOriginalToSorted(MarkerGroup group, std::shared_ptr<SphMarkerDataD_csph> originalSphMarkersD,
                                              std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {
    uint numActive = (uint)m_data_mgr.countersH->numExtendedParticles;
    uint numBlocks, numThreads;
    computeCudaGridSize(numActive, 1024, numBlocks, numThreads);

    OriginalToSorted_csphD<<<numBlocks, numThreads>>>(
        group, mR4CAST(originalSphMarkersD->posRadD), mR3CAST(originalSphMarkersD->velD),
        mR4CAST(originalSphMarkersD->rhoPresEnD), R1CAST(originalSphMarkersD->soundD),
        mR5CAST(m_data_mgr.derivVelRhoEnOriginalD), R1CAST(m_data_mgr.derivRadOriginalD), numActive, mR4CAST(sortedSphMarkersD->posRadD),
        mR3CAST(sortedSphMarkersD->velD), mR4CAST(sortedSphMarkersD->rhoPresEnD), R1CAST(sortedSphMarkersD->soundD),
        mR5CAST(m_data_mgr.derivVelRhoEnD), R1CAST(m_data_mgr.derivRadD),
        U1CAST(m_data_mgr.markersProximity_D->gridMarkerIndexD));
    
    if (m_check_errors) {
        cudaCheckError();
    }
}
    // ------------------------------------------------------
// Inlet and Periodic Boundary Conditions
// ------------------------------------------------------



// REVIEW THIS INLET BOUNDARY CONDITION
// THE ORIGINAL IMPLEMENTATION STILL DO NOT USE IT WHEN APPLYING THE INLET BOUNDARY CONDITIONS.
// Kernel to apply inlet/outlet BC along x. This implements a forced periodicity with given pressure gradient: like
// simulating an infinite tube along x with increasing (Sure? Looks decreasing) pressure. Not implemented also along y and x.
__global__ void ApplyInletBoundaryX_D(Real4* posRadD, Real3* VelD, Real4* rhoPresEnD, const uint numActive) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    Real4 rhoPresEn = rhoPresEnD[index];  // use thread index as particle index here

    // no need to do anything if it is a BCE marker
    if (IsBceMarker(rhoPresEn.w))
        return;

    Real3 posRad = mR3(posRadD[index]);
    Real h = posRadD[index].w;


    // following conditions are exclusive: a particle can't be at same time in both
    if (posRad.x > paramsD_csph.cMax.x) {  // if x position above upper limit we shift it backwards by 1 time the length of the fluid domain
        posRad.x -= (paramsD_csph.cMax.x - paramsD_csph.cMin.x);
        posRadD[index] = mR4(posRad, h);                            // update position
        rhoPresEn.y = rhoPresEn.y + paramsD_csph.delta_pressure.x;  // also add the delta pressure defined along x
        rhoPresEnD[index] = rhoPresEn;                              // update pressure
    }
    if (posRad.x < paramsD_csph.cMin.x) {  // if x position behind lower limit of the fluid domain we shift it upwards by 1
                                           // time the domain length
        posRad.x += (paramsD_csph.cMax.x - paramsD_csph.cMin.x);
        posRadD[index] = mR4(posRad, h);                   // update position
        VelD[index] = mR3(paramsD_csph.V_in.x, 0, 0);  // update velocity to be the specified inlet velocity
        rhoPresEn.y = rhoPresEn.y - paramsD_csph.delta_pressure.x;    // also update pressure by subtracting the parameter delta pressure
        rhoPresEnD[index] = rhoPresEn;
    }

    // following condition may seem to be setting to zero the pressure for all particles.
    // CHECK IT BETTER
    if (posRad.x > -paramsD_csph.x_in)
        rhoPresEnD[index].y = 0;

    if (posRad.x < paramsD_csph.x_in)  // If particle below the inlet, it's velocity set to inlet one
        VelD[index] = mR3(paramsD_csph.V_in.x, 0, 0);
}



// --------------------------------------------
// Kernel to apply periodic BC along x
// --------------------------------------------

__global__ void ApplyPeriodicBoundaryX_D(Real4* posRadD, Real4* rhoPresEnD, const uint numActive) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // use thread index as particle index
    if (index >= numActive)
        return;

    Real4 rhoPresEn = rhoPresEnD[index];
    // no need to do anything if it is a BCE marker
    if (IsBceMarker(rhoPresEn.w))
        return;

    Real3 posRad = mR3(posRadD[index]);
    Real h = posRadD[index].w;

    if (posRad.x > paramsD_csph.cMax.x) {                         // position above upper limit of fluid domain
        posRad.x -= (paramsD_csph.cMax.x - paramsD_csph.cMin.x);  // particle position shifted back
        posRadD[index] = mR4(posRad, h);
        rhoPresEnD[index].y += paramsD_csph.delta_pressure.x;     // consider the defined delta_pressure
        return;
    }
    if (posRad.x < paramsD_csph.cMin.x) {                     // position below lower fluid domain limit
        posRad.x += (paramsD_csph.cMax.x - paramsD_csph.cMin.x);  // particle position shifted upwards
        posRadD[index] = mR4(posRad, h);
        rhoPresEnD[index].y -= paramsD_csph.delta_pressure.x;     // decrease by delta_pressure
        return;
    }
}



// Kernel to apply periodic BC along y, same as along x
__global__ void ApplyPeriodicBoundaryY_D(Real4* posRadD, Real4* rhoPresEnD, const uint numActive) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    Real4 rhoPresEn = rhoPresEnD[index];
    // no need to do anything if it is a BCE marker
    if (IsBceMarker(rhoPresEn.w))
        return;

    Real3 posRad = mR3(posRadD[index]);
    Real h = posRadD[index].w;

    if (posRad.y > paramsD_csph.cMax.y) {
        posRad.y -= (paramsD_csph.cMax.y - paramsD_csph.cMin.y);
        posRadD[index] = mR4(posRad, h);
        rhoPresEn.y = rhoPresEn.y + paramsD_csph.delta_pressure.y;
        rhoPresEnD[index] = rhoPresEn;
        return;
    }
    if (posRad.y < paramsD_csph.cMin.y) {
        posRad.y += (paramsD_csph.cMax.y - paramsD_csph.cMin.y);
        posRadD[index] = mR4(posRad, h);
        rhoPresEn.y = rhoPresEn.y - paramsD_csph.delta_pressure.y;
        rhoPresEnD[index] = rhoPresEn;
        return;
    }
}

// Kernel to apply periodic BC along z, same as above
__global__ void ApplyPeriodicBoundaryZ_D(Real4* posRadD, Real4* rhoPresEnD, const uint numActive) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    Real4 rhoPresEn = rhoPresEnD[index];
    // no need to do anything if it is a BCE marker
    if (IsBceMarker(rhoPresEn.w))
        return;

    Real3 posRad = mR3(posRadD[index]);
    Real h = posRadD[index].w;

    if (posRad.z > paramsD_csph.cMax.z) {
        posRad.z -= (paramsD_csph.cMax.z - paramsD_csph.cMin.z);
        posRadD[index] = mR4(posRad, h);
        rhoPresEn.y = rhoPresEn.y + paramsD_csph.delta_pressure.z;
        rhoPresEnD[index] = rhoPresEn;
        return;
    }
    if (posRad.z < paramsD_csph.cMin.z) {
        posRad.z += (paramsD_csph.cMax.z - paramsD_csph.cMin.z);
        posRadD[index] = mR4(posRad, h);
        rhoPresEn.y = rhoPresEn.y - paramsD_csph.delta_pressure.z;
        rhoPresEnD[index] = rhoPresEn;
        return;
    }
}


// Private class methods that uses Cuda kernels to apply periodic boundary conditions in x, y, and z directions
// Either periodic or inlet conditions
void FluidDynamics_csph::ApplyBoundaryConditions(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {

    uint numActive = (uint)m_data_mgr.countersH->numExtendedParticles;
    uint numBlocks, numThreads;
    computeCudaGridSize(numActive, 1024, numBlocks, numThreads);

    switch (m_data_mgr.paramsH->bc_type.x) {
        case BCType::PERIODIC:
            ApplyPeriodicBoundaryX_D<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD),
                                                                mR4CAST(sortedSphMarkersD->rhoPresEnD), numActive);
            
            if (m_check_errors) {
                cudaCheckError();
            }
            break;
        case BCType::INLET_OUTLET:
            //// TODO - check this and modify as appropriate
            // ApplyInletBoundaryX_D<<<numBlocks, numThreads>>>(mR4CAST(sphMarkersD->posRadD),
            //                                                  mR3CAST(sphMarkersD->velMasD),
            //                                                  mR4CAST(sphMarkersD->rhoPresMuD), numActive);
            // cudaCheckError();
            break;
    }
    // inlet bcs not implemented also for y and z directions
    switch (m_data_mgr.paramsH->bc_type.y) {
        case BCType::PERIODIC:
            ApplyPeriodicBoundaryY_D<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD),
                                                                mR4CAST(sortedSphMarkersD->rhoPresEnD), numActive);
            if (m_check_errors) {
                cudaCheckError();
            }
            break;
    }

    switch (m_data_mgr.paramsH->bc_type.z) {
        case BCType::PERIODIC:
            ApplyPeriodicBoundaryZ_D<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD),
                                                                mR4CAST(sortedSphMarkersD->rhoPresEnD), numActive);
            if (m_check_errors) {
                cudaCheckError();
            }
            break;
    }
}



}  // end namespace compressible
}  // end namespace chrono::fsi::sph