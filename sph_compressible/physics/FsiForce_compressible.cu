// =============================================================================
// Author: Andrea D'Uva
// 
// Derived class to compute inter particles forces depending on the specified problem 
// =============================================================================

#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include "chrono_fsi/sph/physics/FsiForceWCSPH.cuh"
#include "chrono_fsi/sph/physics/SphGeneral.cuh"
#include "chrono_fsi/sph/math/ExactLinearSolvers.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiForce_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"


namespace chrono::fsi::sph {
namespace compressible {

// =============================================================================

// device function to correct the pressure of each marker depending on a distance parameter dist3Alpha.
// If such distance is greater than half the box dimension, then the pressure will be corercted by summing or
// subtracting a delta_pressure parameter. Since delta_pressure in ParamsSPH is defined as a change in pressure for
// periodic bc, this is a correction in pressure for particles that are outside the boundary if the periodic bc are
// enforced
__device__ __inline__ void modifyPressure(Real4& rhoPresEnB, const Real3& dist3Alpha) {
    // body force in x direction
    rhoPresEnB.y = (dist3Alpha.x > 0.5 * paramsD_csph.boxDims.x) ? (rhoPresEnB.y - paramsD_csph.delta_pressure.x) : rhoPresEnB.y;
    rhoPresEnB.y = (dist3Alpha.x < -0.5 * paramsD_csph.boxDims.x) ? (rhoPresEnB.y + paramsD_csph.delta_pressure.x) : rhoPresEnB.y;
    // body force in y direction
    rhoPresEnB.y = (dist3Alpha.y > 0.5 * paramsD_csph.boxDims.y) ? (rhoPresEnB.y - paramsD_csph.delta_pressure.y) : rhoPresEnB.y;
    rhoPresEnB.y = (dist3Alpha.y < -0.5 * paramsD_csph.boxDims.y) ? (rhoPresEnB.y + paramsD_csph.delta_pressure.y) : rhoPresEnB.y;
    // body force in z direction
    rhoPresEnB.y = (dist3Alpha.z > 0.5 * paramsD_csph.boxDims.z) ? (rhoPresEnB.y - paramsD_csph.delta_pressure.z) : rhoPresEnB.y;
    rhoPresEnB.y = (dist3Alpha.z < -0.5 * paramsD_csph.boxDims.z) ? (rhoPresEnB.y + paramsD_csph.delta_pressure.z) : rhoPresEnB.y;
}


// =============================================================================
// Constructor
FsiForce_csph::FsiForce_csph(FsiDataManager_csph& data_mgr, BceManager_csph& bce_mgr, bool verbose, bool check_errors)
    : FsiForce_base(data_mgr, bce_mgr, verbose), m_check_errors(check_errors) {
    CopyParametersToDevice_csph(m_data_mgr.paramsH, m_data_mgr.countersH);
    density_initialization = 0;
}


FsiForce_csph::~FsiForce_csph() {}


void FsiForce_csph::Initialize() {
    FsiForce_base::Initialize();
    cudaMemcpyToSymbolAsync(paramsD_csph, m_data_mgr.paramsH.get(), sizeof(ChFsiParamsSPH_csph));
    cudaMemcpyToSymbolAsync(countersD_csph, m_data_mgr.countersH.get(), sizeof(countersD_csph));
}


// Public method that computes interparticle sph forces. "public" because base class has it public
// even if here private.
// No need to actually modify this methods, but change the subroutines instead.
void FsiForce_csph::ForceSPH(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD, Real time, Real step) {
    
    // Calculate CUDA execution configuration
    // All kernels in FsiForce_csph work on a total of numExtendedParticles threads, in blocks of size 1024 (or 256)
    numActive = (uint)m_data_mgr.countersH->numExtendedParticles;
    computeCudaGridSize(numActive, 1024, numBlocks, numThreads);  // computes an optimal block size for the number of total threads = numExtendedMarkers.

    //
    m_bce_mgr.updateBCEAcc();  // BceManager method that computes the acceleration of bce markers from the rigid body
                               // dynamics

    // Perform density re-initialization depending on problem parameters:
    if (m_data_mgr.paramsH->density_reinit_switch && 
        m_data_mgr.paramsH->rho_evolution == Rho_evolution_csph::DIFFERENTIAL) {

        if (density_initialization >= m_data_mgr.paramsH->density_reinit_steps) {
            density_reinitialization(sortedSphMarkersD);          // calls private method
            density_initialization = 0 ;                         // reset the counter
        }
        density_initialization++;   // update the counter at each time step
    }

#ifndef NDEBUG
    /*
    thrust::host_vector<Real4> debugPos_pre = sortedSphMarkersD->posRadD;
    thrust::host_vector<Real4> debugRho_pre = sortedSphMarkersD->rhoPresEnD;
    std::cout << "After calc ADKE" << std::endl;
    std::cout << "h of particles:" << std::endl;
    for (int i = 0; i < debugRho_pre.size(); i++)
        if (debugPos_pre[i].x < 0.1 && debugPos_pre[i].x > -0.1)
            std::cout << debugPos_pre[i].w << ", ";
    std::cout << std::endl;
    */
#endif

    // If ADKE method for h update, perform it before computing derivatives and also before computing density by summation, if the case:
    if (m_data_mgr.paramsH->h_evolution == H_evolution_csph::ADKE) {
        CfdCalcRadADKE(sortedSphMarkersD);
    }

    // If density through summation, compute it before differential equations
    if (m_data_mgr.paramsH->rho_evolution == Rho_evolution_csph::SUMMATION) {
        CfdCalcRhoSum(sortedSphMarkersD);      // updates density and recomputes pressure and speed of sound
    }

 #ifndef NDEBUG  
    /*
    thrust::host_vector<Real4> debugPos = sortedSphMarkersD->posRadD;
    thrust::host_vector<Real4> debugRho = sortedSphMarkersD->rhoPresEnD;
    std::cout << "After calc RHO sum\nrho component of rhoPresEn:" << std::endl;
    for (int i = 0; i < debugRho.size(); i++)
        if (debugPos[i].x < 0.1 && debugPos[i].x > -0.1)
            std::cout << debugRho[i].x << ", ";
    std::cout << " Size = " << debugRho.size() << std::endl;
    std::cout << "Pres component of rhoPresEn:" << std::endl;
    for (int i = 0; i < debugRho.size(); i++)
        if (debugPos[i].x < 0.1 && debugPos[i].x > -0.1)
            std::cout << debugRho[i].y << ", ";
    std::cout << std::endl;
    std::cout << "h of particles:" << std::endl;
    for (int i = 0; i < debugRho.size(); i++)
        if (debugPos[i].x < 0.1 && debugPos[i].x > -0.1)
            std::cout << debugPos[i].w << ", ";
    std::cout << std::endl;
    */
#endif

    // Impose boundary conditions and calculate derivatives
    CfdApplyBC(sortedSphMarkersD);  // for fluid problem first applies the bc and then computes the actual force at
                                    // each particle (computes rhs of Dvel/Dt equation)
    CfdCalcRHS(sortedSphMarkersD);  // This computes also DrhoDt, but without using XSPH correction in it.


    // If evolving h with differential equation based on mass conservation, compute its derivatives after computing DrhoDt:
    // the called method performs internal checks
    if (m_data_mgr.paramsH->h_evolution == H_evolution_csph::DIFFERENTIAL) {
        CfdCalcRadRHS(sortedSphMarkersD);
     }


    // Perform particle shifting if specified. (Just computes the DeltaV, doesn't apply it)
    if (m_data_mgr.paramsH->shifting_method != ShiftingMethod::NONE) {
        CalculateShifting(sortedSphMarkersD);
    }
}


// -----------------------------------------------------------------------------
// density_reinitialization
// -----------------------------------------------------------------------------

// kernel function that performs the density reinitialization by computing the particle density based on its neighbors
// values. The density of each particle is computed by evolving the differential continuity equation, but after some
// time steps it is corrected by computing it with the explicit equation. Original SPH equation for density is: rho_i =
// sum_j(m_j*W_ij) Here used a different equation: rho_i = sum_j( m_j * W_ij) / sum_j(m_j/rho_j * W_ij) aka Shepard
// filtering. The corrected density is computed, for each particle, based on the uncorrected (old) density for all its
// neighbors).

// Modification: since in general h will be allowed to vary, for consistency need to use a symmetric version 
// of the kernel during particle interaction. So kernel not evaluated at h_A but at (h_a + h_b)/2

// Also updates values of pressure and speed of sound after computation of density
template <bool IsUniform>
__global__ void calcRho_kernel(Real4* sortedPosRad,
                               Real4* sortedRhoPresEn,      // where new densities will be stored
                               Real4* sortedRhoPresEn_old,  // current/old density values
                               Real*  sortedSound,
                               Real*  sortedMassD,
                               const uint* numNeighborsPerPart,
                               const uint* neighborList,
                               const uint numActive,       // actual input passed is numExtendedPart ??
                               int density_reinit_steps) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;    // thread index
    if (index >= numActive)
        return;

    if (sortedRhoPresEn[index].w > -0.5)  // skip case of wall or rigid bce markers
        return;

    // Modified to use Eos of ideal gases.
    // computes (old) pressure (field y) using the Equation of State and old density (field x)
    // with old energy (field z).
    sortedRhoPresEn_old[index].y = Eos_csph(sortedRhoPresEn_old[index].x,
                                            sortedRhoPresEn_old[index].z, 
                                            EosType_csph::IDEAL_RHOEN);

    Real3 posRadA = mR3(sortedPosRad[index]);              // position of particle a (or i)
    Real h_partA = sortedPosRad[index].w;                  // extract h of current particle
    Real h_mean = h_partA;                                 // initialize h_mean to h_partA
    Real SuppRadii = paramsD_csph.h_multiplier * h_partA;  // initialize support kernel radius and its square value
    Real SqRadii = SuppRadii * SuppRadii;

    // initialize quantities:
    Real sum_mW = 0;                            // sum_j(m_j*W_ij)
    Real sum_mW_rho = 0.0000001;                // sum_j(m_j/rho_j*W_ij)
    Real sum_W = 0;                             // sum_j(W_ij)  (Used as debug?)
    uint NLStart = numNeighborsPerPart[index];  // neighbor list start and end for each particle
    uint NLEnd = numNeighborsPerPart[index + 1];

    // loop over neighbors, only contributions from actual fluid sph particles
    for (int n = NLStart; n < NLEnd; n++) {  // Loop over neighbor particles
        uint j = neighborList[n];            // index of current neighbor
        
        // When computing density by summation, should also include particle itself
        // if (j == index) {                    // neighbor list contains also particle itself, in case skip computations
        //     continue;
        // }
        Real3 posRadB = mR3(sortedPosRad[j]);      // position of particle b

        // symmetry of interaction
        
        if (paramsD_csph.h_variation) {
                Real h_partB = sortedPosRad[j].w;  // kernel radius of particle b
                h_mean = (h_partA + h_partB) / 2;  // update h_mean
                SuppRadii = paramsD_csph.h_multiplier * h_mean;
                SqRadii = SuppRadii * SuppRadii;
        }

        Real3 dist3 = Distance_csph(posRadA, posRadB);  // position of b and squared distance ab^2
        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
        if (dd > SqRadii)  // check on particle distance to be lower than kernel support radius
            continue;
        if (sortedRhoPresEn_old[j].w > -1.5 && sortedRhoPresEn_old[j].w < -0.5) {  // Actual fluid sph particle
           
            Real d = length(dist3);              // distance ab
            Real W3 = W3h_csph(paramsD_csph.kernel_type, d, 1/h_mean, paramsD_csph.num_dim);  // computes W_ij with proper symmetric kernel function
            sum_W += W3;
            
            if constexpr (IsUniform) {
                sum_mW += paramsD_csph.markerMass * W3;
                sum_mW_rho += paramsD_csph.markerMass * W3 / sortedRhoPresEn_old[j].x;
            } else {
                sum_mW += sortedMassD[j] * W3;
                sum_mW_rho += sortedMassD[j] * W3 / sortedRhoPresEn_old[j].x;
            }
        }
    }

    // USE FILTERED VERSION OF SUMMATION OR ORIGINAL ONE?
    // sortedRhoPresEn[index].x = sum_mW;
    // reinitializes density of fluid particles only
    if ((density_reinit_steps == 0) && (sortedRhoPresEn[index].w > -1.5) && (sortedRhoPresEn[index].w < -0.5)) {
        // if conditions met actually updates the density (normal mode and fluid marker)
        sortedRhoPresEn[index].x = sum_mW / sum_mW_rho;
        // update pressure and speed of sound
        sortedRhoPresEn[index].y = Eos_csph(sortedRhoPresEn[index].x, sortedRhoPresEn[index].z, EosType_csph::IDEAL_RHOEN);
        sortedSound[index] = SoundFromPresRho(sortedRhoPresEn[index].y, sortedRhoPresEn[index].x);
    }

}


// private method (called by the public one if needed) than performs density reinitialization.
// Basically just calls the previous Cuda kernel
void FsiForce_csph::density_reinitialization(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {

    // Re-Initialize the density after several time steps if needed
    thrust::device_vector<Real4> rhoPresEnD_old = sortedSphMarkersD->rhoPresEnD;
    printf("Re-initializing density after %d steps.\n", m_data_mgr.paramsH->density_reinit_steps);
    
    if (m_data_mgr.paramsH->is_uniform)
        calcRho_kernel<true><<<numBlocks, numThreads>>>(
            mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD), mR4CAST(rhoPresEnD_old),
            R1CAST(sortedSphMarkersD->soundD), nullptr, U1CAST(m_data_mgr.numNeighborsPerPart),
            U1CAST(m_data_mgr.neighborList), numActive, density_initialization);
    else
        calcRho_kernel<false><<<numBlocks, numThreads>>>(
            mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD), mR4CAST(rhoPresEnD_old),
            R1CAST(sortedSphMarkersD->soundD), R1CAST(m_data_mgr.sortedMassD),
            U1CAST(m_data_mgr.numNeighborsPerPart),
            U1CAST(m_data_mgr.neighborList), numActive, density_initialization);
    // density_initialization is the member variable used as counter
}




// -----------------------------------------------------------------------------
// CfdApplyBC - Uses a version Adami BCs modified for compressibility
// -----------------------------------------------------------------------------


// MODIFIED THE VARIABLES NAMES AND TYPES. No need for symmetric kernel here.
// Boundary condition application for compressible Euler Equations with Adami's method
// The pressure, velocity, energy, kernel radius are extrapolated from neighboring fluid particles using the 
// interpolation: f_bce = Sum_fluid (f_fluid * W_ij) / Sum_fluid (W_ij).
// Pressure extrapolation is a bit different as it also considers gravity and the acceleration of the bce markers.
// LATER REVISIT difference between Adami formulation and proper compressible equation for pressure extrapolation.
// Density computed using the EOS.
// Mass of bce markers allowed to vary while the associated volume is kept constant to have a consistent 
// discretization of space.
// To properly apply the free-slip condition one should also provide the local normal vector with respect to the interface.
// See if it can be introduced later on, for now use a simple extrapolation without modifications to BCE markers velocity
// which is left as the prescribed wall velocity obtained from solid body motion.
// See https://www.sciencedirect.com/science/article/pii/S002199911200229X?ref=cra_js_challenge&fr=RR-1
__global__ void CfdAdamiBC_csph(const uint* numNeighborsPerPart,
                                const uint* neighborList,
                                Real4* sortedPosRadD,
                                const uint numActive,
                                Real3* bceAcc,
                                Real4* sortedRhoPresEnD,
                                Real*  sortedSoundD,
                                Real3* sortedVelD) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive)
        return;

    // Ignore all fluid particles --> extrapolation for both rigid body and Wall bce markers 
    if (IsFluidParticle(sortedRhoPresEnD[index].w)) {
        return;
    }

    Real3 posRadA = mR3(sortedPosRadD[index]);
    Real h_partA = sortedPosRadD[index].w;        // kernel radius of particle a
    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];
    Real sum_pw = 0;             //  p_fluid*W_ij
    Real sum_enw = 0;            //  en_fluid*W_ij
    Real sum_radw = 0;           //  h_fluid*W_ij
    // Real3 sum_vw = mR3(0);       //  v_fluid*W_ij
    Real3 sum_rhorw = mR3(0);    //  rho_fluid*pos*W_ij
    Real sum_w = 0;              //  W_ij
    
    /*
    // DEBUG!!!!!
    if (index == 0)
        printf("Inside CfdAdamiBC. h_partA = %g, NLstart = %d, NLEnd = %d\n", h_partA, NLStart, NLEnd);
    */

    for (int n = NLStart + 1; n < NLEnd; n++) {
        uint j = neighborList[n];

        
        // only consider fluid neighbors (wall bce excluded as well)
        if (IsBceMarker(sortedRhoPresEnD[j].w)) {
            continue;
        }

        /*
        // DEBUG!!!!
        if (index == 0)
            printf("Neighbor of particle0 is: %d\n", j);
        */

        Real3 posRadB = mR3(sortedPosRadD[j]);
        Real h_partB = sortedPosRadD[j].w;      // kernel radius of particle b
        Real3 rij = Distance_csph(posRadA, posRadB);
        Real d = length(rij);

        Real W3 = W3h_csph(paramsD_csph.kernel_type, d, 1 / h_partA, paramsD_csph.num_dim);   // No need for symmetric version of kernel
        sum_w += W3;
        sum_pw += sortedRhoPresEnD[j].y * W3;              // p_j*W_ij
        sum_enw += sortedRhoPresEnD[j].z * W3;             // en_j*W_ij
        sum_radw += sortedPosRadD[j].w * W3;               // h_j*W_ij
        sum_rhorw += sortedRhoPresEnD[j].x * rij * W3;     // rho_j*pos_ij*W3
        // sum_vw += sortedVelD[j] * W3;                      // vel_j*W_ij
    }

    // If the sum of kernels is above a minimum value we can actually update the bce properties,
    // otherwise set them to zero as not enough fluid neighbors
    if (sum_w > EPSILON) {
        // Now actually compute the pressure from Adami's formula and density using the equation of state
        // For now follow the original Adami's paper in which says that free-slip condition is implicitely obtained
        // by simply omitting the viscous interaction of a fluid particle with adjacent bce ones.
        // In more refined work we would need to use a prescribed velocity without the component normal to the interface.
        // For now just leave the velocity as the prescribed solid body one.
        Real3 prescribedVel = (IsBceSolidMarker(sortedRhoPresEnD[index].w)) ? ( sortedVelD[index]) : mR3(0);    // wall or solid body bce
        sortedVelD[index] = prescribedVel;
        sortedRhoPresEnD[index].y = (sum_pw + dot(paramsD_csph.gravity - bceAcc[index], sum_rhorw)) / sum_w;    // pressure update, does not consider body forces
        sortedRhoPresEnD[index].z = sum_enw / sum_w;       // energy extrapolation
        sortedPosRadD[index].w = sum_radw / sum_w;         // kernel radius extrapolation
        // density, temperature and speed of sound update:
        sortedRhoPresEnD[index].x = InvEos_csph(sortedRhoPresEnD[index].y, sortedRhoPresEnD[index].z, EosType_csph::IDEAL_RHOEN);   // density from pressure and energy
    
        sortedSoundD[index] = SoundFromPresRho(sortedRhoPresEnD[index].y, sortedRhoPresEnD[index].x);

    } else {
        sortedVelD[index] = mR3(0);

        // CHECK IT LATER!!!!!
         //sortedRhoPresEnD[index].y = 0;
    }
}



// Private method for application of bc in fluid problems. Calls Adami Cuda kernel depending on the bc
// type defined in the problem parameters.
void FsiForce_csph::CfdApplyBC(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {

    if (m_data_mgr.paramsH->boundary_method == BoundaryMethod_csph::ORIGINAL_ADAMI) {

        /*
        thrust::host_vector<Real4> debugRhoH_pre = sortedSphMarkersD->rhoPresEnD;
        thrust::host_vector<Real3> debugVelH_pre = sortedSphMarkersD->velD;
        thrust::host_vector<uint> NumNeighH = m_data_mgr.numNeighborsPerPart;
        thrust::host_vector<uint> NeighListH = m_data_mgr.neighborList;

        std::cout << "BEFORE FsiForce_csph.CfdAdamiBC()" << std::endl;
        std::cout << "Neighbor List of sorted particle 30:" << std::endl;
        int numFluidNeigh = 0;
        uint NLStart = NumNeighH[30];
        uint NLEnd = NumNeighH[30 + 1];
        for (int n = NLStart + 1; n < NLEnd; n++) {
            uint j = NeighListH[n];
            std::cout << j << ", ";
            if (IsFluidParticle(debugRhoH_pre[j].w))
                numFluidNeigh++;

        }
        std::cout << "Number of neighbors: " << NLEnd - NLStart - 1 << ", number of fluid neighbors: " << numFluidNeigh
                  << std::endl;
                  */

        /*
        std::cout << "X component of Velocity:" << std::endl;
        for (int i = 0; i < debugVelH_pre.size(); i++)
            std::cout << debugVelH_pre[i].x << ", ";
        std::cout << " Size = " << debugVelH_pre.size() << std::endl;

        std::cout << "RHO component of rhoPresEn:" << std::endl;
        for (int i = 0; i < debugRhoH_pre.size(); i++)
            std::cout << debugRhoH_pre[i].x << ", ";
        std::cout << std::endl;
        std::cout << "PRES component of rhoPresEn:" << std::endl;
        for (int i = 0; i < debugRhoH_pre.size(); i++)
            std::cout << debugRhoH_pre[i].y << ", ";
        std::cout << std::endl;
        std::cout << "En component of rhoPresEn:" << std::endl;
        for (int i = 0; i < debugRhoH_pre.size(); i++)
            std::cout << debugRhoH_pre[i].z << ", ";
        std::cout << std::endl;
        */




        CfdAdamiBC_csph<<<numBlocks, numThreads>>>(U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList),
                                              mR4CAST(sortedSphMarkersD->posRadD), numActive,
                                              mR3CAST(m_data_mgr.bceAcc), mR4CAST(sortedSphMarkersD->rhoPresEnD),
                                              R1CAST(sortedSphMarkersD->soundD),
                                              mR3CAST(sortedSphMarkersD->velD));


        /*
        // DEBUG ------------------------------------------------
        thrust::host_vector<Real4> debugRhoH_post = sortedSphMarkersD->rhoPresEnD;
        thrust::host_vector<Real3> debugVelH_post = sortedSphMarkersD->velD;
        */

        /*
        std::cout << "AFTER FsiForce_csph.CfdAdamiBC()" << std::endl;
        std::cout << "X component of Velocity:" << std::endl;
        for (int i = 0; i < debugVelH_post.size(); i++)
            std::cout << debugVelH_post[i].x << ", ";
        std::cout << " Size = " << debugVelH_post.size() << std::endl;

        std::cout << "RHO component of rhoPresEn:" << std::endl;
        for (int i = 0; i < debugRhoH_post.size(); i++)
            std::cout << debugRhoH_post[i].x << ", ";
        std::cout << std::endl;
        std::cout << "PRES component of rhoPresEn:" << std::endl;
        for (int i = 0; i < debugRhoH_post.size(); i++)
            std::cout << debugRhoH_post[i].y << ", ";
        std::cout << std::endl;
        std::cout << "En component of rhoPresEn:" << std::endl;
        for (int i = 0; i < debugRhoH_post.size(); i++)
            std::cout << debugRhoH_post[i].z << ", ";
        std::cout << std::endl;

        std::cout << "Changes in velocity after CfdAdamiBC_csph:" << std::endl;
        for (int i = 0; i < debugVelH_post.size(); i++) {
            if (debugVelH_pre[i].x != debugVelH_post[i].x)
                std::cout << "Pre value of x velocity: " << debugVelH_pre[i].x << " post value: " << debugVelH_post[i].x
                          << " index = " << i << std::endl;
            if (debugVelH_pre[i].y != debugVelH_post[i].y)
                std::cout << "Pre value of y velocity: " << debugVelH_pre[i].y << " post value: " << debugVelH_post[i].y
                          << " index = " << i << std::endl;
            if (debugVelH_pre[i].z != debugVelH_post[i].z)
                std::cout << "Pre value of z velocity: " << debugVelH_pre[i].z << " post value: " << debugVelH_post[i].z
                          << " index = " << i << std::endl;
        }
        std::cout << "Changes in rho after CfdAdamiBC_csph:" << std::endl;
        for (int i = 0; i < debugRhoH_post.size(); i++) {
            if (debugRhoH_pre[i].x != debugRhoH_post[i].x)
                std::cout << "Pre value of rho: " << debugRhoH_pre[i].x << " post value: " << debugRhoH_post[i].x
                          << " index = " << i << std::endl;
        }
        std::cout << "Changes in pressure after CfdAdamiBC_csph:" << std::endl;
        for (int i = 0; i < debugRhoH_post.size(); i++) {
            if (debugRhoH_pre[i].y != debugRhoH_post[i].y)
                std::cout << "Pre value of pressure: " << debugRhoH_pre[i].y << " post value: " << debugRhoH_post[i].y
                          << " index = " << i << std::endl;
        }
        std::cout << "Changes in energy after CfdAdamiBC_csph:" << std::endl;
        for (int i = 0; i < debugRhoH_post.size(); i++) {
            if (debugRhoH_pre[i].z != debugRhoH_post[i].z)
                std::cout << "Pre value of energy: " << debugRhoH_pre[i].z << " post value: " << debugRhoH_post[i].z
                          << " index = " << i << std::endl;
        }
        // -------------------------------------------------------------
        */

        /*
        thrust::host_vector<Real4> debugPosH = sortedSphMarkersD->posRadD;
        Real3 pos30 = make_Real3(debugPosH[30]);
        Real3 pos0 = make_Real3(debugPosH[0]);
        Real4 rhoPresEn30 = debugRhoH_post[30];
        Real3 vel30 = debugVelH_post[30];
        Real SuppRadii30 = m_data_mgr.paramsH->h_multiplier * debugPosH[30].w;
        std::cout << "After CfdAdamiBC_csph. Consider properties of particle 30:\n" << std::endl;
        std::cout << " h = " << debugPosH[30].w << ", SuppRadii = " << SuppRadii30
                  << ", SqRadii = " << SuppRadii30 * SuppRadii30 << std::endl;
        std::cout << "Coordinates = " << pos30.x << ", " << pos30.y << ", " << pos30.z << std::endl;
        std::cout << "Velocity = " << vel30.x << ", " << vel30.y << ", " << vel30.z << std::endl;
        std::cout << "rho, pres, en = " << rhoPresEn30.x << ", " << rhoPresEn30.y << ", " << rhoPresEn30.z << "\n" << std::endl;
        std::cout << "Coordinates of particle 0 = " << pos0.x << ", " << pos0.y << ", " << pos0.z << "\n" << std::endl;
        //thrust::host_vector<int> pos = {0,  31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                                       // 44, 45, 46, 47, 60, 61, 62, 63, 64, 65, 72, 73, 74};
        
        thrust::host_vector<int> pos = {60};
        double dd;
        Real3 dist;
        Real SuppRadii;
        std::cout << "Properties of particles neighbors of particle 30:" << std::endl;
        for (const auto& i : pos) {
            std::cout << "Particle " << i << std::endl;
            std::cout << "position = " << debugPosH[i].x << ", " << debugPosH[i].y << ", " << debugPosH[i].z
                      << std::endl;
            // dist = pos30 - make_Real3(debugPosH[i]);
            dist = Distance_csph(pos30, make_Real3(debugPosH[i]), *m_data_mgr.paramsH );
            dd = dist.x * dist.x + dist.y * dist.y * dist.z * dist.z;
            SuppRadii = m_data_mgr.paramsH->h_multiplier * debugPosH[i].w;
            std::cout << "dd = " << dd << ", SuppRadii = " << SuppRadii << ", SqRadii = " << SuppRadii*SuppRadii << std::endl;
            std::cout << "velocity = " << debugVelH_post[i].x << ", " << debugVelH_post[i].y << ", "
                      << debugVelH_post[i].z << std::endl;
            std::cout << "rho, pres, en = " << debugRhoH_post[i].x << ", " << debugRhoH_post[i].y << ", "
                      << debugRhoH_post[i].z << "\n" << std::endl;

        }
        */

    } else {
        std::cout << "For now this BCE option is not implemented." << std::endl;
    }

    if (m_check_errors) {
        cudaCheckError();
    }
}


// -----------------------------------------------------------------------------
// CfdCalcRhoSum --> computation of density for each particle in fluid problems using 
// classical summation equation
// -----------------------------------------------------------------------------


// Cuda kernel to compute the density at the current time step with the summation equation.
// rho_i = sum_j(m_j * W_ij)
// Depending on the problem parameters it may use a symmetric version of the kernel
// After computing the density of a particle, update the pressure with EOS (using save thermal energy) and speed of sound.
template<bool IsUniform>
__global__ void calcRho_sum(Real4* sortedPosRad,
                            Real4* sortedRhoPresEn,      // where new densities will be stored
                            Real*  sortedSound,      // when density is updated, also update pressure and speed of sound
                            const Real* __restrict__ sortedMassD,
                            const uint* numNeighborsPerPart,
                            const uint* neighborList,
                            const uint numActive,       // actual input passed is numExtendedPart
                            volatile bool* error_flag) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // thread index
    if (index >= numActive)
        return;

    if (sortedRhoPresEn[index].w > -0.5)  // skip case of wall and rigid body bce
        return;

    Real3 posRadA = mR3(sortedPosRad[index]);              // position of particle a (or i)
    Real h_partA = sortedPosRad[index].w;                  // extract h of current particle
    Real h_mean = h_partA;                                 // initialize h_mean to h_partA
    Real SuppRadii = paramsD_csph.h_multiplier * h_partA;   // initialize support kernel radius to h_partA. If h varies modified to symmetri version.
    Real SqRadii = SuppRadii * SuppRadii;

    // initialize quantities:
    Real sum_mW = 0;                            // sum_j(m_j*W_ij)
    uint NLStart = numNeighborsPerPart[index];  // neighbor list start and end for each particle
    uint NLEnd = numNeighborsPerPart[index + 1];

    // loop over neighbors
    for (int n = NLStart; n < NLEnd; n++) {  // Loop over neighbor particles
        uint j = neighborList[n];            // index of current neighbor
        
        // When computing density with summation, should include also the particle itself
        //if (j == index) {                    // neighbor list contains also particle itself, in case skip computations
        //     continue;
         //}
        Real3 posRadB = mR3(sortedPosRad[j]);      // position of particle b
        Real3 dist3 = Distance_csph(posRadA, posRadB);  // position of b and squared distance ab^2
        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;

        // symmetric interaction
        if (paramsD_csph.h_variation) {
            Real h_partB = sortedPosRad[j].w;
            h_mean = (h_partA + h_partB) / 2;   // actual mean kernel radius
            SuppRadii = paramsD_csph.h_multiplier * h_mean;
            SqRadii = SuppRadii * SuppRadii;
        }

        if (dd > SqRadii)  // check on particle distance to be lower than kernel support radius
            continue;

        // do we take into account boundary particles in computation of density?
        // if (sortedRhoPresEn[j].w > -1.5 && sortedRhoPresEn[j].w < -0.5) {  // Actual fluid sph particle - solid particles do not contribute to density
            
            Real d = length(dist3);                 // distance ab
            
            // compute kernel value
            Real W3 = W3h_csph(paramsD_csph.kernel_type, d, 1 / h_mean, paramsD_csph.num_dim);  // computes W_ij with proper symmetric kernel function
            if constexpr (IsUniform)
                sum_mW += paramsD_csph.markerMass * W3;  // accumulate values.
            else
                sum_mW += sortedMassD[j] * W3;
        // }
    }

    // update the density, no matter which type of particle under consideration. Wall bce already returned.
    sortedRhoPresEn[index].x = sum_mW;
    // Since density has been updated, also update the particle pressure using the equation of state:
    sortedRhoPresEn[index].y = Eos_csph(sortedRhoPresEn[index].x, sortedRhoPresEn[index].z, EosType_csph::IDEAL_RHOEN);
    // Update speed of sound:
    sortedSound[index] = SoundFromPresRho(sortedRhoPresEn[index].y, sortedRhoPresEn[index].x);
}


void FsiForce_csph::CfdCalcRhoSum(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {
    // Call the corresponding Cuda kernel
    cudaResetErrorFlag(m_errflagD);
    if (m_data_mgr.paramsH->is_uniform)
        calcRho_sum<true><<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
                                           R1CAST(sortedSphMarkersD->soundD), nullptr, U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList),
                                           numActive, m_errflagD);
    else 
        calcRho_sum<false><<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
                                        R1CAST(sortedSphMarkersD->soundD), R1CAST(m_data_mgr.sortedMassD), U1CAST(m_data_mgr.numNeighborsPerPart),
                                        U1CAST(m_data_mgr.neighborList), numActive, m_errflagD);
    cudaCheckErrorFlag(m_errflagD, "Calc_Rho_Sum_D");
}



// -----------------------------------------------------------------------------
// CfdCalcRHS --> computation of forces on each particle in fluid problems using
// compressible SPH approach
// -----------------------------------------------------------------------------



// Cuda kernel to compute the divergence of velocity for each particle.
// DivVel_i = 1/rho_i * Sum_j( m_J * dot(v_i - v_j, gradW_ij) ).
// Used in optional artificial heating term
template<bool IsUniform>
__global__ void calcDivVel(const Real4* __restrict__ sortedPosRad,
                           const Real3* __restrict__ sortedVel,
                           const Real4* __restrict__ sortedRhoPresEn,
                           const Real* __restrict__  sortedMassD,
                           Real* divVel,
                           const uint* numNeighborsPerPart,
                           const uint* neighborList,
                           const uint numActive) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // thread (and particle) index
    if (index >= numActive)
        return;

    if (!IsFluidParticle(sortedRhoPresEn[index].w))  // skip if not a fluid particle
        return;

    Real3 posA = mR3(sortedPosRad[index]);  // position of particle A
    Real3 velA = sortedVel[index];          // velocity of particle A
    Real h_partA = sortedPosRad[index].w;   // h of particle A
    Real h_mean = h_partA;                  // initialize h_mean to h_partA
    Real SuppRadii = paramsD_csph.h_multiplier * h_partA;    // defaults to radius of part A. Modified if h varies.
    Real SqRadii = SuppRadii * SuppRadii;  // Initialize support radius of kernel function

    // initialize quantity:
    Real DivVelA = 0;  // Initial divergence set to zero
    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];

    // loop over neighbors
    for (int n = NLStart; n < NLEnd; n++) {
        uint j = neighborList[n];

        if (j == index) {  // no need to consider the particle itself
            continue;
        }
        Real3 posB = mR3(sortedPosRad[j]);
        Real3 velB = sortedVel[j];
        Real3 dist3 = Distance_csph(posA, posB);  // position of b and squared distance ab^2
        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;

        // if h varies, makes the interaction symmetric
        if (paramsD_csph.h_variation) {
            Real h_partB = sortedPosRad[j].w;
            h_mean = (h_partA + h_partB) / 2;
            SuppRadii = paramsD_csph.h_multiplier * h_mean;
            SqRadii = SuppRadii * SuppRadii;
        }
        
        if (dd > SqRadii)  // check on particle distance to be lower than kernel support radius
            continue;
        if (sortedRhoPresEn[j].w > -1.5 && sortedRhoPresEn[j].w < -0.5) {  // Actual fluid sph particle - solid particles do not contribute to density
            Real3 gradW = GradW3h_csph(paramsD_csph.kernel_type, dist3, 1 / h_mean, paramsD_csph.num_dim);      // symmetric gradient of kernel function
            
            if constexpr (IsUniform)
                DivVelA += paramsD_csph.markerMass * dot(velB - velA, gradW);
            else
                DivVelA += sortedMassD[j] * dot(velB - velA, gradW);

        }
    }
    DivVelA *= 1 / (sortedRhoPresEn[index].x);  // divide accumulated value by density of particle A
    divVel[index] = DivVelA;
    return;                             // return divergence of velocity for particle A
}


// Lightweight kernel to compute the individual particles' term H_a used in the artificial heating term:
__global__ void calcHeatCoeff(const Real4* __restrict__ sortedPosRad,
                              const Real4* __restrict__ sortedRhoPresEn,
                              const Real* __restrict__  sortedSound,
                              const Real* __restrict__  divVel,
                              Real* __restrict__  heatCoeff,
                              const uint numActive) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // thread (and particle) index
    if (index >= numActive)
        return;

    if (!IsFluidParticle(sortedRhoPresEn[index].w)) {  // skip if not a fluid particle
        heatCoeff[index] = 0;
        return;
    }

    Real h_a = sortedPosRad[index].w;
    Real c_a = sortedSound[index];
    Real dv_a = divVel[index];

    // linear term
    Real H = paramsD_csph.Ar_heat_g1 * h_a * c_a;

    // quadratic term activate only if divVel < 0
    if (dv_a < 0)
        H += paramsD_csph.Ar_heat_g2 * h_a * h_a * 2 * (-dv_a);

    heatCoeff[index] = H;

    return;
}


// Device function to compute DvDt  and DrhoDt following classic compressible SPH.
// Given a couple of particles a and b it gives back a vetor with DvDt and DrhoDt.
// Expression of DvDt depends on the type of viscosity chosen: artificial monaghan only implemented for now. No stresses computed.
// Here gradients computed with original formulation, no correction matrices are used.
// Additional artificial heating term in the thermal energy equation.
__device__ inline Real5 cfdDvDt_csph(Real3 dist3,       // distance between two particles a and b
                                     Real d,            // d = length(dist3)
                                     Real h_mean,       // h to be used, either proper h_mean or just h_partA depending on h_variation
                                     Real4 posRadA,     // positions (and h)
                                     Real4 posRadB,
                                     Real3 velA,        // velocity
                                     Real3 velB,
                                     Real4 rhoPresEnA,  // rho, P, en, and particle type
                                     Real4 rhoPresEnB,
                                     Real  soundA,  // temperature and speed of sound
                                     Real  soundB,
                                     Real massB,
                                     Real heatCoeffA,
                                     Real heatCoeffB,
                                     bool debug_flag) {  
    if (IsBceMarker(rhoPresEnA.w) && IsBceMarker(rhoPresEnB.w))  // If both particles are bce markers return a 0 acceleration
        return mR5(0);
   
    Real3 gradW = GradW3h_csph(paramsD_csph.kernel_type, dist3, 1/h_mean, paramsD_csph.num_dim);   // symmetric gradient of kernel function
    Real vAB_dot_gradW = dot(velA - velB, gradW);
    // Continuity equation

    Real derivRho;

    //  contributions fluid-solid allowed
    if (paramsD_csph.rho_evolution == Rho_evolution_csph::DIFFERENTIAL) {
        derivRho = massB * vAB_dot_gradW;  // classic sph continuity equation
    } else {
        derivRho = 0;   // density updated through summation, derivative set to zero.
    }


    // Derivative of velocity:
    Real3 derivV;  // initialize derivative of velocity
    Real vAB_dot_rAB = dot(velA - velB, dist3);
    Real Pi_ab = 0;
    Real eta2 = paramsD_csph.epsMinMarkersDis * paramsD_csph.epsMinMarkersDis * h_mean * h_mean;
    //  pressure component
    derivV = - massB *
             (rhoPresEnA.y / (rhoPresEnA.x * rhoPresEnA.x) + rhoPresEnB.y / (rhoPresEnB.x * rhoPresEnB.x)) * gradW;

    
    // DEBUG !!!!
    if (debug_flag && false) {
        // printf("Inside Cfd_DvDt_csph for interaction of particles 30 and 60.\n");
        // printf("grad_w = %g, %g, %g\nvAB_dot_rAB = %g\nderivV pressure components = %g, %g, %g\n", gradW.x, gradW.y, gradW.z, vAB_dot_rAB, derivV.x, derivV.y,
        //        derivV.z);
        printf("P/rho^2 = %g, |gradW| = %g\n", rhoPresEnA.y / (rhoPresEnA.x * rhoPresEnA.x), length(gradW));
    }
    

    // When following Adami original method for free-slip condition, viscous contribution added 
    // only to fluid-fluid interaction. Omit viscous interaction of a fluid particle with surrounding bce ones, and viceversa.
    if (paramsD_csph.boundary_method == BoundaryMethod_csph::ORIGINAL_ADAMI && IsFluidParticle(rhoPresEnA.w) &&
        IsFluidParticle(rhoPresEnB.w)) {

        switch (paramsD_csph.viscosity_method) {
            case ViscosityMethod_csph::ARTIFICIAL_MONAGHAN: {
                 // artificial viscosity part, see Monaghan formulation                 
                 if (vAB_dot_rAB < 0) {
                     Real c_mean = (soundA + soundB) / 2;   // mean speed of sound
                     Real rho_mean = (rhoPresEnA.x + rhoPresEnB.x) / 2; // mean density
                     Real mu_ab = h_mean * vAB_dot_rAB / (d * d + eta2);
                     Pi_ab = massB *                  
                             (-paramsD_csph.Ar_vis_alpha * c_mean * mu_ab + paramsD_csph.Ar_vis_beta * mu_ab * mu_ab) / rho_mean; 
                     derivV.x -= Pi_ab * gradW.x;             // marker mass already included in Pi_ab
                     derivV.y -= Pi_ab * gradW.y;
                     derivV.z -= Pi_ab * gradW.z;
                  }
                 break;
            }
            case ViscosityMethod_csph::NONE: {
                break;
            }
        }  // end switch
    }   // end if
    


    // Derivative of thermal energy. The 1/2 term will be applied in subsequent kernels.
    Real derivEn;
    // pressure term
    derivEn = massB *
              (rhoPresEnA.y / (rhoPresEnA.x * rhoPresEnA.x) + rhoPresEnB.y / (rhoPresEnB.x * rhoPresEnB.x)) *
              vAB_dot_gradW;

    // Artificial contributions only if both are fluid particles (in the adami formulation) for consistency with DvDt
    if (paramsD_csph.boundary_method == BoundaryMethod_csph::ORIGINAL_ADAMI && IsFluidParticle(rhoPresEnA.w) &&
        IsFluidParticle(rhoPresEnB.w)) {

        switch (paramsD_csph.viscosity_method) {
            case ViscosityMethod_csph::ARTIFICIAL_MONAGHAN: {  // activate if both are fluid particles
                if (vAB_dot_rAB < 0) {                         // reuse existing variables

                     derivEn += Pi_ab * vAB_dot_gradW;  // marker mass already present in Pi_ab
                }
                break;
            }
            case ViscosityMethod_csph::NONE: {
                break;
            }
        }

        if (paramsD_csph.Ar_heat_switch) {
            Real H_ab = (heatCoeffA + heatCoeffB) * (rhoPresEnA.z - rhoPresEnB.z)  / (0.5 * (rhoPresEnA.x + rhoPresEnB.x));
            H_ab = H_ab / (eta2 + d * d);
            derivEn +=  2* massB * H_ab * dot(dist3, gradW);
        }

    }

    

    return mR5(derivV.x, derivV.y, derivV.z, derivRho, derivEn);
}



// Cuda Kernel function that provides implementation of the Euler equations for CFD.
// For each particle in input vectors it computes the vector sortedDerivVelRho of DvDt, DrhoDt, DenDt.
// It also computes the two different time steps, to be used in caso of variable dT subjected to the CFL condition.
template <bool IsUniform>
__global__ void CfdRHS_csph(Real5* __restrict__ sortedDerivVelRhoEn,
                            const Real4* __restrict__ sortedPosRad,
                            const Real3* __restrict__ sortedVel,
                            const Real4* __restrict__ sortedRhoPresEn,
                            const Real*  __restrict__ sortedSound,
                            const Real*  __restrict__ sortedMassD,
                            const Real*  __restrict__ heatCoeffD,
                            const uint*  __restrict__ numNeighborsPerPart,  // number of neighbors particles for particle a
                            const uint*  __restrict__ neighborList,         // index of neighbors for particle a. Index refers to order by hash
                                  Real*  __restrict__ CourantTimeStep,      // corresponding dT_cfl = h/c_S
                                  Real*  __restrict__ AccTimeStep,          // dT_force = C_force*sqrt( h / norm(dVeldT) )
                            const uint numActive,      // Will kernel be called with numActive or with numExtendedMarkers?
                            volatile bool* error_flag) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;  // thread index
    if (id >= numActive)
        return;

    uint index = id;

    // Do nothing if a is a fixed wall BCE particle
    if (sortedRhoPresEn[index].w > -0.5 && sortedRhoPresEn[index].w < 0.5) {
        sortedDerivVelRhoEn[index] = mR5(0);
        return;
    }

    // get position, velocity, rho, P, En, type of particle a
    Real3 posA = mR3(sortedPosRad[index]);
    Real3 velA = sortedVel[index];
    Real4 rhoPresEnA = sortedRhoPresEn[index];
    Real soundA = sortedSound[index];
    Real h_partA = sortedPosRad[index].w;           // h of particle A. Symmetrization of kernel function occurs inside cfdDvDt_csph
    Real h_mean = h_partA;                         // initialize h_mean to h_partA

    Real heatCoeffA = paramsD_csph.Ar_heat_switch ? heatCoeffD[index] : 0;

    Real5 derivVelRhoEn = mR5(0);                          // Initialize rhs (DvDt, DrhoDt, DenDt)
    Real SuppRadii = paramsD_csph.h_multiplier * h_partA;                       // Initialize kernel radius for particle A
    Real SqRadii = SuppRadii * SuppRadii;

    uint NLStart = numNeighborsPerPart[index];  // start and end index of neighbors for particle a
    uint NLEnd = numNeighborsPerPart[index + 1];

    // Loop over neighbor particles
    for (int n = NLStart; n < NLEnd; n++) {
        uint j = neighborList[n];  // index of neighbor particle b. refers to vectors ordered by hash
        if (j == index) {  // by construction there is also particle a in its own list of neighbors. When met, skip.
            continue;
        }

        // position of current particle B:
        Real3 posB = mR3(sortedPosRad[j]);  // position of particle b
        Real3 dist3 = Distance_csph(posA, posB);
        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;

        // If h varies, make the interaction symmetric
        if (paramsD_csph.h_variation) {
            Real h_partB = sortedPosRad[j].w;
            h_mean = (h_partA + h_partB) / 2;
            SuppRadii = paramsD_csph.h_multiplier * h_mean;
            SqRadii = SuppRadii * SuppRadii;
        }

        if (dd > SqRadii)  // safety check: if outside kernel radius of particle a, skip
            continue;
        // Extract density and type of current particle B
        Real4 rhoPresEnB = sortedRhoPresEn[j];
       
        // no solid-solid force
        if (IsBceMarker(rhoPresEnA.w) && IsBceMarker(rhoPresEnB.w))  // skip if both particles are solid markers.
            continue;

        // other properties of particle B
        Real d = length(dist3);                  // actual distance with sqrt()
        Real3 velB = sortedVel[j];               // velocity of b
        Real soundB = sortedSound[j];   // T and speed of sound of B

        // accumulate rhs for vel, rho, energy with compressible sph formulation. Does not apply XSPH correction to
        // density equation.
        Real massB;
        if constexpr (IsUniform)
            massB = paramsD_csph.markerMass;
        else
            massB = sortedMassD[j];

        Real heatCoeffB = paramsD_csph.Ar_heat_switch ? heatCoeffD[j] : 0;

        derivVelRhoEn += cfdDvDt_csph(dist3, d, h_mean, sortedPosRad[index], sortedPosRad[j], velA, velB, rhoPresEnA,
                                      rhoPresEnB, soundA, soundB, massB, heatCoeffA, heatCoeffB, false);

 
        if (!IsFinite(derivVelRhoEn) && (IsBceSolidMarker(rhoPresEnA.w) || IsBceSolidMarker(rhoPresEnB.w)))
           printf("NaN term at index %d, with neighbor %d, one is rigid bce: true\n", index, j);
            
        else if (!IsFinite(derivVelRhoEn) && (IsFluidParticle(rhoPresEnA.w) && IsFluidParticle(rhoPresEnB.w)))
            printf("NaN term at index %d, with neighbor %d, one is rigid bce: false\n", index, j);
   

    }  // end of for loop


    if (!IsFinite(derivVelRhoEn)) {
        printf("Error! particle derivVelRhoEn is NAN: thrown from FsiForce_csph.cu CfdRHS_csph !\n");
        //*error_flag = true;
    }

    // add gravity and other body force to fluid markers(only)
    // Notice: current implementations only supports constant body forces and gravity
    if (IsSphParticle(rhoPresEnA.w)) {
        Real3 totalFluidBodyForce3 = paramsD_csph.bodyForce3 + paramsD_csph.gravity;
        derivVelRhoEn += mR5(totalFluidBodyForce3);
    }

    // Compute the time steps for CFL condition:
    if (IsFluidParticle(rhoPresEnA.w)) {
        CourantTimeStep[index] = h_partA / soundA;  // h / c_s
        Real norm_dV = sqrtf(
            derivVelRhoEn.x * derivVelRhoEn.x + derivVelRhoEn.y * derivVelRhoEn.y + derivVelRhoEn.z * derivVelRhoEn.z );
        AccTimeStep[index] = sqrtf(h_partA / norm_dV);
    }

    if (paramsD_csph.rho_evolution == Rho_evolution_csph::SUMMATION)
        derivVelRhoEn.w = 0;

    sortedDerivVelRhoEn[index] = derivVelRhoEn;
    sortedDerivVelRhoEn[index].t *= (Real)0.5 ;
}


// actual method of the class that computes the rhs of fluid equations. Works on sorted markers. Calls the Cuda kernel.
void FsiForce_csph::CfdCalcRHS(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {
    cudaResetErrorFlag(m_errflagD);

    computeCudaGridSize(numActive, 256, numBlocks, numThreads);

    // If artificial heating is on, pre-compute some quantities
    Real* heatCoeffptr = nullptr;
    if (m_data_mgr.paramsH->Ar_heat_switch) {
        if (m_data_mgr.paramsH->is_uniform) {
            calcDivVel<true><<<numBlocks, numThreads>>>(
                mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velD),
                mR4CAST(sortedSphMarkersD->rhoPresEnD), nullptr,
                R1CAST(m_data_mgr.divVelD),
                U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList), numActive);
        } else {
            calcDivVel<false><<<numBlocks, numThreads>>>(
                mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velD),
                mR4CAST(sortedSphMarkersD->rhoPresEnD), R1CAST(m_data_mgr.sortedMassD), R1CAST(m_data_mgr.divVelD),
                U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList), numActive);
        }
        cudaCheckError();
        // pre-compute H_i coefficients:
        calcHeatCoeff<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD),
                                                 mR4CAST(sortedSphMarkersD->rhoPresEnD),
                                                 R1CAST(sortedSphMarkersD->soundD), R1CAST(m_data_mgr.divVelD),
                                                 R1CAST(m_data_mgr.heatCoeffD), numActive);
        heatCoeffptr = R1CAST(m_data_mgr.heatCoeffD);
        cudaCheckError();
    }
 
    // call the Cuda kernel to perform computations. vector derivVelRhoEnD in data manager is the sorted one.
    if (m_data_mgr.paramsH->is_uniform)
        CfdRHS_csph<true><<<numBlocks, numThreads>>>(mR5CAST(m_data_mgr.derivVelRhoEnD), mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velD),
            mR4CAST(sortedSphMarkersD->rhoPresEnD), R1CAST(sortedSphMarkersD->soundD),
            nullptr, heatCoeffptr, U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList), 
            R1CAST(m_data_mgr.dT_velD), R1CAST(m_data_mgr.dT_forceD), numActive, m_errflagD);
    else
        CfdRHS_csph<false><<<numBlocks, numThreads>>>(
            mR5CAST(m_data_mgr.derivVelRhoEnD), mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velD),
            mR4CAST(sortedSphMarkersD->rhoPresEnD), R1CAST(sortedSphMarkersD->soundD),
            R1CAST(m_data_mgr.sortedMassD), heatCoeffptr,
            U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList), R1CAST(m_data_mgr.dT_velD),
            R1CAST(m_data_mgr.dT_forceD), numActive, m_errflagD);

    cudaCheckErrorFlag(m_errflagD, "RhsCFD");
}


// ----------------------------------------------------------------------------------
// Update kernel radius h of each particle. If ADKE, done through summation directly,
// if DIFFERENTIAL, computes the rhs of derivRad.
// ----------------------------------------------------------------------------------

// Cuda kernel function to compute the right hand side of DradDt_a = (-h_a/3rho_a)*DrhoDt_a
// Works only if density is evolved with the mass conservation equation.
__global__ void calcDradDt_kernel(const Real4* sortedPosRad,
                                  const Real4* sortedRhoPresEn,      
                                  Real* sortedDerivRad,      // store output                              
                                  const Real5* sortedDerivVelRhoEn,
                                  const uint numActive,
                                  volatile bool* error_flag) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // thread index
    if (index >= numActive)
        return;

    if (!IsFluidParticle(sortedRhoPresEn[index].w)) {  // skip case of non fluid particle
        sortedDerivRad[index] = 0;
        return;
    }

    Real rhoA = sortedRhoPresEn[index].x;                  // density of particle A
    Real3 posRadA = mR3(sortedPosRad[index]);              // position of particle a (or i)
    Real h_partA = sortedPosRad[index].w;                  // extract h of current particle
    
    // derivative of h linked to derivative of rho
    sortedDerivRad[index] = -(h_partA / (paramsD_csph.num_dim * rhoA)) * sortedDerivVelRhoEn[index].w;

}


// private method that computes the derivative of kernel radius for each particle:
void FsiForce_csph::CfdCalcRadRHS(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {
    // If evolving h with differential equation based on mass conservation, compute its derivatives after computing
    // DrhoDt:
        if (m_data_mgr.paramsH->rho_evolution != Rho_evolution_csph::DIFFERENTIAL) {
            std::cout
                << "Error: evolution of h with differential equation only if density computed with mass conservation.\n"
                << "Falling back to case of constant h." << std::endl;   // actually corrected in fluidsystem.checksphparameters()
        } else {
            cudaResetErrorFlag(m_errflagD);
            calcDradDt_kernel<<<numBlocks, numThreads>>>(
                mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
                R1CAST(m_data_mgr.derivRadD), mR5CAST(m_data_mgr.derivVelRhoEnD), numActive, m_errflagD);
            cudaCheckErrorFlag(m_errflagD, "CfdUpdateRadDifferential");
        }
}


// First step of ADKE procedure
// Cuda kernel to compute the pilot density at the current time step with the summation equation.
// rho_i_pilot = sum_j(m_j * W_ij)
// It uses the default value of the kernel radius h stored in paramsD_csph.

// No need for symmetric interaction or to recompute the neighbor list with the initial value of h0 because
// the pilot density is only a rough estimate and no specific properties or smoothness are of interest.
template <bool IsUniform>
__global__ void ADKE_calcRho_pilot(const Real4* __restrict__ sortedPosRad,
                                   const Real4* __restrict__ sortedRhoPresEn,
                                   const Real* __restrict__ sortedMassD,
                                   Real*  pilotRho,                      // where pilot densities will be stored
                                   Real*  log_pilotRho,                  // stores log of pilot densities
                                   int*   index_pilot,                   // control values for subsequent sum reduction
                                   const uint* numNeighborsPerPart,
                                   const uint* neighborList,
                                   const uint numActive, 
                                   volatile bool* error_flag) {
 
    uint index = blockIdx.x * blockDim.x + threadIdx.x; // thread index 

    if (index >= numActive)
        return;

    if (!IsFluidParticle(sortedRhoPresEn[index].w)) {   // if not fluid particle, set pilot density to zero
        pilotRho[index] = 0;
        log_pilotRho[index] = 0;
        index_pilot[index] = -1;
        return;
    }

    // properties of particle a
    Real3 posRadA = mR3(sortedPosRad[index]);  // position of particle a (or i)
    // The pilot density is calcolated using h0 = D * delta_x0:
    Real h0 = paramsD_csph.ADKE_D * paramsD_csph.d0;  
    Real SuppRadii = paramsD_csph.h_multiplier * h0;  // support kernel radius and its square value
    Real SqRadii = SuppRadii * SuppRadii;

    // initialize quantities:
    Real result = 0;                               // sum_j(m_j*W_ij) with initial h
    uint NLStart = numNeighborsPerPart[index];     // neighbor list start and end for each particle
    uint NLEnd = numNeighborsPerPart[index + 1];

    // loop over neighbors
    for (int n = NLStart; n < NLEnd; n++) {  // Loop over neighbor particles
        uint j = neighborList[n];            // index of current neighbor
        
        // when computing density by summation, include also the particle itself
        // if (j == index) {                    // neighbor list contains also particle itself, in case skip computations
        //     continue;
        // }

        Real3 posRadB = mR3(sortedPosRad[j]);      // position of particle b
        Real3 dist3 = Distance_csph(posRadA, posRadB);  // position of b and squared distance ab^2
        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
        if (dd > SqRadii)  // check on particle distance to be lower than kernel support radius
            continue;
        // do we include bce markers when computing pilot density of a fluid particle?
        // if (sortedRhoPresEn[j].w > -1.5 && sortedRhoPresEn[j].w < -0.5) {  // Actual fluid sph particle - solid particles do not contribute to density
            Real d = length(dist3);              // distance ab
            // compute kernel value
            Real W3 = W3h_csph(paramsD_csph.kernel_type, d, 1/h0, paramsD_csph.num_dim);      // computes W_ij with initial kernel radius h_0
            // accumulate values.
            if constexpr (IsUniform)
                result += paramsD_csph.markerMass * W3;
            else
                result += sortedMassD[j] * W3;
                           
        // }
    }

    // safety check on minimum possible density
    result = (result > 1e-4) ? result : 1e-4;
    // update the pilot density
    pilotRho[index] = result ;
    log_pilotRho[index] = log(result);
    index_pilot[index] = index;
}



// Define functors for subsequent operations in the ADKE scheme::


struct valid_value {
    __host__ __device__ Real operator()(const thrust::tuple<int, Real>& t) const { 
        int key = thrust::get<0>(t);
        Real value = thrust::get<1>(t);
        return (key != -1) ? value : (Real)0.0;
    }
};


struct calc_lambda {
    Real k;
    Real eps;
    Real q_value;
    Real lambda_min;
    Real lambda_max;
    __host__ __device__ calc_lambda(Real k_, Real eps_ , Real q, Real min = 0.3, Real max = 2.5) :
        k(k_), eps(eps_), q_value(q), lambda_min(min), lambda_max(max) {}

    __host__ __device__ Real operator()(const thrust::tuple<int, Real>& t) const { 
        int key = thrust::get<0>(t);
        Real rho_pilot = thrust::get<1>(t);
        Real lambda = k * std::pow(rho_pilot / q_value,-eps);
        if (key == -1) {
            return (Real)1.0;
        } 
        else {
            if (lambda < lambda_min)
                return lambda_min;
            else if (lambda > lambda_max)
                return lambda_max;
            else
                return lambda;
        }
    }
};



struct multiply_h {
    __host__ __device__ multiply_h(Real init) : h0(init){};

    __host__ __device__ Real4 operator()(const Real4& posrad, const Real lambda) const { 
        Real4 result = posrad;
        result.w = lambda * h0;
        return result;
    }

    Real h0;
};


// private method to evolve each particle's h according to the ADKE method
void FsiForce_csph::CfdCalcRadADKE(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {
    cudaResetErrorFlag(m_errflagD);

    // first step: compute the pilot density for each particle using the value of h at 
    // the beginning of the time step:
    thrust::device_vector<Real> pilotRhoD(numActive);         // work with the number of active particles
    thrust::device_vector<Real> log_pilotRhoD(numActive);
    thrust::device_vector<int> index_pilotRhoD(numActive);
    thrust::fill(pilotRhoD.begin(), pilotRhoD.end(), 0.0);   // initialize to zero
    thrust::fill(log_pilotRhoD.begin(), log_pilotRhoD.end(), 0.0);
    thrust::fill(index_pilotRhoD.begin(), index_pilotRhoD.end(), -1);

    // call the Cuda kernel that computes the pilot density (and its log) for each fluid particle
    if (m_data_mgr.paramsH->is_uniform)  
        ADKE_calcRho_pilot<true><<<numBlocks, numThreads>>> (mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
            nullptr, R1CAST(pilotRhoD), R1CAST(log_pilotRhoD), I1CAST(index_pilotRhoD),
            U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList), numActive, m_errflagD);
    else 
        ADKE_calcRho_pilot<false><<<numBlocks, numThreads>>> (mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
            R1CAST(m_data_mgr.sortedMassD), R1CAST(pilotRhoD), R1CAST(log_pilotRhoD), I1CAST(index_pilotRhoD),
            U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList), numActive, m_errflagD);
    cudaCheckError();

    // step 2: compute the mean value of all pilot densities by means of a thrust reduce
    auto begin_indLog = thrust::make_zip_iterator(thrust::make_tuple(index_pilotRhoD.begin(), log_pilotRhoD.begin()));
    auto end_indLog = thrust::make_zip_iterator(thrust::make_tuple(index_pilotRhoD.end(), log_pilotRhoD.end()));

    Real init_value = (Real)0.0;
    auto sum = thrust::transform_reduce(begin_indLog, end_indLog, valid_value(), init_value, thrust::plus<Real>());
    
    if (m_data_mgr.countersH->numFluidMarkers == 0) {
        std::cerr << "In ADKE dividing by 0 because numFluidMarkers = 0, returning early" << std::endl;
        return;
    }

    Real q = std::exp(sum / m_data_mgr.countersH->numFluidMarkers);

    // Initialize vector of lambdas and utility tuple iterators
    thrust::device_vector<Real> lambdaD(pilotRhoD.size());
    thrust::fill(lambdaD.begin(), lambdaD.end(), (Real)1.0);

    // step 3: compute the lambda for each particle and update the kernel radius h
    auto begin_indRho = thrust::make_zip_iterator(thrust::make_tuple(index_pilotRhoD.begin(), pilotRhoD.begin()));
    auto end_indRho = thrust::make_zip_iterator(thrust::make_tuple(index_pilotRhoD.end(), pilotRhoD.end()));
    // compute lambda for each particle
    calc_lambda functor(m_data_mgr.paramsH->ADKE_k, m_data_mgr.paramsH->ADKE_eps, q);
    thrust::transform(begin_indRho, end_indRho, lambdaD.begin(), functor);
    // now multiply each h for each corresponding lambda value. No need for checks, lambda of non-fluid markers = 1
    multiply_h h_functor(m_data_mgr.paramsH->ADKE_D * m_data_mgr.paramsH->d0);
    thrust::transform(sortedSphMarkersD->posRadD.begin(), sortedSphMarkersD->posRadD.end(), lambdaD.begin(), sortedSphMarkersD->posRadD.begin(), h_functor);

}


// -----------------------------------------------------------------------------
// CalculateShifting
// -----------------------------------------------------------------------------

// Templated device cuda functions to compute shifting depending on which method we want
// This function computes the contributions given by neighbor particles and accumulates them.
// Recall xsph works by correcting a particle velocity: v_i_xsph = v_I + delta_v_i_xsph
// Then one uses this corrected velocity in the update of particle position, leaving continuity and momentum equation to
// use original velocities. Another philosophy is to use the corrected velocity also in the equations of density and
// momentum.
template <ShiftingMethod SHIFT, bool IsUniform>
__device__ void ShiftingAccumulateNeighborContrib_csph( uint index,
                                                        const Real3& posA,          // variables for particle a given as references
                                                        const Real4& rhoPresEnA,
                                                        const Real3& velA,
                                                        const Real4* sortedPosRad,  // sorted arrays for neighbor particles' variables
                                                        const Real3* sortedVel,
                                                        const Real4* sortedRhoPresEn,
                                                        const Real*  sortedMassD,  
                                                        const uint* neighborList,
                                                        uint NLStart,
                                                        uint NLEnd,
                                                        bool consider_bce,  // true if bce particles to be considered in the accumulation of corrective terms. Some methods
                                                                            // use them, others don't
                                                        Real3& deltaV,      // output is the velocity correction term - for xsph only output
                                                        Real3& inner_sum,   // additional output for other shifting techiques
                                                        Real& nabla_r) {    // additional output for other shifting techiques
    
    Real h_partA = sortedPosRad[index].w;        // h of particle A
    Real h_mean = h_partA;                 // initialize h_mean to h_partA
    Real SuppRadii = paramsD_csph.h_multiplier * h_partA;
    Real SqRadii = SuppRadii * SuppRadii;  // squared radius support of kernel function using standard h

    // Loop over neighbors
    for (uint n = NLStart + 1; n < NLEnd; n++) {
        uint j = neighborList[n];

        // Only proceed if neighbor is fluid (this check is inlined for brevity)
        if (!IsFluidParticle(sortedRhoPresEn[j].w) && !consider_bce) {
            continue;
        }

        // Distance check
        Real3 posB = mR3(sortedPosRad[j]);
        Real3 dist3 = Distance_csph(posA, posB);
        Real dd = dot(dist3, dist3);

        // check if need to symmetrize the interaction:
        if (paramsD_csph.h_variation) {
            Real h_partB = sortedPosRad[j].w;
            h_mean = (h_partA + h_partB) / 2;
            SuppRadii = paramsD_csph.h_multiplier * h_mean;
            SqRadii = SuppRadii * SuppRadii;
        }

        if (dd > SqRadii) {  // safety check on particles distance
            continue;
        }
        Real d = sqrt(dd);

        // If XSPH is required
        if constexpr (SHIFT == ShiftingMethod::XSPH) {
            Real3 velB = sortedVel[j];                          // state of particle b
            Real4 rhoPresEnB = sortedRhoPresEn[j];
            Real rho_bar = 0.5f * (rhoPresEnA.x + rhoPresEnB.x);  // average density between a and b
            
            // accumulate xsph correction terms
            if constexpr (IsUniform)     // if uniform multiply once outside the loop
                deltaV += 1 * (velB - velA) * W3h_csph(paramsD_csph.kernel_type, d, 1 / h_mean, paramsD_csph.num_dim) / rho_bar;
            else
                deltaV += sortedMassD[j] * (velB - velA) * W3h_csph(paramsD_csph.kernel_type, d, 1 / h_mean, paramsD_csph.num_dim) / rho_bar;
             
        }      
    }
}


// Cuda kernel function that computes the corrective delta_vel to be applied for each particle, based on contribution of
// its neighbors.
template <ShiftingMethod SHIFT, bool IsUniform>
__global__ void Calc_Shifting_D_csph(Real3* vel_XSPH_Sorted_D,  // output is array of delta velocities
                                     Real4* sortedPosRad,
                                     Real3* sortedVel,
                                     Real4* sortedRhoPresEn,
                                     const Real* sortedMassD,
                                     const uint* numNeighborsPerPart,
                                     const uint* neighborList,
                                     const uint numActive,
                                     volatile bool* error_flag) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // thread index = particle index
    if (index >= numActive)
        return;

    // If not fluid, do nothing
    if (!IsFluidParticle(sortedRhoPresEn[index].w))
        return;

    // Gather data of particle a
    Real4 rhoPresEnA = sortedRhoPresEn[index];
    Real3 velA = sortedVel[index];
    Real3 posA = mR3(sortedPosRad[index]);

    // Range for neighbors
    uint NLStart = numNeighborsPerPart[index];
    uint NLEnd = numNeighborsPerPart[index + 1];

    // Accumulators for different methods. Xsph will use only deltaV
    Real3 deltaV = mR3(0);
    Real3 inner_sum = mR3(0);
    Real nabla_r = 0;

    // depending on method may need bce contributions. False for XSPH
    bool consider_bce = false;

    // Accumulate neighbor contribution for each particle a, using the specific shifting technique.
    // results stored in deltaV, inner_sum and nabla_r (depending on the method)
    ShiftingAccumulateNeighborContrib_csph<SHIFT,IsUniform>(index, posA, rhoPresEnA, velA, sortedPosRad, sortedVel,
                                                  sortedRhoPresEn, sortedMassD, neighborList, NLStart, NLEnd, consider_bce, deltaV,
                                                  inner_sum, nabla_r);

    // Post-process resulting deltaV depending on SHIFT
    // Correction terms stored in result variable
    Real3 result = mR3(0);

    if constexpr (SHIFT == ShiftingMethod::XSPH) {
        // XSPH velocity is just a scaling of the deltaV obtained
        if constexpr (IsUniform)
            result = paramsD_csph.markerMass * paramsD_csph.shifting_xsph_eps * deltaV;
        else
            result = 1 * paramsD_csph.shifting_xsph_eps * deltaV;    // mass already present in single terms.
         
    } 
 

    // Write out - This is the delta velocity to be applied for particle A
    vel_XSPH_Sorted_D[index] = result;

    // Check for NaNs
    if (!IsFinite(result)) {
        printf("Error! Shifting produce  NAN. Particle: %u\n", index);
        *error_flag = true;
    }
}



// Actual private method that computes the shifting depending on the method stored in FsiDataManager.parameters
// The result is a vector of deltaV values (1 for each particle) stored in the vel_XSPH_D vector, to be summed to actual
// particle velocity in a later step.
void FsiForce_csph::CalculateShifting(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {
    cudaResetErrorFlag(m_errflagD);

    computeCudaGridSize(numActive, 1024, numBlocks, numThreads);
    // Initialize the corrective velocity to 0
    thrust::fill(m_data_mgr.vel_XSPH_D.begin(), m_data_mgr.vel_XSPH_D.begin() + numActive, mR3(0));
    // Select the desired shifting technique. For now use only XSPH with compressibility.
    switch (m_data_mgr.paramsH->shifting_method) {
        case ShiftingMethod::XSPH:  
            if (m_data_mgr.paramsH->is_uniform)
                Calc_Shifting_D_csph<ShiftingMethod::XSPH, true><<<numBlocks, numThreads>>>(
                    mR3CAST(m_data_mgr.vel_XSPH_D), mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velD),
                    mR4CAST(sortedSphMarkersD->rhoPresEnD), nullptr, 
                    U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList), numActive, m_errflagD);
            else
                Calc_Shifting_D_csph<ShiftingMethod::XSPH, false><<<numBlocks, numThreads>>>(
                    mR3CAST(m_data_mgr.vel_XSPH_D), mR4CAST(sortedSphMarkersD->posRadD),
                    mR3CAST(sortedSphMarkersD->velD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
                    R1CAST(m_data_mgr.sortedMassD), U1CAST(m_data_mgr.numNeighborsPerPart),
                    U1CAST(m_data_mgr.neighborList), numActive, m_errflagD);
            break;
    }
    cudaCheckErrorFlag(m_errflagD, "Calc_Shifting_D");
}


}  // end namespace compressible
}  // end namespace chrono::fsi::sph
