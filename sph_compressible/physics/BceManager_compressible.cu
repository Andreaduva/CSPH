// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Extension of base class for processing boundary condition enforcing (bce) markers forces
// in FSI system.
// =============================================================================

//// TODO: There need to be a better way to compute bce marker forces for different solvers.
//// For explicit solver, it is essentially derivVelRhoD times the marker mass,

#include <type_traits>
#include <fstream>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/copy.h>

#include "chrono_fsi/sph/physics/BceManager.cuh"
#include "chrono_fsi/sph/physics/SphGeneral.cuh"

#include "chrono_fsi/sph_compressible/physics/BceManager_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"

namespace chrono::fsi::sph{
namespace compressible {

BceManager_csph::BceManager_csph(FsiDataManager_csph& data_mgr, bool verbose, bool check_errors)
    : m_data_mgr(data_mgr),
      m_verbose(verbose),
      m_check_errors(check_errors) {
    m_totalForceRigid.resize(0);
    m_totalTorqueRigid.resize(0);
    m_rigid_block_size = 512;
    m_rigid_grid_size = 0;
}

BceManager_csph::~BceManager_csph() {}

// -----------------------------------------------------------------------------

// Initializes the system and construct BCE at the initial configuration of the system
void BceManager_csph::Initialize(std::vector<int> fsiBodyBceNum) {
    // initialize values in the global static variables paramsD_csph, countersD_csph defined in SphGeneral_compressible.cuh
    cudaMemcpyToSymbolAsync(paramsD_csph, m_data_mgr.paramsH.get(),
                            sizeof(ChFsiParamsSPH_csph));  // first is destination to copy, second is source, third is size
    cudaMemcpyToSymbolAsync(countersD_csph, m_data_mgr.countersH.get(),
                            sizeof(Counters_csph));        

    // Resizing the arrays used to modify the BCE velocity and pressure according to Adami
    // ADAPT TO COMPRESSIBLE TREATMENT OF BCE
    m_totalForceRigid.resize(m_data_mgr.countersH->numFsiBodies);
    m_totalTorqueRigid.resize(m_data_mgr.countersH->numFsiBodies);

    ////int haveGhost = (m_data_mgr.countersH->numGhostMarkers > 0) ? 1 : 0;
    ////int haveHelper = (m_data_mgr.countersH->numHelperMarkers > 0) ? 1 : 0;
    int haveRigid = (m_data_mgr.countersH->numFsiBodies > 0) ? 1 : 0;

    // Populate local position of BCE markers - on rigid bodies
    if (haveRigid) {
        SetForceAccumulationBlocks(fsiBodyBceNum);

        m_data_mgr.rigid_BCEcoords_D = m_data_mgr.rigid_BCEcoords_H;
        m_data_mgr.rigid_BCEsolids_D = m_data_mgr.rigid_BCEsolids_H;
        //// TODO (Huzaifa): Try to see if this additional function is needed
        UpdateBodyMarkerStateInitial();
    }

}

// -----------------------------------------------------------------------------

void BceManager_csph::SetForceAccumulationBlocks(std::vector<int> fsiBodyBceNum) {
    // 1 zero is pre added so that in a block with invalid threads, there is no need to map back the invalid threads in
    // the global array. This is only required in the very next block
    thrust::host_vector<uint> rigid_valid_threads(0);
    thrust::host_vector<uint> rigid_accumulated_threads(1, 0);

    uint accumulatedPaddedThreads = 0;
    for (int irigid = 0; irigid < fsiBodyBceNum.size(); irigid++) {
        // Calculate block requirements with thread padding to ensure that during rigid body force accumulation each
        // block only handles one rigid body.
        //  - for bodies with > m_rigid_block_size BCE markers, split the work into multiple blocks and pad the last
        //    block with invalid threads.
        //  - for bodies with <= m_rigid_block_size BCE markers, need only one block and pad that block with invalid
        //    threads.
        // Additionally, accumulate the number of padded thread in each block to ensure we go to the right global index
        // which does not account for thread padding.
        uint numBlocks = (fsiBodyBceNum[irigid] + m_rigid_block_size - 1) / m_rigid_block_size;
        for (uint blockNum = 0; blockNum < numBlocks; blockNum++) {
            uint numValidThreads = min(m_rigid_block_size, fsiBodyBceNum[irigid] - blockNum * m_rigid_block_size);
            rigid_valid_threads.push_back(numValidThreads);
            uint numPaddedThreadsInThisBlock = m_rigid_block_size - numValidThreads;
            accumulatedPaddedThreads += numPaddedThreadsInThisBlock;
            rigid_accumulated_threads.push_back(accumulatedPaddedThreads);
        }
        m_rigid_grid_size += numBlocks;
    }

    // Copy vectors to device
    m_rigid_valid_threads = rigid_valid_threads;
    m_rigid_accumulated_threads = rigid_accumulated_threads;
}

// -----------------------------------------------------------------------------
// CalcRigidBceAcceleration
// -----------------------------------------------------------------------------

// kernel function used as helper to compute accelerations in parallel. Not the private method in class BceManager.
// Computes accelerations of rigid bce markers only. Wall bce markers excluded from output
// Accelerations are compute using laws of rigid body motion.
__global__ void CalcRigidBceAccelerationD(
    Real3* accelerations,        // BCE marker accelerations (output). This vector/array is sorted by hash
    const Real3* BCE_pos_local,  // BCE body-local coordinates
    const uint* body_IDs,        // rigid body ID for each BCE marker
    const Real4* body_rot,       // body orientation (relative to global frame)
    const Real3* body_angvel,    // body ang. vels. (relative to global frame)
    const Real3* body_linacc,    // body lin. acels. (relative to global frame)
    const Real3* body_angacc,    // body ang. acels. (relative to global frame)
    const uint* mapOriginalToSorted) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= countersD_csph.numRigidMarkers)  // number of bce markers associated to rigid bodies (different from bce
                                             // markers on boundaries)
        return;

    uint marker_index = index + countersD_csph.startRigidMarkers;  // offset so that we access only bce rigid markers. This
                                                              // is the "linear index" meaning ordered by index
    uint sorted_index = mapOriginalToSorted[marker_index];    // access the location of such index in the vactor of
                                                              // indexes ordered by hash

    int body_ID = body_IDs[index];  // body ID. bodies don't get sorted by hash
    Real3 local = BCE_pos_local[index];

    Real3 u, v, w;
    RotationMatrixFromQuaternion(u, v, w, body_rot[body_ID]);

    // linear acceleration
    Real3 acc = body_linacc[body_ID];

    // centrifugal acceleration
    Real3 omega = body_angvel[body_ID];
    Real3 omega_cross = cross(omega, local);
    Real3 omega_cross_cross = cross(omega, omega_cross);

    acc += mR3(dot(u, omega_cross_cross), dot(v, omega_cross_cross), dot(w, omega_cross_cross));

    // tangential acceleration
    Real3 alpha_cross = cross(body_angacc[body_ID], local);
    acc += mR3(dot(u, alpha_cross), dot(v, alpha_cross), dot(w, alpha_cross));

    accelerations[sorted_index] = acc;  // writes the acceleration in order by hash
}


// private method of BceManager class, host function. Calls the associated Cuda kernel.
void BceManager_csph::CalcRigidBceAcceleration() {

    uint numThreads, numBlocks;
    // computes proper cuda grid sizes to work with numRigidMarkers threads
    computeCudaGridSize((uint)m_data_mgr.countersH->numRigidMarkers, 256, numBlocks,numThreads);  
    // calls the Cuda kernel to actually perform computations
    CalcRigidBceAccelerationD<<<numBlocks, numThreads>>>(  //
        mR3CAST(m_data_mgr.bceAcc),                        // kernel will populate it with numRigidMarkers accelerations (boundary bce excluded) //
        mR3CAST(m_data_mgr.rigid_BCEcoords_D), U1CAST(m_data_mgr.rigid_BCEsolids_D),               
        mR4CAST(m_data_mgr.fsiBodyState_D->rot), mR3CAST(m_data_mgr.fsiBodyState_D->ang_vel),      
        mR3CAST(m_data_mgr.fsiBodyState_D->lin_acc), mR3CAST(m_data_mgr.fsiBodyState_D->ang_acc),  
        U1CAST(m_data_mgr.markersProximity_D->mapOriginalToSorted)                                 
    );

    cudaDeviceSynchronize();
    if (m_check_errors) {
        cudaCheckError();
    }

}


// -----------------------------------------------------------------------------
// Public method to calculate accelerations of solid BCE markers -> load m_data_mgr.bceAcc
// Boundary bce markers excluded.
// -----------------------------------------------------------------------------

void BceManager_csph::updateBCEAcc() {
    if (m_data_mgr.countersH->numRigidMarkers > 0)
        CalcRigidBceAcceleration();
}

// -----------------------------------------------------------------------------
// Rigid_Forces_Torques
// -----------------------------------------------------------------------------

// kernel function to compute forces and torques on rigid bodies in parallel. Invoked by the methods in BceManager class
// TO BE UPDATED DEPENDING ON PROPER LAW FOR COMPRESSIBLE SPH FLUID DYNAMICS?
// DON'T KNOW AS HERE FORCE ACTING ON BODY DUE TO A RIGID MARKER IS JUST THE ACCELERATION DUE TO FLUID DYNAMICS
// LAWS TIMES ITS MASS
template <bool IsUniform>
__global__ void CalcRigidForces_D( Real3* __restrict__ body_forces,                             // write - output is one force vector for each body
                                   Real3* __restrict__ body_torques,                            // write - output is one torque vector for each body
                                   const uint* __restrict__ rigidBodyBlockValidThreads,         // read
                                   const uint* __restrict__ rigidBodyAccumulatedPaddedThreads,  // read
                                   const Real5* __restrict__ derivatives,                       // read - (velocity, rho, en) derivative of markers
                                   const Real4* __restrict__ positions,                         // read - (position, h) of markers
                                   const uint* __restrict__ body_IDs,                           // read - ID of body the marker belongs to
                                   const Real3* __restrict__ body_pos,                          // read - position of bodies
                                   const uint* __restrict__ mapOriginalToSorted,                // read - map original index to sorted of bce markers
                                   const uint numRigidMarkers,                                  // read
                                   const uint startRigidMarkers,                                // read - starting index of rigid markers in the particle vector
                                   const Real markerMass,
                                   const Real* __restrict__ sortedMassD = nullptr) {                             

    extern __shared__ char sharedMem[];  // declaration of dynamic shared memory (not known at compile time)
    const uint blockSize = blockDim.x;   // threads per block

    // Shared memory allocations
    Real3* sharedForces = (Real3*)sharedMem;                  // Size: blockSize
    Real3* sharedTorques = (Real3*)&sharedForces[blockSize];  // Size: blockSize

    uint threadIdx_x = threadIdx.x;
    uint global_index = blockIdx.x * blockDim.x + threadIdx_x;

    // Valid threads in current block
    uint validThreads = rigidBodyBlockValidThreads[blockIdx.x];
    // Valid threads in previous block
    uint paddedThreads = rigidBodyAccumulatedPaddedThreads[blockIdx.x];

    uint global_index_padded = global_index - paddedThreads;

    if (global_index_padded >= numRigidMarkers)
        return;

    uint marker_index = global_index_padded + startRigidMarkers;
    uint sorted_index = mapOriginalToSorted[marker_index];

    const Real Mass = IsUniform ? markerMass : sortedMassD[sorted_index];

    // Get body ID for the current marker
    uint body_ID = body_IDs[global_index_padded];

    Real3 Force = make_Real3(0.0f, 0.0f, 0.0f);
    Real3 Torque = make_Real3(0.0f, 0.0f, 0.0f);
    // removed ISPH check
    // Force is simply computed as the current marker acceleration due to fluid dynamics times its mass
    if (threadIdx_x < validThreads) {
        Force = mR3(derivatives[sorted_index]) * Mass;             // Force computed as Dv/Dt*mass
        Real3 dist3 = mR3(positions[sorted_index]) - body_pos[body_ID];  // distance marker - center of mass of its body
        Torque = cross(dist3, Force);
    }

    sharedForces[threadIdx_x] = Force;
    sharedTorques[threadIdx_x] = Torque;

    __syncthreads();

    // Reduce to one force vector per rigid body
    // Standard block wise reduction - each block only contains elements from a single rigid body
    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx_x < stride && threadIdx_x + stride < validThreads) {
            sharedForces[threadIdx_x].x += sharedForces[threadIdx_x + stride].x;
            sharedForces[threadIdx_x].y += sharedForces[threadIdx_x + stride].y;
            sharedForces[threadIdx_x].z += sharedForces[threadIdx_x + stride].z;

            sharedTorques[threadIdx_x].x += sharedTorques[threadIdx_x + stride].x;
            sharedTorques[threadIdx_x].y += sharedTorques[threadIdx_x + stride].y;
            sharedTorques[threadIdx_x].z += sharedTorques[threadIdx_x + stride].z;
        }
        __syncthreads();
    }

    if (threadIdx_x == 0) {
        // Atomic addition to global arrays
        atomicAdd(&body_forces[body_ID].x, sharedForces[0].x);
        atomicAdd(&body_forces[body_ID].y, sharedForces[0].y);
        atomicAdd(&body_forces[body_ID].z, sharedForces[0].z);

        atomicAdd(&body_torques[body_ID].x, sharedTorques[0].x);
        atomicAdd(&body_torques[body_ID].y, sharedTorques[0].y);
        atomicAdd(&body_torques[body_ID].z, sharedTorques[0].z);
    }
}



// Public class method, host function, that computes forces and torques exerted on rigid bodies by fluid. Calls the
// CalcRigidForces Cuda kernel.
void BceManager_csph::Rigid_Forces_Torques() {

    if (m_data_mgr.countersH->numFsiBodies == 0)   // no fsi bodies
        return;

    thrust::fill(m_data_mgr.rigid_FSI_ForcesD.begin(), m_data_mgr.rigid_FSI_ForcesD.end(), mR3(0));
    thrust::fill(m_data_mgr.rigid_FSI_TorquesD.begin(), m_data_mgr.rigid_FSI_TorquesD.end(), mR3(0));

    // Calculate shared memory size
    size_t sharedMemSize = 2 * m_rigid_block_size * sizeof(Real3);

    if (m_data_mgr.paramsH->is_uniform) {
        CalcRigidForces_D<true><<<m_rigid_grid_size, m_rigid_block_size, sharedMemSize>>>(
            mR3CAST(m_data_mgr.rigid_FSI_ForcesD), mR3CAST(m_data_mgr.rigid_FSI_TorquesD),
            U1CAST(m_rigid_valid_threads), U1CAST(m_rigid_accumulated_threads), mR5CAST(m_data_mgr.derivVelRhoEnD),
            mR4CAST(m_data_mgr.sortedSphMarkers2_D->posRadD), U1CAST(m_data_mgr.rigid_BCEsolids_D),
            mR3CAST(m_data_mgr.fsiBodyState_D->pos), /* rigid_BCEsolids_d is a vector with body ID for each marker */
            U1CAST(m_data_mgr.markersProximity_D->mapOriginalToSorted), (uint)m_data_mgr.countersH->numRigidMarkers,
            (uint)m_data_mgr.countersH->startRigidMarkers, m_data_mgr.paramsH->markerMass);
    } 
    else if (!m_data_mgr.paramsH->is_uniform) {
        CalcRigidForces_D<false><<<m_rigid_grid_size, m_rigid_block_size, sharedMemSize>>>(
            mR3CAST(m_data_mgr.rigid_FSI_ForcesD), mR3CAST(m_data_mgr.rigid_FSI_TorquesD),
            U1CAST(m_rigid_valid_threads), U1CAST(m_rigid_accumulated_threads), mR5CAST(m_data_mgr.derivVelRhoEnD),
            mR4CAST(m_data_mgr.sortedSphMarkers2_D->posRadD), U1CAST(m_data_mgr.rigid_BCEsolids_D),
            mR3CAST(m_data_mgr.fsiBodyState_D->pos), /* rigid_BCEsolids_d is a vector with body ID for each marker */
            U1CAST(m_data_mgr.markersProximity_D->mapOriginalToSorted), (uint)m_data_mgr.countersH->numRigidMarkers,
            (uint)m_data_mgr.countersH->startRigidMarkers, m_data_mgr.paramsH->markerMass, R1CAST(m_data_mgr.sortedMassD));
    }

    if (m_check_errors) {
        cudaCheckError();
    }
}



// -----------------------------------------------------------------------------
// UpdateBodyMarkerState
// UpdateBodyMarkerStateInitial
// -----------------------------------------------------------------------------

// cuda kernel which actively updates the state of bce markers depending on rigid body positions and velocities
// Looks like it requires the array of positions and velocities to have the order: fluid bce, boundary bce, rigid bce,
// etc. This updates the states of particles ordered by hash
__global__ void UpdateBodyMarkerState_D(Real4* positions,            // global marker positions (sorted)
                                        Real3* velocities,           // global marker velocities (sorted)
                                        const Real3* BCE_pos_local,  // BCE body-local coordinates
                                        const uint* body_IDs,        // rigid body ID for each BCE marker
                                        const Real3* body_pos,       // body positions (relative to global frame)
                                        const Real4* body_rot,       // body orientation (relative to global frame)
                                        const Real3* body_linvel,    // body lin. vels. (relative to global frame)
                                        const Real3* body_angvel,    // body ang. vels. (relative to global frame)
                                        const uint* mapOriginalToSorted) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= countersD_csph.numRigidMarkers)
        return;

    uint marker_index = index + countersD_csph.startRigidMarkers;  // index of rigid bce in original order (assume first fluid then boundary etc)
    uint sorted_index = mapOriginalToSorted[marker_index];  // index of rigid bce when order by hash

    int body_ID = body_IDs[index];
    Real3 local = BCE_pos_local[index];

    Real3 u, v, w;
    RotationMatrixFromQuaternion(u, v, w, body_rot[body_ID]);

    // BCE marker position
    Real h = positions[sorted_index].w;
    Real3 pos = body_pos[body_ID] + mR3(dot(u, local), dot(v, local), dot(w, local));
    positions[sorted_index] = mR4(pos, h);

    // BCE marker velocity
    Real3 omega_cross = cross(body_angvel[body_ID], local);
    velocities[sorted_index] = body_linvel[body_ID] + mR3(dot(u, omega_cross), dot(v, omega_cross), dot(w, omega_cross));
}


// Unsorted referring to vectors not ordered by hash but by index. Still assume first fluid markers, then boundary bce,
// then rigid bce markers, etc
__global__ void UpdateBodyMarkerStateUnsorted_D(Real4* positions,            // global marker positions (original)
                                                Real3* velocities,           // global marker velocities (original)
                                                const Real3* BCE_pos_local,  // BCE body-local coordinates
                                                const uint* body_IDs,        // rigid body ID for each BCE marker
                                                const Real3* body_pos,       // body positions (relative to global frame)
                                                const Real4* body_rot,       // body orientation (relative to global frame)
                                                const Real3* body_linvel,    // body lin. vels. (relative to global frame)
                                                const Real3* body_angvel) {  // body ang. vels. (relative to global frame)
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= countersD_csph.numRigidMarkers)
        return;

    uint marker_index = index + countersD_csph.startRigidMarkers;

    int body_ID = body_IDs[index];
    Real3 local = BCE_pos_local[index];

    Real3 u, v, w;
    RotationMatrixFromQuaternion(u, v, w, body_rot[body_ID]);

    // BCE marker position
    Real h = positions[marker_index].w;
    Real3 pos = body_pos[body_ID] + mR3(dot(u, local), dot(v, local), dot(w, local));
    positions[marker_index] = mR4(pos, h);

    // BCE marker velocity
    Real3 omega_cross = cross(body_angvel[body_ID], local);
    velocities[marker_index] =
        body_linvel[body_ID] + mR3(dot(u, omega_cross), dot(v, omega_cross), dot(w, omega_cross));
}



void BceManager_csph::UpdateBodyMarkerState() {
    if (m_data_mgr.countersH->numFsiBodies == 0)
        return;

    uint nBlocks, nThreads;
    computeCudaGridSize((uint)m_data_mgr.countersH->numRigidMarkers, 256, nBlocks, nThreads);

    UpdateBodyMarkerState_D<<<nBlocks, nThreads>>>(
        mR4CAST(m_data_mgr.sortedSphMarkers2_D->posRadD), mR3CAST(m_data_mgr.sortedSphMarkers2_D->velD),
        mR3CAST(m_data_mgr.rigid_BCEcoords_D),
        U1CAST(m_data_mgr.rigid_BCEsolids_D),  // 4th input contains body ID for each rigid marker
        mR3CAST(m_data_mgr.fsiBodyState_D->pos), mR4CAST(m_data_mgr.fsiBodyState_D->rot),
        mR3CAST(m_data_mgr.fsiBodyState_D->lin_vel), mR3CAST(m_data_mgr.fsiBodyState_D->ang_vel),
        U1CAST(m_data_mgr.markersProximity_D->mapOriginalToSorted));

    if (m_check_errors) {
        cudaCheckError();
    }
}



void BceManager_csph::UpdateBodyMarkerStateInitial() {
    if (m_data_mgr.countersH->numFsiBodies == 0)
        return;

    uint nBlocks, nThreads;
    computeCudaGridSize((uint)m_data_mgr.countersH->numRigidMarkers, 256, nBlocks, nThreads);

    UpdateBodyMarkerStateUnsorted_D<<<nBlocks, nThreads>>>(
        mR4CAST(m_data_mgr.sphMarkers_D->posRadD), mR3CAST(m_data_mgr.sphMarkers_D->velD),
        mR3CAST(m_data_mgr.rigid_BCEcoords_D),
        U1CAST(m_data_mgr.rigid_BCEsolids_D),  // 4th input contains body ID for each rigid marker
        mR3CAST(m_data_mgr.fsiBodyState_D->pos), mR4CAST(m_data_mgr.fsiBodyState_D->rot),
        mR3CAST(m_data_mgr.fsiBodyState_D->lin_vel), mR3CAST(m_data_mgr.fsiBodyState_D->ang_vel));

    if (m_check_errors) {
        cudaCheckError();
    }
}


}  // end namespace compressible
}  // end namespace chrono::fsi::sph
