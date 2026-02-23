// ==========================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Modification of base class for proximity computations to account for compressible
// sph data types
// =============================================================================

#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <fstream>
#include <cmath>
#include "chrono_fsi/sph/physics/CollisionSystem.cuh"
#include "chrono_fsi/sph/physics/SphGeneral.cuh"
#include "chrono_fsi/sph/utils/UtilsDevice.cuh"

#include "chrono_fsi/sph_compressible/physics/CollisionSystem_compressible.cuh"
// Is following header file NEEDED ????
#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"
#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {

// ===============================================================================================
// Public methods of class CollisionSystem_csph are split in multiple, less intensive, CUDA kernel functions 
// ===============================================================================================

struct compareRadValue {
    __host__ __device__ bool operator()(const Real4& left, const Real4& right) { return left.w < right.w; }
};


// If h is allowed to vary, we have to change the grid parameters and base them on current
// maximum value of h
void CollisionSystem_csph::UpdateGridParams(std::shared_ptr<SphMarkerDataD_csph> sphMarkersD) {

    // find max value of h among input markers
    auto it = thrust::max_element(sphMarkersD->posRadD.begin(), sphMarkersD->posRadD.end(), compareRadValue());
    // index of max_h and its value:
    int idx = it - sphMarkersD->posRadD.begin();
    Real4 maxRadMarker = sphMarkersD->posRadD[idx];
    Real h_max = maxRadMarker.w;

    auto m_paramsH = m_data_mgr.paramsH;  // type is a shared pointer
    
    // If ADKE scheme, h_max is maximum between max h of particles and h0 used in pilot density estimate
    if (m_paramsH->h_evolution == H_evolution_csph::ADKE) {
        Real d = m_paramsH->ADKE_D;
        Real d0 = m_paramsH->d0;
        Real h_ref = d * d0;
        h_max = std::max(h_max, m_paramsH->ADKE_D * m_paramsH->d0);
    }
        


    // Set up subdomains and grid parameters for faster neighbor particle search
    // Ok to use default h parameter as this is the initialization phase.

    // if a grid cell has size of about 1 or 2 kernel radius, side0 represents how many cells we have along a direction:
    int3 side0 = mI3( max(1, (int)floor((m_paramsH->cMax.x - m_paramsH->cMin.x) / (m_paramsH->h_multiplier * h_max)) ),
                      max(1, (int)floor((m_paramsH->cMax.y - m_paramsH->cMin.y) / (m_paramsH->h_multiplier * h_max)) ),
                      max(1, (int)floor((m_paramsH->cMax.z - m_paramsH->cMin.z) / (m_paramsH->h_multiplier * h_max)) ) );
    // fixing number of cells (being side0) for each direction, binsize3 tells the sizes of each cell.
    Real3 binSize3 = mR3((m_paramsH->cMax.x - m_paramsH->cMin.x) / side0.x, (m_paramsH->cMax.y - m_paramsH->cMin.y) / side0.y,
                          (m_paramsH->cMax.z - m_paramsH->cMin.z) / side0.z);

    m_paramsH->binSize0 = (binSize3.x > binSize3.y) ? binSize3.x : binSize3.y;  // take maximum between x and y
    m_paramsH->binSize0 = (m_paramsH->binSize0 > binSize3.z) ? m_paramsH->binSize0 : binSize3.z;  // maximum between z and previous value

    // fixing the length of the cell (cubic here) as being binsize0, SIDE tells how many cells along each direction
    // of the computational domain.
    int3 SIDE = mI3( max(1, int((m_paramsH->cMax.x - m_paramsH->cMin.x) / m_paramsH->binSize0 + .1) ),
                     max(1, int((m_paramsH->cMax.y - m_paramsH->cMin.y) / m_paramsH->binSize0 + .1) ),
                     max(1, int((m_paramsH->cMax.z - m_paramsH->cMin.z) / m_paramsH->binSize0 + .1) ) );
    Real mBinSize = m_paramsH->binSize0;
    m_paramsH->gridSize = SIDE;                               // number of cells along each domain direction
    m_paramsH->cellSize = mR3(mBinSize, mBinSize, mBinSize);  // dimension of the single cell (cubic)

    // Precompute grid min and max bounds considering whether we have periodic boundaries or not
    m_paramsH->minBounds = make_int3(m_paramsH->x_periodic ? INT_MIN : 0, m_paramsH->y_periodic ? INT_MIN : 0,
                                     m_paramsH->z_periodic ? INT_MIN : 0);

    m_paramsH->maxBounds = make_int3(m_paramsH->x_periodic ? INT_MAX : m_paramsH->gridSize.x - 1,
                                     m_paramsH->y_periodic ? INT_MAX : m_paramsH->gridSize.y - 1,
                                     m_paramsH->z_periodic ? INT_MAX : m_paramsH->gridSize.z - 1);

    // copy modified parameters to device:
    cudaMemcpyToSymbolAsync(paramsD_csph, m_paramsH.get(), sizeof(ChFsiParamsSPH_csph));
    cudaCheckError();
}




//------------------------------ Create the active list --------------------------------------------
// Writes only if active - thus active list has all active partices at the front
// The index's are the index of the original particle arrangement
// After the active particles, we have the random values that were present at initialization.
// __restrict__ keyword allows compiler to make optimization and caching. Does not modify code behavior.
// ExtendedActivityIdD vector of size numAllMarkers with 0 or 1 to mark if particle is active or not.
// activeListD is an array which stores the index of active particles only (so values of 0 in extendedAct are ignored)
// prefix_sum is array obtained by an exclusive prefix sum where we can find the active particle indexes inside
//  the array ExtendedActivity --> E.g. extendedActivityId = [1,0,1,1,0,1,0]  -> only 4 active particles so activeList will have length
// 4 instead of 7. Then make a prefix sum on exteAct and get [0,1,1,2,3,3,4] It means: if particle i (tid) is active
// (extendedActivityId(tid) == 1), then place its index i (tid) at position prefixSum(tid) into output array. Does not
// give back numActive, the number of active particles.
__global__ void fillActiveListD( const uint* __restrict__ prefixSum,        // read - has position of output activeList where to insert index of particle
                                                                            // if this is active. (computed externally with an exclusive prefix scan)
                                 const int32_t* __restrict__ extendedActivityIdD,  // read
                                 uint* __restrict__ activeListD,                   // write
                                 uint numAllMarkers)      {
    
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // thread index also used as particle index
    if (tid >= numAllMarkers)
        return;  // index out of bounds

    // Check if the value is 1 (active)
    if (extendedActivityIdD[tid] == 1) {
        uint writePos = prefixSum[tid];  // an integer in [0..(numActive-1)]. prefixSum tells where to write the index
                                         // in activeListD
        activeListD[writePos] = tid;     // List with index of active particle.
    }
}



// --------------------------------- Compute particle hash value -----------------------------------
// calcHashD :
// 1. Get particle index determined by the block and thread we are in.
// 2. From x, y, z position, determine which bin it is in.
// 3. Calculate hash from bin index.
// 4. Store hash and particle index associated with it.
// Again only the active particles are stored upfront in gridMarkerHashD and gridMarkerIndexD, values after numActive
// index (if any behind it) are meaningless.
__global__ void calcHashD(uint* gridMarkerHashD,    // gridMarkerHash Store particle hash here
                          uint* gridMarkerIndexD,   // gridMarkerIndex Store particle index here
                          const uint* activeListD,  // active list --> see above. It has the indexes of active particles.
                          const Real4* posRad,      // positions of all particles (SPH and BCE)
                          const Real4* rhoPresEn,
                          uint numActive,           // number of active particles (activeList has numParticles size, but after numActive values makes
                                                    // no sense), suppose computed outside
                          volatile bool* error_flag) {
    // Computee the thread index.
    uint globalIndex =
        blockIdx.x * blockDim.x +
        threadIdx.x;  // Now not directly associated to a particle index, because need to work on active particles only.
    if (globalIndex >= numActive)
        return;  // more stringent condition: not just >= numAllParticles

    uint index = activeListD[globalIndex];  // from thread index get corresponding active particle index
    Real3 p = mR3(posRad[index]);           // active particle pos

    if (!IsFinite(p)) {
        printf("[calcHashD] index %d position is NaN\n", index);
        *error_flag = true;
        return;
    }

    // Check particle is inside the domain.
    // Assuming worldOrigin is set properly with respect to the computational domain
    // I guess the convention is that the domain extends from worldOrigin along positive direction of the three axes.
    // So this check may have sense: a particle position can't be too out of bounds in the negative directions
    // The 40*h here acts as a buffer: particles are allowed to be a bit out of border, but not much.
    Real3 boxCorner = paramsD_csph.worldOrigin - mR3(40 * paramsD_csph.h);
    if (p.x < boxCorner.x || p.y < boxCorner.y || p.z < boxCorner.z && IsFluidParticle(rhoPresEn[index].w) ) {
        printf("[calcHashD] index %u (%f %f %f) out of min boundary (%f %f %f)\n",  //
               index, p.x, p.y, p.z, boxCorner.x, boxCorner.y, boxCorner.z);
        *error_flag = true;
        return;
    }
    // See here to compute the maximum boundary the box dimensions are added to worldOrigin.
    // in paramsD the boxDims are commented as the dimensions of the domain (not half dimensions)
    // Again add a small buffer of 40*h.
    boxCorner = paramsD_csph.worldOrigin + paramsD_csph.boxDims + mR3(40 * paramsD_csph.h);
    if (p.x > boxCorner.x || p.y > boxCorner.y || p.z > boxCorner.z && IsFluidParticle(rhoPresEn[index].w) ) {
        printf("[calcHashD] index %u (%f %f %f) out of max boundary (%f %f %f)\n",  //
               index, p.x, p.y, p.z, boxCorner.x, boxCorner.y, boxCorner.z);
        *error_flag = true;
        return;
    }

    // Get x,y,z bin index in grid
    int3 gridPos = calcGridPos_csph(p);  // These functions are defined in SphGeneral.cuh
    // Calculate a hash from the bin index
    uint hash = calcGridHash_csph(gridPos);
    // Store grid hash
    // grid hash is a scalar cell ID
    gridMarkerHashD[globalIndex] = hash;  // at position = thread_index store corresponding cell hash
    // Store particle index associated to the hash we stored in gridMarkerHashD
    gridMarkerIndexD[globalIndex] = index;  // at position = thread_index we have the particle index
}






// --------------------------- Compute start/end index of particles inside bins with equal hash value------------------------
// kernel function requiring array of hash values and corresponding marker index, sorted by hash.
// start and end index refer to positions in arrays which are sorted by hash.
__global__ void findCellStartEndD(uint* cellStartD,        // output: cell start index
                                  uint* cellEndD,          // output: cell end index
                                  uint* gridMarkerHashD,   // input: sorted grid hashes
                                  uint* gridMarkerIndexD,  // input: sorted particle indices
                                  uint numActive) {

    extern __shared__ uint sharedHash[];  // blockSize + 1 elements. extern __shared__ used to declare dynamic shared memory: size not
                                          // known at compile time. Shared memory is a per-block resource.
    // Get the particle index the current thread is supposed to be looking at.
    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // thread index used as particle index
    uint hash;

    if (index >= numActive)  // only work on active particles
        return;

    hash = gridMarkerHashD[index];  // hash value of current particle. Specific to each thread.
    if (hash == UINT_MAX)
        return;
    // Load hash data into shared memory so that we can look at neighboring
    // particle's hash value without loading two hash values per thread
    // shared memory works per block, so use just threadIdx.x
    // The +1 is to be able to get also the hash value from neighbour particles (neighbour in terms of block position).
    // Kind of padding to avoid two loadings per threads from shared memory. In this way, the thread thid will later
    // read from sharedHash[thid] the value actually loaded by thid-1 and can compare such value with its own hash value
    // stored in the variable hash.
    sharedHash[threadIdx.x + 1] = hash;

    // first thread in block must load neighbor particle hash (the neighbor here means the previous one)
    if (index > 0 && threadIdx.x == 0)  // so does not consider block 0, considers thread 0 in blocks different from 0
                                        // Makes sense: thread 0 in block 0 does not have any neighbors, to be treated ad hoc.
        sharedHash[0] = gridMarkerHashD[index - 1];  // loads hash value of previous particle.

    __syncthreads();
    if (sharedHash[threadIdx.x] == UINT_MAX)
        return;

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell. As it
    // isn't the first particle, it must also be the cell end of
    // the previous particle's cell.
    // hash corresponds to the current thid particle, while sharedHash[threadIdx.x] was loaded by threadIdx.x - 1.
    // We save the global index because refers to the actual particle index.
    if (index == 0 || hash != sharedHash[threadIdx.x]) {
        cellStartD[hash] = index;  // save at hash position because we have one stard index and one end index for each
                                   // unique hash value.
        if (index > 0)
            cellEndD[sharedHash[threadIdx.x]] =
                index;  // sharedHash[threadIdx.x] gives the hash value of particle threadIdx.x-1, so the previous one.
    }

    if (index == numActive - 1)
        cellEndD[hash] = index + 1;
}




//-------------------------------- mapping original to sorted indices----------------------
// gridMarkerIndex contains particles indexes sorted by hash.
// example illustrating the logic. Assume 4 particles with indexes [0,1,2,3]. After sorting them by hash we have
// gridMarkerIndex = [2,0,3,1]. So particle 0 will be at position 1 after such sorting -> mapOriginalToSorted[0] = 1.
// the mapOriginalToSorted array will then be = [1,3,0,2].
// It's the inverse mapping of gridMarkerIndex, which maps from sorted to original.
__global__ void OriginalToSortedD(uint* mapOriginalToSorted, uint* gridMarkerIndex, uint numActive) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numActive)
        return;

    uint index = gridMarkerIndex[id];  // index of particle associated to id, so ordered by hash.

    mapOriginalToSorted[index] = id;  // access mapOriginalToSorted with index, so the array will be ordered following
                                      // the particles indexes only. At each position (particle indexes) will be
                                      // associated the position it has in the array of indexes ordered by hash.
}



// ---------------------------------- Cuda kernel function to sort particles by hash ----------------------------------
// reorder data from sorted by particle index to sorted by hash values.
template <bool isUniform>
__global__ void reorderDataD(
    const uint* __restrict__ gridMarkerIndexD,                // read - indices of particles, sorted by hash
    Real4* __restrict__ sortedPosRadD,                        // write - ordered by hash
    Real3* __restrict__ sortedVelD,                           // write - ordered by hash
    Real4* __restrict__ sortedRhoPreEnD,                      // write - ordered by hash
    Real* __restrict__ sortedSoundD,                     // write - ordered by hash
    Real* __restrict__ sortedMassD,
    int32_t* __restrict__ activityIdentifierSortedD,          // write - ordered by hash
    const Real4* __restrict__ posRadD,                        // read - ordered by particle index
    const Real3* __restrict__ velD,                           // read - ordered by particle index
    const Real4* __restrict__ rhoPresEnD,                     // read - ordered by particle index
    const Real* __restrict__ soundD,                     // read - ordered by particle index
    const Real* __restrict__ massD,
    const int32_t* __restrict__ activityIdentifierOriginalD,  // read - ordered by particle index.
    uint numActive) {

    uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // threadindex, now working on position ordered by hash
    if (tid >= numActive)
        return;

    uint originalIndex = gridMarkerIndexD[tid];  // gets the original particle index currently considered.

    // Read from original arrays
    Real4 posRadVal = posRadD[originalIndex];
    Real3 velVal = velD[originalIndex];
    Real4 rhoPreEnVal = rhoPresEnD[originalIndex];
    int32_t activityIdentifierVal = activityIdentifierOriginalD[originalIndex];
    Real soundVal = soundD[originalIndex];

    if (!IsFinite(mR3(posRadVal))) {
        printf("Error! reorderDataD_ActiveOnly: posRad is NAN at original index %u\n", originalIndex);
    }

    // Write to sorted arrays at index 'tid'
    sortedPosRadD[tid] = posRadVal;
    sortedVelD[tid] = velVal;
    sortedRhoPreEnD[tid] = rhoPreEnVal;
    sortedSoundD[tid] = soundVal;
    activityIdentifierSortedD[tid] = activityIdentifierVal;

    if (!isUniform) {
        sortedMassD[tid] = massD[originalIndex];
    }
}

// =============================================================================
//                            Actual neighbor search kernels 
 
 
 
// counts the number of neighbour particles, inside the support radius of the kernel function W_h
__global__ void neighborSearchNum_csph(const Real4* sortedPosRad,    // read - sorted by hash
                                  const Real4* sortedRhoPreEn,  // read - sorted by hash - appears unused
                                  const uint* cellStart,        // read - for each cell ordered by hash, denotes the particle
                                                                // index which first belong to such hash
                                  const uint* cellEnd,          // read
                                  const uint numActive,         // read
                                  uint* numNeighborsPerPart) {  // write - number of neighbor particles for each particle
                                  
    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // thread index, associated to a particle ordered by hash.
    if (index >= numActive) {
        return;
    }

    // extract position of the ordered particle
    Real3 posRadA = mR3(sortedPosRad[index]);
    Real h_part = sortedPosRad[index].w;  // extract kernel radius of particle
    int3 gridPos = calcGridPos_csph(posRadA);  // call function to get corresponding position in the grid    
    Real SuppRadii = paramsD_csph.h_multiplier * h_part;    // support radius Changed from paramsD.h to PosRad.w (h of individual particles)
    Real SqRadii = SuppRadii * SuppRadii;
    uint j_num = 0;


    /*
    // DEBUG!!!!!
    if (index == 0)
        printf("Inside neighborSearchNum. SuppRadii used is: %g, SqRadii = %g, h_part = %g\n", SuppRadii, SqRadii, h_part);
    */

    // loop over the 27 neighbor cells. Just add and subtract 1 to grid coordinates since they are integers.
    // for each cell extract the particles inside and loop over them.
    // It considers also the cell of current particle and particles in it.
    // done for each particle since it's a parallel kernel function.

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighborPos = gridPos + mI3(x, y, z);  // position of neighbor cell
                // Check if we need to skip this neighbor position (out of bounds for non-periodic dimensions)
                if (neighborPos.x < paramsD_csph.minBounds.x || neighborPos.x > paramsD_csph.maxBounds.x ||
                    neighborPos.y < paramsD_csph.minBounds.y || neighborPos.y > paramsD_csph.maxBounds.y ||
                    neighborPos.z < paramsD_csph.minBounds.z || neighborPos.z > paramsD_csph.maxBounds.z) {
                    continue;
                }
                uint gridHash = calcGridHash_csph(neighborPos);  // compute hash of neighbor cell
                uint startIndex = cellStart[gridHash];  // get start index of particle in the considered neighbor cell
                uint endIndex = cellEnd[gridHash];      // get end index
              
                // loop over particles inside the neighbor cell
                for (uint j = startIndex; j < endIndex; j++) {
                    Real3 posRadB = mR3(sortedPosRad[j]);  // start and end indexes refers to arrays sorted by hash
                    Real3 dist3 = Distance_csph(posRadA, posRadB);  // position between current particle and current neighbor particle
                    Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;  // square of distance
                    
                    if (dd < SqRadii) {  // Here comparing squares of distances to avoid expensive sqrts computations.
                        j_num++;  // if norm of distance lower than square of support radius, it is an actual neighbor
                                  // particle
                    }
                    
                }
            }
        }
    }
    numNeighborsPerPart[index] = j_num;  // register the number of actual neighbor particles inside the support radius
}  // I think that numNeighborsPerPart will always be at least 1, since the particle itself is considered.



// --------------------------second step: after neighborSearchNum now this-----------------------------------------
// Notice that neighborSearchNum results will always be >= 1, because at some point the loop will compare the studied
// particle with itself, and will always count it as a neighbor. Approach consistent with second sted neighborSearchID,
// where the first particle index in each neighbor lists is always forced to be the particle itself. Common approach
// used.
__global__ void neighborSearchID(const Real4* sortedPosRad,        // read - sorted by hash
                                 const Real4* sortedRhoPreEn,      // read - sorted by hash - appears unused
                                 const uint* cellStart,            // read - sorted by hash
                                 const uint* cellEnd,              // read - sorted by hash
                                 const uint numActive,             // read
                                 const uint* numNeighborsPerPart,  // read - sorted by hash
                                 uint* neighborList) {             // write
                                 
    uint index = blockIdx.x * blockDim.x + threadIdx.x;  // thread index associated to ordered particles
    if (index >= numActive) {
        return;
    }
    Real3 posRadA = mR3(sortedPosRad[index]);  // get position of particle
    Real h_part = sortedPosRad[index].w;       // h of actual particle
    int3 gridPos = calcGridPos_csph(posRadA);       // associated position in the grid
    Real SuppRadii = 2.0f * h_part;            // support radius. From ParamsD_csph.h to h_part
    Real SqRadii = SuppRadii * SuppRadii;      // squared to avoid sqrt computations
    uint j_num = 1;  // counter for neighbors. initialized to 1 because first neighbor is the particle itself
                     // since j_num used as offset in neighborList, and particle itself always saved first, then j_num
                     // starts from 1.
       
      
    /*
    // DEBUG!!!!!
    if (index == 0)
        printf("Inside neighborSearchID. SuppRadii used is: %g, SqRadii = %g, h_part = %g\n", SuppRadii, SqRadii,
               h_part);
               */


    // Here a bit more involved. numNeighborsPerPart[index] not used as a counter but as start offset for the neighbor
    // list of our particle. This offset should come from a prefix sum scan of counters computed in the kernel
    // neighborSearchNum. So neighborList will be a 1D array containing all contatenated lists, and
    // numNeighborsPerPart[index] tells where the list of particle index starts. neighborList[offset] = index -> the
    // first element in the list is the particle itself
    neighborList[numNeighborsPerPart[index]] = index;

    // loop over 27 neighbor cells
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighborPos = gridPos + mI3(x, y, z);  // grid position of neighbor cell
                // Check if we need to skip this neighbor position (out of bounds for non-periodic dimensions)
                if (neighborPos.x < paramsD_csph.minBounds.x || neighborPos.x > paramsD_csph.maxBounds.x ||
                    neighborPos.y < paramsD_csph.minBounds.y || neighborPos.y > paramsD_csph.maxBounds.y ||
                    neighborPos.z < paramsD_csph.minBounds.z || neighborPos.z > paramsD_csph.maxBounds.z) {
                    continue;
                }
                uint gridHash = calcGridHash_csph(neighborPos);  // hash of neighbor cell
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];  // start and end indexes of neighbor particles
                // loop over neighbor particles
                for (uint j = startIndex; j < endIndex; j++) {
                    if (j != index) {  // NOW WE EXCLUDE THE PARTICLE TO BE COMPARED WITH ITSELF, since it was already
                                       // accounted for.
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance_csph(posRadA, posRadB);
                        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
                        if (dd < SqRadii) {  // condition for a particle to be an actual neighbor
                            neighborList[numNeighborsPerPart[index] + j_num] =
                                j;    // save the neighbor particle index. j_num defaults to 1 for the first neighbor
                                      // found (if any)
                            j_num++;  // update j_num counter(offset)
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
//                  Actual methods of class CollisionSystem_csph
// =============================================================================


CollisionSystem_csph::CollisionSystem_csph(FsiDataManager_csph& data_mgr) : m_data_mgr(data_mgr), m_sphMarkersD(nullptr) {}

CollisionSystem_csph::~CollisionSystem_csph() {}

void CollisionSystem_csph::Initialize() {
    // paramsD and countersD are defined as global static variable in SphGeneral_compressible. Here 
    // we actually give them their values based on the equivalent structures stored on the host
    cudaMemcpyToSymbolAsync(paramsD_csph, m_data_mgr.paramsH.get(), sizeof(ChFsiParamsSPH_csph));
    cudaMemcpyToSymbolAsync(countersD_csph, m_data_mgr.countersH.get(), sizeof(Counters_csph));
}

// sorts particles based on their hash value
// Needs the collision system class to have already been initialized: i.e. the fsiDataManager attribute correctly
// created. output is the SOA ordered by hash
void CollisionSystem_csph::ArrangeData(std::shared_ptr<SphMarkerDataD_csph> sphMarkersD,
                                       std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {
    bool* error_flagD;
    cudaMallocErrorFlag(error_flagD);
    cudaResetErrorFlag(error_flagD);


    // Create active list where all active particles are at the front of the array
    // compute blocknum and threadnum for kernel launch
    uint numThreads, numBlocks;  // initialize
    // function in UtilsDevice.cu that given number of elements to work on(numAllmarkers), and the blocksize(1024 here),
    // determines the num of blocks and threads. Here given numAllMarkers
    computeCudaGridSize((uint)m_data_mgr.countersH->numAllMarkers, 1024, numBlocks, numThreads);

    // call fillActiveListD kernel, first kernel defined in this page. Reduces the extended activity list to just active
    // list with active indices at the front. first input is the prefix sum of the extended activity identifier, second
    // is the extended activity identifier array itself, third is the reduce activity list.
    // The PrefixSumExtendedActivity vector is computed in FluidDynamics::CheckActivityArrayResize()
    // Extended activity markers should mean all active markers including also
    // bce and ridig ones. num of blocks and threads computed with allMarkers, the maximum number of markers so far in
    // the simulation. The extendedActivityIdentifierOriginalD should have size numAllMarkers and be a vector of 0 or 1
    // flags. Doing a prefix sum on it should then compute the actual number of active particles, of each type (fluid or
    // else)
    // ExtendedActivityIdentifierOriginalD is populated by method in FluidDynamics class.
    fillActiveListD<<<numBlocks, numThreads>>>(
        U1CAST(m_data_mgr.prefixSumExtendedActivityIdD), INT_32CAST(m_data_mgr.extendedActivityIdentifierOriginalD),
        U1CAST(m_data_mgr.activeListD), (uint)m_data_mgr.countersH->numAllMarkers);


    // Reset cell size HERE MODIFY FOR VARIABLE GRID SIZE
    int3 cellsDim = m_data_mgr.paramsH->gridSize;
    int numCells = cellsDim.x * cellsDim.y * cellsDim.z;
    m_data_mgr.markersProximity_D->cellStartD.resize(numCells);
    m_data_mgr.markersProximity_D->cellEndD.resize(numCells);

    cudaCheckError();
    // Calculate Hash
    computeCudaGridSize((uint)m_data_mgr.countersH->numExtendedParticles, 1024, numBlocks, numThreads);
    calcHashD<<<numBlocks, numThreads>>>(U1CAST(m_data_mgr.markersProximity_D->gridMarkerHashD),
                                         U1CAST(m_data_mgr.markersProximity_D->gridMarkerIndexD),
                                         U1CAST(m_data_mgr.activeListD), mR4CAST(sphMarkersD->posRadD),
                                         mR4CAST(sphMarkersD->rhoPresEnD),
                                         (uint)m_data_mgr.countersH->numExtendedParticles, error_flagD);
    cudaCheckErrorFlag(error_flagD, "calcHashD");
    cudaCheckError();

    // Sort Particles based on Hash
    thrust::sort_by_key(
        m_data_mgr.markersProximity_D->gridMarkerHashD.begin(),
        m_data_mgr.markersProximity_D->gridMarkerHashD.begin() + m_data_mgr.countersH->numExtendedParticles,
        m_data_mgr.markersProximity_D->gridMarkerIndexD.begin());

    // Find the start index and the end index of the sorted array in each cell
    //
    thrust::fill(m_data_mgr.markersProximity_D->cellStartD.begin(), m_data_mgr.markersProximity_D->cellStartD.end(), 0);
    thrust::fill(m_data_mgr.markersProximity_D->cellEndD.begin(), m_data_mgr.markersProximity_D->cellEndD.end(), 0);
    cudaCheckError();
    // TODO - Check if 256 is optimal here
    computeCudaGridSize((uint)m_data_mgr.countersH->numExtendedParticles, 256, numBlocks, numThreads);
    uint smemSize = sizeof(uint) * (numThreads + 1);
    // launch kernel to populate the start and end indexes of the sorted cells
    findCellStartEndD<<<numBlocks, numThreads, smemSize>>>(
        U1CAST(m_data_mgr.markersProximity_D->cellStartD), U1CAST(m_data_mgr.markersProximity_D->cellEndD),
        U1CAST(m_data_mgr.markersProximity_D->gridMarkerHashD), U1CAST(m_data_mgr.markersProximity_D->gridMarkerIndexD),
        (uint)m_data_mgr.countersH->numExtendedParticles);

    cudaCheckError();
    // Launch a kernel to find the location of original particles in the sorted arrays
    // This is faster than using thrust::sort_by_key()
    computeCudaGridSize((uint)m_data_mgr.countersH->numExtendedParticles, 1024, numBlocks, numThreads);
    OriginalToSortedD<<<numBlocks, numThreads>>>(U1CAST(m_data_mgr.markersProximity_D->mapOriginalToSorted),
                                                 U1CAST(m_data_mgr.markersProximity_D->gridMarkerIndexD),
                                                 (uint)m_data_mgr.countersH->numExtendedParticles);


    // Reorder the arrays according to the sorted index of all particles
    computeCudaGridSize((uint)m_data_mgr.countersH->numExtendedParticles, 1024, numBlocks, numThreads);
    cudaCheckError();
    if (m_data_mgr.paramsH->is_uniform)
        reorderDataD<true> <<<numBlocks, numThreads>>>(
            U1CAST(m_data_mgr.markersProximity_D->gridMarkerIndexD), mR4CAST(sortedSphMarkersD->posRadD),
            mR3CAST(sortedSphMarkersD->velD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
            R1CAST(sortedSphMarkersD->soundD), nullptr,
            INT_32CAST(m_data_mgr.activityIdentifierSortedD), mR4CAST(sphMarkersD->posRadD),
            mR3CAST(sphMarkersD->velD), mR4CAST(sphMarkersD->rhoPresEnD),
            R1CAST(sphMarkersD->soundD), nullptr, INT_32CAST(m_data_mgr.activityIdentifierOriginalD),
            (uint)m_data_mgr.countersH->numExtendedParticles);
    else
        reorderDataD<false><<<numBlocks, numThreads>>>(
            U1CAST(m_data_mgr.markersProximity_D->gridMarkerIndexD), mR4CAST(sortedSphMarkersD->posRadD),
            mR3CAST(sortedSphMarkersD->velD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
            R1CAST(sortedSphMarkersD->soundD), R1CAST(m_data_mgr.sortedMassD),
            INT_32CAST(m_data_mgr.activityIdentifierSortedD),
            mR4CAST(sphMarkersD->posRadD), mR3CAST(sphMarkersD->velD), mR4CAST(sphMarkersD->rhoPresEnD), R1CAST(sphMarkersD->soundD),
            R1CAST(m_data_mgr.massD),
            INT_32CAST(m_data_mgr.activityIdentifierOriginalD),
            (uint)m_data_mgr.countersH->numExtendedParticles);

    cudaCheckError();

    cudaFreeErrorFlag(error_flagD);

}

// actually creates the neighbor lists
void CollisionSystem_csph::NeighborSearch(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD) {

    uint numActive = (uint)m_data_mgr.countersH->numExtendedParticles;
    uint numBlocksShort, numThreadsShort;
    computeCudaGridSize(numActive, 1024, numBlocksShort, numThreadsShort);

    // Execute the kernel
    thrust::fill(m_data_mgr.numNeighborsPerPart.begin(), m_data_mgr.numNeighborsPerPart.end(), 0);

    // start neighbor search
    // first pass to get number of neighbors for each particle
    neighborSearchNum_csph<<<numBlocksShort, numThreadsShort>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
        U1CAST(m_data_mgr.markersProximity_D->cellStartD), U1CAST(m_data_mgr.markersProximity_D->cellEndD), numActive,
        U1CAST(m_data_mgr.numNeighborsPerPart));
    cudaCheckError();

    // In-place exclusive scan for num of neighbors, result placed back in numNeighborsPerPart
    // which is then overwritten.
    // Used as input to work on the concatenated list of neighbors.
    // Each element will contain the total number of neighbors computed for all particles before it.
    // First element of result output is 0.
    // Last element of input not summed to last element of output.
    thrust::exclusive_scan(m_data_mgr.numNeighborsPerPart.begin(), m_data_mgr.numNeighborsPerPart.end(),
                           m_data_mgr.numNeighborsPerPart.begin());

    // back() gives a reference to the last element of the vector. Since vector is now a prefix sum, the last element
    // also tells how many neighbors we have globally so it will be the actual length of the vector neighborList.
    m_data_mgr.neighborList.resize(m_data_mgr.numNeighborsPerPart.back());
    thrust::fill(m_data_mgr.neighborList.begin(), m_data_mgr.neighborList.end(), 0);

    // second pass, build the actual neighbor list
    neighborSearchID<<<numBlocksShort, numThreadsShort>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresEnD),
        U1CAST(m_data_mgr.markersProximity_D->cellStartD), U1CAST(m_data_mgr.markersProximity_D->cellEndD), numActive,
        U1CAST(m_data_mgr.numNeighborsPerPart), U1CAST(m_data_mgr.neighborList));



}



// All kernels work with numExtendedParticles and not with numActive or numAllMarkers for following reason.
// The neighbor search is called by FluidDynamics class after it has performed the activity update and the prefix sum on the extended activity index vector.
// So it has set numExtendedParticles as all active particles in the extended (original + buffer) domain. Now this number is in general different from numAllMarkers.
// If numExtendedParticles < numAllMarkers no issues, if numExtendedParticles > numAllParticles the data array have already been resized accordingly.
// Now, the neighbor list works only with the active particles in the extended domain, and their total number is indeed the numExtendedParticles mentioned before.
// However in the first kernel FillActiveList we pass numAllMarkers, because that is the actual length of the vector extendedACtivityId,
// nut when we build ActiveList, it will still have numAllMarkers length, but only numExtendedParticles meaningful terms.


}  // end namespace compressible
}  // end namespace chrono::fsi::sph

