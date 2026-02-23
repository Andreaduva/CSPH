// Author: Andrea D'Uva - 2025
//
// -------------------------------------------------------------
// Extending the DataManager class to allow for compressibility
// -------------------------------------------------------------

#ifndef CH_FSI_DATA_MANAGER_COMPRESSIBLE_H
#define CH_FSI_DATA_MANAGER_COMPRESSIBLE_H

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "chrono_fsi/sph/ChFsiParamsSPH.h"
#include "chrono_fsi/sph/physics/MarkerType.cuh"
#include "chrono_fsi/sph/utils/UtilsDevice.cuh"
#include "chrono_fsi/sph/physics/FsiDataManager.cuh"

#include "chrono_fsi/sph_compressible/ChFsiParamsSPH_compressible.h"
#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"
#include "chrono_fsi/sph_compressible/math/CustomMath_compressible.cuh"
// 
// 
// work in compressible namespace
namespace chrono::fsi::sph {
namespace compressible {

// define auxiliary types, extending ones defined in original DataManager:



//-------------------------------------------------
//    SPH markers data structs
//-------------------------------------------------


// typedef device tuple for holding compressible SPH data: {pos,vel,[rho,pressure,en,type],[T,c]}
// type is a tuple containing iterators to device vectors.
typedef thrust::device_vector<Real>::iterator rIterD;
typedef thrust::tuple<r4IterD, r3IterD, r4IterD, rIterD> iterTupleSphD_csph;
// zip_iterator to a sequence of tuples (thus creating a virtual array) containing iterators of device vectors
typedef thrust::zip_iterator<iterTupleSphD_csph> zipIterSphD_csph;

// typedef host tuple for holding compressible SPH data: {pos,vel,[rho,pressure,type],[T,c]}
// Similar to above. zip iterator to a sequence of tuples containing iterators to host vectors.
typedef thrust::host_vector<Real>::iterator rIterH;
typedef thrust::tuple<r4IterH, r3IterH, r4IterH, rIterH> iterTupleH_csph;
typedef thrust::zip_iterator<iterTupleH_csph> zipIterSphH_csph;

// struct to store data of compressible sph markers on device, modify the original one
struct SphMarkerDataD_csph {
    thrust::device_vector<Real4> posRadD;      //< Vector of the positions of particles + characteristic radius
    thrust::device_vector<Real3> velD;         //< Vector of the velocities of particles. Here pure velocity, original chrono sph should be velocity/density
    thrust::device_vector<Real4> rhoPresEnD;   //< Vector of the rho+pressure+thermal energy+type of particles
    thrust::device_vector<Real>  soundD;       //< Vector of speed of sound

    zipIterSphD_csph iterator(int offset);     //< Function to make the zip iterator for tuple of iterators of device vectors
    void resize(size_t s);
};

// struct to store data of sph markers on host
struct SphMarkerDataH_csph {
    thrust::host_vector<Real4> posRadH;      //< Vector of the positions of particles
    thrust::host_vector<Real3> velH;         //< Vector of the velocities of particles
    thrust::host_vector<Real4> rhoPresEnH;   //< Vector of the rho+pressure+thermal energy+type of particles
    thrust::host_vector<Real>  soundH;       //< Vector of speed of sound
   
    zipIterSphH_csph iterator(int offset);   //< Function to make the zip iterator for tuple of iterators of host vectors
    void resize(size_t s);
};



//--------------------------------------------------------------
// Rigid FSI bodies data structs - same as Chrono::fsi::sph ones
//--------------------------------------------------------------

// typedef device iterators for shorthand rigid body states:
// pos,lin_vel,lin_acc,rot,ang_Vel,ang_acc
typedef thrust::tuple<r3IterD, r3IterD, r3IterD, r4IterD, r3IterD, r3IterD> iterTupleRigidD;
typedef thrust::zip_iterator<iterTupleRigidD> zipIterRigidD;

// typedef host iterators for shorthand rigid body states:
typedef thrust::tuple<r3IterH, r3IterH, r3IterH, r4IterH, r3IterH, r3IterH> iterTupleRigidH;
typedef thrust::zip_iterator<iterTupleRigidH> zipIterRigidH;

// typedef host iterators for shorthand chrono bodies operations
typedef thrust::tuple<r3IterH, r3IterH, r3IterH, r4IterH, r3IterH, r3IterH> iterTupleChronoBodiesH;
typedef thrust::zip_iterator<iterTupleChronoBodiesH> zipIterChronoBodiesH;

// Rigid body states on host, one state for each FSI body.
struct FsiBodyStateH_csph {
    thrust::host_vector<Real3> pos;      ///< body positions
    thrust::host_vector<Real3> lin_vel;  ///< body linear velocities
    thrust::host_vector<Real3> lin_acc;  ///< body linear accelerations
    thrust::host_vector<Real4> rot;      ///< body orientations (quaternions)
    thrust::host_vector<Real3> ang_vel;  ///< body angular velocities (local frame)
    thrust::host_vector<Real3> ang_acc;  ///< body angular accelerations (local frame)

    zipIterRigidH iterator(int offset);  ///< Function to make the zip iterator for tuple of iterators of host vectors
    void Resize(size_t s);
};

// Rigid body states on device.
// Data are managed in an SOA (structure of array), with each array containing corresponding data from all bodies in
// the system.
struct FsiBodyStateD_csph {
    thrust::device_vector<Real3> pos;      ///< body linear positions
    thrust::device_vector<Real3> lin_vel;  ///< body linear velocities
    thrust::device_vector<Real3> lin_acc;  ///< body linear accelerations
    thrust::device_vector<Real4> rot;      ///< body orientations (quaternions)
    thrust::device_vector<Real3> ang_vel;  ///< body angular velocities (local frame)
    thrust::device_vector<Real3> ang_acc;  ///< body angular accelerations (local frame)

    zipIterRigidD iterator(int offset);  ///< Function to make the zip iterator for tuple of iterators of device vectors

    void CopyFromH(const FsiBodyStateH_csph& bodyStateH);       /// Copy from the same struct but made by host vectors
    FsiBodyStateD_csph& operator=(const FsiBodyStateD_csph& other);  /// As above but with the = operator. Not a copy constructor.
    void Resize(size_t s);
};


//-------------------------------------------------------
// Auxiliary data structs: proximity data and counters
//-------------------------------------------------------


// Struct to store neighbor search information on the device.
struct ProximityDataD_csph {
    thrust::device_vector<uint> gridMarkerHashD;      //< gridMarkerHash=s(i,j,k)= k*n_x*n_y + j*n_x + i (numAllMarkers in Counter struct);
    thrust::device_vector<uint> gridMarkerIndexD;     //< Marker's index, can be original or sorted (numAllMarkers);
    thrust::device_vector<uint> cellStartD;           //< Index of the particle starts a cell in sorted list (m_numGridCells ??)
    thrust::device_vector<uint> cellEndD;             //< Index of the particle ends a cell in sorted list (m_numGridCells)
    thrust::device_vector<uint> mapOriginalToSorted;  //< Index mapping from the original to the sorted (numAllMarkers);

    void resize(size_t s);
};


// Struct to store CUDA device information.
struct CudaDeviceInfo {
    int deviceID;               //< CUDA device ID
    cudaDeviceProp deviceProp;  //< CUDA device properties
};


// Number of rigid and flexible solid bodies, fluid SPH particles, solid SPH particles, boundary SPH particles.
// This structure holds the number of SPH particles and rigid/flexible bodies.
//  Note that the order of makers in the memory is as follows:
//  -  (1) fluid particles (type = -1)
//  -  (2) particles attached to fixed objects (boundary particles with type = 0)
//  -  (3) particles attached to rigid bodies (type = 1)
//  -  (4) particles attached to flexible bodies (type = 2) - not supported in this implementation
struct Counters_csph {
    size_t numFsiBodies;          //< number of rigid bodies

    size_t numGhostMarkers;      //< number of Ghost SPH particles for Variable Resolution methods
    size_t numHelperMarkers;     //< number of helper SPH particles used for merging particles
    size_t numFluidMarkers;      //< number of fluid SPH particles (including ghost and helper particles)
    size_t numBoundaryMarkers;   //< number of BCE markers on boundaries
    size_t numRigidMarkers;      //< number of BCE markers on rigid bodies
    size_t numBceMarkers;        //< total number of BCE markers (boundary + rigid + flexible)
    size_t numAllMarkers;        //< total number of particles in the simulation (of all kinds)

    size_t startBoundaryMarkers;   //< index of first BCE marker on boundaries (corresponds to size of fluidMarkers)
    size_t startRigidMarkers;      //< index of first BCE marker on first rigid body (startBoundaryMarkers +
                                   //< numBoundaymarkers)
    size_t numActiveParticles;     //< number of active particles - means ones inside the interaction domain of at least
                                   //< one Fsi body.
    size_t numExtendedParticles;   //< number of extended particles (so in the buffered domain) that have 1 as activity flag. Different from numAllMarkers
};


// -----------------------------------------------------------------------------
// Data manager for the SPH-based FSI system. Compressible version based on the standard Chrono::fsi::sph one

struct FsiDataManager_csph {
  public:
    FsiDataManager_csph(std::shared_ptr<ChFsiParamsSPH_csph> params);  // Construction via a shared pointer to a ChFsiParamsSPH struct.
    virtual ~FsiDataManager_csph();

    // Set the growth factor for buffer resizing.
    void SetGrowthFactor(float factor) { GROWTH_FACTOR = factor; }

    // Add an SPH particle given its position, physical properties, velocity, kernel radius.
    // Does not set or update any counter.
    // Particle is added only to the host side of data.
    // Speed of sound computed internally from rho, pres, en.
    void AddSphParticle(Real3 pos,
                        Real rho,
                        Real pres,
                        Real en,           // thermal energy
                        Real3 vel = mR3(0.0),
                        Real h = -1,       // default h is global one
                        Real mass = -1);   // default mass is global one  

    //  Add a BCE marker of given type at the specified position and with specified velocity.
    //  Does not set or update any counter.
    //  This adds the Bce markers pos and velocity to the global marker vectors posRadH, velH, rhoPresEnH (using
    //  default parameters)
    //  Since thermodynamic properties are computed by interpolation from fluid particles, need to pass only
    //  kernel radius and mass as additional parameters
    void AddBceMarker(MarkerType type, Real3 pos, Real3 vel = mR3(0.0), Real h = -1, Real mass = -1);

    // Initialize the underlying FSI system.
    // Set reference arrays, set counters, and resize simulation arrays.
    // vectors referring to sph markers variables are all initialized with numAllMarkers length, no matter what actual
    // length should be. Also copies data referring to particles position, velocity, density and pressure from host
    // vectors to device vectors. But host vectors should already be populated. Host vectors are populated from calls
    // to AddSphParticle and AddBceMarker, but if they have some order (like fluid first) it's not done here.
    void Initialize(unsigned int num_fsi_bodies);

    /// Find indices of all SPH particles inside the specified OBB.
    // hsize = box half dimensions, pos = center of box in global frame, ax,ay,az = box axes direction in global frame.
    std::vector<int> FindParticlesInBox(const Real3& hsize,
                                        const Real3& pos,
                                        const Real3& ax,
                                        const Real3& ay,
                                        const Real3& az);

    // Extract positions of all markers (SPH and BCE).
    std::vector<Real3> GetPositions();

    // Extract velocities of all markers (SPH and BCE).
    std::vector<Real3> GetVelocities();

    // Extract accelerations of all markers (SPH and BCE). From implementation however look only fluid sph markers.
    std::vector<Real3> GetAccelerations();

    // Extract forces applied to all markers (SPH and BCE). Again looks only fluid markers are extracted.
    std::vector<Real3> GetForces();

    // Extract fluid properties of all markers (SPH and BCE).
    // For each SPH particle, the 3-dimensional vector contains density, pressure, and thermal energy.
    std::vector<Real3> GetProperties();

    // Extract local sound speed of all markers.
    std::vector<Real> GetSound();

    // Extract kernel radius of all particles:
    std::vector<Real> GetKernelRad();

    // Extract maximum kernel radius among all fluid particles:
    Real GetMaxFluidRad();

    // Extract maximum kernel radius among all particles (SPH and BCE):
    // Field x of output Real2 is the h value found, field y represents the particle type.
    Real2 GetMaxRad();

    // Extract positions of all markers (SPH and BCE) with indices in the provided array.
    std::vector<Real3> GetPositions(const std::vector<int>& indices);

    // Extract velocities of all markers (SPH and BCE) with indices in the provided array.
    std::vector<Real3> GetVelocities(const std::vector<int>& indices);

    // Extract accelerations of all markers (SPH and BCE) with indices in the provided array. Implementation looks like
    // extracts only fluid markers
    std::vector<Real3> GetAccelerations(const std::vector<int>& indices);

    // Extract forces applied to all markers (SPH and BCE) with indices in the provided array. Implementations looks
    // like extracts only fluid markers
    std::vector<Real3> GetForces(const std::vector<int>& indices);

    // Extract FSI forces on rigid bodies.
    std::vector<Real3> GetRigidForces();

    // Extract FSI torques on rigid bodies.
    std::vector<Real3> GetRigidTorques();

    void ConstructReferenceArray();  // Builds reference array based on sphMarkers_H elements (all sph particles).
    void SetCounters(unsigned int num_fsi_bodies);

    // Reset device data at beginning of a step.
    // Initializes device vectors to zero.
    void ResetData();

    // Resize data arrays based on particle activity. May allocate more space on device memory or shrink the device
    // vectors.
    //  Doesn't like resize bceAcc from numAllMarkers to numRigidMarkers + numFlexMarkers.
    void ResizeArrays(uint numExtended);

    // Return device memory usage.
    size_t GetCurrentGPUMemoryUsage() const;

    private:
    // Check if mass data is consistent
      void CheckMassConsistency();

    public:
    // ------------------------

    std::shared_ptr<CudaDeviceInfo> cudaDeviceInfo;  //< CUDA device information

    std::shared_ptr<ChFsiParamsSPH_csph> paramsH;  //< simulation parameters (host)
    std::shared_ptr<Counters_csph> countersH;           //< problem counters (host)
    // following struct of arrays store data for all sph markers (fluid sph and bce)
    std::shared_ptr<SphMarkerDataD_csph> sphMarkers_D;  //< information of SPH particles at state 1 on device, not in hash
                                                        //< order. See struct definitions above.
    std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkers1_D;  //< sorted information of SPH particles at state 1 on device. Probably state 1 and state
                                                               //< 2 introduced because used for different computational purposes, for instance in the integration phase.
    std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkers2_D;  //< sorted information of SPH particles at state 2 on device. State 1 are used as temporary states for midpoint computation. State 2 are actually advanced.
    std::shared_ptr<SphMarkerDataH_csph> sphMarkers_H;         //< information of SPH particles on host, all particles, fluid +
                                                               //< bce. Are comments above correct??
    
    // vectors that keep track of individual mass particles
    thrust::host_vector<Real> massH;
    thrust::device_vector<Real> massD;
    thrust::device_vector<Real> sortedMassD;

    // ------------------------

    // FSI solid states (FsiBodyState structs defined in FsiDataManager.cuh)
    std::shared_ptr<FsiBodyStateH_csph> fsiBodyState_H;  //< rigid body state (host)
    std::shared_ptr<FsiBodyStateD_csph> fsiBodyState_D;  //< rigid body state (device)

    // FSI solid BCEs
    thrust::host_vector<Real3> rigid_BCEcoords_H;     //< local coordinates for BCE markers on rigid bodies (host)
    thrust::device_vector<Real3> rigid_BCEcoords_D;   //< local coordinates for BCE markers on rigid bodies (device)

    thrust::host_vector<uint> rigid_BCEsolids_H;      //< body ID for BCE markers on rigid bodies (host)
    thrust::device_vector<uint> rigid_BCEsolids_D;    //< body ID for BCE markers on rigid bodies (device)

    // FSI solid forces
    thrust::device_vector<Real3> rigid_FSI_ForcesD;   //< surface-integrated fluid forces acting on rigid bodies
    thrust::device_vector<Real3> rigid_FSI_TorquesD;  //< surface-integrated torques to rigid bodies

    // ------------------------

    std::shared_ptr<ProximityDataD_csph> markersProximity_D;  //< information of neighbor search on the device

    thrust::host_vector<int4> referenceArray;  //< phases in the array of SPH particles. Vector of int4 with each int4 associated to a
                                               //< component type. Field x is start index of the phase in the Sphvector,
                                               //  field y is end index of the phase in same vector, field z is the component type
                                               //  (helper,ghost,boundary,etc) while field w is phase type (fluid,solid,boundary) indexes
                                               //  refers to vectors in the struct of SphMarkers_H, whose vectors are by construction sorted
                                               //  by component type.
 
    // Fluid data (device)
    thrust::device_vector<Real5> derivVelRhoEnD;          //< particle dv/dt, d(rho)/dt, d(en)/dt - sorted
    thrust::device_vector<Real5> derivVelRhoEnOriginalD;  //< particle dv/dt, d(rho)/dt, d(en)/dt - unsorted
    thrust::device_vector<Real>  derivRadD;               //< particle dRad/dt - sorted   ( rad = kernel radius h)
    thrust::device_vector<Real>  derivRadOriginalD;       //< particle dRad/dt - unsorted
    thrust::device_vector<Real3> vel_XSPH_D;              //< XSPH velocity for particles
    thrust::device_vector<Real3> bceAcc;                  //< acceleration for boundary rigid body particles. Wall particles indeed excluded as they have fixed position.
                                                          // Initialized with numAllMarkers length. Needed in Adami style BC enforcement.

    thrust::device_vector<Real> divVelD;                  //< divergence of velocity. Sorted. Needed in case artificial conduction is on
    thrust::device_vector<Real> heatCoeffD;                //< coefficients of thermal conduction.

    // ActivityIdentifier vectors are processed in FLuidDynamics class.
    thrust::device_vector<int32_t> activityIdentifierOriginalD;          //< active particle flags - unsorted - 1 for active, 0 for inactive, -1 for
                                                                         //< zombie (outside domain). length should be numAllMarkers.
    thrust::device_vector<int32_t> activityIdentifierSortedD;            //< active particle flags - sorted   - active list only
    thrust::device_vector<int32_t> extendedActivityIdentifierOriginalD;  //< extended active particle flags - unsorted -> I think it has size
                                                                         //< numAllMarkers. now the identifier 1 also applied to particles that
                                                                         //< are inside the buffered interaction domain with at least one Fsi
                                                                         //< solid.
    thrust::device_vector<uint> prefixSumExtendedActivityIdD;            //< prefix sum of extended particles - computed in FluidDynamics::CheckActivityArrayResize using custom functor that treats -1 flags as 0.
    thrust::device_vector<uint> activeListD;                             //< active list of particles. Only first numActive indexes make sense.
                                                                         //< (In theory this vector not resized but have length numAllMarkers.
    
    thrust::device_vector<uint> numNeighborsPerPart;  //< number of neighbors for each particle. Result of an exclusive scan.
                                                      //< It tells, for a given particle, which elements of neighborList to access. This is because neighborList
                                                      //< is a concatenated array. So for particle 12, numNeighborsPerPart[12] = 19 means
                                                      //< first actual neighbor ID in neighborList is at position 19. Last neighbor Id is at position 12+1.

    // List of all neighbors (indexed with information from numNeighborsPerPart)
    thrust::device_vector<uint> neighborList;    //< neighbor list for all particles - concatenated array.
    thrust::device_vector<uint> freeSurfaceIdD;  //< identifiers for particles close to free surface

    // Vectors for timesteps used in case of variable time step based on CFL condition
    thrust::device_vector<Real> dT_velD;         //< dT_vel = h/c_s
    thrust::device_vector<Real> dT_forceD;       //< dT_force = C_force*sqrt( h / norm(dVeldT) )

  private:
    // Dynamic Memory management parameters for device memory (so check on device vector sizes)
    uint m_max_extended_particles;  //< Maximum number of extended particles seen so far
    uint m_resize_counter;          //< Counter for number of resizes since last shrink
    float GROWTH_FACTOR;            //< Buffer factor for growth (20%)
    float SHRINK_THRESHOLD;         //< Shrink if using less than 50% of capacity
    uint SHRINK_INTERVAL;           //< Shrink every N resizes


    // DEBUG
    // GpuTimer timer_debug;
};







} // namespace end
}



#endif



