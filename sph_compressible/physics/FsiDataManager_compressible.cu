// =============================================================================
// Author: Andrea D'Uva - 2025
//
// Implementation of FsiDataManager_compressible.
//
// =============================================================================

////#define DEBUG_LOG
#include <iostream>
#include <algorithm>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/partition.h>
#include <thrust/zip_function.h>
#include <thrust/extrema.h>
#include <thrust/count.h>

#include "chrono/utils/ChUtils.h"

#include "chrono_fsi/sph/physics/FsiDataManager.cuh"
#include "chrono_fsi/sph/math/CustomMath.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"


namespace chrono::fsi::sph {
namespace compressible {

//-----------------------------------------------------
// Methods for sph data structs
//-----------------------------------------------------

// SphMarkerDataD_csph methods
zipIterSphD_csph SphMarkerDataD_csph::iterator(int offset) {
    return thrust::make_zip_iterator(thrust::make_tuple(posRadD.begin() + offset, velD.begin() + offset,
                                                        rhoPresEnD.begin() + offset, soundD.begin() + offset));
}

void SphMarkerDataD_csph::resize(size_t s) {
    posRadD.resize(s);
    velD.resize(s);
    rhoPresEnD.resize(s);
    soundD.resize(s);
}

// SphMarkerDataH_csph methods
zipIterSphH_csph SphMarkerDataH_csph::iterator(int offset) {
    return thrust::make_zip_iterator(thrust::make_tuple(posRadH.begin() + offset, velH.begin() + offset,
                                                        rhoPresEnH.begin() + offset, soundH.begin() + offset));
}

void SphMarkerDataH_csph::resize(size_t s) {
    posRadH.resize(s);
    velH.resize(s);
    rhoPresEnH.resize(s);
    soundH.resize(s);
}

//------------------------------------------------------
// Methods for Fsi rigid bodies data structs
//------------------------------------------------------

zipIterRigidH FsiBodyStateH_csph::iterator(int offset) {
    return thrust::make_zip_iterator(thrust::make_tuple(pos.begin() + offset, lin_vel.begin() + offset,
                                                        lin_acc.begin() + offset, rot.begin() + offset,
                                                        ang_vel.begin() + offset, ang_acc.begin() + offset));
}

void FsiBodyStateH_csph::Resize(size_t s) {
    pos.resize(s);
    lin_vel.resize(s);
    lin_acc.resize(s);
    rot.resize(s);
    ang_vel.resize(s);
    ang_acc.resize(s);
}

zipIterRigidD FsiBodyStateD_csph::iterator(int offset) {
    return thrust::make_zip_iterator(thrust::make_tuple(pos.begin() + offset, lin_vel.begin() + offset,
                                                        lin_acc.begin() + offset, rot.begin() + offset,
                                                        ang_vel.begin() + offset, ang_acc.begin() + offset));
}

void FsiBodyStateD_csph::Resize(size_t s) {
    pos.resize(s);
    lin_vel.resize(s);
    lin_acc.resize(s);
    rot.resize(s);
    ang_vel.resize(s);
    ang_acc.resize(s);
}

void FsiBodyStateD_csph::CopyFromH(const FsiBodyStateH_csph& bodyStateH) {
    thrust::copy(bodyStateH.pos.begin(), bodyStateH.pos.end(), pos.begin());
    thrust::copy(bodyStateH.lin_vel.begin(), bodyStateH.lin_vel.end(), lin_vel.begin());
    thrust::copy(bodyStateH.lin_acc.begin(), bodyStateH.lin_acc.end(), lin_acc.begin());
    thrust::copy(bodyStateH.rot.begin(), bodyStateH.rot.end(), rot.begin());
    thrust::copy(bodyStateH.ang_vel.begin(), bodyStateH.ang_vel.end(), ang_vel.begin());
    thrust::copy(bodyStateH.ang_acc.begin(), bodyStateH.ang_acc.end(), ang_acc.begin());
}

FsiBodyStateD_csph& FsiBodyStateD_csph::operator=(const FsiBodyStateD_csph& other) {
    if (this == &other) {
        return *this;
    }
    thrust::copy(other.pos.begin(), other.pos.end(), pos.begin());
    thrust::copy(other.lin_vel.begin(), other.lin_vel.end(), lin_vel.begin());
    thrust::copy(other.lin_acc.begin(), other.lin_acc.end(), lin_acc.begin());
    thrust::copy(other.rot.begin(), other.rot.end(), rot.begin());
    thrust::copy(other.ang_vel.begin(), other.ang_vel.end(), ang_vel.begin());
    thrust::copy(other.ang_acc.begin(), other.ang_acc.end(), ang_acc.begin());
    return *this;
}

//-------------------------------------------
// ProximityDataD resize method
//-------------------------------------------

void ProximityDataD_csph::resize(size_t s) {
    mapOriginalToSorted.resize(s);
}

//---------------------------------------------------------------------------------------
// FsiDataManager_csph methods
//---------------------------------------------------------------------------------------

// constructor
FsiDataManager_csph::FsiDataManager_csph(std::shared_ptr<ChFsiParamsSPH_csph> params) : 
                     paramsH(params),
                     derivVelRhoEnD(),
                     derivVelRhoEnOriginalD(),
                     derivRadD(),
                     derivRadOriginalD() {
    // initialize vectors
    countersH = chrono_types::make_shared<Counters_csph>();

    sphMarkers_D = chrono_types::make_shared<SphMarkerDataD_csph>();
    sortedSphMarkers1_D = chrono_types::make_shared<SphMarkerDataD_csph>();
    sortedSphMarkers2_D = chrono_types::make_shared<SphMarkerDataD_csph>();
    sphMarkers_H = chrono_types::make_shared<SphMarkerDataH_csph>();

    fsiBodyState_D = chrono_types::make_shared<FsiBodyStateD_csph>();
    fsiBodyState_H = chrono_types::make_shared<FsiBodyStateH_csph>();

    markersProximity_D = chrono_types::make_shared<ProximityDataD_csph>();

    cudaDeviceInfo = chrono_types::make_shared<CudaDeviceInfo>();

    // Resizing parameters
    m_max_extended_particles = 0;
    m_resize_counter = 0;
    GROWTH_FACTOR = 1.2f;
    SHRINK_THRESHOLD = 0.75f;
    SHRINK_INTERVAL = 50;

}

// destructor
FsiDataManager_csph::~FsiDataManager_csph() {}

// note that a call to FsiDataManager::AddSphParticle does not increment or modify any counter
// Also the particle is just added at the end, not sorted or inserted depending on type.
// Particle is added only to the host side.
// Input parameters are rho, pres, energy because they are the ones that are going to be evolved.
// Sound speed computed as consequence.
void FsiDataManager_csph::AddSphParticle(Real3 pos,
                                         Real  rho,
                                         Real  pres,
                                         Real  en,
                                         Real3 vel,
                                         Real  h,
                                         Real  mass) {
    if (h == -1) { // means default value
        h = paramsH->h;
    }
    sphMarkers_H->posRadH.push_back(mR4(pos, h));
    sphMarkers_H->velH.push_back(vel);
    sphMarkers_H->rhoPresEnH.push_back(mR4(rho, pres, en, -1));  // -1 is the type of particle: according to MarkerType an Sph particle has code -1
    // compute corresponding sound speed:
    Real Cs = sqrt(paramsH->gamma * pres / rho);  // Speed of sound
    sphMarkers_H->soundH.push_back(Cs);
    massH.push_back(mass);
}

// note that a call to FsiDataManager::AddBceMarker does not increment or modify any counter.
// Bce marker added with default parameters except for pos and vel.
void FsiDataManager_csph::AddBceMarker(MarkerType type, Real3 pos, Real3 vel, Real h, Real mass) {
    if (h == -1)
        h = paramsH->h;
    sphMarkers_H->posRadH.push_back(mR4(pos, h));
    sphMarkers_H->velH.push_back(vel);
    sphMarkers_H->rhoPresEnH.push_back(mR4(paramsH->rho0, paramsH->p0, paramsH->e0,
                                           GetMarkerCode(type)));  // GetMarkerCode returns the numeric code depending on the enum type in type variable
    sphMarkers_H->soundH.push_back(paramsH->Cs0);
    massH.push_back(mass);
}

// Uses referenceArray to set the counters of various particles type.
void FsiDataManager_csph::SetCounters(unsigned int num_fsi_bodies) {
    // meshes not considered in this model, counters present in original Counters struct, but here won't be modified.
    countersH->numFsiBodies = num_fsi_bodies;


    countersH->numGhostMarkers = 0;       // Number of ghost particles
    countersH->numHelperMarkers = 0;      // Number of helper particles
    countersH->numFluidMarkers = 0;       // Number of fluid SPH particles
    countersH->numBoundaryMarkers = 0;    // Number of boundary BCE markers
    countersH->numRigidMarkers = 0;       // Number of rigid BCE markers
    countersH->numBceMarkers = 0;         // Total number of BCE markers
    countersH->numAllMarkers = 0;         // Total number of SPH + BCE particles
    countersH->startBoundaryMarkers = 0;  // Start index of the boundary BCE markers
    countersH->startRigidMarkers = 0;     // Start index of the rigid BCE markers

    size_t rSize = referenceArray.size();

    for (size_t i = 0; i < rSize; i++) {
        int4 rComp4 = referenceArray[i];       // components of phase i
        int numMarkers = rComp4.y - rComp4.x;  // number of particles present in phase i (end index - start index)

        switch (rComp4.z) {  // referenceArray.z is the particle type
            case -3:
                countersH->numHelperMarkers += numMarkers;
                break;
            case -2:
                countersH->numGhostMarkers += numMarkers;
                break;
            case -1:
                countersH->numFluidMarkers += numMarkers;
                break;
            case 0:
                countersH->numBoundaryMarkers += numMarkers;
                break;
            case 1:
                countersH->numRigidMarkers += numMarkers;
                break;
            case 2:
                std::cerr << "ERROR SetCounters: 1D flexible elements not supported";
                throw std::runtime_error("SetCounters: 1D flexible element not supported.");
                break;
            case 3:
                std::cerr << "ERROR SetCounters: 2D flexible elements not supported";
                throw std::runtime_error("SetCounters: 2D flexible element not supported.");
                break;
            default:
                std::cerr << "ERROR SetCounters: particle type not defined." << std::endl;
                throw std::runtime_error("SetCounters: Particle type not defined.");
                break;
        }
    }

    countersH->numFluidMarkers += countersH->numGhostMarkers + countersH->numHelperMarkers;
    countersH->numBceMarkers = countersH->numBoundaryMarkers + countersH->numRigidMarkers;  //
                               
    countersH->numAllMarkers = countersH->numFluidMarkers + countersH->numBceMarkers;

    countersH->startBoundaryMarkers = countersH->numFluidMarkers;
    countersH->startRigidMarkers = countersH->startBoundaryMarkers + countersH->numBoundaryMarkers;
}

struct sphTypeCompEqual {
    __host__ __device__ bool operator()(const Real4& o1, const Real4& o2) { return o1.w == o2.w; }
};


// build reference_array from the sph particles data
void FsiDataManager_csph::ConstructReferenceArray() {
    auto numAllMarkers = sphMarkers_H->rhoPresEnH.size();  // Recall rhoPresEnH is a host vector of 4 Reals
                                                           // rhoPresEnH should include all Sph markers in the system.
    thrust::host_vector<int> numComponentMarkers(numAllMarkers);  // host vector of size = number of sph particles
    thrust::fill(numComponentMarkers.begin(), numComponentMarkers.end(), 1);  // fill it with 1
    thrust::host_vector<Real4> dummyRhoPresEnH = sphMarkers_H->rhoPresEnH;    // dummy host vector

    // reduction of input vectors (key,value) by the keys values. thurst::pair<OutputIter1,OutputIter2> =
    // reduce_by_key(InputIter key_first, InputIter key_last, Iter value_first, OutputIter key_output, OutputIter
    // value_out, binary_pred_equality) Each group of consecutive keys in the range [key_first, key_last) that are
    // equal, the algorithm copies the key first element in key_output. The corresponding values in the range are
    // reduced by sum and copied in values_output. Here use sphTypeCompEqual to test for equality of keys -> key are
    // equal if w component is the same, here it's the type of particle.
    auto new_end = thrust::reduce_by_key(dummyRhoPresEnH.begin(), dummyRhoPresEnH.end(),  // keys first, last
                                         numComponentMarkers.begin(),                     // values first
                                         dummyRhoPresEnH.begin(),                         // keys out
                                         numComponentMarkers.begin(),                     // values out
                                         sphTypeCompEqual());

    // since we reduced by keys with the key being the particle values and value vector being a set of 1, we have in
    // numComponentMarkers the number of sph particles for each particle type.

    size_t numberOfComponents = new_end.first - dummyRhoPresEnH.begin();
    // number of unique type components found (assume particles in rhoPresMuH were sorted by type??)
    // specifically new_end.first points to the end of the key values processed.
    // RhoPresEnH IS ORDERED BY PARTICLE TYPE BY CONSTRUCTION SO WE GET UNIQUE KEYS ???
    // NOT SURE! ADDSPHPARTICLE AND ADDBCEPASRTICLE DO NOT ADD PARTICLES INTO sphMarkers_H IN ANY ORDER

    dummyRhoPresEnH.resize(numberOfComponents);
    numComponentMarkers.resize(numberOfComponents);  // resize, so for each particle type in dummyRhoPresMuH.w the corresponding element in
                                                     // numComponentMarkers tells how many particles there are

    referenceArray.clear();
 
    // Loop through all components loading referenceArray and referenceArray_FEA.
    // phaseType depends on componentType.
    int start_index = 0;
    for (size_t i = 0; i < numberOfComponents; i++) {
        int compType = (int)std::floor(dummyRhoPresEnH[i].w + .1);
        int phaseType = -1;
        if (compType == -3) {
            phaseType = -1;  // For helper
        } else if (compType == -2) {
            phaseType = -1;  // For ghost
        } else if (compType == -1) {
            phaseType = -1;  // For fluid/granular
        } else if (compType == 0) {
            phaseType = 0;  // For boundary
        } else if (compType == 1) {
            phaseType = 1;  // For rigid
        } else if (compType == 2) {
            phaseType = 1;  // For 1D cable elements
        } else if (compType == 3) {
            phaseType = 1;  // For 2D shell elements
        } else {
            phaseType = 1;
        }
        auto new_entry = mI4(start_index, start_index + numComponentMarkers[i], compType,
                             phaseType);  // the reference vector value for a specific component type.
        start_index += numComponentMarkers[i];

        referenceArray.push_back(new_entry);
        if (compType == 2 || compType == 3)
            ;   // mesh-related reference array absent in this implementation
    }

    dummyRhoPresEnH.clear();
    numComponentMarkers.clear();
}


// Reset stored data to 0. Doesn't resize any vector
void FsiDataManager_csph::ResetData() {
    auto zero3 = mR3(0);
    auto zero5 = mR5(0);
    Real zero = 0.0;
    thrust::fill(derivVelRhoEnD.begin(), derivVelRhoEnD.end(), zero5);
    thrust::fill(derivVelRhoEnOriginalD.begin(), derivVelRhoEnOriginalD.end(), zero5);;
    thrust::fill(derivRadD.begin(), derivRadD.end(), zero);     
    thrust::fill(derivRadOriginalD.begin(), derivRadOriginalD.end(), zero);

    thrust::fill(freeSurfaceIdD.begin(), freeSurfaceIdD.end(), 0);
    thrust::fill(vel_XSPH_D.begin(), vel_XSPH_D.end(), zero3);


    //// bceAcc needed in Adami bcs
    thrust::fill(bceAcc.begin(), bceAcc.end(), zero3);

    thrust::fill(dT_velD.begin(), dT_velD.end(), FLT_MAX);       // then compute dT only for fluid particles, will be less than FLT_MAX
    thrust::fill(dT_forceD.begin(), dT_forceD.end(), FLT_MAX);

    if (paramsH->Ar_heat_switch) {
        thrust::fill(divVelD.begin(), divVelD.end(), zero);
        thrust::fill(heatCoeffD.begin(), heatCoeffD.end(), zero);
    }
}

// ------------------------------------------------------------------------------
// Resize length of vectors depending on particle activity
void FsiDataManager_csph::ResizeArrays(uint numExtended) {
    bool should_shrink = false;
    m_resize_counter++;

    // On first allocation or if we exceed current capacity, grow with buffer
    if (numExtended > m_max_extended_particles) {
        // Add buffer for future growth
        uint new_capacity = static_cast<uint>(numExtended * GROWTH_FACTOR);

        // Reserve space in all arrays
        markersProximity_D->gridMarkerHashD.reserve(new_capacity);
        markersProximity_D->gridMarkerIndexD.reserve(new_capacity);

        sortedSphMarkers2_D->posRadD.reserve(new_capacity);
        sortedSphMarkers2_D->velD.reserve(new_capacity);
        sortedSphMarkers2_D->rhoPresEnD.reserve(new_capacity);
        sortedSphMarkers2_D->soundD.reserve(new_capacity);

        sortedSphMarkers1_D->posRadD.reserve(new_capacity);
        sortedSphMarkers1_D->velD.reserve(new_capacity);
        sortedSphMarkers1_D->rhoPresEnD.reserve(new_capacity);
        sortedSphMarkers1_D->soundD.reserve(new_capacity);

        freeSurfaceIdD.reserve(new_capacity);
        vel_XSPH_D.reserve(new_capacity);

        dT_velD.reserve(new_capacity);
        dT_forceD.reserve(new_capacity);

        if (!paramsH->is_uniform) {
            massD.reserve(new_capacity);
            sortedMassD.reserve(new_capacity);
        }

        if (paramsH->Ar_heat_switch) {
            divVelD.reserve(new_capacity);
            heatCoeffD.reserve(new_capacity);
        }

        m_max_extended_particles = new_capacity;
    }

    // Check if we should shrink based on counter and memory usage
    if (m_resize_counter >= SHRINK_INTERVAL) {
        float usage_ratio = float(numExtended) / markersProximity_D->gridMarkerHashD.capacity();
        should_shrink = (usage_ratio < SHRINK_THRESHOLD);
        m_resize_counter = 0;
    }

    // Always resize to actual size needed
    markersProximity_D->gridMarkerHashD.resize(numExtended);
    markersProximity_D->gridMarkerIndexD.resize(numExtended);

    sortedSphMarkers2_D->posRadD.resize(numExtended);
    sortedSphMarkers2_D->velD.resize(numExtended);
    sortedSphMarkers2_D->rhoPresEnD.resize(numExtended);
    sortedSphMarkers2_D->soundD.resize(numExtended);

    sortedSphMarkers1_D->posRadD.resize(numExtended);
    sortedSphMarkers1_D->velD.resize(numExtended);
    sortedSphMarkers1_D->rhoPresEnD.resize(numExtended);
    sortedSphMarkers1_D->soundD.resize(numExtended);

    derivVelRhoEnD.resize(numExtended);
    if (paramsH->h_evolution == H_evolution_csph::DIFFERENTIAL) {
        derivRadD.resize(numExtended);
    }
   

    freeSurfaceIdD.resize(numExtended);
    vel_XSPH_D.resize(numExtended);

    dT_velD.resize(numExtended);
    dT_forceD.resize(numExtended);

    if (!paramsH->is_uniform) {
        massD.resize(numExtended);
        sortedMassD.resize(numExtended);
    }

    if (paramsH->Ar_heat_switch) {
        divVelD.resize(numExtended);
        heatCoeffD.resize(numExtended);
    }

    // Only shrink periodically if needed
    if (should_shrink) {
        markersProximity_D->gridMarkerHashD.shrink_to_fit();
        markersProximity_D->gridMarkerIndexD.shrink_to_fit();

        sortedSphMarkers2_D->posRadD.shrink_to_fit();
        sortedSphMarkers2_D->velD.shrink_to_fit();
        sortedSphMarkers2_D->rhoPresEnD.shrink_to_fit();
        sortedSphMarkers2_D->soundD.shrink_to_fit();

        sortedSphMarkers1_D->posRadD.shrink_to_fit();
        sortedSphMarkers1_D->velD.shrink_to_fit();
        sortedSphMarkers1_D->rhoPresEnD.shrink_to_fit();
        sortedSphMarkers1_D->soundD.shrink_to_fit();

        derivVelRhoEnD.shrink_to_fit();
        if (paramsH->h_evolution == H_evolution_csph::DIFFERENTIAL) {
            derivRadD.shrink_to_fit();
        }
        
        freeSurfaceIdD.shrink_to_fit();
        vel_XSPH_D.shrink_to_fit();

        dT_velD.shrink_to_fit();
        dT_forceD.shrink_to_fit();

        if (!paramsH->is_uniform) {
            massD.shrink_to_fit();
            sortedMassD.shrink_to_fit();
        }

        if (paramsH->Ar_heat_switch) {
            divVelD.shrink_to_fit();
            heatCoeffD.shrink_to_fit();
        }

        // Update max particles to match new capacity
        m_max_extended_particles = (uint)markersProximity_D->gridMarkerHashD.capacity();
    }
}


//--------------------------------------------------------------
// Function to check consistency of mass input
void FsiDataManager_csph::CheckMassConsistency(){

    struct MassPositive {
        __host__ bool operator()(Real val) { return val > 0; }
    };

    int countM1 = thrust::count(massH.begin(), massH.end(), -1);
    int Size = massH.size();

    bool AllPos = thrust::count_if(massH.begin(), massH.end(), MassPositive()) == Size;
    bool AllMin1 = countM1 == Size;
    bool AllEqual = thrust::count(massH.begin(), massH.end(), massH[0]) == Size;
    bool AllEqualMarker = thrust::count(massH.begin(), massH.end(), paramsH->markerMass) == Size; 
    bool AnyMinus1 = countM1 > 0;


    if (paramsH-> is_uniform) {
        if (paramsH->markerMass < 0) {
            std::cerr << "Something went wrong in the mass computation. Problem assumed uniform but markerMass < 0. "
                         "Check again the input mass"
                      << std::endl;
            std::exit(-10);
        } else if (AllEqualMarker)
            return;        // ok, no need to do anything
        else if (AllMin1) {  // ok
            thrust::fill(massH.begin(), massH.end(), paramsH->markerMass);
            return;
        }
        else if (AllEqual && !AllEqualMarker) {
            std::cout << "All masses in the vector are positive and equal but different from markerMass. Changing "
                         "the markerMass value to massH[0]"
                      << std::endl;
            paramsH->markerMass = massH[0];
            return;
        }
        else if (!AllEqual && AllPos) {
            std::cout << "Problem set to uniform but all masses are > 0 and at lest one is different. Setting problem "
                         "to non-uniform"
                      << std::endl;
            paramsH->is_uniform = false;
            return;
        }
        else if (!AllEqual && !AllPos ) {
            std::cout << "Problem set to uniform but not all masses are equal and at least one mass set to -1. Check again the input masses"
                      << std::endl;
            std::exit(-11);
        } 
    }

    if ( !paramsH->is_uniform) {
        if (AllMin1) {
            std::cout << "Problem set as non-uniform but all mass values are equal to -1. Check again the input."
                      << std::endl;
            std::exit(-12);
        } else if (AllEqual && AllPos && !AllEqualMarker) {
            std::cerr << "Problem set as non-uniform but all masses are equal to a value >0 and different to "
                         "markerMass. Setting problem as uniform and markerMass = mass[0]"
                      << std::endl;
            paramsH->is_uniform = true;
            paramsH->markerMass = massH[0];
            return;
                      
        } else if (!AllEqual && AllPos) { // ok, do nothing
            return;
        } else if (!AllEqual && AnyMinus1) {
            std::cerr
                << "Error! Problem set as non uniform, but at least one mass set to -1. Check again the input masses."
                << std::endl;
            std::exit(-13);
        } else if (AllEqual && AllPos && AllEqualMarker) {
            std::cout << "Problem set to non-uniform but alla masses are equal to markerMass, which is > 0. Setting "
                         "the problem to uniform"
                      << std::endl;
            paramsH->is_uniform = true;
        }
    }

    return;

}



//--------------------------------------------------------------------------------------------------------------------------------
// Function to initialize the Data Manager system. 
void FsiDataManager_csph::Initialize(unsigned int num_fsi_bodies) {
    ConstructReferenceArray();
    SetCounters(num_fsi_bodies);
    CheckMassConsistency();

    // check consistency in number of sph markers
    if (countersH->numAllMarkers != sphMarkers_H->rhoPresEnH.size()) {
        std::cerr << "ERROR (Initialize): mismatch in total number of markers." << std::endl;
        throw std::runtime_error("Mismatch in total number of markers.");
    }

    sphMarkers_D->resize(countersH->numAllMarkers);
    sphMarkers_H->resize(countersH->numAllMarkers);
    markersProximity_D->resize(countersH->numAllMarkers);

    derivVelRhoEnOriginalD.resize(countersH->numAllMarkers);
    if (paramsH->h_evolution == H_evolution_csph::DIFFERENTIAL) {
        derivRadOriginalD.resize(countersH->numAllMarkers);
    }
    

    
    // needed in Adami bcs
    bceAcc.resize(countersH->numAllMarkers, mR3(0));  // Rigid/flex body accelerations from motion. Notice like other
                                                      // vectors it's initialized to numAllMarkers length.

    activityIdentifierOriginalD.resize(countersH->numAllMarkers, 1);
    activityIdentifierSortedD.resize(countersH->numAllMarkers, 1);
    extendedActivityIdentifierOriginalD.resize(countersH->numAllMarkers, 1);
    prefixSumExtendedActivityIdD.resize(countersH->numAllMarkers, 1);
    activeListD.resize(countersH->numAllMarkers, 1);

    // Number of neighbors for the particle of given index
    numNeighborsPerPart.resize(countersH->numAllMarkers + 1, 0);
    // copy marker data from host vector to device vectors. Host vectors should already be populated by calls to
    // AddSphParticle and AddBceMarker. But are data in the host vector sorted in some ways? Like fluid first and bce
    // after?
    thrust::copy(sphMarkers_H->posRadH.begin(), sphMarkers_H->posRadH.end(), sphMarkers_D->posRadD.begin());
    thrust::copy(sphMarkers_H->velH.begin(), sphMarkers_H->velH.end(), sphMarkers_D->velD.begin());
    thrust::copy(sphMarkers_H->rhoPresEnH.begin(), sphMarkers_H->rhoPresEnH.end(), sphMarkers_D->rhoPresEnD.begin());
    thrust::copy(sphMarkers_H->soundH.begin(), sphMarkers_H->soundH.end(), sphMarkers_D->soundD.begin());

    fsiBodyState_D->Resize(countersH->numFsiBodies);
    fsiBodyState_H->Resize(countersH->numFsiBodies);

    rigid_FSI_ForcesD.resize(countersH->numFsiBodies);
    rigid_FSI_TorquesD.resize(countersH->numFsiBodies);


    // dealing with mass arrays. Called by ChFluidSystem_csph.Initialize() after optional computation of mass = rho/sum(W)

    if (!paramsH->is_uniform) {
        massH.resize(countersH->numAllMarkers);
        massD.resize(countersH->numAllMarkers);
        thrust::copy(massH.begin(), massH.end(), massD.begin());
    } 

    if (paramsH->Ar_heat_switch) {
        divVelD.resize(countersH->numAllMarkers);
        heatCoeffD.resize(countersH->numAllMarkers);
    }
}

//--------------------------------------------------------------------------------------------------
// Methods to access properties of sph particles
//--------------------------------------------------------------------------------------------------
struct extract_functor {
    extract_functor() {}
    __host__ __device__ Real3 operator()(Real4& x) const { return mR3(x); }
};


struct extract_functor_deriv {
    extract_functor_deriv() {}
    __host__ __device__ Real3 operator()(Real5& x) const { return mR3(x.x, x.y, x.z); }
};


struct scale_functor {
    scale_functor(Real a) : m_a(a) {}
    __host__ __device__ Real3 operator()(Real3& x) const { return m_a * x; }
    const Real m_a;
};


// Position of all markers (sph and bce)
std::vector<Real3> FsiDataManager_csph::GetPositions() {
    auto& pos4_D = sphMarkers_D->posRadD;

    // Extract positions only (drop radius)
    thrust::device_vector<Real3> pos_D(pos4_D.size());
    thrust::transform(pos4_D.begin(), pos4_D.end(), pos_D.begin(), extract_functor());

    // Copy to output
    std::vector<Real3> pos_H(pos_D.size());
    thrust::copy(pos_D.begin(), pos_D.end(), pos_H.begin());
    return pos_H;
}


// Velocity of all markers (sph and bce)
std::vector<Real3> FsiDataManager_csph::GetVelocities() {
    const auto& vel_D = sphMarkers_D->velD;

    // Copy to output
    std::vector<Real3> vel_H(vel_D.size());
    thrust::copy(vel_D.begin(), vel_D.end(), vel_H.begin());
    return vel_H;
}


// declaration says Acceleration of all markers (sph and bce),
// but here looks acceleration for fluid markers only.
std::vector<Real3> FsiDataManager_csph::GetAccelerations() {
    // Copy data for SPH particles only
    const auto n = countersH->numFluidMarkers;
    thrust::device_vector<Real5> acc5_D(n);
    thrust::copy_n(derivVelRhoEnD.begin(), n, acc5_D.begin());

    // Extract positions only (drop radius)
    thrust::device_vector<Real3> acc3_D(acc5_D.size());
    thrust::transform(acc5_D.begin(), acc5_D.end(), acc3_D.begin(), extract_functor_deriv());

    // Copy to output
    std::vector<Real3> acc_H(acc3_D.size());
    thrust::copy(acc3_D.begin(), acc3_D.end(), acc_H.begin());
    return acc_H;
}


// declaration says Forces of all markers (sph and bce),
// but here looks forces for fluid markers only, because GetAcceleration works with fluid particles only
std::vector<Real3> FsiDataManager_csph::GetForces() {
    std::vector<Real3> frc_H = GetAccelerations();
    std::transform(frc_H.begin(), frc_H.end(), frc_H.begin(), scale_functor(paramsH->markerMass));
    return frc_H;
}


// Access properties (rho, pres, energy) of all markers (sph and bce),
std::vector<Real3> FsiDataManager_csph::GetProperties() {
    auto& prop4_D = sphMarkers_D->rhoPresEnD;

    // Extract fluid properties only (drop particle type)
    thrust::device_vector<Real3> prop_D(prop4_D.size());
    thrust::transform(prop4_D.begin(), prop4_D.end(), prop_D.begin(), extract_functor());

    // Copy to output
    std::vector<Real3> prop_H(prop_D.size());
    thrust::copy(prop_D.begin(), prop_D.end(), prop_H.begin());
    return prop_H;
}


// access sound speed of all markers (sph and bce)
std::vector<Real> FsiDataManager_csph::GetSound() {
    auto& sound_D = sphMarkers_D->soundD;
    std::vector<Real> sound_H(sound_D.size());
    thrust::copy(sound_D.begin(), sound_D.end(), sound_H.begin());
    return sound_H;
}


// functor to extract kernel radius from Real4
struct extract_functor_kernrad {
    extract_functor_kernrad() {}
    __host__ __device__ Real operator()(Real4& x) const { return x.w; }
};


// access kernel radius of all particles (sph and bce)
std::vector<Real> FsiDataManager_csph::GetKernelRad() {
    auto& temp_D = sphMarkers_D->posRadD;

    // Extract kernel radius only (drop position)
    thrust::device_vector<Real> kernRad_D(temp_D.size());
    thrust::transform(temp_D.begin(), temp_D.end(), kernRad_D.begin(), extract_functor_kernrad());

    // Copy to output
    std::vector<Real> kernRad_H(kernRad_D.size());
    thrust::copy(kernRad_D.begin(), kernRad_D.end(), kernRad_H.begin());
    return kernRad_H;
}


struct compareRadValue {
    __host__ __device__ bool operator() (const Real4& left, const Real4& right) {
        return left.w < right.w;
    }
};


// access maximum kernel radius of all particles (sph and bce)
Real2 FsiDataManager_csph::GetMaxRad() {
    // iterate through posRadD to find maximum h
    auto it = thrust::max_element(sphMarkers_D->posRadD.begin(), sphMarkers_D->posRadD.end(), compareRadValue());
    // index of max_h and its value:
    int idx = it - sphMarkers_D->posRadD.begin();
    Real4 maxRad = sphMarkers_D->posRadD[idx];
    // associated particle type:
    Real4 type = sphMarkers_D->rhoPresEnD[idx];
    return mR2(maxRad.w, type.w);
}

// access maximum kernel radius among all fluid particles
Real FsiDataManager_csph::GetMaxFluidRad() {
    // number of fluid particle is:
    const auto n = countersH->numFluidMarkers;
    // find maximum element:
    auto it = thrust::max_element(sphMarkers_D->posRadD.begin(), sphMarkers_D->posRadD.begin() + n, compareRadValue());
    // return corresponding h:
    int idx = it - sphMarkers_D->posRadD.begin();
    Real4 maxRad = sphMarkers_D->posRadD[idx];
    return maxRad.w;
}


//---------------------------------------------------------------------------
// Methods to access properties of sph particles in the given index vector.
//---------------------------------------------------------------------------

std::vector<Real3> FsiDataManager_csph::GetPositions(const std::vector<int>& indices) {
    // pass vector of indices to device
    thrust::device_vector<int> indices_D(indices.size());
    thrust::copy(indices.begin(), indices.end(), indices_D.begin());

    // Get all extended positions
    auto& allpos4_D = sphMarkers_D->posRadD;  // initialized with maximum possible length

    // Gather only those for specified indices
    thrust::device_vector<Real4> pos4_D(allpos4_D.size());
    auto end = thrust::gather(thrust::device,                      // execution policy
                              indices_D.begin(), indices_D.end(),  // range of gather locations
                              allpos4_D.begin(),                   // beginning of source
                              pos4_D.begin()                       // beginning of destination
    );

    // Extract positions only (drop radius)
    thrust::device_vector<Real3> pos_D(pos4_D.size());
    thrust::transform(pos4_D.begin(), pos4_D.end(), pos_D.begin(), extract_functor());

    // Trim the output vector of particle positions.
    // The actual size of the indices array may be lower than the length of sphMarkers vector. num_active doesn't refer
    // to the actual distinction between active and inactive particles
    size_t num_active = (size_t)(end - pos4_D.begin());
    assert(num_active == indices_D.size());
    pos_D.resize(num_active);

    // Copy to output
    std::vector<Real3> pos_H(pos_D.size());
    thrust::copy(pos_D.begin(), pos_D.end(), pos_H.begin());
    return pos_H;
}



std::vector<Real3> FsiDataManager_csph::GetVelocities(const std::vector<int>& indices) {
    // create device vector of indices
    thrust::device_vector<int> indices_D(indices.size());
    thrust::copy(indices.begin(), indices.end(), indices_D.begin());

    // Get all velocities
    auto allvel_D = sphMarkers_D->velD;

    // Gather only those for specified indices
    thrust::device_vector<Real3> vel_D(allvel_D.size());
    auto end = thrust::gather(thrust::device,                      // execution policy
                              indices_D.begin(), indices_D.end(),  // range of gather locations
                              allvel_D.begin(),                    // beginning of source
                              vel_D.begin()                        // beginning of destination
    );

    // Trim the output vector of particle positions. Again num_active refers to the number of indexes requested, not to
    // distinction between active and inactive particles.
    size_t num_active = (size_t)(end - vel_D.begin());
    assert(num_active == indices_D.size());
    vel_D.resize(num_active);

    // Copy to output
    std::vector<Real3> vel_H(vel_D.size());
    thrust::copy(vel_D.begin(), vel_D.end(), vel_H.begin());
    return vel_H;
}



std::vector<Real3> FsiDataManager_csph::GetAccelerations(const std::vector<int>& indices) {
    // pass vector of indices to device
    thrust::device_vector<int> indices_D(indices.size());
    thrust::copy(indices.begin(), indices.end(), indices_D.begin());

    // Get all extended derivatives
    auto& allAcc5_D = derivVelRhoEnD;  // initialized with maximum possible length

    // Gather only those for specified indices
    thrust::device_vector<Real5> acc5_D(allAcc5_D.size());
    auto end = thrust::gather(thrust::device,                      // execution policy
                              indices_D.begin(), indices_D.end(),  // range of gather locations
                              allAcc5_D.begin(),                   // beginning of source
                              acc5_D.begin()                       // beginning of destination
    );

    // Extract acceleration only (drop rho and en)
    thrust::device_vector<Real3> acc3_D(acc5_D.size());
    thrust::transform(acc5_D.begin(), acc5_D.end(), acc3_D.begin(), extract_functor_deriv());

    // Trim the output vector of particle acceleration.
    // The actual size of the indices array may be lower than the length of sphMarkers vector. num_active doesn't refer
    // to the actual distinction between active and inactive particles
    size_t num_active = (size_t)(end - acc5_D.begin());
    assert(num_active == indices_D.size());
    acc3_D.resize(num_active);

    // Copy to output
    std::vector<Real3> acc3_H(acc3_D.size());
    thrust::copy(acc3_D.begin(), acc3_D.end(), acc3_H.begin());
    return acc3_H;
}

 

std::vector<Real3> FsiDataManager_csph::GetForces(const std::vector<int>& indices) {
    std::vector<Real3> frc_H = GetAccelerations(indices);
    std::transform(frc_H.begin(), frc_H.end(), frc_H.begin(), scale_functor(paramsH->markerMass));
    return frc_H;
}


//-------------------------------------------------------------------
// Methods to access forces and torques acting on rigid bodies
//-------------------------------------------------------------------

std::vector<Real3> FsiDataManager_csph::GetRigidForces() {
    std::vector<Real3> out_H(rigid_FSI_ForcesD.size());
    thrust::copy(rigid_FSI_ForcesD.begin(), rigid_FSI_ForcesD.end(), out_H.begin());
    return out_H;
}

std::vector<Real3> FsiDataManager_csph::GetRigidTorques() {
    std::vector<Real3> out_H(rigid_FSI_TorquesD.size());
    thrust::copy(rigid_FSI_TorquesD.begin(), rigid_FSI_TorquesD.end(), out_H.begin());
    return out_H;
}


//--------------------------------------------------------------------------------------------------------------------------------

struct in_box {
    in_box() {}

    __device__ bool operator()(const Real4 v) {
        // Convert location in box frame
        auto d = mR3(v) - pos;  // input position relative to center of box
        auto w = mR3(           // Turns the relative position into the box frame (oriented along its axis)
            ax.x * d.x + ax.y * d.y + ax.z * d.z,  //
            ay.x * d.x + ay.y * d.y + ay.z * d.z,  //
            az.x * d.x + az.y * d.y + az.z * d.z   //
        );
        // Check w between all box limits
        return (w.x >= -hsize.x && w.x <= +hsize.x) && (w.y >= -hsize.y && w.y <= +hsize.y) &&
               (w.z >= -hsize.z && w.z <= +hsize.z);
    }

    Real3 hsize;  // Semi dimensions of the box
    Real3 pos;    // Position of center of box
    Real3 ax;     // Local x,y,z axes of the box with respect to global coordinates
    Real3 ay;
    Real3 az;
};

std::vector<int> FsiDataManager_csph::FindParticlesInBox(const Real3& hsize,
                                                    const Real3& pos,
                                                    const Real3& ax,
                                                    const Real3& ay,
                                                    const Real3& az) {
    // Extract indices of SPH particles contained in the OBB
    auto& ref = referenceArray;           // reference to a host vector of int4
    auto& pos_D = sphMarkers_D->posRadD;  // reference to a thrust device vector

    // Find start and end locations for SPH particles (exclude ghost and BCE markers)
    int haveHelper = (ref[0].z == -3) ? 1 : 0;  // if present, assumed located at beginnig of reference array
    int haveGhost = (ref[0].z == -2 || ref[1].z == -2) ? 1 : 0;
    auto sph_start = ref[haveHelper + haveGhost].x;  // if present, the int is 1, so we skip them in the reference array
    auto sph_end = ref[haveHelper + haveGhost].y;
    auto num_sph = sph_end - sph_start;  // number of fluid sph particles which are not ghost or helper or bce.

    // Preallocate output vector of indices
    thrust::device_vector<int> indices_D(num_sph);

    // Extract indices of SPH particles inside OBB
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last(num_sph);
    in_box predicate;
    predicate.hsize = hsize;
    predicate.pos = pos;
    predicate.ax = ax;
    predicate.ay = ay;
    predicate.az = az;

    // this copy_if copies the indices between first and last into indices_D vector, if the predicate applied to the
    // corresponding stencil element evaluates to true.
    auto end = thrust::copy_if(thrust::device,     // execution policy
                               first, last,        // range of all particle indices
                               pos_D.begin(),      // stencil vector
                               indices_D.begin(),  // beginning of destination
                               predicate           // predicate for stencil elements
    );
    // the end output iterator points to indices_D.begin() + number of times predicate evaluated to true.

    // Trim the output vector of indices
    size_t num_active = (size_t)(end - indices_D.begin());  // number of particles inside the box.
    indices_D.resize(num_active);  // shrink/resize array to actual number of particles in the box.

    // Copy to output
    std::vector<int> indices_H;
    thrust::copy(indices_D.begin(), indices_D.end(), indices_H.begin());
    return indices_H;
}

// Method to compute amount of GPU (device) memory allocated by FsiDataManager_csph
size_t FsiDataManager_csph::GetCurrentGPUMemoryUsage() const {
    size_t total_bytes = 0;

    // SPH marker data
    total_bytes += sphMarkers_D->posRadD.capacity() * sizeof(Real4);
    total_bytes += sphMarkers_D->velD.capacity() * sizeof(Real3);
    total_bytes += sphMarkers_D->rhoPresEnD.capacity() * sizeof(Real4);
    total_bytes += sphMarkers_D->soundD.capacity() * sizeof(Real);

    // Sorted SPH marker data (state 1)
    total_bytes += sortedSphMarkers1_D->posRadD.capacity() * sizeof(Real4);
    total_bytes += sortedSphMarkers1_D->velD.capacity() * sizeof(Real3);
    total_bytes += sortedSphMarkers1_D->rhoPresEnD.capacity() * sizeof(Real4);
    total_bytes += sortedSphMarkers1_D->soundD.capacity() * sizeof(Real);

    // Sorted SPH marker data (state 2)
    total_bytes += sortedSphMarkers2_D->posRadD.capacity() * sizeof(Real4);
    total_bytes += sortedSphMarkers2_D->velD.capacity() * sizeof(Real3);
    total_bytes += sortedSphMarkers2_D->rhoPresEnD.capacity() * sizeof(Real4);
    total_bytes += sortedSphMarkers2_D->soundD.capacity() * sizeof(Real);

    // Mass vectors data
    total_bytes += massD.capacity() * sizeof(Real);
    total_bytes += sortedMassD.capacity() * sizeof(Real);

    // Proximity data
    total_bytes += markersProximity_D->gridMarkerHashD.capacity() * sizeof(uint);
    total_bytes += markersProximity_D->gridMarkerIndexD.capacity() * sizeof(uint);
    total_bytes += markersProximity_D->cellStartD.capacity() * sizeof(uint);
    total_bytes += markersProximity_D->cellEndD.capacity() * sizeof(uint);
    total_bytes += markersProximity_D->mapOriginalToSorted.capacity() * sizeof(uint);

    // FSI body state data
    total_bytes += fsiBodyState_D->pos.capacity() * sizeof(Real3);
    total_bytes += fsiBodyState_D->lin_vel.capacity() * sizeof(Real3);
    total_bytes += fsiBodyState_D->lin_acc.capacity() * sizeof(Real3);
    total_bytes += fsiBodyState_D->rot.capacity() * sizeof(Real4);
    total_bytes += fsiBodyState_D->ang_vel.capacity() * sizeof(Real3);
    total_bytes += fsiBodyState_D->ang_acc.capacity() * sizeof(Real3);


    // Fluid data
    total_bytes += derivVelRhoEnD.capacity() * sizeof(Real5);
    total_bytes += derivVelRhoEnOriginalD.capacity() * sizeof(Real5);
    total_bytes += derivRadD.capacity() * sizeof(Real);
    total_bytes += derivRadOriginalD.capacity() * sizeof(Real);
    total_bytes += vel_XSPH_D.capacity() * sizeof(Real3);

    total_bytes += bceAcc.capacity() * sizeof(Real3);

    total_bytes += divVelD.capacity() * sizeof(Real);
    total_bytes += heatCoeffD.capacity() * sizeof(Real);

    // Activity and neighbor data
    total_bytes += activityIdentifierOriginalD.capacity() * sizeof(int32_t);
    total_bytes += activityIdentifierSortedD.capacity() * sizeof(int32_t);
    total_bytes += extendedActivityIdentifierOriginalD.capacity() * sizeof(int32_t);
    total_bytes += prefixSumExtendedActivityIdD.capacity() * sizeof(uint);
    total_bytes += activeListD.capacity() * sizeof(uint);
    total_bytes += numNeighborsPerPart.capacity() * sizeof(uint);
    total_bytes += neighborList.capacity() * sizeof(uint);
    total_bytes += freeSurfaceIdD.capacity() * sizeof(uint);

    // BCE data
    total_bytes += rigid_BCEcoords_D.capacity() * sizeof(Real3);
    total_bytes += rigid_BCEsolids_D.capacity() * sizeof(uint);

    // FSI forces
    total_bytes += rigid_FSI_ForcesD.capacity() * sizeof(Real3);
    total_bytes += rigid_FSI_TorquesD.capacity() * sizeof(Real3);



    return total_bytes;
}

}  // end namespace compressible
}  // end namespace chrono::fsi::sph
