// =============================================================================
// Author: Andrea D'Uva- 2025
// =============================================================================
//
// Adapt base class FsiParticleRelocator to compressible data structures
//
// =============================================================================

////#define DEBUG_LOG

#include <cstdio>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/partition.h>
#include <thrust/zip_function.h>

#include "chrono/utils/ChUtils.h"

#include "chrono_fsi/sph/physics/FsiParticleRelocator.cuh"
#include "chrono_fsi/sph/utils/UtilsDevice.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiParticleRelocator_compressible.cuh"
#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {


// constructor
FsiParticleRelocator_csph::FsiParticleRelocator_csph(FsiDataManager_csph& data_mgr, const DefaultProperties_csph& props)
    : m_data_mgr(data_mgr), m_props(props) {}


// Relocation function to shift marker position by a given vector.
// Implements a Thrust unary function to be used with thrust::for_each.
struct shift_op {
    // constructor that initializes internal variables
    shift_op(const Real3& shift, const FsiParticleRelocator_csph::DefaultProperties_csph& props) : s(shift), p(props) {}

    // the functor operator() is templated
    // However working logic is built to work on a SoA tuple, like ones for SphMarkerData
    // Notice it is  a device function
    template <typename T>
    __device__ void operator()(const T& a) const {
        // Modify position
        Real4 posw = thrust::get<0>(a);        // thrust::get<n> extracts the n-th element of a tuple. 0 is posRad
        Real3 pos = mR3(posw);                 // drop radius
        pos += s;                              // adds the shift value to the position
        thrust::get<0>(a) = mR4(pos, posw.w);  // this call allows to modify the tuple value in-place

        // Reset all other marker properties - marker type preserved
        Real3 zero3 = mR3(0);
        Real zero = Real(0.0);
        thrust::get<1>(a) = zero3;                                             // velocity
        thrust::get<2>(a) = mR4(p.rho0, p.p0, p.e0, thrust::get<2>(a).w);      // rho, pres, en, type
        thrust::get<3>(a) = zero;                                              // Cs
        }

    // internal variables or states
    Real3 s;                                              // shift in position
    FsiParticleRelocator_csph::DefaultProperties_csph p;  // properties after relocation
};





void FsiParticleRelocator_csph::Shift(MarkerType type, const Real3& shift) {
    // Get start and end indices in marker data vectors based on specified type
    // Only types allowd are SPH_PARTICLE and BCE_WALL
    int start_idx = 0;
    int end_idx = 0;
    switch (type) {
        case MarkerType::BCE_WALL:
            start_idx = (int)m_data_mgr.countersH->startBoundaryMarkers;
            end_idx = start_idx + (int)m_data_mgr.countersH->numBoundaryMarkers;
            break;
        case MarkerType::SPH_PARTICLE:
            start_idx = 0;
            end_idx = start_idx + (int)m_data_mgr.countersH->numFluidMarkers;
            break;
    }

    // Transform all markers in the specified range by passing the zip-iterator to the tuple formed by
    // the arrays in the SphMarkerData struct.
    thrust::for_each(m_data_mgr.sphMarkers_D->iterator(start_idx), m_data_mgr.sphMarkers_D->iterator(end_idx),
                     shift_op(shift, m_props));
}


// Selector function to find particles in a given AABB.
// Implements a Thrust predicate to be used with thrust::transform_if or thrust::partition.
struct inaabb_op {
    // constructor to initialize the internal AABB state
    inaabb_op(const RealAABB& aabb_src) : aabb(aabb_src) {}

    // functor operator() returns true if particle inside AABB, false in other case
    // notice it is a device function
    template <typename T>
    __device__ bool operator()(const T& a) const {
        Real4 posw = thrust::get<0>(a);  // extract posRad from input tuple
        Real3 pos = mR3(posw);           // drops radius
        // is position inside AABB?
        if (pos.x < aabb.min.x || pos.x > aabb.max.x)
            return false;
        if (pos.y < aabb.min.y || pos.y > aabb.max.y)
            return false;
        if (pos.z < aabb.min.z || pos.z > aabb.max.z)
            return false;
        return true;
    }
    // internal state is the Axis_aligned Bounding Box of interest
    RealAABB aabb;
};




// Relocation function to move particles to a given integer AABB.
// Implements a Thrust unary function to be used with thrust::for_each.
// Operates on a tuple {index, data_tuple}.
// Index is then used to go back to a 3d position, so assume methods works on indexes ordered by hash
struct togrid_op {
    // constructor to initialize internal state
    togrid_op(const IntAABB& aabb_dest, Real spacing, const FsiParticleRelocator_csph::DefaultProperties_csph& props)
        : aabb(aabb_dest), delta(spacing), p(props) {}

    // notice it is a device function
    // operatores on a tuple {index, data_tuple} where "index" is a linear index
    // So input is a nested-tuple being data_tuple a tuple itself
    template <typename T>
    __device__ T operator()(const T& t) const {
        int index = thrust::get<0>(t);  // index of input tuple
        auto a = thrust::get<1>(t);     // data of input tuple

        // 1. Convert linear index to 3D grid coordinates in an AABB of same size as destination AABB
        int idx = index;
        auto dim = aabb.max - aabb.min;  // aabb is integer based, so .max - .min can be seen as number of grid cells along
                                         // each direction x,y,z and the grid has a total of dim.x*dim.y*dim.z cells
        // "inverse hashing" to go from linear index back to a 3d point
        int x = idx % (dim.x + 1);  // modulo operator for grid dimension along x
        idx /= (dim.x + 1);
        int y = idx % (dim.y + 1);
        idx /= (dim.y + 1);
        int z = idx;

        // 2. Shift marker grid ccordinates to current destination AABB
        x += aabb.min.x;
        y += aabb.min.y;
        z += aabb.min.z;

        // Modify marker position in real coordinates (preserve marker type)
        auto w = thrust::get<0>(a).w;                      // extract marker type
        Real3 pos = mR3(delta * x, delta * y, delta * z);  // modify position by multiplication
        thrust::get<0>(a) = mR4(pos, w);                   // modify position in-place

        // Reset all other marker properties - preserve marker type
        Real3 zero3 = mR3(0);
        Real zero = Real(0.0);
        thrust::get<1>(a) = zero3;                                              // velocity
        thrust::get<2>(a) = mR4(p.rho0, p.p0, p.e0, thrust::get<2>(a).w);       // rho, pres, en, type
        thrust::get<3>(a) = zero;                                               // Cs

        return t;
    }
    // internal states
    IntAABB aabb;       // destination AABB having integer coordinates
    Real delta;         // delta or spacing value, to be multiplied by original coordinates x,y,z
    FsiParticleRelocator_csph::DefaultProperties_csph p;   // default properties after relocation
};

// move all particles of given type inside the source RealAABB box into destination IntAABB box
void FsiParticleRelocator_csph::MoveAABB2AABB(MarkerType type,
                                              const RealAABB& aabb_src,
                                              const IntAABB& aabb_dest,
                                              Real spacing) {
    // Get start and end indices in marker data vectors based on specified type
    int start_idx = 0;
    int end_idx = 0;
    switch (type) {
        case MarkerType::BCE_WALL:
            start_idx = (int)m_data_mgr.countersH->startBoundaryMarkers;
            end_idx = start_idx + (int)m_data_mgr.countersH->numBoundaryMarkers;
            break;
        case MarkerType::SPH_PARTICLE:
            start_idx = 0;
            end_idx = start_idx + (int)m_data_mgr.countersH->numFluidMarkers;
            break;
    }

    // Move markers to be relocated at beginning of data structure
    // partition reorders elements in [first,last] based on predicate() such that elements
    // that verify it precede ones that failto satisy it.
    // returns an iterator "middle" such that each iterator in [first,middle) the predicate is true
    // and false for [middle, last]
    // The relative order of output is not necessarily the same as in original case.
    // The operator actually changes the input data
    // Predicate here is true if particle is inside aabb_src.
    auto middle = thrust::partition(m_data_mgr.sphMarkers_D->iterator(start_idx),
                                    m_data_mgr.sphMarkers_D->iterator(end_idx), inaabb_op(aabb_src));
    // number of markers moved
    auto n_move = (int)(middle - m_data_mgr.sphMarkers_D->iterator(start_idx));

    ChDebugLog("Num candidate markers: " << m_data_mgr.sphMarkers_D->iterator(end_idx) -
                                                m_data_mgr.sphMarkers_D->iterator(start_idx));
    ChDebugLog("Num moved markers:     " << n_move);

    // Relocate markers based on their index
    thrust::counting_iterator<uint> idx_first(0);
    thrust::counting_iterator<uint> idx_last = idx_first + n_move;

    auto data_first = m_data_mgr.sphMarkers_D->iterator(start_idx);
    auto data_last = m_data_mgr.sphMarkers_D->iterator(start_idx + n_move);

    // for_each operator on zip-iterators acting on tuples {index, data}
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(idx_first, data_first)),
                     thrust::make_zip_iterator(thrust::make_tuple(idx_last, data_last)),
                     togrid_op(aabb_dest, spacing, m_props));
}

}  // end namespace compressible
}  // end namespace chrono::fsi::sph
