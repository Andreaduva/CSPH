
// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Base class for processing proximity in an FSI system.
// Modification of original class to consider compressible data structures
// =============================================================================

#ifndef CH_COLLISIONSYSTEM_FSI_COMPRESSIBLE_H_
#define CH_COLLISIONSYSTEM_FSI_COMPRESSIBLE_H_

#include "chrono_fsi/ChApiFsi.h"
#include "chrono_fsi/sph/physics/FsiDataManager.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"
// FsiDataManager_compressible.cuh defines the data types used in this class.

namespace chrono::fsi::sph {
namespace compressible {


// Base class for processing proximity computation in an FSI system.
// Has a pointer to the FsiDataManager_csph struct.
// Missing where some of the underlying variables are manipulated. E.g extendedActivityList?
class CollisionSystem_csph {
  public:
    CollisionSystem_csph(FsiDataManager_csph& data_mgr);
    ~CollisionSystem_csph();

    // Complete construction.
    void Initialize();

    // Sort particles based on their bins (broad phase proximity search).
    // This function encapsulates calcHash, findCellStartEndD, and reorderDataD.
    void ArrangeData(std::shared_ptr<SphMarkerDataD_csph> sphMarkersD, std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);

    // Construct neighbor lists (narrow phase proximity search). To be called after ArrangeData
    void NeighborSearch(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkersD);

    // Updates the parameters related to the neighbor search grid, like binsize, cellsize etc.
    // Needed if h is allowed to vary, in order to guarantee that all neighbors will be considered.
    // Conservative approach based on using the maximum value of h.
    // Input markers may be sorted by hash or not, does not make a difference.
    void UpdateGridParams(std::shared_ptr<SphMarkerDataD_csph> sphMarkersD);

  private:
    FsiDataManager_csph& m_data_mgr;  //< FSI data manager reference to an object
    // Note: this is cached on every call to ArrangeData() 'WHY
    std::shared_ptr<SphMarkerDataD_csph> m_sphMarkersD;  //< Information of the particles in the original array
};



}  // end namespace compressible
}  // end namespace chrono::fsi::sph


#endif