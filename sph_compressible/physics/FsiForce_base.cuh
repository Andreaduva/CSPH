// =============================================================================
// Author: Andrea D'Uva - 2025
// 
// Base class for processing SPH force in a FSI system.
// Same as original base class FsiForce. Reported here to avoid link errors
// =============================================================================

#ifndef CH_FSI_FORCE_BASE_COMPRESSIBLE_H
#define CH_FSI_FORCE_BASE_COMPRESSIBLE_H

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "chrono_fsi/sph/physics/BceManager.cuh"
#include "chrono_fsi/sph/physics/FsiDataManager.cuh"

#include "chrono_fsi/sph_compressible/physics/BceManager_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {

// Base class to calculate force between SPH particles. Renamed for clarity.
//
// This is an abstract class that defines an interface that various SPH methods should implement. The class owns a
// collision system fsi which takes care of GPU based proximity computation of the particles. It also holds a pointer
// to external data of SPH particles, proximity data, parameters, and numbers.
class FsiForce_base {
  public:
    // Base constructor for the FsiForce class.
    // The constructor instantiates the force system
    // and initializes the pointer to external data.
    FsiForce_base(FsiDataManager_csph& data_mgr,  //< FSI data manager
                  BceManager_csph& bce_mgr,       //< BCE manager
                  bool verbose);                  //< verbose output
    

    // Destructor of the FsiForce.
    virtual ~FsiForce_base();


    // Function to calculate forces on SPH particles. Pure virtual method
    virtual void ForceSPH(std::shared_ptr<SphMarkerDataD_csph> sortedSphMarkers_D, Real time, Real step) = 0;


    // Synchronize the copy of the data (parameters and number of objects) between device (GPU) and host (CPU).
    // This function needs to be called once the host data are modified.
    virtual void Initialize();

    // Copy sorted data into original data (Real).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is Real. The class is invasive, meaning that the sorted
    // data will be modified (and will be equivalent to the original). Therefore, this
    // function should be used whenever sorted data is not needed, but efficiency is preferred.
    // GridMarkerIndex vector is the mapping of indexes sorted -> original
    static void CopySortedToOriginal_Invasive_R1(thrust::device_vector<Real>& original,
                                                 thrust::device_vector<Real>& sorted,
                                                 const thrust::device_vector<uint>& gridMarkerIndex);

    // Copy sorted data into original data (Real).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is Real. The class is non-invasive, meaning that the
    // sorted data will not be modified. This comes at the expense of lower efficiency.
    static void CopySortedToOriginal_NonInvasive_R1(thrust::device_vector<Real>& original,
                                                    const thrust::device_vector<Real>& sorted,
                                                    const thrust::device_vector<uint>& gridMarkerIndex);


    // Copy sorted data into original data (real3).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is real3. The class is invasive, meaning that the sorted
    // data will be modified (and will be equivalent to the original). Therefore, this
    // function should be used whenever sorted data is not needed, but efficiency is preferred.
    // GridMarkerIndex vector is the mapping of indexes sorted -> original
    static void CopySortedToOriginal_Invasive_R3(thrust::device_vector<Real3>& original,
                                                 thrust::device_vector<Real3>& sorted,
                                                 const thrust::device_vector<uint>& gridMarkerIndex);


    // Copy sorted data into original data (real3).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is real3. The class is non-invasive, meaning that the
    // sorted data will not be modified. This comes at the expense of lower efficiency.
    static void CopySortedToOriginal_NonInvasive_R3(thrust::device_vector<Real3>& original,
                                                    const thrust::device_vector<Real3>& sorted,
                                                    const thrust::device_vector<uint>& gridMarkerIndex);


    // Copy sorted data into original data (real4).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is real4. The class is invasive, meaning that the sorted
    // data will be modified (and will be equivalent to the original). Therefore,  this
    // function should be used whenever sorted data is not needed, but efficiency is preferred.
    static void CopySortedToOriginal_Invasive_R4(thrust::device_vector<Real4>& original,
                                                 thrust::device_vector<Real4>& sorted,
                                                 const thrust::device_vector<uint>& gridMarkerIndex);


    // Copy sorted data into original data (real4).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is real4. The class is non-invasive, meaning that the
    // sorted data will not be modified. This comes at the expense of lower efficiency.
    static void CopySortedToOriginal_NonInvasive_R4(thrust::device_vector<Real4>& original,
                                                    thrust::device_vector<Real4>& sorted,
                                                    const thrust::device_vector<uint>& gridMarkerIndex);


    // Copy sorted data into original data (real2).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is real2. The class is invasive, meaning that the sorted
    // data will be modified (and will be equivalent to the original). Therefore,  this
    // function should be used whenever sorted data is not needed, but efficiency is preferred.
    static void CopySortedToOriginal_Invasive_R2(thrust::device_vector<Real2>& original,
                                                 thrust::device_vector<Real2>& sorted,
                                                 const thrust::device_vector<uint>& gridMarkerIndex);

    // Copy sorted data into original data (real2).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is real2. The class is non-invasive, meaning that the
    // sorted data will not be modified. This comes at the expense of lower efficiency.
    static void CopySortedToOriginal_NonInvasive_R2(thrust::device_vector<Real2>& original,
                                                    thrust::device_vector<Real2>& sorted,
                                                    const thrust::device_vector<uint>& gridMarkerIndex);


    // Copy sorted data into original data (real5).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is real5. The class is invasive, meaning that the sorted
    // data will be modified (and will be equivalent to the original). Therefore,  this
    // function should be used whenever sorted data is not needed, but efficiency is preferred.
    static void CopySortedToOriginal_Invasive_R5(thrust::device_vector<Real5>& original,
                                                 thrust::device_vector<Real5>& sorted,
                                                 const thrust::device_vector<uint>& gridMarkerIndex);

    // Copy sorted data into original data (real5).
    // This function copies the data that are sorted in the collision system, into the
    // original data, where data is real5. The class is non-invasive, meaning that the
    // sorted data will not be modified. This comes at the expense of lower efficiency.
    static void CopySortedToOriginal_NonInvasive_R5(thrust::device_vector<Real5>& original,
                                                    thrust::device_vector<Real5>& sorted,
                                                    const thrust::device_vector<uint>& gridMarkerIndex);
 
  protected:
    FsiDataManager_csph& m_data_mgr;  //< FSI data manager
    BceManager_csph& m_bce_mgr;       //< BCE manager

    bool m_verbose;
    bool* m_errflagD;

    friend class FluidDynamics_csph;
};



}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif