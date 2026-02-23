// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Utility function to print the save fluid, bce, and boundary data into file
// Modified to include compressibility-related properties
// =============================================================================

#ifndef CHUTILSPRINTSPH_COMPRESSIBLE_H
#define CHUTILSPRINTSPH_COMPRESSIBLE_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "chrono_fsi/sph/utils/UtilsDevice.cuh"
#include "chrono_fsi/sph/physics/FsiDataManager.cuh"

#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"


namespace chrono::fsi::sph {
namespace compressible {


// Save current CFD SPH data to files.
// Create separate files to write fluid, boundary BCE, rigid BCE, information.
// The amount of data saved for each marker is controlled by the specified OutputLevel
void saveParticleDataCFD_csph(const std::string& dir, OutputLevel level, FsiDataManager_csph& data_mgr);
void saveParticleDataCFD_csph(const std::string& dir,
                              OutputLevel level,
                              H_evolution_csph h_evolution_type,
                              const thrust::device_vector<Real4>& posRadD,
                              const thrust::device_vector<Real3>& velD,
                              const thrust::device_vector<Real5>& derivVelRhoEnD,
                              const thrust::device_vector<Real>&  derivRadD,
                              const thrust::device_vector<Real4>& rhoPresEnD,
                              const thrust::device_vector<Real>&  soundD,
                              const thrust::host_vector<int4>& referenceArray);

// Save current FSI solid data.
// Append states and fluid forces at current time for all solids in the FSI problem.
void saveSolidData_csph(const std::string& dir, double time, FsiDataManager_csph& data_mgr);
void saveSolidData_csph(const std::string& dir,
                        double time,
                        const thrust::device_vector<Real3>& posRigidD,
                        const thrust::device_vector<Real4>& rotRigidD,
                        const thrust::device_vector<Real3>& velRigidD,
                        const thrust::device_vector<Real3>& forceRigidD,
                        const thrust::device_vector<Real3>& torqueRigidD);

// Save variable time steps in a CSV file.
void saveVariableTimeStep_csph(const std::string& dir,
                               const std::vector<double>& min_time_steps,
                               const std::vector<double>& min_courant_steps,
                               const std::vector<double>& min_force_steps);

// Save current particle data to a CSV file.
// Writes particle positions, velocities, rho, pressure, energy, temperature and speed of sound. No derivatives are saved.
void writeParticleFileCSV_csph(const std::string& filename, FsiDataManager_csph& data_mgr);
void writeParticleFileCSV_csph(const std::string& filename,
                               thrust::device_vector<Real4>& posRadD,
                               thrust::device_vector<Real3>& velD,
                               thrust::device_vector<Real4>& rhoPresEnD,
                               thrust::device_vector<Real>&  soundD,
                               thrust::host_vector<int4>& referenceArray);


// Save current particle data to a JSON file for use with Splashsurf.
// Write particle positions only.
void writeParticleFileJSON_csph(const std::string& filename, FsiDataManager_csph& data_mgr);
void writeParticleFileJSON_csph(const std::string& filename,
                                thrust::device_vector<Real4>& posRadD,
                                thrust::host_vector<int4>& referenceArray);


}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif
