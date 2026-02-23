// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Utility function to print the save fluid, bce, and boundary data to files
// =============================================================================

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#include <thrust/reduce.h>

#include "chrono_fsi/sph/utils/UtilsDevice.cuh"
#include "chrono_fsi/sph/utils/UtilsPrintSph.cuh"

#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"
#include "chrono_fsi/sph_compressible/utils/UtilsPrintSph_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {

// -----------------------------------------------------------------------------
// CFD particle data. Particles are saved by index order, not by hash order!
// -----------------------------------------------------------------------------


// Worker function to output CFD information on file for markers in specified range depending on the desired output
// level. Host vectors as input

void SaveFileCFD_csph(const std::string& filename,  // the file will have this name
                      OutputLevel level,
                      const H_evolution_csph h_evolution,
                      thrust::host_vector<Real4> pos,
                      thrust::host_vector<Real3> vel,
                      thrust::host_vector<Real5> deriv,           // derivVelRhoEn
                      thrust::host_vector<Real>  derivRad,        
                      thrust::host_vector<Real4> rhoPresEn,
                      thrust::host_vector<Real>  sound,
                      int start_index,
                      int end_index) {
    std::ofstream file(filename);
    std::stringstream sstream;

    // Store name of variables saved
    switch (level) {
        case OutputLevel::STATE:
            sstream << "x,y,z,|U|,acc\n";
            break;
        case OutputLevel::STATE_PRESSURE:
            sstream << "x,y,z,v_x,v_y,v_z,|U|,acc_v,rho,pressure,energy\n";
            break;
        case OutputLevel::CFD_FULL:  
            sstream << "x,y,z,h,v_x,v_y,v_z,|U|,acc_v,rho,pressure,energy,c_s";
            if (h_evolution == H_evolution_csph::DIFFERENTIAL) {
                sstream << ",dhdt\n";
            } else {
                sstream << "\n";
            }
            break;
    }
    // iterate over particles in the provided range of indexes
    for (size_t i = start_index; i < end_index; i++) {
        Real4 p = pos[i];
        Real3 v = vel[i];
        Real3 a = mR3(deriv[i]);
        Real4 rp = rhoPresEn[i];
        Real  c_s = sound[i];
        Real dh = 0;
        if (h_evolution == H_evolution_csph::DIFFERENTIAL) {
            dh = derivRad[i];
        }

        Real v_len = length(v);
        Real a_len = length(a);
        // see above the list of variables saved for each output level
        switch (level) {
            case OutputLevel::STATE:
                sstream << p.x << ", " << p.y << ", " << p.z << ", " << v_len << ", " << a_len;
                break;
            case OutputLevel::STATE_PRESSURE:
                sstream << p.x << ", " << p.y << ", " << p.z << ", " << v.x << ", " << v.y << ", " << v.z << ", "
                        << v_len << ", " << a_len << ", " << rp.x << ", " << rp.y << ", " << rp.z;
                break;
            case OutputLevel::CFD_FULL:
                sstream << p.x << ", " << p.y << ", " << p.z << ", " << p.w << ", " << v.x << ", " << v.y << ", " << v.z
                        << ", " << v_len << ", " << a_len << ", " << rp.x << ", " << rp.y << ", " << rp.z << ", "
                        << c_s;
                if (h_evolution == H_evolution_csph::DIFFERENTIAL) {
                    sstream << ", " << dh;
                }
                break;
        }
        sstream << std::endl;
    }
    // save on file as string
    file << sstream.str();
    file.close();
}




// Worker function to output CFD files at current frame. Filenames are standard, only directory name to be provided.
// Inside it calls above function SaveFileCFD
// Host vectors as input
void SaveAllCFD_csph(const std::string& dir,  // directory where to save the file. Output will generate files named with
                                              // current frame in different subdirectories of dir
                     int frame,               // current frame/timestep. Used to generate unique filenames
                     OutputLevel level,
                     H_evolution_csph h_evolution,
                     thrust::host_vector<Real4> pos,
                     thrust::host_vector<Real3> vel,
                     thrust::host_vector<Real5> deriv, 
                     thrust::host_vector<Real>  derivRad,
                     thrust::host_vector<Real4> rhoPresEn,
                     thrust::host_vector<Real>  sound,
                     const thrust::host_vector<int4>& referenceArray) {

    bool haveHelper = (referenceArray[0].z == -3) ? true : false;  // referenceArray constructed by reduction and sorted by keys, so
                                                                   // particle type starts from -3,-2,-1,.. (if present)
    bool haveGhost = (referenceArray[0].z == -2 || referenceArray[1].z == -2) ? true : false;

    // recall referenceArray is int4 vector of {start index, end index, particle type, phase type}
    // Save helper and ghost particles to a separate file in the /others subdirectory of dir
    if (haveHelper || haveGhost) {
        std::string filename = dir + "/others" + std::to_string(frame) + ".csv";
        SaveFileCFD_csph(filename, level, h_evolution,   //
                         pos, vel, deriv, derivRad, rhoPresEn, sound,  //
                         referenceArray[0].x,
                         referenceArray[haveHelper + haveGhost].y);  // Start and end indices obtained by reference array (stores number of particles of each
                                                                     // type) because we save particles by index order, not by hash
    }

    // Save fluid/granular SPH particles to files in the /fluid subdirectory of dir
    {
        std::string filename = dir + "/fluid" + std::to_string(frame) + ".csv";
        SaveFileCFD_csph(filename, level, h_evolution, //
                         pos, vel, deriv, derivRad, rhoPresEn, sound, 
                         referenceArray[haveHelper + haveGhost].x,
                         referenceArray[haveHelper + haveGhost].y);  // offset to get correct fluid indices
        
    }

    // Save boundary BCE particles to files in /boundary subdirectory
    if (frame == 0) {
        std::string filename = dir + "/boundary" + std::to_string(frame) + ".csv";
        SaveFileCFD_csph(filename, level, h_evolution,                                                                 //
                         pos, vel, deriv, derivRad, rhoPresEn, sound,                                              //
                         referenceArray[haveHelper + haveGhost + 1].x, referenceArray[haveHelper + haveGhost + 1].y);  //
    }

    // Save rigid BCE particles to files in /rigidBCE subdirectory
    int refSize = (int)referenceArray.size();
    if (refSize > haveHelper + haveGhost + 2 && referenceArray[2].z == 1) {
        std::string filename = dir + "/rigidBCE" + std::to_string(frame) + ".csv";
        SaveFileCFD_csph(filename, level, h_evolution,                                                  //
                         pos, vel, deriv, derivRad, rhoPresEn, sound,                               //
                         referenceArray[haveHelper + haveGhost + 2].x, referenceArray[refSize - 1].y);  //
    }

}



// Function declared in the header, this version takes as input just the FsiDataManager object reference
// Notice we provide the variable vectors that are in index order.
// This function is actually just a wrapper: the real function is the saveParticleDataCFD with explicit input variables.
void saveParticleDataCFD_csph(const std::string& dir, OutputLevel level, FsiDataManager_csph& data_mgr) {

    saveParticleDataCFD_csph(dir, level, data_mgr.paramsH->h_evolution,                                                       //
                             data_mgr.sphMarkers_D->posRadD, data_mgr.sphMarkers_D->velD,            //
                             data_mgr.derivVelRhoEnOriginalD, data_mgr.derivRadOriginalD,            //
                             data_mgr.sphMarkers_D->rhoPresEnD, data_mgr.sphMarkers_D->soundD,   //
                             data_mgr.referenceArray);             
}


// calls in the SaveAllCFD and so saveFileCFD functions in cascade
// Takes device vectors as input, need to copy into host vectors to be passed to saveAllCFD
void saveParticleDataCFD_csph(const std::string& dir,
                              OutputLevel level,
                              H_evolution_csph h_evolution,
                              const thrust::device_vector<Real4>& posRadD,
                              const thrust::device_vector<Real3>& velD,
                              const thrust::device_vector<Real5>& derivVelRhoEnD,
                              const thrust::device_vector<Real>&  derivRadD,
                              const thrust::device_vector<Real4>& rhoPresEnD,
                              const thrust::device_vector<Real>&  soundD,
                              const thrust::host_vector<int4>& referenceArray) {

    // converting device vectors to host
    thrust::host_vector<Real4> pos = posRadD;  
    thrust::host_vector<Real3> vel = velD;
    thrust::host_vector<Real5> acc = derivVelRhoEnD;
    thrust::host_vector<Real> dh = derivRadD;
    thrust::host_vector<Real4> rhoPresEn = rhoPresEnD;
    thrust::host_vector<Real>  sound = soundD;

    // Current frame number
    static int frame = -1;  // Static variable, so initialized only once and persisting in memory
    frame++;

    // Start printing in a separate thread and detach the thread to allow independent execution
    std::thread th(SaveAllCFD_csph,                          //
                   dir, frame, level, h_evolution,   //
                   pos, vel, acc, dh, rhoPresEn, sound,  //
                   referenceArray    //
    );
    th.detach();
}




// -----------------------------------------------------------------------------
// FSI solids data
// -----------------------------------------------------------------------------
//
// Worker function to write current data for all FSI solids. Each body has its own file.
// Contrary to fluid files, where using one file for each timestep and saving there all particles' properties,
// here we have one file per each body and we save there all properties at each different time step.
// Host vectors as input
void SaveAllSolid_csph(const std::string& dir,
                       const std::string& delim,
                       double time,
                       thrust::host_vector<Real3> posRigid,
                       thrust::host_vector<Real4> rotRigid,
                       thrust::host_vector<Real3> velRigid,
                       thrust::host_vector<Real3> forceRigid,
                       thrust::host_vector<Real3> torqueRigid) {

    // Number of rigids, 1D nodes, and 2D nodes
    size_t numRigids = posRigid.size();


    // Write information for each FSI rigid body
    for (size_t i = 0; i < numRigids; i++) {
        Real3 pos = posRigid[i];
        Real4 rot = rotRigid[i];
        Real3 vel = velRigid[i];
        Real3 force = forceRigid[i];
        Real3 torque = torqueRigid[i];

        std::string filename = dir + "/FSI_body" + std::to_string(i) + ".csv";  // One file for each rigid body
        std::ofstream file(filename, std::fstream::app);                        // notice opened in append mode
        file << time << delim << pos.x << delim << pos.y << delim << pos.z << delim << rot.x << delim << rot.y << delim
             << rot.z << delim << rot.w << delim << vel.x << delim << vel.y << delim << vel.z << delim << force.x
             << delim << force.y << delim << force.z << delim << torque.x << delim << torque.y << delim << torque.z
             << std::endl;  // save properties corresponding to the input time
        file.close();
    }

}


// function that saves solid states taking as input the time and FsiDataManager& reference.
// Again just a wrapper for the same function that receives explicitely all parameters
void saveSolidData_csph(const std::string& dir, double time, FsiDataManager_csph& data_mgr) {
    saveSolidData_csph(dir, time,                                                       //
                       data_mgr.fsiBodyState_D->pos, data_mgr.fsiBodyState_D->rot,      //
                       data_mgr.fsiBodyState_D->lin_vel,                                //
                       data_mgr.rigid_FSI_ForcesD, data_mgr.rigid_FSI_TorquesD);                                    //
}

// function to call, under the hood uses saveAllSolids() function.
// Device vectors as input, to be converted into host vectors for the saveAllSolid() input
void saveSolidData_csph(const std::string& dir,
                        double time,
                        const thrust::device_vector<Real3>& posRigidD,
                        const thrust::device_vector<Real4>& rotRigidD,
                        const thrust::device_vector<Real3>& velRigidD,
                        const thrust::device_vector<Real3>& forceRigidD,
                        const thrust::device_vector<Real3>& torqueRigidD) {
    const std::string delim = ",";

    // Copy data arrays to host
    thrust::host_vector<Real3> posRigidH = posRigidD;
    thrust::host_vector<Real3> velRigidH = velRigidD;
    thrust::host_vector<Real4> rotRigidH = rotRigidD;
    thrust::host_vector<Real3> forceRigidH = forceRigidD;
    thrust::host_vector<Real3> torqueRigidH = torqueRigidD;


    // Number of rigids, 1D nodes, and 2D nodes
    size_t numRigids = posRigidH.size();

    // Create files if needed
    static bool create_files = true;  // the first time this function is called it will create the files and write the
                                      // variables names. When called next, it will just open it and append the values.
    if (create_files) {
        for (size_t i = 0; i < numRigids; i++) {
            std::string filename = dir + "/FSI_body" + std::to_string(i) + ".csv";
            std::ofstream file(filename);
            file << "Time" << delim << "x" << delim << "y" << delim << "z" << delim << "q0" << delim << "q1" << delim
                 << "q2" << delim << "q3" << delim << "Vx" << delim << "Vy" << delim << "Vz" << delim << "Fx" << delim
                 << "Fy" << delim << "Fz" << delim << "Tx" << delim << "Ty" << delim << "Tz" << std::endl;
            file.close();
        }
    }
    create_files = false;

    // Start printing in a separate thread and detach the thread to allow independent execution
    std::thread th(SaveAllSolid_csph,                                           //
                   dir, delim, time,                                            //
                   posRigidH, rotRigidH, velRigidH, forceRigidH, torqueRigidH);                       //
    th.detach();
}


// -----------------------------------------------------------------
// Save minimum courant and force time steps along with actual variable time step used
// at each integration point
// -----------------------------------------------------------------

void saveVariableTimeStep_csph(const std::string& dir,
                               const std::vector<double>& min_time_steps,
                               const std::vector<double>& min_courant_steps,
                               const std::vector<double>& min_force_steps) {
    std::string filename = dir + "/timesteps.csv";
    std::ofstream file(filename);
    std::stringstream sstream;
    // name of values saved
    sstream << "min_dt, min_courant_dt, min_force_dt\n";
    // print values
    for (size_t i = 0; i < min_time_steps.size(); i++) {
        sstream << min_time_steps[i] << ", " << min_courant_steps[i] << ", " << min_force_steps[i] << std::endl;
    }
    // save on file as string
    file << sstream.str();
    file.close();
}


// -----------------------------------------------------------------------------
// write particle file and data in CSV format
// Actually also previous functions do save files in csv format.
//------------------------------------------------------------------------------

// again this reduced input size function is a wrapper of the extended input one.
void writeParticleFileCSV_csph(const std::string& filename, FsiDataManager_csph& data_mgr) {
    writeParticleFileCSV_csph(filename, data_mgr.sphMarkers_D->posRadD, data_mgr.sphMarkers_D->velD,
                              data_mgr.sphMarkers_D->rhoPresEnD, data_mgr.sphMarkers_D->soundD,
                              data_mgr.referenceArray);
}


// extended input function that contains actual implementation
// Does not call other functions
// It writes only fluid particles data, not the derivatives.
// Previous functions work in same way: separate values by commas and save as csv format.
void writeParticleFileCSV_csph(const std::string& outfilename,
                               thrust::device_vector<Real4>& posRadD,
                               thrust::device_vector<Real3>& velD,
                               thrust::device_vector<Real4>& rhoPresEnD,
                               thrust::device_vector<Real>&  soundD,
                               thrust::host_vector<int4>& referenceArray) {
    thrust::host_vector<Real4> posRadH = posRadD;
    thrust::host_vector<Real3> velH = velD;
    thrust::host_vector<Real4> rhoPresEnH = rhoPresEnD;
    thrust::host_vector<Real>  soundH = soundD;
    double eps = 1e-20;

    bool haveHelper = (referenceArray[0].z == -3) ? true : false;
    bool haveGhost = (referenceArray[0].z == -2 || referenceArray[1].z == -2) ? true : false;

    std::ofstream ofile;
    ofile.open(outfilename);
    std::stringstream ss;
    ss << "x,y,z,v_x,v_y,v_z,|U|,rho,pressure,energy,c_s\n";
    // loop over all fluid particles
    for (size_t i = referenceArray[haveHelper + haveGhost].x; i < referenceArray[haveHelper + haveGhost].y; i++) {
        Real4 rP = rhoPresEnH[i];
        if (rP.w != -1)  // ensures we work with fluid particles only
            continue;
        Real4 pos = posRadH[i];
        Real3 vel = velH[i] + mR3(Real(1e-20));  // why?
        Real velMag = length(vel);
        Real c_s = soundH[i];

        ss << pos.x << ", " << pos.y << ", " << pos.z << ", " << vel.x + eps << ", " << vel.y + eps << ", "
           << vel.z + eps << ", " << velMag + eps << ", " << rP.x << ", " << rP.y + eps << ", " 
            << rP.z << ", " << c_s << std::endl;
    }

    ofile << ss.str();  // writes values and saves file
    ofile.close();
}



// -----------------------------------------------------------------------------

void writeParticleFileJSON_csph(const std::string& filename, FsiDataManager_csph& data_mgr) {
     writeParticleFileJSON_csph(filename, data_mgr.sphMarkers_D->posRadD, data_mgr.referenceArray);
}

void writeParticleFileJSON_csph(const std::string& outfilename,
                                thrust::device_vector<Real4>& posRadD,
                                thrust::host_vector<int4>& referenceArray) {

    thrust::host_vector<Real4> posRadH = posRadD;

    bool haveHelper = (referenceArray[0].z == -3) ? true : false;
    bool haveGhost = (referenceArray[0].z == -2 || referenceArray[1].z == -2) ? true : false;

    std::ofstream ofile;
    ofile.open(outfilename);
    std::stringstream ss;

    ss << "[\n";
    size_t i;
    for (i = referenceArray[haveHelper + haveGhost].x; i < referenceArray[haveHelper + haveGhost].y - 1; i++) {
        Real4 pos = posRadH[i];
        ss << "[" << pos.x << ", " << pos.y << ", " << pos.z << "],\n";
    }
    {
        i = referenceArray[haveHelper + haveGhost].y - 1;
        Real4 pos = posRadH[i];
        ss << "[" << pos.x << ", " << pos.y << ", " << pos.z << "]\n";
    }
    ss << "]\n";

    ofile << ss.str();
    ofile.close();
}

}  // end namespace compressible
}  // end namespace chrono::fsi::sph
