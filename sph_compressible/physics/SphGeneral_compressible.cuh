//-----------------------------------------------------------------------------
// Author: Andrea D'Uva - 2025
//
//
// Extension of base file SphGeneral.cuh to account for compressibility related
// utility functions
//------------------------------------------------------------------------------

#ifndef CH_SPH_GENERAL_COMPRESSIBLE_CUH
#define CH_SPH_GENERAL_COMPRESSIBLE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "chrono_fsi/sph/physics/FsiDataManager.cuh"

#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"

namespace chrono::fsi::sph {
namespace compressible {

//--------------------------------------------------------------------------------------------------------------------------------

// Declared as const variables static in order to be able to use them in a different translation units in the utils
// These are declared here, along with the function to set them equal to the parameters of the problem, which are
// stored on the host.
// Then actually set and used by other functions in other files, for instance in calcHashD of CollisionSystem_compressible.cu
// THe __constant__ declaration makes them Device variables, defined in the Constant memory space. 
__constant__ static ChFsiParamsSPH_csph paramsD_csph;
__constant__ static Counters_csph countersD_csph;

void CopyParametersToDevice_csph(std::shared_ptr<ChFsiParamsSPH_csph> paramsH, std::shared_ptr<Counters_csph> countersH);

//--------------------------------------------------------------------------------------------------------------------------------

#ifndef INVPI
#define INVPI Real(0.31830988618379)
#endif
#ifndef EPSILON
#define EPSILON Real(1e-8)
#endif


//--------------------------------------------------------------------------------------------------------------------------------
// Cubic Spline SPH kernel function
// d > 0 is the distance between 2 particles. h is the SPH kernel length

inline __host__ __device__ Real W3h_CubicSpline_csph(Real d, Real invh, int dim) {
    
    Real q = fabs(d) * invh;  // R = r_vec / h. also abs of r_vec since kernel is symmetric.
    Real alpha = 3 * INVPI * cube(invh) / 2;
    
    if (dim == 1)
        alpha = invh;
    else if (dim == 2)
        alpha = 15 * INVPI * invh * invh / 7;

    if (q < 1) {
        return alpha * ( 2/Real(3) - q*q + 0.5*cube(q));
    }
    if (q < 2) {
        return alpha / 6 * cube(2-q);
    }
    return 0;
}

inline __host__ __device__ Real3 GradW3h_CubicSpline_csph(Real3 d, Real invh, int dim) {
   
    Real beta = 3 * INVPI * quartic(invh) / 2;
    
    if (dim == 1) {
        beta = invh * invh;
        d.y = 0;
        d.z = 0;
    } else if (dim == 2) {
        beta = 15 * INVPI * cube(invh) / 7;
        d.z = 0;
    }
    
    Real r_len = length(d);
    Real q = r_len * invh;
    if (abs(q) < EPSILON)
        return mR3(0);

    if (q < 1) {
        return (beta * (3/Real(2) * q*q - 2*q)) * d/r_len;  // It's a Real3 because d is Real3
    }
    if (q < 2) {
        return ( -0.5 * beta * square(2 - q)) * d/r_len;
    }
    return mR3(0);
}


//--------------------------------------------------------------------------------------------------------------------------------
// Quintic Spline SPH kernel function
// d > 0 is the distance between 2 particles. h is the SPH kernel length

inline __host__ __device__ Real W3h_QuinticSpline_csph(Real d, Real invh, int dim) {
    
    Real alpha = 3*INVPI * cube(invh) / 359;
    
    if (dim == 1)
        alpha = invh / 120;
    else if (dim == 2)
        alpha = 7 * INVPI * square(invh) / 478;

    Real q = fabs(d) * invh;
    
    if (q < 1) {
        return alpha * (quintic(3 - q) - 6 * quintic(2 - q) + 15 * quintic(1 - q));
    }
    if (q < 2) {
        return alpha * (quintic(3 - q) - 6 * quintic(2 - q));
    }
    if (q < 3) {
        return alpha * (quintic(3 - q));
    }
    return 0;
}

inline __host__ __device__ Real3 GradW3h_QuinticSpline_csph(Real3 d, Real invh, int dim) {
    
    // beta = -5 * alpha / h^2
    Real beta = -5 * INVPI * 3 * quartic(invh) / 359;
    if (dim == 1) {
        beta = -5 * square(invh) /120;
        d.y = 0;
        d.z = 0;
    } else if (dim == 2) {
        beta = -35 * INVPI * cube(invh) / 478;
        d.z = 0;
    }

    Real r_len = length(d);
    Real q = r_len * invh;
    if (fabs(q) < 1e-10)
        return mR3(0);

    if (q < 1) {
        return (beta * (quartic(3 - q) - 6 * quartic(2 - q) + 15 * quartic(1 - q))) * d / r_len;
    }
    if (q < 2) {
        return (beta * (quartic(3 - q) - 6 * quartic(2 - q))) * d / r_len;
    }
    if (q < 3) {
        return (beta * (quartic(3 - q))) * d / r_len;
    }
    return mR3(0);
}

//--------------------------------------------------------------------------------------------------------------------------------
// Wendland Quintic SPH kernel function
// d > 0 is the distance between 2 particles. h is the SPH kernel length

inline __host__ __device__ Real W3h_Wendland_csph(Real d, Real invh, int dim) {
    
    Real alpha = 21 * INVPI * cube(invh) / 256;
    if (dim == 1)
        alpha = 5 * invh / 128;
    else if (dim == 2)
        alpha = 7 * INVPI * square(invh) / 56;
    
    Real q = fabs(d) * invh;

    if (q < 2) {;
        return alpha * quartic(2 - q) * (2 * q + 1);
    }
    return 0;
}

inline __host__ __device__ Real3 GradW3h_Wendland_csph(Real3 d, Real invh, int dim) {
    
    Real beta = - 5 * 21 * INVPI * quartic(invh) / 256;
    
    if (dim == 1) {
        beta = -5 * 5 * square(invh) / 128;
        d.y = 0;
        d.z = 0;
    } else if (dim == 2) {
        beta = -5 * 7 * INVPI * cube(invh) / 56;
        d.z = 0;
    }
    
    Real r_len = length(d);
    Real q = r_len* invh;
    if (fabs(q) < 1e-10)
        return mR3(0);

    if (q < 2) {
        return (beta * q * cube(1 - 0.5 * q)) * d / r_len;
    }
    return mR3(0);
}

//--------------------------------------------------------------------------------------------------------------------------------
// KernelType enum class defined in ChFsiDefinitions.h and included in class ChFsiParamsSPH.h
// Returns a kernel value given position, 1/h and KernelType
inline __host__ __device__ Real W3h_csph(KernelType_csph type, Real d, Real invh, int dim) {
    switch (type) {
        case KernelType_csph::CUBIC_SPLINE:
            return W3h_CubicSpline_csph(d, invh, dim);
        case KernelType_csph::QUINTIC_SPLINE:
            return W3h_QuinticSpline_csph(d, invh, dim);
        case KernelType_csph::WENDLAND:
            return W3h_Wendland_csph(d, invh, dim);
    }

    return -1;
}
// returns the kernel gradient at given pos, 1/h, kerneltype
inline __host__ __device__ Real3 GradW3h_csph(KernelType_csph type, Real3 d, Real invh, int dim) {
    switch (type) {
        case KernelType_csph::CUBIC_SPLINE:
            return GradW3h_CubicSpline_csph(d, invh, dim);
        case KernelType_csph::QUINTIC_SPLINE:
            return GradW3h_QuinticSpline_csph(d, invh, dim);
        case KernelType_csph::WENDLAND:
            return GradW3h_Wendland_csph(d, invh, dim);
    }

    return mR3(-1, -1, -1);
}

//--------------------------------------------------------------------------------------------------------------------------------

// Equations of state for compressible ideal gases
// Input variables in1, in2 represent different quantities depending on the EosType chosen
// This is the Device side function, callable only from __global__ or __device__ Cuda functions
// Doesn't deduce the EosType from paramsD to allow for more flexibility.
inline  __device__ Real Eos_csph(Real in1, Real in2, const EosType_csph eos_type) {
    switch (eos_type) {
        case EosType_csph::IDEAL_RHOEN: {
            // P = (gamma-1)*rho*u 
            // in1 = rho, in2 = u, out = P
            return in1 * in2 * (paramsD_csph.gamma - 1);
        }
    }
    return -1;
}

// This is the Host side function, notice it asks as input the parameters struct
inline __host__ Real Eos_csph(Real in1, Real in2, const ChFsiParamsSPH_csph& paramsH) {
    switch (paramsH.eos_type) {
        case EosType_csph::IDEAL_RHOEN: {
            // P = (gamma-1)*rho*u
            // in1 = rho, in2 = u, out = P
            return in1 * in2 * (paramsH.gamma - 1);
        }
    }
    return -1;
}


// More concise expressions for the equation of state:

/* Device side:
inline __device__ Real EosPV_csph(Real V, Real T, Real mass, Real R_spec) {
    return mass * R_spec * (T / V);  // P = n*R*(T/V) = m*R_spec*T/V
}

inline __device__ Real EosRhoT_csph(Real rho, Real T, Real R_spec) {
    return rho * T * R_spec;  // P = rho*R_spec*T
}

inline __device__ Real EosRhoEn_csph(Real rho, Real en, Real gamma) {
    return rho * en * (gamma - 1);  // P = rho*u*(gamma-1)
}
*/

// Device side with less output:
inline __device__ Real EosRhoEn_csph(Real rho, Real en) {
    return rho * en * (paramsD_csph.gamma - 1);  // P = rho*u*(gamma-1)
}


// Host side functions
inline __host__ __device__ Real EosPV_csph(Real V, Real T, Real mass, Real R_spec) {
    return mass * R_spec * (T / V);  // P = n*R*(T/V) = m*R_spec*T/V
}

inline __host__ __device__ Real EosRhoT_csph(Real rho, Real T, Real R_spec) {
    return rho * T * R_spec;   // P = rho*R_spec*T
}

inline __host__ __device__ Real EosRhoEn_csph(Real rho, Real en, Real gamma) {
    return rho * en * (gamma - 1);   // P = rho*u*(gamma-1)
}




//------------------------------------------------------------------------------------------------
// Inverse eqution of state: used in Adami style BCs. Compute density given other state variables.
//------------------------------------------------------------------------------------------------

// device function
inline __device__ Real InvEos_csph(Real in1, Real in2, const EosType_csph eos_type) {
    switch (eos_type) {
        case EosType_csph::IDEAL_RHOEN: {
            // P = (gamma - 1)*rho*u  -> rho = P/( u * (gamma - 1))
            // in1 = P, in2 = u, out = rho
            return in1 / (in2 * (paramsD_csph.gamma - 1));
        }
    }
    return -1;
}



// host side function
inline __host__ Real InvEos_csph(Real in1, Real in2, const ChFsiParamsSPH_csph& paramsH) {
    switch (paramsH.eos_type) {
        case EosType_csph::IDEAL_RHOEN: {
            // P = (gamma - 1)*rho*u  -> rho = P/( u * (gamma - 1))
            // in1 = P, in2 = u, out = rho
            return in1 / (in2 * (paramsH.gamma - 1));
        }
    }
    return -1;
}



//------------------------------------------------------------------------------
// Equations to compute temperature and the speed of sound of compressible gases
//------------------------------------------------------------------------------

// Device functions:

inline __device__ Real SoundFromPresRho(Real pres, Real rho) {
    return sqrt(paramsD_csph.gamma * (pres / rho));
}

inline __device__ Real SoundFromEn(Real en) {
    return sqrt(paramsD_csph.gamma * (paramsD_csph.gamma - 1) * en);
}


// Host functions: notice they ask the parameters struct:

inline __host__ Real SoundFromPresRho(Real pres, Real rho, const ChFsiParamsSPH_csph& paramsH) {
    return sqrt(paramsH.gamma * (pres / rho));
}

inline __host__ Real SoundFromEn(Real en, const ChFsiParamsSPH_csph& paramsH) {
    return sqrt(paramsH.gamma * (paramsH.gamma - 1) * en);
}



//--------------------------------------------------------------------------------------------------------------------------------
// modify position of particle b wrt particle a for beriodic bcs and if they overlap, to retrieve a minimum distance.
__device__ inline Real3 Modify_Local_PosB_csph(Real3& b, Real3 a) {
    Real3 dist3 = a - b;
    b.x += ((dist3.x > 0.5f * paramsD_csph.boxDims.x) ? paramsD_csph.boxDims.x : 0);
    b.x -= ((dist3.x < -0.5f * paramsD_csph.boxDims.x) ? paramsD_csph.boxDims.x : 0);

    b.y += ((dist3.y > 0.5f * paramsD_csph.boxDims.y) ? paramsD_csph.boxDims.y : 0);
    b.y -= ((dist3.y < -0.5f * paramsD_csph.boxDims.y) ? paramsD_csph.boxDims.y : 0);

    b.z += ((dist3.z > 0.5f * paramsD_csph.boxDims.z) ? paramsD_csph.boxDims.z : 0);
    b.z -= ((dist3.z < -0.5f * paramsD_csph.boxDims.z) ? paramsD_csph.boxDims.z : 0);

    dist3 = a - b;
    // modifying the markers perfect overlap
    Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
    Real MinD = paramsD_csph.epsMinMarkersDis * paramsD_csph.h;  // ok to use baseline h parameter?
    Real sq_MinD = MinD * MinD;
    if (dd < sq_MinD) {
        dist3 = mR3(MinD, 0, 0);
    }
    b = a - dist3;
    return (dist3);
}

__device__ inline Real3 Distance_csph(Real3 a, Real3 b) {
    return Modify_Local_PosB_csph(b, a);
}


// Same functions but on the host side:
__host__ inline Real3 Modify_Local_PosB_csph(Real3& b, Real3 a, const ChFsiParamsSPH_csph& paramsH) {
    Real3 dist3 = a - b;
    b.x += ((dist3.x > 0.5f * paramsH.boxDims.x) ? paramsH.boxDims.x : 0);
    b.x -= ((dist3.x < -0.5f * paramsH.boxDims.x) ? paramsH.boxDims.x : 0);

    b.y += ((dist3.y > 0.5f * paramsH.boxDims.y) ? paramsH.boxDims.y : 0);
    b.y -= ((dist3.y < -0.5f * paramsH.boxDims.y) ? paramsH.boxDims.y : 0);

    b.z += ((dist3.z > 0.5f * paramsH.boxDims.z) ? paramsH.boxDims.z : 0);
    b.z -= ((dist3.z < -0.5f * paramsH.boxDims.z) ? paramsH.boxDims.z : 0);

    dist3 = a - b;
    // modifying the markers perfect overlap
    Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
    Real MinD = paramsH.epsMinMarkersDis * paramsH.h;  // ok to use baseline h parameter?
    Real sq_MinD = MinD * MinD;
    if (dd < sq_MinD) {
        dist3 = mR3(MinD, 0, 0);
    }
    b = a - dist3;
    return (dist3);
}

__host__ inline Real3 Distance_csph(Real3 a, Real3 b, const ChFsiParamsSPH_csph& paramsH) {
    return Modify_Local_PosB_csph(b, a, paramsH);
}
//--------------------------------------------------------------------------------------------------------------------------------
// Function for neighbor search. Given the position p of a particle, uses the worldOrigin and the elementary cell size,
// to express the cell indexes in which this particle falls into Returns an int3, with int3.x = cell number in x
// direction and so on.

// Used many times in CollisionSystem and in FsiForce computations
// Will need a dynamic parameter for the cellsize.
__device__ inline int3 calcGridPos_csph(Real3 p) {
    int3 gridPos;
    if (paramsD_csph.cellSize.x * paramsD_csph.cellSize.y * paramsD_csph.cellSize.z == 0)
        printf("calcGridPos=%f,%f,%f\n", paramsD_csph.cellSize.x, paramsD_csph.cellSize.y, paramsD_csph.cellSize.z);

    gridPos.x = (int)floor((p.x - paramsD_csph.worldOrigin.x) / (paramsD_csph.cellSize.x));  // translates to get position relative to world origin and divide by cell size.
    gridPos.y = (int)floor((p.y - paramsD_csph.worldOrigin.y) / (paramsD_csph.cellSize.y));
    gridPos.z = (int)floor((p.z - paramsD_csph.worldOrigin.z) / (paramsD_csph.cellSize.z));
    return gridPos;
}
//--------------------------------------------------------------------------------------------------------------------------------
// Computes hash value of a grid cell based on its position. Maps a 3D cell into an unique uint hash value.

// used many times in collision system neighbor search
__device__ inline uint calcGridHash_csph(int3 gridPos) {

    // perform wrapping back inside domain if particle is outside and the direction is markerd as periodic.
    // performs clamping (to avoid illegal positions) if particle is outside domain and direction is not periodic.
    // a ? b : c ? d : e reads as: if (a) then (b), elseif (c) then (d), else e 
    gridPos.x = (gridPos.x >= paramsD_csph.gridSize.x && paramsD_csph.x_periodic) ? gridPos.x - paramsD_csph.gridSize.x :
                (gridPos.x >= paramsD_csph.gridSize.x && !paramsD_csph.x_periodic) ? paramsD_csph.gridSize.x - 1 : gridPos.x;

    gridPos.y = (gridPos.y >= paramsD_csph.gridSize.y && paramsD_csph.y_periodic) ? gridPos.y - paramsD_csph.gridSize.y :
                (gridPos.y >= paramsD_csph.gridSize.y && !paramsD_csph.y_periodic) ? paramsD_csph.gridSize.y - 1 : gridPos.y;

    gridPos.z = (gridPos.z >= paramsD_csph.gridSize.z && paramsD_csph.z_periodic) ? gridPos.z - paramsD_csph.gridSize.z :
                (gridPos.z >= paramsD_csph.gridSize.z && !paramsD_csph.z_periodic) ? paramsD_csph.gridSize.z - 1 : gridPos.z;

    gridPos.x = (gridPos.x < 0 && paramsD_csph.x_periodic) ? gridPos.x + paramsD_csph.gridSize.x :
                (gridPos.x < 0 && !paramsD_csph.x_periodic) ? 0 : gridPos.x;

    gridPos.y = (gridPos.y < 0 && paramsD_csph.y_periodic) ? gridPos.y + paramsD_csph.gridSize.y :
                (gridPos.y < 0 && !paramsD_csph.y_periodic) ? 0 : gridPos.y;

    gridPos.z = (gridPos.z < 0 && paramsD_csph.z_periodic) ? gridPos.z + paramsD_csph.gridSize.z :
                (gridPos.z < 0 && !paramsD_csph.z_periodic) ? 0 : gridPos.z;

    return gridPos.z * paramsD_csph.gridSize.y * paramsD_csph.gridSize.x + gridPos.y * paramsD_csph.gridSize.x + gridPos.x;  // returns a linear hash = z*(gridsize.x*gridsize.y) + y*gridsize.x + x
}


} // namespace compressible
}  // namespace fsi::sph

#endif