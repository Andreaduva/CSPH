// Author: Andrea D'Uva - 2025
//
//---------------------------------------------------------------------------------
// Extending the content of ChFsiDefinitionsSPH.h for compressible Euler equations
//---------------------------------------------------------------------------------

#ifndef CH_FSI_DEFINITIONS_SPH_COMPRESSIBLE_H
#define CH_FSI_DEFINITIONS_SPH_COMPRESSIBLE_H

namespace chrono::fsi::sph {
namespace compressible {

// Compressible sph problem type
enum class PhysicsProblem_csph {
    EULER  // compressible inviscid Euler equations
};

// How to evolve density in the scheme
enum class Rho_evolution_csph {
    SUMMATION,    // Density is updated based on the summation equation at each time step
    DIFFERENTIAL  // Density evolved using the mass conservation equation
};

// Integration scheme. Similar to chrono::fsi::sph one but removed ISPH
// All explicit integratioon schemes use a formulation in which the density is
// integrated and an equation of state is used to calculate the corresponding pressure.
enum class IntegrationScheme_csph {
    EULER,       //< Explicit Euler
    RK2,         //< Runge-Kutta 2
    VERLET,      //< Velocity Verlet - not actually implemented in original Chrono::fsi::sph
    SYMPLECTIC,  //< Symplectic Euler
};

// SPH kernel type.
enum class KernelType_csph { CUBIC_SPLINE, QUINTIC_SPLINE, WENDLAND };


// Equation of state type for compressible gasdynamics assuming gas ideal and perfect (e = Cv*T)
enum class EosType_csph {
    IDEAL_RHOEN  // P = (gamma-1)*rho*u (gamma = Cp/Cv, R_spec = Cp - Cv)
};

// Viscosity methods
enum class ViscosityMethod_csph {
    NONE,
    ARTIFICIAL_MONAGHAN  // Monaghan formulation for artificial viscosity
};

enum class HeatingMethod_csph {
    NONE,
    ARTIFICIAL           // Artificial heating in energy equation
};

// How to treat smoothing length h
enum class H_evolution_csph {
    CONSTANT,     // constant smoothing lenght h, equal for all particles
    ADKE,         // Adaptive Density Kernel Estimation
    DIFFERENTIAL  // smoothing length of each particle update through differential equation linked to density
};

// How to treat solid boundary markers. Either original suggestion from Adami for compressible flows or a 
// modified version (to be implemented).
enum class BoundaryMethod_csph {
    ORIGINAL_ADAMI,                 // bce velocity kept as prescribed solid body one. 
    MODIFIED_ADAMI
};
//-------------------------------------------------------------
// New data type Real5 to compute all derivatives in one kernel
//-------------------------------------------------------------



}  // end namespace compressible
}  // end namespace chrono::fsi::sph



#endif