// Author: Andrea D'Uva - 2025
//
//--------------------------------------------------------------------
// Extend the class ChFsiParamsSPH to deal with compressible problems
//--------------------------------------------------------------------

#ifndef CH_FSI_PARAMS_COMPRESSIBLE_H
#define CH_FSI_PARAMS_COMPRESSIBLE_H

#include <ctime>

#include "chrono_fsi/sph/ChFsiDefinitionsSPH.h"
#include "chrono_fsi/sph/ChFsiDataTypesSPH.h"
#include "chrono_fsi/sph/ChFsiParamsSPH.h"

#include "chrono_fsi/sph_compressible/ChFsiDefinitionsSPH_compressible.h"



namespace chrono::fsi::sph {
namespace compressible {

    // struct with simulation parameters. Rework of original one. 
struct ChFsiParamsSPH_csph {

    PhysicsProblem_csph physics_problem;         // Actual physical problem
    Rho_evolution_csph rho_evolution;            // How density is evolved in time
    H_evolution_csph h_evolution;                //< smoothing length evolution treatment (constant or density based)
    IntegrationScheme_csph integration_scheme;   //< Integration scheme
    EosType_csph eos_type;                       //< Equation of state type (Ideal gas law, either PV, RhoT, RhoEn)
    ViscosityMethod_csph viscosity_method;       //< Viscosity treatment type (NONE or artificial Monaghan)
    HeatingMethod_csph heating_method;           //< Artificial heating term in energy equation (NONE or ARTIFICIAL)
    BoundaryMethod_csph boundary_method;         //< Boundary type, for now implemented only ORIGINAL_ADAMI
    KernelType_csph kernel_type;                 //< Kernel type (Quadratic, cubic spline, quintinc spline, quintic Wendland)
    sph::ShiftingMethod shifting_method;         //< Shifting method (NONE, PPST, XSPH, PPST_XSPH) - only XSPH planned for support
                                                 //< in compressible sph
   
    int3 gridSize;           //< Probably it's the number of cells in each direction of the grid. E.g. 20 cells along x, 10 along y, 30 along z.
    Real3 worldOrigin;       //< Origin point - set in FLuidSystem as = cMin
    Real3 cellSize;          //< Cell size for the neighbor particle search (domain decomposed into cells of dimension cellSize). Real3 but actually same size along each direction
    uint  numBodies;         //< Number of FSI bodies.
    Real3 boxDims;           //< Dimensions (AABB) of the domain. Always set as cMax - cMin.
    Real binSize0;   //< Suggests the length of the bin each particle occupies. Normally this would be 2*hsml since
                     //< hsml is the radius of the particle, but when we have periodic boundary condition varies a
                     //< little from 2 hsml.This may change slightly due to the location of the periodic BC.
    // binSize0 and cellSize are numerically redundant since the implementation is cellSize = (binSize0, binSize0, binSize0)
    // Still saved to be directly used in different context without need to always perform conversion.
    // ACtually binSize0 is only used to define cellSize, never used/considered in other places.                 
                     
    // These parameters are never actually used at any point in original chrono::sph code
    // Real3 zombieBoxDims;     //< Dimensions (AABB) of the zombie domain
    // Real3 zombieOrigin;      //< Origin point of the zombie domain


    Real  d0;                //< Initial separation of SPH particles
    Real  ood0;              //< Initial 1 / d0
    Real  d0_multiplier;     //< Initial Multiplier to obtain the interaction length, h = d0_multiplier * d0
    Real  h;                 //< Initial Kernel interaction length
    Real  ooh;               //< Initial 1 / h
    Real  h_multiplier;      //< Multiplier to obtain kernel radius, r = h_multiplier * h (depends on kernel type)
    int   num_neighbors;     //< Number of neighbor particles
    Real  epsMinMarkersDis;  //< Multiplier for minimum distance between markers (d_min = eps * h). Used also in artificial viscosity at denominator.
    int   num_bce_layers;    //< Number of BCE marker layers attached to boundary and solid surfaces (default: 3)
    Real  toleranceZone;     //< Helps determine the particles that are in the domain but are outside the boundaries, so
                             //< they are not considered fluid particles and are dropped at the beginning of the simulation.
    
    
    Real3 delta_pressure;    //< Change in Pressure for periodic BC (when particle moves from one side to the other)

    bool h_variation;             //< if using a formulation with h varying in space and time, need to use symmetric kernels in
                                  //computations.
    bool midpoint_neigh_search;   //< if need to recompute the neighbor search parameters during time integration midpoint step

    Real3 V_in;   //< Inlet velocity for inlet BC
    Real x_in;    //< Inlet position for inlet BC

    Real3 gravity;      //< Gravitational acceleration
    Real3 bodyForce3;   //< Constant force applied to the fluid particles (solids not directly affected)

    Real rho0;      //< Initial Density
    Real invrho0;   //< 1 / rho0
    Real volume0;   //< Initial particle volume, particle volume in sph computed as mass/density. However initial
                    //< particle volume computed as (init_spacing)^3
    // new:
    Real p0;           //< Initial (homogeneous) pressure
    Real e0;           //< Initial (homogeneous) specific thermal energy
    Real Cs0;          //< Initial (homogeneous) sound speed
    Real gamma;        // Ratio of heat capacities Cp/Cv

    bool is_uniform;   //< flag to set if the problem is (initially) uniform
    Real markerMass;   //< marker mass

    
    Real dT;   //< Time step. Depending on the model this will vary and the only way to determine what time step to
               //< use is to run simulations multiple time and find which one is the largest dT that produces a
               //< stable simulation.

    bool use_variable_time_step;  //< Set to true if use variable time step for integration based on the CFL condition
    Real C_cfl;                   //< Integrator specific constants used in deriving the time step from the CFL condition.
    Real C_force;

    int num_dim;  //< number of dimensions of the fluid problem. Vectors are still 3D but uses different  
               // coefficients for kernel normalization


    bool use_default_limits;  //< true if cMin and cMax are not user-provided (default: true)
    bool use_init_pressure;   //< true if (initial) pressure set based on height (default: false)




    double pressure_height;  //< height for pressure initialization. Used if use_init_pressure = true in the equation
                             //< pressure0 = rho0*g_z*(z-pressure_height)

    bool density_reinit_switch;    //< if true, allows reinitialization of density after density_reinit_steps steps.
    int density_reinit_steps;  //< Reinitialize density after density_reinit_steps steps. Note that doing this more frequently helps
                        // in getting more accurate incompressible fluid, but more stable solution is obtained for
                        // larger density_reinit_steps

    // again following ones need REVISION
    bool Apply_BC_U;        //< This option lets you apply a velocity BC on the BCE markers
    Real L_Characteristic;  //< Characteristic for Re number computation

    Real Ar_vis_alpha;       //< Artificial viscosity alpha coefficient
    Real Ar_vis_beta;        //< Artificial viscosity beta coefficient
    Real Ar_heat_g1;         //< Artificial heating coefficients
    Real Ar_heat_g2;
    bool Ar_heat_switch;         //< true if artificial heating on and g1,g2 != 0

    Real shifting_xsph_eps;  //< Coefficient for XSPH shifting

    Real ADKE_k;            //< k parameter in ADKE calculation of lambdas
    Real ADKE_eps;          //< epsilon parameter in ADKE calculation
    Real ADKE_D;            //< term that relates h0 with reference delta_x0.

    // Real C_Wi;    //< Threshold of the integration of the kernel function. Works as treshold when computing the 
                     //  sum of W_ij. If sum_Wij < C_Wi means particle is close to free surface. In Chrono used only in CRM 
                     //  but planned to be used also in CFD. For now left unused.

    // In FluidSystemSPH added as the dimension of the fluid container.
    // Must be specified as input parameter.
    Real boxDimX;   //< Dimension of the space domain - X
    Real boxDimY;   //< Dimension of the space domain - Y
    Real boxDimZ;   //< Dimension of the space domain - Z

    BoundaryConditions bc_type;  //< boundary condition types in the 3 domain directions
    bool x_periodic;             //< periodic boundary conditions in x direction?
    bool y_periodic;             //< periodic boundary conditions in y direction?
    bool z_periodic;             //< periodic boundary conditions in z direction?

    int3 minBounds;   //< Lower limit point of the grid (in grid index)
    int3 maxBounds;   //< Upper limit point of the grid (in grid index)

    // Represent the computational domain in FluidSystemSPH.
    // Must be given as input, otherwise use the default rule for which 
    Real3 cMin;   //< Lower limit point (in world coordinates)
    Real3 cMax;   //< Upper limit point (in world coordinates)

    // Following variables are not used at all as far as I know.
    // Real3 zombieMin;   //< Lower limit point of the zombie domain -> All particles outside this will be frozen
    // Real3 zombieMax;   //< Upper limit point of the zombie domain -> All particles outside this will be frozen

    // in theory described as used in CRM to avoid simulating particles too far away, but in pratice used in FluidDynamics to update the activity lists.
    Real3 bodyActiveDomain;   //< Size of the active domain that influenced by an FSI body
    bool use_active_domain;   //< Set to true if active domain is used. CHECK IF I CAN DELETE IT FOR FLUID PROBLEMS

    int num_proximity_search_steps;   //< Number of steps between updates to neighbor lists
}; // struct

} // end namespace compressible 
} // end namespace chrono::fsi::sph


#endif






