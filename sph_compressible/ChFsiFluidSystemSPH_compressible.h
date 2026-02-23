// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Implementation of an FSI-aware SPH fluid solver. Modified to allow compressibility
//
// =============================================================================

#ifndef CH_FLUID_SYSTEM_SPH_COMPRESSIBLE_H
#define CH_FLUID_SYSTEM_SPH_COMPRESSIBLE_H

#include "chrono_fsi/ChFsiFluidSystem.h"

#include "chrono_fsi/sph/ChFsiParamsSPH.h"
#include "chrono_fsi/sph/ChFsiDefinitionsSPH.h"
#include "chrono_fsi/sph/ChFsiDataTypesSPH.h"

#include "chrono_fsi/sph_compressible/ChFsiParamsSPH_compressible.h"
#include "chrono_fsi/sph_compressible/ChFsiDefinitionsSPH_compressible.h"

namespace chrono::fsi::sph {
namespace compressible {

class ChFsiInterfaceSPH_csph;
class FluidDynamics_csph;
class BceManager_csph;
struct FsiDataManager_csph;


// Physical system for an FSI-aware SPH fluid solver.
class CH_FSI_API ChFsiFluidSystemSPH_csph : public ChFsiFluidSystem {
  public:
    // Structure with ideal gas properties.
    struct CH_FSI_API FluidProperties_csph {
        double density;      //< density (default: 1)
        double pressure;     //< pressure (default: 0.4)
        double energy;       //< energy (default:1)
        double gamma;        //< ratio of specific heats (default 1.4)
        double char_length;  //< characteristic length (default: 1.0)

        FluidProperties_csph();  // default empty constructor
    };


    /// Structure with SPH method parameters. Used as shortening for the ChFsiParamsSPH_csph structure.
    //  In a program the users defines the following parameters, then calls the method SetSPHParameters that
    //  automatically creates and complete the whole ChFsiParamsSPH_csph object, which is the one stored in the
    //  FluidSystemSPH class
    struct CH_FSI_API SPHParameters_csph {
        Rho_evolution_csph rho_evolution;           //< How density is evolved (default: DIFFERENTIAL)
        H_evolution_csph h_evolution;               //< How h if evolved (default: CONSTANT)
        IntegrationScheme_csph integration_scheme;  //< Integration scheme (default: RK2)
        EosType_csph eos_type;                      //< equation of state (default: ISOTHERMAL)
        ViscosityMethod_csph viscosity_method;      //< viscosity treatment (default: ARTIFICIAL_UNILATERAL)
        HeatingMethod_csph heating_method;          //< artificial heating method (default: NONE)
        BoundaryMethod_csph boundary_method;        //< boundary treatment (default: ADAMI)
        KernelType_csph kernel_type;                //< kernel type (default: CUBIC_CPLINE)
        sph::ShiftingMethod shifting_method;        //< shifting method (default: XSPH)
        int num_bce_layers;                    //< number of BCE layers (boundary and solids, default: 3)
        double initial_spacing;                //< initial particle spacing (default: 0.01)
        double d0_multiplier;       //< kernel length multiplier, h = d0_multiplier * initial_spacing (default: 1.2)
        
        double shifting_xsph_eps;   //< XSPH coefficient (default: 0.5)
        
        double min_distance_coefficient;  //< min inter-particle distance as fraction of kernel radius (default: 0.01)
        bool density_reinit_switch;       //< use or not density reinitialization technique
        int density_reinit_steps;         //< number of steps between density reinitializations (default: 2e8)
        
        double artificial_viscosity_alpha;         //< artificial viscosity alpha coefficient (default: 1.)
        double artificial_viscosity_beta;          //< artificial viscosity beta coefficient (default: 2.)
        
        double artificial_heating_g1;              //< artificial heating coefficients (default: 0.)
        double artificial_heating_g2;              //< default: 0.

        double ADKE_k;                    //< parameters for ADKE update of smoothing length.
        double ADKE_eps;                  //< default: 1, 0.5, 1.5.
        double ADKE_D;

        int num_proximity_search_steps;            //< number of steps between updates to neighbor lists (default: 4)

        bool use_variable_time_step;       //< parameters for variable time step based on CFL condition. (default: false)
        double C_cfl;                      //< default: 0.5
        double C_force;                    //< default: 0.5

        int num_dim;                       //< number of dimensions for sph simulation (default: 3)
        bool is_uniform;                   //< flag to check if the problem is (initially) uniform (default:true)
        double markerMass;                 //< Left as -1 to use params->volume * params->density
        SPHParameters_csph();  // default empty constructor
    };


    ChFsiFluidSystemSPH_csph();  // default empty constructor
    ~ChFsiFluidSystemSPH_csph();

    // Read Chrono::FSI parameters from the specified JSON file.
    void ReadParametersFromFile(const std::string& json_file);

    // Enable/disable CUDA error checks (default: enabled).
    void EnableCudaErrorCheck(bool val) { m_check_errors = val; }

    // ------------------------ Set methods ----------------------------
    
    // Set method to evolve density
    void SetRhoEvolution(Rho_evolution_csph rho_evolution);

    // Set method to evolve h parameter
    void SetKernRadEvolution(H_evolution_csph h_evolution);

    // Set coefficients for ADKE scheme
    void SetADKECoeff(double k, double eps, double D = 1.5);

    // Set preferred form of equation of state
    void SetEosType(EosType_csph eos_type);

    // Set initial spacing parameter d0 and compute associated quantities.
    void SetInitialSpacing(double spacing);

    // Set parameter marker mass explicitely
    void SetMarkerMass(double mass);

    // Set multiplier for interaction length.
    // h = multiplier * initial_spacing.
    void SetKernelMultiplier(double multiplier);

    // Set the shifting method.
    void SetShiftingMethod(ShiftingMethod shifting_method);

    // Set boolean switch to update the neighbor grid at integration midpoint
    void SetMidpointGridUpdate(bool midpoint_switch);

    // Set the fluid container dimension
    void SetContainerDim(const ChVector3d& box_dim);

    // Set computational domain and boundary conditions on its sides.
    // `bc_type` indicates the types of BCs imposed in the three directions of the computational domain.
    // By default, no special boundary conditions are imposed in any direction (BCType::NONE).
    // BoundaryConditions different from BoundaryMethod. See ChFsiDefinitionSPH.h
    void SetComputationalDomain(const ChAABB& computational_AABB, BoundaryConditions bc_type);

    // Set computational domain.
    // Note that this version leaves the setting for BC type unchanged.
    void SetComputationalDomain(const ChAABB& computational_AABB);

    // Set dimensions of the active domain AABB.
    // This value activates only those SPH particles that are within an AABB of the specified size from an object
    // interacting with the "fluid" phase.
    // Note that this setting should *not* be used for CFD simulations, but rather only when solving problems using the
    // CRM (continuum representation of granular dynamics) for terramechanics simulations.
    void SetActiveDomain(const ChVector3d& box_dim);

    // Set number of BCE marker layers (default: 3).
    void SetNumBCELayers(int num_layers);

    // Set (initial) density.
    void SetDensity(double rho0);

    // Set prescribed initial height for pressure based on depth.
    void SetPressureHeight(const double fzDim);

    // Set (initial) pressure
    void SetPressure(double p0);

    // Set (initial) energy
    void SetEnergy(double e0);

    // Set gas ratio of specific heats
    void SetGamma(double gamma);

    // Set (initial) fluid properties.
    void SetFluidProperties(const FluidProperties_csph& fluid_props);

    // Set the XSPH Shifting parameters
    // eps: coefficient for the XSPH shifting method
    void SetShiftingXSPHParameters(double eps);

    // Set gravity for the FSI system.
    virtual void SetGravitationalAcceleration(const ChVector3d& gravity) override;

    // Set a constant force applied to the fluid.
    // Solid bodies are not explicitly affected by this force, but they are affected indirectly through the fluid.
    void SetBodyForce(const ChVector3d& force);

    // Set the integration scheme (default: RK2).
    void SetIntegrationScheme(IntegrationScheme_csph scheme);

    // Set flag to denote if the problem is at least initially uniform and can use only one value of mass
    void SetIsUniform(bool is_uniform);

    // Set the number of steps between successive updates to neighbor lists (default: 4).
    void SetNumProximitySearchSteps(int steps);

    // Set the number of steps before density reinitialization (default is off)
    void SetDensityReinitSteps(int steps);

    // Set use of variable time step
    void SetUseVariableTimeStep(bool use_variable_time_step);

    // Set values for constants in the CFL condition
    void SetCFLParams(const double C_cfl, const double C_force);

    // Set value of minimum marker distance parameter (eta)
    void SetMinMarkerDistance(double eps);
   
    // Check consistency of fluid parameters (defaults and user-provided)
    void CheckFluidParameters();

    // Checks the applicability of user set parameters for SPH and throws an exception if necessary.
    void CheckSPHParameters();

    // Set compressible SPH method parameters from the SPHParameters_csph struct
    void SetSPHParameters(const SPHParameters_csph& sph_params);

    // Set simulation data output level (default: STATE_PRESSURE).
    // Options:
    // - STATE           marker state, velocity, and acceleration
    // - STATE_PRESSURE  STATE plus density, pressure, temperature and energy
    // - CFD_FULL        STATE_PRESSURE plus derivatives
    void SetOutputLevel(OutputLevel output_level);

    // Set boundary treatment type (default: Original Adami).
    void SetBoundaryType(BoundaryMethod_csph boundary_method);

    // Set viscosity treatment type (default: artificial Monaghan).
    void SetViscosityMethod(ViscosityMethod_csph viscosity_method);

    // Set artificial viscosity coefficients alpha and beta (default: 1.0 and 2.0).
    void SetArtificialViscosityCoefficient(double alpha, double beta);

    // Set artificial heating treatment type (default: NONE).
    void SetHeatingMethod(HeatingMethod_csph heating_method);

    // Set artificial heating coefficients g1, g2 (default: 0.0 and 0.0)
    void SetArtificialHeatingCoefficient(double g1, double g2);

    // Set kernel type.
    void SetKernelType(KernelType_csph kernel_type);

    // set number of dimensions used for the fluid problem
    void SetNumDim(int n_dim);


    // ----------------------- Get methods for sph parameters --------------------------------------
    // Return the default SPH kernel length of kernel function.
    double GetKernelLength() const;

    // Return the initial spacing of the SPH particles.
    double GetInitialSpacing() const;

    // Return the value used for mass and stored in the paramsH class
    double GetMarkerMass() const;

    // Return the number of BCE layers.
    int GetNumBCELayers() const;

    // Get the fluid container dimensions.
    ChVector3d GetContainerDim() const;

    // Get the computational domain.
    ChAABB GetComputationalDomain() const;

    // Return initial density.
    double GetBaseDensity() const;
    
    // Return initial pressure.
    double GetBasePressure() const;

    // Return initial energy.
    double GetBaseEnergy() const;

    // Return initial gamma parameter
    double GetGamma() const;

    // Return SPH particle mass.
    double GetParticleMass() const;

    // Return gravitational acceleration.
    ChVector3d GetGravitationalAcceleration() const;

    // Return the constant force applied to the fluid (if any).
    ChVector3d GetVolumetricForce() const;

    // Get the number of steps between successive updates to neighbor lists.
    int GetNumProximitySearchSteps() const;

    // Get number of steps before density reinitialization (0 if not used)
    int GetDensityReinitSteps() const;

    // Get use of density reinitialization
    bool GetDensityReinitSwitch() const;

    // Get use of variable time step
    bool GetUseVariableTimeStep() const;

    // Get parameters used in CFL condition
    double2 GetCFLParams() const;

    // Return the current system parameters (debugging only).
    const ChFsiParamsSPH_csph& GetParams() const { return *m_paramsH; }

    // Get the current number of fluid SPH particles.
    size_t GetNumFluidMarkers() const;

    // Get the current number of boundary BCE markers.
    size_t GetNumBoundaryMarkers() const;

    // Get the current number of rigid body BCE markers.
    size_t GetNumRigidBodyMarkers() const;

    // Return the SPH particle positions. NOTICE following vectors are actually stored as Real3 but reported
    // as vectors of ChVector3d
    std::vector<ChVector3d> GetParticlePositions() const;

    // Return the SPH particle velocities.
    std::vector<ChVector3d> GetParticleVelocities() const;

    // Return the accelerations of SPH particles.
    std::vector<ChVector3d> GetParticleAccelerations() const;

    // Return the forces acting on SPH particles.
    std::vector<ChVector3d> GetParticleForces() const;

    // Return the SPH particle fluid properties for all particles (fluid + bce).
    // For each SPH particle, the 3-dimensional vector contains density, pressure, and energy.
    // Difference wrt GetProperties() is this returns vector of ChVector3d.
    std::vector<ChVector3d> GetParticleFluidProperties() const;

    
    // Return the boundary treatment type.
    BoundaryMethod_csph GetBoundaryType() const { return m_paramsH->boundary_method; }

    // Return the viscosity treatment type.
    ViscosityMethod_csph GetViscosityMethod() const { return m_paramsH->viscosity_method; }

    // Return the viscosity treatment type.
    HeatingMethod_csph GetHeatingMethod() const { return m_paramsH->heating_method; }

    // Return the kernel type.
    KernelType_csph GetKernelType() const { return m_paramsH->kernel_type; }

    // Return the number of dimension used for the fluid problem
    int GetNumDim() const { return m_paramsH->num_dim;  }

    // Write FSI system particle output.
    void WriteParticleFile(const std::string& filename) const;

    // Save current SPH particle and BCE marker data to files.
    // This function creates three CSV files for SPH particles, boundary BCE markers, and solid BCE markers data.
    void SaveParticleData(const std::string& dir) const;

    // Save current FSI solid data to files.
    // This function creates CSV files for force and torque on rigid bodies and flexible nodes.
    void SaveSolidData(const std::string& dir, double time) const;

    // Print the three time step quantities and the final time step to a file
    // Only valid in variable time step mode
    void SaveTimeSteps(const std::string& dir) const;

    // ----------- Functions for adding SPH particles------------------

    // Add an SPH particle with given properties to the FSI system.
    // Temperature and speed of sound are computed internally.
    // If h = 0 means use default value in Parameters struct.
    void AddSPHParticle(const ChVector3d& pos,
                        double rho,
                        double pres,
                        double en,
                        const ChVector3d& vel = ChVector3d(0),
                        const double h = -1,
                        const double mass = -1);

    // Add an SPH particle with current properties to the SPH system.
    // Uses default values for thermodynamic properties.
    void AddSPHParticle(const ChVector3d& pos,
                        const ChVector3d& vel = ChVector3d(0));

    // Create SPH particles in the specified box volume.
    // The SPH particles are created on a uniform grid with resolution equal to the FSI initial separation.
    void AddBoxSPH(const ChVector3d& boxCenter, const ChVector3d& boxHalfDim);

    // -----------

    // Add WALL BCE markers at the specified points.
    // The points are assumed to be provided relative to the specified frame.
    // These BCE markers are not associated with a particular FSI body and, as such, cannot be used to extract fluid
    // forces and moments. If fluid reaction forces are needed, create an FSI body with the desired geometry or list
    // of BCE points and add it through the containing FSI system (AddBceFsiBody method)
    void AddBCEBoundary(const std::vector<ChVector3d>& points, const ChFramed& frame);

    // Similar as above but also explicitely pass h and mass.
    void AddBCEBoundary(const std::vector<ChVector3d>& points,
                        const ChFramed& frame,
                        const double h,
                        const double mass);

    // Similar as above but also explicitely pass h and mass for each bce particle
    void AddBCEBoundary(const std::vector<ChVector3d>& points, const ChFramed& frame, const std::vector<double>& h_vec, const std::vector<double>& mass_vec);
    // ----------- Utility functions for extracting information at specific SPH particles -----------------

    // Utility function for finding indices of SPH particles inside a given OBB.
    // The object-oriented box, of specified size, is assumed centered at the origin of the provided frame and aligned
    // with the axes of that frame.
    std::vector<int> FindParticlesInBox(const ChFrame<>& frame, const ChVector3d& size);

    // Extract positions of all markers (SPH and BCE).
    std::vector<Real3> GetPositions() const;

    // Extract velocities of all markers (SPH and BCE).
    std::vector<Real3> GetVelocities() const;

    // Extract accelerations of all markers (SPH and BCE).
    std::vector<Real3> GetAccelerations() const;

    // Extract forces applied to all markers (SPH and BCE).
    std::vector<Real3> GetForces() const;

    // Extract fluid properties of all markers (SPH and BCE).
    // For each SPH particle, the 3-dimensional vector contains density, pressure, and energy.
    // Difference wrt GetParticleFluidProperties is this returns vector of Real3.
    std::vector<Real3> GetProperties() const;

    // Extract speed of sound of all markers (SPH and BCE).
    std::vector<Real> GetSound() const;

    // Extract kernel radius (h) of all particles (SPH and BCE).
    std::vector<Real> GetKernelRad() const;

    // Extract positions of all markers (SPH and BCE) with indices in the provided array.
    std::vector<Real3> GetPositions(const std::vector<int>& indices) const;

    // Extract velocities of all markers (SPH and BCE) with indices in the provided array.
    std::vector<Real3> GetVelocities(const std::vector<int>& indices) const;

    // Extract accelerations of all markers (SPH and BCE) with indices in the provided array.
    std::vector<Real3> GetAccelerations(const std::vector<int>& indices) const;

    // Extract forces applied to allmarkers (SPH and BCE) with indices in the provided array.
    std::vector<Real3> GetForces(const std::vector<int>& indices) const;

    // Extract maximum kernel radius among all fluid particles:
    Real GetMaxFluidRad() const;

    // Extract maximum kernel radius among all particles (SPH and BCE):
    // Field x of output Real2 is the h value found, field y represents the particle type.
    Real2 GetMaxRad() const;

    // ----------- Utility functions for creating points in various volumes --------------------
    // All following methods just create geometrical points. Their output is to be added to the system
    // by calling the proper Add method
    

    // Create marker points on a rectangular plate of specified X-Y dimensions, assumed centered at the origin.
    // Markers are created in a number of layers (in the negative Z direction) corresponding to system parameters.
    std::vector<ChVector3d> CreatePointsPlate(const ChVector2d& size) const;

    // Create marker points for a box container of specified dimensions.
    // The box volume is assumed to be centered at the origin. The 'faces' input vector specifies which faces of the
    // container are to be created: for each direction, a value of -1 indicates the face in the negative direction, a
    // value of +1 indicates the face in the positive direction, and a value of 2 indicates both faces. Setting a value
    // of 0 does not create container faces in that direction. Markers are created in a number of layers corresponding
    // to system parameters.
    std::vector<ChVector3d> CreatePointsBoxContainer(const ChVector3d& size, const ChVector3i& faces) const;

    // Create interior marker points for a box of specified dimensions, assumed centered at the origin.
    // Markers are created inside the box, in a number of layers corresponding to system parameters.
    std::vector<ChVector3d> CreatePointsBoxInterior(const ChVector3d& size) const;

    // Create exterior marker points for a box of specified dimensions, assumed centered at the origin.
    // Markers are created outside the box, in a number of layers corresponding to system parameters.
    std::vector<ChVector3d> CreatePointsBoxExterior(const ChVector3d& size) const;

    // Create interior marker points for a sphere of specified radius, assumed centered at the origin.
    // Markers are created inside the sphere, in a number of layers corresponding to system parameters.
    // Markers are created using spherical coordinates (polar=true), or else on a uniform Cartesian grid.
    std::vector<ChVector3d> CreatePointsSphereInterior(double radius, bool polar) const;

    // Create exterior marker pointss for a sphere of specified radius, assumed centered at the origin.
    // Markers are created outside the sphere, in a number of layers corresponding to system parameters.
    // Markers are created using spherical coordinates (polar=true), or else on a uniform Cartesian grid.
    std::vector<ChVector3d> CreatePointsSphereExterior(double radius, bool polar) const;

    // Create interior marker points for a cylinder of specified radius and height.
    // The cylinder is assumed centered at the origin and aligned with the Z axis.
    // Markers are created inside the cylinder, in a number of layers corresponding to system parameters.
    // Markers are created using cylindrical coordinates (polar=true), or else on a uniform Cartesian grid.
    std::vector<ChVector3d> CreatePointsCylinderInterior(double rad, double height, bool polar) const;

    // Create exterior marker points for a cylinder of specified radius and height.
    // The cylinder is assumed centered at the origin and aligned with the Z axis.
    // Markers are created outside the cylinder, in a number of layers corresponding to system parameters.
    // Markers are created using cylindrical coordinates (polar=true), or else on a uniform Cartesian grid.
    std::vector<ChVector3d> CreatePointsCylinderExterior(double rad, double height, bool polar) const;

    // Create interior marker points for a cone of specified radius and height.
    // The cone is assumed centered at the origin and aligned with the Z axis.
    // Markers are created inside the cone, in a number of layers corresponding to system parameters.
    // Markers are created using cylinderical coordinates (polar=true), or else on a uniform Cartesian grid.
    std::vector<ChVector3d> CreatePointsConeInterior(double rad, double height, bool polar) const;

    // Create exterior marker points for a cone of specified radius and height.
    // The cone is assumed centered at the origin and aligned with the Z axis.
    // Markers are created outside the cone, in a number of layers corresponding to system parameters.
    // Markers are created using cylinderical coordinates (polar=true), or else on a uniform Cartesian grid.
    std::vector<ChVector3d> CreatePointsConeExterior(double rad, double height, bool polar) const;

    // Create marker points filling a cylindrical annulus of specified radii and height.
    // The cylinder annulus is assumed centered at the origin and aligned with the Z axis.
    // Markers are created using cylindrical coordinates (polar=true), or else on a uniform Cartesian grid.
    std::vector<ChVector3d> CreatePointsCylinderAnnulus(double rad_inner,
                                                        double rad_outer,
                                                        double height,
                                                        bool polar) const;

    // Create marker points filling a closed mesh.
    // Markers are created on a Cartesian grid with a separation corresponding to system parameters.
    std::vector<ChVector3d> CreatePointsMesh(ChTriangleMeshConnected& mesh) const;

  public:
    PhysicsProblem_csph GetPhysicsProblem() const;
    std::string GetPhysicsProblemString() const;
    std::string GetSphIntegrationSchemeString() const;


    // DEBUG ONLY 
     FsiDataManager_csph* GetDataManager() const { return m_data_mgr.get(); }
  private:


    // SPH specification of an FSI rigid solid. This struct represents a rigid body through its associated BCE markers.
    // Different from the data structure FsiBody
    struct FsiSphBody {
        std::shared_ptr<FsiBody> fsi_body;   //< underlying FSI solid. FsiBody defined in ChFsiDefinitions.h
        std::vector<ChVector3d> bce;         //< BCE initial global positions
        std::vector<ChVector3d> bce_coords;  //< local BCE coordinates: (x, y, z) in body frame
        std::vector<int> bce_ids;            //< BCE identification (body ID)
        bool check_embedded;                 //< if true, check for overlapping SPH particles
    };


    // Initialize simulation parameters with default values.
    void InitParams();

    // ---------------------------- Add Fsi Body to system -------------------------------

    // SPH solver-specific actions taken when a rigid solid is added as an FSI object.
    // Given the input FsiBody, this method creates the fsisphbody struct that also adds bce markers
    // by calling the CreateBCEFsiBody method.
    virtual void OnAddFsiBody(std::shared_ptr<FsiBody> fsi_body, bool check_embedded) override;


    // Create the the local BCE coordinates, their body associations, and the initial global BCE positions for the
    // given FSI rigid body.
    void CreateBCEFsiBody(std::shared_ptr<FsiBody> fsi_body,  // FsiBody different from FsiSphBody
                          std::vector<int>& bce_ids,
                          std::vector<ChVector3d>& bce_coords,
                          std::vector<ChVector3d>& bce);

    // -----------------------------------------------------------------------

    // Initialize the SPH fluid system. Just initializes empty mesh objects and call the proper virtual method
    void Initialize(const std::vector<FsiBodyState>& body_states);

    // Initialize the SPH fluid system with FSI support.
    // Need to leave the mesh input for polymorphism purposes, but will not be used.
    virtual void Initialize(const std::vector<FsiBodyState>& body_states,
                            const std::vector<FsiMeshState>& empty_mesh1,
                            const std::vector<FsiMeshState>& empty_mesh2) override;

    // Struct FsiSphBody is defined above at beginning of private field.
    // Add the rigid body and associated bce markers to the underlying data manager fields.
    // Notice that the bce markers are added to the data manager with zero input velocity.
    // Guess it will be properly set when enforcing boundary conditions.
    void AddBCEFsiBody(const FsiSphBody& fsisph_body);


    // -----------------------------------------------------------------------------

    // Mesh data structures left for polymorphism reasons. The base class defines following methods as pure virtual.
    // So input left untouched to avoid change in function signature. 
    // In current implementation no mesh is actually used.
    
    // Load the given body and mesh node states in the SPH data manager structures.
    // This function converts FEA mesh states from the provided AOS records to the SOA layout used by the SPH data
    // manager. LoadSolidStates is always called once during initialization. If the SPH fluid solver is paired with the
    // generic FSI interface, LoadSolidStates is also called from ChFsiInterfaceGeneric::ExchangeSolidStates at each
    // co-simulation data exchange. If using the custom SPH FSI interface, MBS states are copied directly to the
    // device memory in ChFsiInterfaceSPH::ExchangeSolidStates.
    virtual void LoadSolidStates(const std::vector<FsiBodyState>& body_states,
                                 const std::vector<FsiMeshState>& empty_mesh1,
                                 const std::vector<FsiMeshState>& empty_mesh2) override;


    // Actual function that this method will use. Inside it will initialize empty vectors of type FsiMeshState,
    // and give them as input to the omonimous virtual function.
    void LoadSolidStates(const std::vector<FsiBodyState>& body_states);


    // Store the body and mesh node forces from the SPH data manager to the given vectors.
    // If the SPH fluid solver is paired with the generic FSI interface, StoreSolidForces is also called from
    // ChFsiInterfaceGeneric::ExchangeSolidForces at each co-simulation data exchange. If using the custom SPH FSI
    // interface, MBS forces are copied directly from the device memory in ChFsiInterfaceSPH::ExchangeSolidForces.
    virtual void StoreSolidForces(std::vector<FsiBodyForce> body_forces,
                                  std::vector<FsiMeshForce> empty_mesh1,
                                  std::vector<FsiMeshForce> empty_mesh2) override;

    // Actual function that this method will use. Inside it will initialize empty vectors of type FsiMeshState,
    // and give them as input to the omonimous virtual function.
    void StoreSolidForces(const std::vector<FsiBodyForce> body_forces);

    // ----------

    // Function to integrate the fluid system from `time` to `time + step`.
    virtual void OnDoStepDynamics(double time, double step) override;

    // get the variable step size.
    // If problem doesn't use a variable time step, just returns the constant one.
    double GetVariableStepSize() override;

    // Additional actions taken before applying fluid forces to the solid phase.
    virtual void OnExchangeSolidForces() override;

    // Additional actions taken after loading new solid phase states.
    virtual void OnExchangeSolidStates() override;

    // ----------

    std::shared_ptr<ChFsiParamsSPH_csph> m_paramsH;  //< simulation parameters stored into ChFsiParamsSPH_csph object
    bool m_force_proximity_search;

    std::unique_ptr<FsiDataManager_csph> m_data_mgr;       //< FSI data manager
    std::unique_ptr<FluidDynamics_csph> m_fluid_dynamics;  //< fluid system
    std::unique_ptr<BceManager_csph> m_bce_mgr;            //< BCE manager

    unsigned int m_num_rigid_bodies;     //< number of rigid bodies

    std::vector<FsiSphBody> m_bodies;       //< list of FSI rigid bodies
    std::vector<int> m_fsi_bodies_bce_num;  //< number of BCE particles on each fsi body

    // First value is always the input time step for CFD.
    std::vector<double> m_time_steps;    //< collect values of variable time steps used in the simulation
    std::vector<double> m_courant_steps;  //< minimum values of courant and acceleration time steps.
    std::vector<double> m_force_steps;    
    
    OutputLevel m_output_level;
    bool m_check_errors;

    friend class ChFsiSystemSPH_csph;
    friend class ChFsiInterfaceSPH_csph;
    friend class ChFsiProblemSPH_csph;

    // bool parameters to check which properties has been set explicitely by proper methods.
    bool set_rho;
    bool set_pres;
    bool set_en;
    bool set_all;     // properties set with SetFLuidProperties.
    bool def_prop;    // using default fluid properties 

    // inherited from base class:
    // bool m_verbose;        
    // std::string m_outdir;  

    // bool m_use_node_directions;  
    // bool m_is_initialized;       
    // double m_step;         
    // unsigned int m_frame;  

    // ChTimer m_timer_step;  
    // double m_RTF;          
    // double m_time;         
};



}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif
