// =============================================================================
// Authors: Andrea D'Uva - 2025
// =============================================================================
//
// Utility class to set up a compressible SPH-based Chrono::FSI problem.
// This class is at highest level of the hierarchy, as it rules the various classes
// that govern the MBS, the fluid and the Fsi systems.
// =============================================================================

#ifndef CH_FSI_PROBLEM_SPH_COMPRESSIBLE_H
#define CH_FSI_PROBLEM_SPH_COMPRESSIBLE_H

#include <cmath>
#include <unordered_set>
#include <unordered_map>

#include "chrono/physics/ChSystem.h"
#include "chrono/utils/ChBodyGeometry.h"
#include "chrono/functions/ChFunction.h"

#include "chrono_fsi/ChApiFsi.h"
#include "chrono_fsi/sph/ChFsiSystemSPH.h"
#include "chrono_fsi/sph/physics/FsiParticleRelocator.cuh"

#include "chrono_fsi/sph_compressible/ChFsiSystemSPH_compressible.h"
#include "chrono_fsi/sph_compressible/physics/FsiParticleRelocator_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"

namespace chrono::fsi::sph{
namespace compressible {


// Base class to set up a Chrono::FSI problem.
// Actually it's an abstract class that can0t be instantiated because it has two pure virtual methods (snap2grid and grid2point).
class CH_FSI_API ChFsiProblemSPH_csph {
  public:

    // Enable verbose output during construction of ChFsiProblemSPH (default: false).
    void SetVerbose(bool verbose);

    // Access the underlying FSI system. (On its own has a reference to a ChFsiFluidSystemSPH_csph)
    ChFsiSystemSPH_csph& GetSystemFSI() { return m_sysFSI; }

    // Access the underlying SPH system.
    ChFsiFluidSystemSPH_csph& GetFluidSystemSPH() { return m_sysSPH; }

    // Access the underlying MBS system.
    ChSystem& GetMultibodySystem() { return m_sysFSI.GetMultibodySystem(); }

    // Set the fluid properties.
    void SetFluidProperties(const ChFsiFluidSystemSPH_csph::FluidProperties_csph& fluid_props);

    // Set SPH method parameters.
    void SetSPHParameters(const ChFsiFluidSystemSPH_csph::SPHParameters_csph& sph_params);

    // Add a rigid body to the FSI problem.
    // BCE markers are created for the provided geometry (which may or may not match the body collision geometry).
    // By default, where applicable, BCE markers are created using polar coordinates (in layers starting from the shape
    // surface). Generation of BCE markers on a uniform Cartesian grid can be enforced setting use_grid_bce=true.
    // Creation of FSI bodies embedded in the fluid phase is allowed (SPH markers inside the body geometry volume are
    // pruned). To check for possible overlap with SPH particles, set 'check_embedded=true'.
    // This function must be called before Initialize().
    void AddRigidBody(std::shared_ptr<ChBody> body,
                      std::shared_ptr<utils::ChBodyGeometry> geometry,
                      bool check_embedded,
                      bool use_grid_bce = false);

    void AddRigidBodySphere(std::shared_ptr<ChBody> body,
                            const ChVector3d& pos,
                            double radius,
                            bool use_grid_bce = false);
    void AddRigidBodyBox(std::shared_ptr<ChBody> body, const ChFramed& pos, const ChVector3d& size);
    void AddRigidBodyCylinderX(std::shared_ptr<ChBody> body,
                               const ChFramed& pos,
                               double radius,
                               double length,
                               bool use_grid_bce = false);
    void AddRigidBodyMesh(std::shared_ptr<ChBody> body,
                          const ChVector3d& pos,
                          const std::string& obj_file,
                          const ChVector3d& interior_point,
                          double scale);

    // Return the number of BCE markers associated with the specified rigid body.
    size_t GetNumBCE(std::shared_ptr<ChBody> body) const;

    // Interface for callback to set initial particle pressure, density, energy, and velocity.
    class CH_FSI_API ParticlePropertiesCallback {
      public:
        ParticlePropertiesCallback() : p0(0), rho0(0), e0(0), v0(VNULL) {}
        ParticlePropertiesCallback(const ParticlePropertiesCallback& other) = default;
        virtual ~ParticlePropertiesCallback() {}

        // Set values for particle properties.
        // The default implementation sets pressure and velocity to zero and constant density and energy.
        // If an override is provided, it must set *all* particle properties.
        virtual void set(const ChFsiFluidSystemSPH_csph& sysSPH, const ChVector3d& pos) {
            p0 = sysSPH.GetBasePressure();
            rho0 = sysSPH.GetBaseDensity();
            e0 = sysSPH.GetBaseEnergy();
            v0 = VNULL;
        }

        double p0;
        double rho0;
        double e0;
        ChVector3d v0;
    };

    // Register a callback for setting SPH particle initial properties.
    void RegisterParticlePropertiesCallback(std::shared_ptr<ParticlePropertiesCallback> callback) {
        m_props_cb = callback;
    }

    // Set gravitational acceleration for both multibody and fluid systems.
    void SetGravitationalAcceleration(const ChVector3d& gravity) { m_sysFSI.SetGravitationalAcceleration(gravity); }

    // Set integration step size for fluid dynamics.
    void SetStepSizeCFD(double step) { m_sysFSI.SetStepSizeCFD(step); }

    // Set integration step size for multibody dynamics.
    // If a value is not provided, the MBS system is integrated with the same step used for fluid dynamics.
    void SetStepsizeMBD(double step) { m_sysFSI.SetStepsizeMBD(step); }

    // Explicitly set the computational domain limits.
    // By default, this encompasses all SPH and BCE markers with no boundary conditions imposed in any direction.
    void SetComputationalDomain(const ChAABB& aabb,
                                BoundaryConditions bc_type = {BCType::NONE, BCType::NONE, BCType::NONE}) {
        m_domain_aabb = aabb;
        m_bc_type = bc_type;
    }

    // Complete construction of the FSI problem and initialize the FSI system.
    // After this call, no additional solid bodies should be added to the FSI problem.
    void Initialize();

    // Advance the dynamics of the underlying FSI system by the specified step.
    // Just calls the underlying ChFsiSystem_csph method.
    void DoStepDynamics(double step);

    // Get the ground body.
    std::shared_ptr<ChBody> GetGroundBody() const { return m_ground; }

    // Get number of SPH particles.
    size_t GetNumSPHParticles() const { return m_sph.size(); }

    // Get number of boundary BCE markers.
    size_t GetNumBoundaryBCEMarkers() const { return m_bce.size(); }

    // Get limits of computational domain.
    const ChAABB& GetComputationalDomain() const { return m_domain_aabb; }

    // Get the boundary condition type for the three sides of the computational domain.
    const BoundaryConditions& GetBoundaryConditionTypes() const { return m_bc_type; }

    // Get limits of SPH volume.
    const ChAABB& GetSPHBoundingBox() const { return m_sph_aabb; }

    // Return the FSI applied force on the specified body (as returned by AddRigidBody).
    // The force is applied at the body COM and is expressed in the absolute frame.
    // An exception is thrown if the given body was not added through AddRigidBody.
    const ChVector3d& GetFsiBodyForce(std::shared_ptr<ChBody> body) const;

    // Return the FSI applied torque on on the specified body (as returned by AddRigidBody).
    // The torque is expressed in the absolute frame.
    // An exception is thrown if the given body was not added through AddRigidBody.
    const ChVector3d& GetFsiBodyTorque(std::shared_ptr<ChBody> body) const;

    // Get current estimated RTF (real time factor) for the fluid system.
    double GetRtfCFD() const { return m_sysSPH.GetRtf(); }

    // Get current estimated RTF (real time factor) for the multibody system.
    double GetRtfMBD() const { return m_sysFSI.GetMultibodySystem().GetRTF(); }

    // Set SPH simulation data output level (default: STATE_PRESSURE).
    // Options:
    // - STATE           marker state, velocity, and acceleration
    // - STATE_PRESSURE  STATE plus density and pressure
    // - CFD_FULL        STATE_PRESSURE plus various CFD parameters
    // - CRM_FULL        STATE_PRESSURE plus normal and shear stress
    void SetOutputLevel(OutputLevel output_level);

    // Save current SPH and solid data to files.
    // This functions creates three CSV files (for SPH particles, boundary BCE markers, and solid BCE markers data) in
    // the directory `sph_dir` and two CSV files (for force and torque on rigid bodies) in the
    // directory `fsi_dir`.
    void SaveOutputData(double time, const std::string& sph_dir, const std::string& fsi_dir);

    // Save the set of initial SPH and BCE grid locations to files in the specified output directory.
    void SaveInitialMarkers(const std::string& out_dir) const;

    PhysicsProblem_csph GetPhysicsProblem() const { return m_sysSPH.GetPhysicsProblem(); }
    std::string GetPhysicsProblemString() const { return m_sysSPH.GetPhysicsProblemString(); }
    std::string GetSphIntegrationSchemeString() const { return m_sysSPH.GetSphIntegrationSchemeString(); }

  protected:
    // Create a ChFsiProblemSPH object.
    // No SPH parameters are set.
    // the class constructor is protected by default.
    // Automatically creates a ground fixed body and adds it to the system.
    ChFsiProblemSPH_csph(ChSystem& sys, double spacing);

    // Hash function for a 3D integer grid coordinate.
    //  Return of std::hash is usually a fixed-size hash value, typically size_t
    struct CoordHash {
        std::size_t operator()(const ChVector3i& p) const {
            size_t h1 = std::hash<int>()(p.x());  // first compute hash values of each field
            size_t h2 = std::hash<int>()(p.y());
            size_t h3 = std::hash<int>()(p.z());
            return (h1 ^ (h2 << 1)) ^ h3;  // combine hash values using bitwise operators (XOR ^ or bitwise shift <<)
        }
    };

    // Grid points with integer coordinates.
    // Unordered sets are unordered associative containers that store unique elements. It stores its elements using
    // hashing. This provides almost constant O(1) time search, insert and delete operations, but elements are not
    // sorted in any specific order. The insertion process doesn't accept position, since it is defined based on the
    // hashed value. Also can't access elements by position but need to increment/decrement the begin() and end()
    // iterators. Can be done also by next() or advance() functions. Unordered set provides fast search by value with
    // method find() std::unordered_set is defined as a class template. First argument in template is the type of the
    // key/value (it stores only one element), the second one is a custom hashing functor
    typedef std::unordered_set<ChVector3i, CoordHash> GridPoints;

    // helper functions to convert between real type coordinates and grid type ones (integers)

    virtual ChVector3i Snap2Grid(const ChVector3d& point) = 0;  // convert a vector of doubles into a vector of int by
                                                                // transforming in grid coordinates and rounding.
    virtual ChVector3d Grid2Point(const ChVector3i& p) = 0;  // convert integer grid points into double "physical" values. In Cartesian Problem
                                                             // just multiplies int coordinates by m_spacing.

    // Prune (trim) fluid SPH markers that are inside the solid body volume.
    // Treat separately primitive shapes (use explicit test for interior points) and mesh shapes (use ProcessBodyMesh).
    void ProcessBody(ChFsiFluidSystemSPH_csph::FsiSphBody& b);

    // Prune SPH markers that are inside a body mesh volume.
    // Voxelize the body mesh (at the scaling resolution) and identify grid nodes inside the boundary
    // defined by the body BCEs. Note that this assumes the BCE markers form a watertight boundary.
    int ProcessBodyMesh(ChFsiFluidSystemSPH_csph::FsiSphBody& b,
                        ChTriangleMeshConnected trimesh,
                        const ChVector3d& interior_point);

    // Only derived classes can use the following particle and marker relocation functions

    void CreateParticleRelocator();
    void BCEShift(const ChVector3d& shift_dist);
    void SPHShift(const ChVector3d& shift_dist);
    void SPHMoveAABB2AABB(const ChAABB& aabb_src, const ChIntAABB& aabb_dest);
    void ForceProximitySearch();

    ChFsiFluidSystemSPH_csph m_sysSPH;      //< underlying Chrono SPH system
    ChFsiSystemSPH_csph m_sysFSI;           //< underlying Chrono FSI system
    double m_spacing;                  //< particle and marker spacing
    std::shared_ptr<ChBody> m_ground;  //< ground body
    GridPoints m_sph;                  //< SPH particle grid locations
    GridPoints m_bce;                  //< boundary BCE marker grid locations
    ChVector3d m_offset_sph;           //< SPH particles offset
    ChVector3d m_offset_bce;           //< boundary BCE particles offset
    ChAABB m_domain_aabb;              //< computational domain bounding box. ChAABB is a class defined in standard Chrono.
    BoundaryConditions m_bc_type;      //< boundary conditions in each direction
    ChAABB m_sph_aabb;                 //< SPH volume bounding box

    std::unordered_map<std::shared_ptr<ChBody>, size_t> m_fsi_bodies;  //< unordered (key, value) map with ChBody pointers and index in FSI body list.

    std::shared_ptr<ChFsiProblemSPH_csph::ParticlePropertiesCallback> m_props_cb;  //< callback for particle properties

    std::unique_ptr<FsiParticleRelocator_csph> m_relocator;

    bool m_verbose;      //< if true, write information to standard output
    bool m_initialized;  //< if true, problem was initialized

    friend class SelectorFunctionWrapper;
};



// ----------------------------------------------------------------------------
// Derived class ProblemCartesian
// ----------------------------------------------------------------------------

// Class to set up a Chrono::FSI problem using particles and markers on a Cartesian coordinates grid.
class CH_FSI_API ChFsiProblemCartesian_csph : public ChFsiProblemSPH_csph {
  public:
    // Create a ChFsiProblemSPH object.
    // No SPH parameters are set.
    ChFsiProblemCartesian_csph(ChSystem& sys, double spacing);

    // Construct using information from the specified files.
    // The SPH particle and BCE marker locations are assumed to be provided on an integer grid.
    // Locations in real space are generated using the specified grid separation value and the
    // patch translated to the specified position.
    void Construct(const std::string& sph_file,  //< filename with SPH grid particle positions
                   const std::string& bce_file,  //< filename with BCE grid marker positions
                   const ChVector3d& pos         //< reference position
    );

    // Construct SPH particles and optionally BCE markers in a box of given dimensions.
    // The reference position is the center of the bottom face of the box; in other words, SPH particles are generated
    // above this location and BCE markers for the bottom boundary are generated below this location.
    // If created, the BCE markers for the top, bottom, and side walls are adjacent to the SPH domain; 'side_flags' are
    // boolean combinations of BoxSide enums.
    // Particles are created and stored only in the FsiProblem class until call to Initialize(), 
    // where they are actually added to the FsiDataManager struct.
    void Construct(const ChVector3d& box_size,  //< box dimensions
                   const ChVector3d& pos,       //< reference position
                   int side_flags               //< sides for which BCE markers are created
    );

    // Construct SPH particles and optionally BCE markers from a given heightmap.
    // The image file is read with STB, using the number of channels defined in the input file and reading
    // the image as 16-bit (8-bit images are automatically converted). Supported image formats: JPEG, PNG,
    // BMP, GIF, PSD, PIC, PNM.
    // Create the SPH particle grid locations for a patch of specified X and Y dimensions with optional
    // translation. The height at each grid point is obtained through bilinear interpolation from the gray values in
    // the provided heightmap image (with pure black corresponding to the lower height range and pure white to the
    // upper height range). SPH particle grid locations are generated to cover the specified depth under each grid
    // point. If created, BCE marker layers are generated below the bottom-most layer of SPH particles and on the sides
    // of the patch. 'side_flags' are boolean combinations of BoxSide enums.
    void Construct(const std::string& heightmap_file,  //< filename for the heightmap image
                   double length,                      //< patch length (X direction)
                   double width,                       //< patch width (Y direction)
                   const ChVector2d& height_range,     //< height range (black to white level)
                   double depth,                       //< fluid phase depth
                   bool uniform_depth,                 //< if true, bottom follows surface
                   const ChVector3d& pos,              //< reference position
                   int side_flags                      //< sides for which BCE markers are created
    );

    // Add fixed BCE markers, representing a container for the computational domain.
    // The specified 'box_size' represents the dimensions of the *interior* of the box.
    // The reference position is the center of the bottom face of the box.
    // Boundary BCE markers are created outside this volume, in layers, starting at a distance equal to the spacing.
    // 'side_flags' are boolean combinations of BoxSide enums.
    // It is the caller responsibility to ensure that the container BCE markers do not overlap with any SPH particles.
    size_t AddBoxContainer(const ChVector3d& box_size,  //< box dimensions
                           const ChVector3d& pos,       //< reference positions
                           int side_flags               //< sides for which BCE markers are created
    );


  private:
    virtual ChVector3i Snap2Grid(const ChVector3d& point) override;
    virtual ChVector3d Grid2Point(const ChVector3i& p) override;
};

// ----------------------------------------------------------------------------

// Class to set up a Chrono::FSI problem using particles and markers on a cylindrical coordinates grid.
class CH_FSI_API ChFsiProblemCylindrical_csph : public ChFsiProblemSPH_csph {
  public:
    // Create a ChFsiProblemSPH object.
    // No SPH parameters are set.
    ChFsiProblemCylindrical_csph(ChSystem& sys, double spacing);

    // Construct SPH particles and optionally BCE markers in a cylindrical annulus of given dimensions.
    // Set inner radius to zero to create a cylindrical container.
    // The reference position is the center of the bottom face of the cylinder; in other words, SPH particles are
    // generated above this location and BCE markers for the bottom boundary are generated below this location.
    // If created, the BCE markers for the bottom and side walls are adjacent to the SPH domain; 'side_flags' are
    // boolean combinations of CylSide enums.
    void Construct(double radius_inner,    ///< inner radius
                   double radius_outer,    ///< outer radius
                   double height,          ///< height
                   const ChVector3d& pos,  ///< reference position,
                   int side_flags          ///< sides for which BCE markers are created
    );

    // Add fixed BCE markers, representing a cylindrical annulus container for the computational domain.
    // Set inner radius to zero to create a cylindrical container.
    // The cylinder is constructed with its axis along the global Z axis.
    // The specified dimensions refer to the *interior* of the cylindrical annulus.
    // 'side_flags' are boolean combinations of CylSide enums.
    size_t AddCylindricalContainer(double radius_inner,    ///< inner radius
                                   double radius_outer,    ///< outer radius
                                   double height,          ///< height
                                   const ChVector3d& pos,  ///< reference position
                                   int side_flags          ///< sides for which BCE markers are created
    );

  private:
    virtual ChVector3i Snap2Grid(const ChVector3d& point) override;
    virtual ChVector3d Grid2Point(const ChVector3i& p) override;
};

// ----------------------------------------------------------------------------

// Predefined SPH particle initial properties callback (depth-based pressure).
class CH_FSI_API DepthPressurePropertiesCallback_csph : public ChFsiProblemSPH_csph::ParticlePropertiesCallback {
  public:
    DepthPressurePropertiesCallback_csph(double zero_height) : ParticlePropertiesCallback(), zero_height(zero_height) {}

    virtual void set(const ChFsiFluidSystemSPH_csph& sysSPH, const ChVector3d& pos) override {
        double gz = std::abs(sysSPH.GetGravitationalAcceleration().z());
        // double c2 = sysSPH.GetSoundSpeed() * sysSPH.GetSoundSpeed();
        p0 = sysSPH.GetBaseDensity() * gz * (zero_height - pos.z());
        // rho0 = sysSPH.GetBaseDensity() + p0 / c2;
        e0 = sysSPH.GetBaseEnergy();
        auto params = sysSPH.GetParams();
        auto temp_params = params;
        temp_params.eos_type = EosType_csph::IDEAL_RHOEN;
        rho0 = InvEos_csph(p0, e0, temp_params);
        v0 = VNULL;
    }

  private:
    double zero_height;
};

}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif
