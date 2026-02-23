// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Implementation of an FSI-aware SPH fluid solver for use in compressible sph applications.
//
// =============================================================================

#include <cmath>
#include <algorithm>

#include "chrono/core/ChTypes.h"

#include "chrono/utils/ChUtils.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_fsi/sph/ChFsiFluidSystemSPH.h"

#include "chrono_fsi/sph/physics/SphGeneral.cuh"
#include "chrono_fsi/sph/physics/FsiDataManager.cuh"
#include "chrono_fsi/sph/physics/FluidDynamics.cuh"
#include "chrono_fsi/sph/physics/BceManager.cuh"

#include "chrono_fsi/sph/math/CustomMath.cuh"

#include "chrono_fsi/sph/utils/UtilsTypeConvert.cuh"
#include "chrono_fsi/sph/utils/UtilsPrintSph.cuh"
#include "chrono_fsi/sph/utils/UtilsDevice.cuh"

#include "chrono_thirdparty/filesystem/path.h"
#include "chrono_thirdparty/filesystem/resolver.h"

#include "chrono_thirdparty/rapidjson/document.h"
#include "chrono_thirdparty/rapidjson/filereadstream.h"

#include "chrono_fsi/sph_compressible/ChFsiFluidSystemSPH_compressible.h"

#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/FluidDynamics_compressible.cuh"
#include "chrono_fsi/sph_compressible/physics/BceManager_compressible.cuh"

#include "chrono_fsi/sph_compressible/math/CustomMath_compressible.cuh"

#include "chrono_fsi/sph_compressible/utils/UtilsPrintSph_compressible.cuh"
#include "chrono_fsi/sph_compressible/utils/UtilsDevice_compressible.cuh"

using namespace rapidjson;

using std::cout;
using std::cerr;
using std::endl;

namespace chrono::fsi::sph {
namespace compressible {


//--------------------------------- constructor and destructor ----------------------
// default empty constructor still initializes some of the attributes
ChFsiFluidSystemSPH_csph::ChFsiFluidSystemSPH_csph()
    : ChFsiFluidSystem(),
      m_num_rigid_bodies(0),
      m_output_level(OutputLevel::STATE_PRESSURE),
      m_force_proximity_search(false),
      m_check_errors(true),
      set_rho(false),
      set_pres(false),
      set_en(false),
      set_all(false),
      def_prop(true)   {
      m_paramsH = chrono_types::make_shared<ChFsiParamsSPH_csph>();
      InitParams(); // method that Initializes simulation parameters with default values

      m_data_mgr = chrono_types::make_unique<FsiDataManager_csph>(m_paramsH);
}


ChFsiFluidSystemSPH_csph::~ChFsiFluidSystemSPH_csph() {}


//-------------------------------------------------------------------------------
// Initializes simulation parameters with some default values. recall these are stored in a ChFsiParamsSPH object, and
// class ChFsiFluidSystemSPH has a pointer to it.
//-------------------------------------------------------------------------------
void ChFsiFluidSystemSPH_csph::InitParams() {

    // HERE NO DEFAULT VALUES FOR COMPUTATIONAL DOMAIN RELATED PARAMETERS 
    // Notice by default the cMin and Cmax parameters are set depending on boxDIms/2 + some buffer on z
    // Fluid properties
    m_paramsH->rho0 = Real(1);
    m_paramsH->p0 = Real(0.4);
    m_paramsH->e0 = Real(1);
    m_paramsH->gamma = Real(1.4); 
    m_paramsH->Cs0 = sqrt(m_paramsH->gamma * (m_paramsH->p0 / m_paramsH->rho0));

    m_paramsH->invrho0 = 1 / m_paramsH->rho0;
    m_paramsH->bodyForce3 = mR3(0, 0, 0);
    m_paramsH->gravity = mR3(0, 0, 0);
    m_paramsH->L_Characteristic = Real(1.0);

    // SPH parameters
    m_paramsH->physics_problem = PhysicsProblem_csph::EULER;
    m_paramsH->rho_evolution = Rho_evolution_csph::DIFFERENTIAL;
    m_paramsH->h_evolution = H_evolution_csph::CONSTANT;
    m_paramsH->h_variation = false;
    m_paramsH->integration_scheme = IntegrationScheme_csph::RK2;
    m_paramsH->eos_type = EosType_csph::IDEAL_RHOEN;
    m_paramsH->viscosity_method = ViscosityMethod_csph::ARTIFICIAL_MONAGHAN;
    m_paramsH->heating_method = HeatingMethod_csph::NONE;
    m_paramsH->boundary_method = BoundaryMethod_csph::ORIGINAL_ADAMI;     
    m_paramsH->kernel_type = KernelType_csph::CUBIC_SPLINE;
    m_paramsH->shifting_method = sph::ShiftingMethod::XSPH;
    m_paramsH->bc_type = {BCType::NONE, BCType::NONE, BCType::NONE};

    m_paramsH->d0 = Real(0.01);
    m_paramsH->ood0 = 1 / m_paramsH->d0;
    m_paramsH->d0_multiplier = Real(1.2);
    m_paramsH->h = m_paramsH->d0_multiplier * m_paramsH->d0;
    m_paramsH->ooh = 1 / m_paramsH->h;
    m_paramsH->h_multiplier = 2;      // for the default cubic spline kernel

    m_paramsH->volume0 = cube(m_paramsH->d0);

    m_paramsH->shifting_xsph_eps = Real(0.5);
    m_paramsH->density_reinit_switch = false;
    m_paramsH->density_reinit_steps = 2147483647;
    m_paramsH->Ar_vis_alpha = Real(1.0);
    m_paramsH->Ar_vis_beta = Real(2.0);
    m_paramsH->Ar_heat_g1 = Real(0.0);
    m_paramsH->Ar_heat_g2 = Real(0.0);
    m_paramsH->Ar_heat_switch = false;

    m_paramsH->epsMinMarkersDis = Real(0.01);

    m_paramsH->ADKE_k = 1.0;
    m_paramsH->ADKE_eps = 0.5;
    m_paramsH->ADKE_D = 1.5;

    m_paramsH->is_uniform = true;
    m_paramsH->markerMass = -1;       // Initialize it later

    m_paramsH->num_bce_layers = 3;
    m_paramsH->dT = Real(-1);         // -1 will mean that has not been initilized by user

    // The default bodyactivedomain value will ensure that all particles are always marked as active
    m_paramsH->bodyActiveDomain = mR3(1e10, 1e10, 1e10);
    m_paramsH->use_active_domain = false;

    m_paramsH->use_default_limits = true;       // true if computational domain not set by the user. When using method to set it, this variable will be false.
    m_paramsH->use_init_pressure = false;       // true if initial pressure to be set based on height. Default false
    m_paramsH->num_proximity_search_steps = 4;

    m_paramsH->use_variable_time_step = false;
    m_paramsH->C_cfl = Real(0.5);
    m_paramsH->C_force = Real(0.5); 
    m_paramsH->num_dim = 3;
}


//------------------------------------------------------------------------------
// function that loads a vector from a JSON file usind the RapidJSON library.
Real3 LoadVectorJSON(const Value& a) {
    assert(a.IsArray());
    assert(a.Size() == 3);
    return mR3(a[0u].GetDouble(), a[1u].GetDouble(), a[2u].GetDouble());
}


//-------------------------------------------------------------------------------
// Functions that set problem parameters. They modify attributes of the ChFsiParamsSPH object through the pointer
// m_paramsH
// ------------------------------------------------------------------------------

// Reads Fsi parameters from specified JSON file. Uses the RapidJSON library. Notice we are "using namespace rapidjson".
// Also uses some custom functions defined to interface Chrono and Json files, like LoadVectorJSON() defined above
void ChFsiFluidSystemSPH_csph::ReadParametersFromFile(const std::string& json_file) {
    if (m_verbose)
        cout << "Reading parameters from: " << json_file << endl;

    FILE* fp = fopen(json_file.c_str(), "r");
    if (!fp) {
        cerr << "Invalid JSON file!" << endl;
        return;
    }

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    fclose(fp);

    Document doc;

    doc.ParseStream<ParseFlag::kParseCommentsFlag>(is);
    if (!doc.IsObject()) {
        cerr << "Invalid JSON file!!" << endl;
        return;
    }

    if (doc.HasMember("Physical Properties of Fluid")) {
        if (doc["Physical Properties of Fluid"].HasMember("Density")) {
            m_paramsH->rho0 = doc["Physical Properties of Fluid"]["Density"].GetDouble();
            set_rho = true;
        }

        if (doc["Physical Properties of Fluid"].HasMember("Pressure")) {
            m_paramsH->p0 = doc["Physical Properties of Fluid"]["Pressure"].GetDouble();
            set_pres = true;
        }

        if (doc["Physical Properties of Fluid"].HasMember("Specific energy")) {
            m_paramsH->e0 = doc["Physical Properties of Fluid"]["Specific energy"].GetDouble();
            set_en = true;
        }

        if (doc["Physical Properties of Fluid"].HasMember("Gamma"))
            m_paramsH->gamma = doc["Physical Properties of Fluid"]["Gamma"].GetDouble();
        else
            m_paramsH->gamma = Real(1.4);

        if (doc["Physical Properties of Fluid"].HasMember("Body Force"))
            m_paramsH->bodyForce3 = LoadVectorJSON(doc["Physical Properties of Fluid"]["Body Force"]);

        if (doc["Physical Properties of Fluid"].HasMember("Gravity"))
            m_paramsH->gravity = LoadVectorJSON(doc["Physical Properties of Fluid"]["Gravity"]);

        if (doc["Physical Properties of Fluid"].HasMember("Characteristic Length"))
            m_paramsH->L_Characteristic = doc["Physical Properties of Fluid"]["Characteristic Length"].GetDouble();
    

        
        m_paramsH->Cs0 = std::sqrt(m_paramsH->gamma * m_paramsH->p0 / m_paramsH->rho0);
        m_paramsH->invrho0 = 1 / m_paramsH->rho0;
    }


    if (doc.HasMember("SPH Parameters")) {

        if (doc["SPH Parameters"].HasMember("Initial Spacing"))
            m_paramsH->d0 = doc["SPH Parameters"]["Initial Spacing"].GetDouble();

        if (doc["SPH Parameters"].HasMember("Kernel Multiplier"))
            m_paramsH->d0_multiplier = doc["SPH Parameters"]["Kernel Multiplier"].GetDouble();

        if (doc["SPH Parameters"].HasMember("Epsilon"))
            m_paramsH->epsMinMarkersDis = doc["SPH Parameters"]["Epsilon"].GetDouble();

        if (doc["SPH Parameters"].HasMember("Marker Mass"))
            m_paramsH->markerMass = doc["SPH Parameters"]["Marker Mass"].GetDouble();

        if (doc["SPH Parameters"].HasMember("Shifting Method")) {
            std::string method = doc["SPH Parameters"]["Shifting Method"].GetString();
            if (method == "XSPH")
                m_paramsH->shifting_method = ShiftingMethod::XSPH;
            else {
                m_paramsH->shifting_method = ShiftingMethod::NONE;
            }
        }

        if (doc["SPH Parameters"].HasMember("XSPH Coefficient"))
            m_paramsH->shifting_xsph_eps = doc["SPH Parameters"]["XSPH Coefficient"].GetDouble();

        if (doc["SPH Parameters"].HasMember("Kernel Type")) {
            std::string type = doc["SPH Parameters"]["Kernel Type"].GetString();
            if (type == "Cubic")
                m_paramsH->kernel_type = KernelType_csph::CUBIC_SPLINE;
            else if (type == "Quintic")
                m_paramsH->kernel_type = KernelType_csph::QUINTIC_SPLINE;
            else if (type == "Wendland")
                m_paramsH->kernel_type = KernelType_csph::WENDLAND;
            else {
                cerr << "Incorrect kernel type in the JSON file: " << type << endl;
                cerr << "Falling back to cubic spline." << endl;
                m_paramsH->kernel_type = KernelType_csph::CUBIC_SPLINE;
            }
        }

        if (doc["SPH Parameters"].HasMember("Boundary Treatment Type")) {
            std::string type = doc["SPH Parameters"]["Boundary Treatment Type"].GetString();
            if (type == "Original Adami")
                m_paramsH->boundary_method = BoundaryMethod_csph::ORIGINAL_ADAMI;
            else if (type == "Modified Adami") {
                cerr << " Modified adami boundary treatment not yet implemented. Falling back to original Adami"
                     << endl;
                m_paramsH->boundary_method = BoundaryMethod_csph::ORIGINAL_ADAMI;
            } else {
                cerr << "Incorrect boundary treatment type in the JSON file: " << type << endl;
                cerr << "Falling back to Adami " << endl;
                m_paramsH->boundary_method = BoundaryMethod_csph::ORIGINAL_ADAMI;
            }
        }

        if (doc["SPH Parameters"].HasMember("Viscosity Treatment Type")) {
            std::string type = doc["SPH Parameters"]["Viscosity Treatment Type"].GetString();
            if (m_verbose)
                cout << "viscosity treatment is : " << type << endl;
            if (type == "NONE")
                m_paramsH->viscosity_method = ViscosityMethod_csph::NONE;
            else if (type == "Artificial Monaghan") {
                m_paramsH->viscosity_method = ViscosityMethod_csph::ARTIFICIAL_MONAGHAN;
            } else {
                cerr << "Incorrect viscosity type in the JSON file: " << type << endl;
                cerr << "Falling back to Artificial Monaghan Viscosity" << endl;
                m_paramsH->viscosity_method = ViscosityMethod_csph::ARTIFICIAL_MONAGHAN;
            }
        }

        if (doc["SPH Parameters"].HasMember("Artificial viscosity alpha"))
            m_paramsH->Ar_vis_alpha = doc["SPH Parameters"]["Artificial viscosity alpha"].GetDouble();

        if (doc["SPH Parameters"].HasMember("Artificial viscosity beta"))
            m_paramsH->Ar_vis_beta = doc["SPH Parameters"]["Artificial viscosity beta"].GetDouble();


        if (doc["SPH Parameters"].HasMember("EOS Type")) {
            std::string type = doc["SPH Parameters"]["EOS Type"].GetString();
            if (m_verbose)
                cout << "Eos type is : " << type << endl;
            if (type == "Ideal EhoEn")
                m_paramsH->eos_type = EosType_csph::IDEAL_RHOEN;
            else {
                cerr << "Incorrect eos type in the JSON file: " << type << endl;
                cerr << "Falling back to P(rho,en) Equation of State " << endl;
                m_paramsH->eos_type = EosType_csph::IDEAL_RHOEN;
            }
        }

        if (doc["SPH Parameters"].HasMember("Density evolution")) {
            std::string type = doc["SPH Parameters"]["Density evolution"].GetString();
            if (m_verbose)
                cout << "Method for density evolution is : " << type << endl;
            if (type == "Summation")
                m_paramsH->rho_evolution = Rho_evolution_csph::SUMMATION;
            else if (type == "Continuity equation")
                m_paramsH->rho_evolution = Rho_evolution_csph::DIFFERENTIAL;
            else {
                cerr << "Incorrect method of density evolution in the JSON file: " << type << endl;
                cerr << "Falling back to differential evolution of density " << endl;
                m_paramsH->rho_evolution = Rho_evolution_csph::DIFFERENTIAL;
            }
        }

        if (doc["SPH Parameters"].HasMember("Kernel radius evolution")) {
            std::string type = doc["SPH Parameters"]["Kernel radius evolution"].GetString();
            if (m_verbose)
                cout << "Method for kernel radius treatment is : " << type << endl;
            if (type == "Constant") {
                m_paramsH->h_evolution = H_evolution_csph::CONSTANT;
                m_paramsH->h_variation = false;
            } else if (type == "ADKE") {
                m_paramsH->h_evolution = H_evolution_csph::ADKE;
                m_paramsH->h_variation = true;
            } else if (type == "Differential") {
                m_paramsH->h_evolution = H_evolution_csph::DIFFERENTIAL;
                m_paramsH->h_variation = true;
            }
            else {
                cerr << "Incorrect method of kernel radius treatment in the JSON file: " << type << endl;
                cerr << "Falling back to constant kernel radius " << endl;
                m_paramsH->h_evolution = H_evolution_csph::CONSTANT;
            }
        }

        if (doc["SPH Parameters"].HasMember("ADKE k"))
            m_paramsH->ADKE_k = doc["SPH Parameters"]["ADKE k"].GetInt();

        if (doc["SPH Parameters"].HasMember("ADKE epsilon"))
            m_paramsH->ADKE_eps = doc["SPH Parameters"]["ADKE epsilon"].GetInt();

        if (doc["SPH Parameters"].HasMember("Density Reinitialization")) {
            m_paramsH->density_reinit_steps = doc["SPH Parameters"]["Density Reinitialization"].GetInt();
            m_paramsH->density_reinit_switch = true;
        } else {
            m_paramsH->density_reinit_switch = false;
        }

        if (doc["SPH Parameters"].HasMember("Time steps per proximity search"))
            m_paramsH->num_proximity_search_steps = doc["SPH Parameters"]["Time steps per proximity search"].GetInt();
    }

    if (doc.HasMember("Time Stepping")) {
        if (doc["Time Stepping"].HasMember("Time step")) {
            m_paramsH->dT = doc["Time Stepping"]["Time step"].GetDouble();
            m_step = m_paramsH->dT;
        }
        if (doc["Time Stepping"].HasMember("Use variable time step")) {
            std::string type  = doc["Time Stepping"]["Use variable time step"].GetString();
            if (type == "true")
                m_paramsH->use_variable_time_step = true;
        }
        if (doc["Time Stepping"].HasMember("C_cfl")) {
            m_paramsH->C_cfl = doc["Time Stepping"]["C_cfl"].GetDouble();
            m_paramsH->use_variable_time_step = true;
        }
        if (doc["Time Stepping"].HasMember("C_force")) {
            m_paramsH->C_force = doc["Time Stepping"]["C_force"].GetDouble();
        }
    }

    if (doc.HasMember("Time Stepping")) {
        
    }

    // Geometry Information
    if (doc.HasMember("Geometry Inf")) {
        if (doc["Geometry Inf"].HasMember("BoxDimensionX"))
            m_paramsH->boxDimX = doc["Geometry Inf"]["BoxDimensionX"].GetDouble();

        if (doc["Geometry Inf"].HasMember("BoxDimensionY"))
            m_paramsH->boxDimY = doc["Geometry Inf"]["BoxDimensionY"].GetDouble();

        if (doc["Geometry Inf"].HasMember("BoxDimensionZ"))
            m_paramsH->boxDimZ = doc["Geometry Inf"]["BoxDimensionZ"].GetDouble();
    }

    if (doc.HasMember("Body Active Domain")) {
        auto size = LoadVectorJSON(doc["Body Active Domain"]);
        m_paramsH->use_active_domain = true;
        m_paramsH->bodyActiveDomain = size / 2;
    }


    // Calculate dependent parameters
    m_paramsH->ood0 = 1 / m_paramsH->d0;
    m_paramsH->h = m_paramsH->d0_multiplier * m_paramsH->d0;
    m_paramsH->ooh = 1 / m_paramsH->h;
    m_paramsH->volume0 = cube(m_paramsH->d0);
    m_paramsH->invrho0 = 1 / m_paramsH->rho0;

}



//------------------------------------------------------------------------------
// Different simple Set methods to set various Sph method parameters/properties
//------------------------------------------------------------------------------
void ChFsiFluidSystemSPH_csph::SetRhoEvolution(Rho_evolution_csph rho_evolution) {
    m_paramsH->rho_evolution = rho_evolution;
}

void ChFsiFluidSystemSPH_csph::SetKernRadEvolution(H_evolution_csph h_evolution) {
    m_paramsH->h_evolution = h_evolution;
    if (h_evolution == H_evolution_csph::ADKE || h_evolution == H_evolution_csph::DIFFERENTIAL)
        m_paramsH->h_variation = true;
    else
        m_paramsH->h_variation = false;
}

void ChFsiFluidSystemSPH_csph::SetADKECoeff(double k, double eps, double D) {
    m_paramsH->ADKE_k = k;
    m_paramsH->ADKE_eps = eps;
    m_paramsH->ADKE_D = D;
}


void ChFsiFluidSystemSPH_csph::SetEosType(EosType_csph eos_type) {
    m_paramsH->eos_type = eos_type;
}

void ChFsiFluidSystemSPH_csph::SetBoundaryType(BoundaryMethod_csph boundary_method) {
    m_paramsH->boundary_method = boundary_method;
}

void ChFsiFluidSystemSPH_csph::SetViscosityMethod(ViscosityMethod_csph viscosity_method) {
    m_paramsH->viscosity_method = viscosity_method;
}

void ChFsiFluidSystemSPH_csph::SetArtificialViscosityCoefficient(double alpha, double beta) {
    m_paramsH->Ar_vis_alpha = alpha;
    m_paramsH->Ar_vis_beta = beta;
}

void ChFsiFluidSystemSPH_csph::SetHeatingMethod(HeatingMethod_csph heating_method) {
    m_paramsH->heating_method = heating_method;
}

void ChFsiFluidSystemSPH_csph::SetArtificialHeatingCoefficient(double g1, double g2) {
    m_paramsH->Ar_heat_g1 = g1;
    m_paramsH->Ar_heat_g1 = g1;
}

void ChFsiFluidSystemSPH_csph::SetKernelType(KernelType_csph kernel_type) {
    m_paramsH->kernel_type = kernel_type;
    switch (m_paramsH->kernel_type) {
        case KernelType_csph::CUBIC_SPLINE:
            m_paramsH->h_multiplier = 2;
            break;
        case KernelType_csph::QUINTIC_SPLINE:
            m_paramsH->h_multiplier = 3;
            break;
        case KernelType_csph::WENDLAND:
            m_paramsH->h_multiplier = 2;
            break;
    }
}

void ChFsiFluidSystemSPH_csph::SetNumDim(int n_dim) {
    if (n_dim < 1 || n_dim > 3) {
        std::cerr << "Number of dimensions must be between 1 and 3. Keeping num_dim = 3";
        return;
    }
    m_paramsH->num_dim = n_dim;
}

void ChFsiFluidSystemSPH_csph::SetShiftingMethod(ShiftingMethod shifting_method) {
    if (shifting_method == ShiftingMethod::XSPH) {
            m_paramsH->shifting_method = shifting_method;
        }
    else if(shifting_method == ShiftingMethod::NONE){
            m_paramsH->shifting_method = shifting_method;
    } 
    else {
        cerr << "Invalid Shifting Method set, revert to NONE" << endl;
        m_paramsH->shifting_method = ShiftingMethod::NONE;
    }
}

void ChFsiFluidSystemSPH_csph::SetMidpointGridUpdate(bool midpoint_switch) {
    m_paramsH->midpoint_neigh_search = midpoint_switch;
}

void ChFsiFluidSystemSPH_csph::SetIntegrationScheme(IntegrationScheme_csph scheme) {
    m_paramsH->integration_scheme = scheme;
}

void ChFsiFluidSystemSPH_csph::SetIsUniform(bool is_uniform) {
    m_paramsH->is_uniform = is_uniform;
}

void ChFsiFluidSystemSPH_csph::SetContainerDim(const ChVector3d& box_dim) {
    m_paramsH->boxDimX = box_dim.x();
    m_paramsH->boxDimY = box_dim.y();
    m_paramsH->boxDimZ = box_dim.z();
}

void ChFsiFluidSystemSPH_csph::SetComputationalDomain(const ChAABB& computational_AABB, BoundaryConditions bc_type) {
    m_paramsH->cMin = ToReal3(computational_AABB.min);
    m_paramsH->cMax = ToReal3(computational_AABB.max);
    m_paramsH->use_default_limits = false;
    m_paramsH->bc_type = bc_type;
}

void ChFsiFluidSystemSPH_csph::SetComputationalDomain(const ChAABB& computational_AABB) {
    m_paramsH->cMin = ToReal3(computational_AABB.min);
    m_paramsH->cMax = ToReal3(computational_AABB.max);
    m_paramsH->use_default_limits = false;
}

void ChFsiFluidSystemSPH_csph::SetActiveDomain(const ChVector3d& box_dim) {
    m_paramsH->bodyActiveDomain = ToReal3(box_dim / 2);
    m_paramsH->use_active_domain = true;
}


void ChFsiFluidSystemSPH_csph::SetNumBCELayers(int num_layers) {
    m_paramsH->num_bce_layers = num_layers;
}

    
// gravity in m_paramsH is a Real3 type, input gravity is a ChVector. Is operator= overloaded for Real type? Guess don't
// need it as components of Chvector3d are double and Real is typedef for either double or float, so compiler should
// perform casting without problems
void ChFsiFluidSystemSPH_csph::SetGravitationalAcceleration(const ChVector3d& gravity) {
    m_paramsH->gravity.x = gravity.x();
    m_paramsH->gravity.y = gravity.y();
    m_paramsH->gravity.z = gravity.z();
}

void ChFsiFluidSystemSPH_csph::SetBodyForce(const ChVector3d& force) {
    m_paramsH->bodyForce3.x = force.x();
    m_paramsH->bodyForce3.y = force.y();
    m_paramsH->bodyForce3.z = force.z();
}

void ChFsiFluidSystemSPH_csph::SetInitialSpacing(double spacing) {
    m_paramsH->d0 = (Real)spacing;
    m_paramsH->ood0 = 1 / m_paramsH->d0;       // inverse of initial spacing
    m_paramsH->volume0 = cube(m_paramsH->d0);  // initial particle volume computed as (init_spacing)^3, then computed as mass/density
   

    m_paramsH->h = m_paramsH->d0_multiplier * m_paramsH->d0;  // by definition of d0_multiplier: h = init_spacing*d0_multiplier
    m_paramsH->ooh = 1 / m_paramsH->h;             // inverse of h
}

void ChFsiFluidSystemSPH_csph::SetMarkerMass(double mass) {
    m_paramsH->markerMass = mass;
}

// not the parameter k involved in kernel radius as rad = h * k, which depends only on the kernel type, but the
// "multiplier" parameter such that h = init_spacing*multiplier
void ChFsiFluidSystemSPH_csph::SetKernelMultiplier(double multiplier) {
    m_paramsH->d0_multiplier = Real(multiplier);  // set parameter

    m_paramsH->h = m_paramsH->d0_multiplier * m_paramsH->d0;  // update correlated h
    m_paramsH->ooh = 1 / m_paramsH->h;
}

void ChFsiFluidSystemSPH_csph::SetDensity(double rho0) {
    m_paramsH->rho0 = rho0;
    m_paramsH->invrho0 = 1 / m_paramsH->rho0;

    set_rho = true;
    def_prop = false;
}

void ChFsiFluidSystemSPH_csph::SetPressureHeight(const double height) {
    m_paramsH->pressure_height = height;
    m_paramsH->use_init_pressure = true;
}

void ChFsiFluidSystemSPH_csph::SetPressure(double p0) {
    m_paramsH->p0 = p0;
    set_pres = true;
    def_prop = false;
}


void ChFsiFluidSystemSPH_csph::SetEnergy(double e0) {
    m_paramsH->e0 = e0;
    set_en = true;
    def_prop = false;
}


void ChFsiFluidSystemSPH_csph::SetGamma(double gamma) {
    m_paramsH->gamma = gamma;
    def_prop = false;
}

void ChFsiFluidSystemSPH_csph::SetShiftingXSPHParameters(double eps) {
    m_paramsH->shifting_xsph_eps = eps;
}

void ChFsiFluidSystemSPH_csph::SetOutputLevel(OutputLevel output_level) {
    m_output_level = output_level;
}

// this parameters represents the number of time-steps between unpdates of the neighbor list. Default 4
void ChFsiFluidSystemSPH_csph::SetNumProximitySearchSteps(int steps) {
    m_paramsH->num_proximity_search_steps = steps;
}

// set number of steps before density reinitialization
void ChFsiFluidSystemSPH_csph::SetDensityReinitSteps(int steps) {
    m_paramsH->density_reinit_steps = steps;
    if (steps != 0)
        m_paramsH->density_reinit_switch = true;
    else
        m_paramsH->density_reinit_switch = false;
}

void ChFsiFluidSystemSPH_csph::SetUseVariableTimeStep(bool use_switch) {
    m_paramsH->use_variable_time_step = use_switch;
}

void ChFsiFluidSystemSPH_csph::SetCFLParams(double C_cfl, double C_force) {
    m_paramsH->C_cfl = C_cfl;
    m_paramsH->C_force = C_force;
}

void ChFsiFluidSystemSPH_csph::SetMinMarkerDistance(double eps) {
    m_paramsH->epsMinMarkersDis = eps;
}

void ChFsiFluidSystemSPH_csph::CheckFluidParameters() {
    // Check the combinations of user provided fluid properties
    
    if (set_all)        // All properties user-provided with SetFluidProperties()
        ;               
    else if (def_prop)  // All properties are default ones
        return;
    else if (!set_rho && !set_pres  && !set_en) {  // Changed gamma
        std::cerr << "Error - Using default density, pressure and energy but non-default fluid constants!" << std::endl;
        return;
    }
    else if (set_rho && set_pres && !set_en) {   // Only energy not set
        m_paramsH->e0 = m_paramsH->p0 / m_paramsH->rho0 / (m_paramsH->gamma - 1);
    } 
 
    else if (!set_rho) {  // Density not set
        std::cerr << "Error - Forgot to provide density" << std::endl;
        return;
    }

    else if (!set_pres && set_en) {  // Pressure not set, compute with Eos
        m_paramsH->p0 = EosRhoEn_csph(m_paramsH->rho0, m_paramsH->e0, m_paramsH->gamma);
    } 

    // Set speed of sound for cases that didn't already return
    m_paramsH->Cs0 = std::sqrt(m_paramsH->gamma * m_paramsH->p0 / m_paramsH->rho0);

  
    return;
}



void ChFsiFluidSystemSPH_csph::CheckSPHParameters() {

    if (m_paramsH->num_dim < 1 || m_paramsH->num_dim > 3) {
        std::cerr
            << "WARNING: number of dimensions for fluid problem is not between 1 and 3. Reverting to 3D"
            << std::endl;
        m_paramsH->num_dim = 3;
    }

    // Calculate default cMin and cMax - they depend on provided boxDims (InitParams does not provide default values)
    Real3 default_cMin = mR3(-2 * m_paramsH->boxDims.x, -2 * m_paramsH->boxDims.y, -2 * m_paramsH->boxDims.z) - 10 * mR3(m_paramsH->h);
    Real3 default_cMax = mR3(+2 * m_paramsH->boxDims.x, +2 * m_paramsH->boxDims.y, +2 * m_paramsH->boxDims.z) + 10 * mR3(m_paramsH->h);

    // Check if user-defined cMin and cMax are much larger than defaults
    if (m_paramsH->cMin.x < 2 * default_cMin.x || m_paramsH->cMin.y < 2 * default_cMin.y ||
        m_paramsH->cMin.z < 2 * default_cMin.z || m_paramsH->cMax.x > 2 * default_cMax.x ||
        m_paramsH->cMax.y > 2 * default_cMax.y || m_paramsH->cMax.z > 2 * default_cMax.z) {
        cerr << "WARNING: User-defined cMin or cMax is much larger than the default values. "
             << "This may slow down the simulation." << endl;
    }

    // TODO: Add check for whether computational domain is larger than SPH + BCE layers
    if (m_paramsH->d0_multiplier < 1) {
        cerr << "WARNING: Kernel interaction length multiplier is less than 1. This may lead to numerical "
                "instability due to poor particle approximation."
             << endl;
    }

    if (m_paramsH->kernel_type == KernelType_csph::CUBIC_SPLINE && m_paramsH->d0_multiplier > 1.5) {
        // Check if W3h is defined as W3h_CubicSpline
        cerr << "WARNING: Kernel interaction radius multiplier is greater than 1.5 and the cubic spline kernel is "
                "used. This may lead to pairing instability. See Pg 10. of Ha H.Bui et al. Smoothed particle "
                "hydrodynamics (SPH) and its applications in geomechanics : From solid fracture to granular "
                "behaviour and multiphase flows in porous media. You might want to switch to the Wendland kernel."
             << endl;
    }

    if (m_paramsH->num_bce_layers < 3) {
        cerr << "WARNING: Number of BCE layers is less than 3. This may cause insufficient kernel support at the "
                "boundaries and lead to leakage of particles"
             << endl;
    }

    // Check shifting method and whether defaults have changed
    if (m_paramsH->shifting_method == ShiftingMethod::NONE) {
        if (m_paramsH->shifting_xsph_eps != 0.5) {
            cerr << "WARNING: Shifting method is NONE, but shifting parameters have been modified. These "
                    "changes will not take effect."
                 << endl;
        }
    }

    if (m_paramsH->h_evolution == H_evolution_csph::DIFFERENTIAL &&
        m_paramsH->rho_evolution != Rho_evolution_csph::DIFFERENTIAL) {
        cerr << "WARNING: h evolution is DIFFERENTIAL, but density is evolved through summation. Falling back to "
                "constant h"
             << endl;
        m_paramsH->h_evolution = H_evolution_csph::CONSTANT;
    }

    if (m_paramsH->heating_method == HeatingMethod_csph::ARTIFICIAL && 
        m_paramsH->Ar_heat_g1 == 0 &&
        m_paramsH->Ar_heat_g2 == 0) {
        std::cerr << "WARNING: artificial heating set as on but g1 = g2 = 0. Reverting to "
                                  "Artificial Heating = NONE"
                               << std::endl;
        m_paramsH->Ar_heat_switch = false;
    } else if (m_paramsH->heating_method == HeatingMethod_csph::ARTIFICIAL &&
               (m_paramsH->Ar_heat_g1 != 0 || m_paramsH->Ar_heat_g2 != 0))
        m_paramsH->Ar_heat_switch = true;

}

// default constructor of FluidProperties struct defined in the ChFsiFluidSystemSPH class
ChFsiFluidSystemSPH_csph::FluidProperties_csph::FluidProperties_csph()
    : density(1), pressure(0.4), energy(1), gamma(1.4), char_length(1) {}

// set some problem parameters according to corresponding one in input struct
void ChFsiFluidSystemSPH_csph::SetFluidProperties(const FluidProperties_csph& fluid_props) {

    m_paramsH->rho0 = Real(fluid_props.density);
    m_paramsH->invrho0 = 1 / m_paramsH->rho0;
    m_paramsH->p0 = Real(fluid_props.pressure);
    m_paramsH->e0 = Real(fluid_props.energy);
    m_paramsH->gamma = Real(fluid_props.gamma);
 
    m_paramsH->Cs0 = std::sqrt(m_paramsH->gamma * m_paramsH->p0 / m_paramsH->rho0);
    m_paramsH->invrho0 = 1 / m_paramsH->rho0;
    m_paramsH->L_Characteristic = Real(fluid_props.char_length);
    
    set_all = true;
    def_prop = false;
}


// default constructor of the SPHParameters struct defined in the ChFsiFluidSystemSPH class
ChFsiFluidSystemSPH_csph::SPHParameters_csph::SPHParameters_csph()
    : integration_scheme(IntegrationScheme_csph::RK2),
      rho_evolution(Rho_evolution_csph::DIFFERENTIAL),
      h_evolution(H_evolution_csph::CONSTANT),
      eos_type(EosType_csph::IDEAL_RHOEN),
      viscosity_method(ViscosityMethod_csph::ARTIFICIAL_MONAGHAN),
      heating_method(HeatingMethod_csph::NONE),
      boundary_method(BoundaryMethod_csph::ORIGINAL_ADAMI),
      kernel_type(KernelType_csph::CUBIC_SPLINE),
      shifting_method(ShiftingMethod::NONE),
      num_bce_layers(3),
      initial_spacing(0.01),
      d0_multiplier(1.2),
      shifting_xsph_eps(0.5),
      min_distance_coefficient(0.01),
      density_reinit_switch(false),
      density_reinit_steps(2e8),
      artificial_viscosity_alpha(1.0),
      artificial_viscosity_beta(2.0),
      artificial_heating_g1(0.0),
      artificial_heating_g2(0.0),
      ADKE_k(1.0),
      ADKE_eps(0.5),
      ADKE_D(1.5),
      num_proximity_search_steps(4),
      use_variable_time_step(false),
      C_cfl(0.5),
      C_force(0.5),
      num_dim(3),
      is_uniform(true),
      markerMass(-1) {}



// sets problem parameters given a full input SPHParameters reference
void ChFsiFluidSystemSPH_csph::SetSPHParameters(const SPHParameters_csph& sph_params) {
    m_paramsH->integration_scheme = sph_params.integration_scheme;

    m_paramsH->eos_type = sph_params.eos_type;
    m_paramsH->viscosity_method = sph_params.viscosity_method;
    m_paramsH->heating_method = sph_params.heating_method;
    m_paramsH->boundary_method = sph_params.boundary_method;
    m_paramsH->kernel_type = sph_params.kernel_type;
    m_paramsH->shifting_method = sph_params.shifting_method;
    m_paramsH->rho_evolution = sph_params.rho_evolution;
    m_paramsH->h_evolution = sph_params.h_evolution;

    m_paramsH->d0 = sph_params.initial_spacing;
    m_paramsH->volume0 = cube(m_paramsH->d0);
    m_paramsH->markerMass = m_paramsH->volume0 * m_paramsH->rho0;
    m_paramsH->d0_multiplier = sph_params.d0_multiplier;
    m_paramsH->h = m_paramsH->d0_multiplier * m_paramsH->d0;
    m_paramsH->ood0 = 1 / m_paramsH->d0;
    m_paramsH->ooh = 1 / m_paramsH->h;

    m_paramsH->shifting_xsph_eps = sph_params.shifting_xsph_eps;
    m_paramsH->epsMinMarkersDis = sph_params.min_distance_coefficient;

    m_paramsH->density_reinit_switch = sph_params.density_reinit_switch;
    m_paramsH->density_reinit_steps = sph_params.density_reinit_steps;

    m_paramsH->num_bce_layers = sph_params.num_bce_layers;
    m_paramsH->Ar_vis_alpha = sph_params.artificial_viscosity_alpha;
    m_paramsH->Ar_vis_beta = sph_params.artificial_viscosity_beta;

    m_paramsH->Ar_heat_g1 = sph_params.artificial_heating_g1;
    m_paramsH->Ar_heat_g2 = sph_params.artificial_heating_g2;

    m_paramsH->ADKE_k = sph_params.ADKE_k;
    m_paramsH->ADKE_eps = sph_params.ADKE_eps;
    m_paramsH->ADKE_D = sph_params.ADKE_D;

    m_paramsH->num_proximity_search_steps = sph_params.num_proximity_search_steps;
    
    m_paramsH->use_variable_time_step = sph_params.use_variable_time_step;
    m_paramsH->C_cfl = sph_params.C_cfl;
    m_paramsH->C_force = sph_params.C_force;
    m_paramsH->num_dim = sph_params.num_dim;
    m_paramsH->is_uniform = sph_params.is_uniform;
    m_paramsH->markerMass = sph_params.markerMass;

}

//------------------------------------------------------------------------------
// Public methods that access variuos private variables of the ChFsiFluidSystemSPH class
//------------------------------------------------------------------------------

PhysicsProblem_csph ChFsiFluidSystemSPH_csph::GetPhysicsProblem() const {
    return (m_paramsH->physics_problem);
}

std::string ChFsiFluidSystemSPH_csph::GetPhysicsProblemString() const {
    return ("EULER");
}

std::string ChFsiFluidSystemSPH_csph::GetSphIntegrationSchemeString() const {
    std::string method = "";
    switch (m_paramsH->integration_scheme) {
        case IntegrationScheme_csph::EULER:
            method = "CSPH_EULER";
            break;
        case IntegrationScheme_csph::RK2:
            method = "CSPH_RK2";
            break;
        case IntegrationScheme_csph::VERLET:
            method = "CSPH_VERLET";
            break;
        case IntegrationScheme_csph::SYMPLECTIC:
            method = "CSPH_SYMPLECTIC";
            break;
    }

    return method;
}

//------------------------------------------------------------------------------
// Convert host data from the provided SOA (struct of array) to the data manager's AOS (array of struct) and copy to
// device. We give as input a std::vector of FsiBodyState and we copy it into the struct of vectors fsiBodyState_H
// inside FsiDataManager.

void ChFsiFluidSystemSPH_csph::LoadSolidStates(const std::vector<FsiBodyState>& body_states) {
    // initialize empty vector for the mesh elements:
    std::vector<FsiMeshState> empty_mesh;

    // Call the complete function:
    LoadSolidStates(body_states, empty_mesh, empty_mesh);
}


void ChFsiFluidSystemSPH_csph::LoadSolidStates(const std::vector<FsiBodyState>& body_states,
                                               const std::vector<FsiMeshState>& empty_mesh1,
                                               const std::vector<FsiMeshState>& empty_mesh2) {
    {  // rigid bodies
        size_t num_bodies = body_states.size();
        // copy input data into data manager
        for (size_t i = 0; i < num_bodies; i++) {
            m_data_mgr->fsiBodyState_H->pos[i] = ToReal3(body_states[i].pos);
            m_data_mgr->fsiBodyState_H->lin_vel[i] = ToReal3(body_states[i].lin_vel);
            m_data_mgr->fsiBodyState_H->lin_acc[i] = ToReal3(body_states[i].lin_acc);
            m_data_mgr->fsiBodyState_H->rot[i] = ToReal4(body_states[i].rot);
            m_data_mgr->fsiBodyState_H->ang_vel[i] = ToReal3(body_states[i].lin_acc);
            m_data_mgr->fsiBodyState_H->ang_acc[i] = ToReal3(body_states[i].ang_acc);
        }

        if (num_bodies > 0)  // if bodies present, copy host vector into device vector
            m_data_mgr->fsiBodyState_D->CopyFromH(*m_data_mgr->fsiBodyState_H);
    }

}


void ChFsiFluidSystemSPH_csph::StoreSolidForces(std::vector<FsiBodyForce> body_forces) {
    // initialize empty mesh vector
    std::vector<FsiMeshForce> empty_mesh;

    StoreSolidForces(body_forces, empty_mesh, empty_mesh);
}
   
// Copy from device and convert host data from the data manager's AOS to the output SOA.
// here the input vectors are overwritten to store data already in data manager object
void ChFsiFluidSystemSPH_csph::StoreSolidForces(std::vector<FsiBodyForce> body_forces,
                                                std::vector<FsiMeshForce> empty_mesh1,
                                                std::vector<FsiMeshForce> empty_mesh2) {
    {  // rigid bodies forces and torques
        auto forcesH = m_data_mgr->GetRigidForces();
        auto torquesH = m_data_mgr->GetRigidTorques();

        size_t num_bodies = body_forces.size();
        for (size_t i = 0; i < num_bodies; i++) {
            body_forces[i].force = ToChVector(forcesH[i]);  // from a Real3 to a ChVector3d
            body_forces[i].torque = ToChVector(torquesH[i]);
        }
    }

}



//------------------------------------------------------------------------------
// Two print utilities. One for hardware, other for problem's parameters
//------------------------------------------------------------------------------

void PrintDeviceProperties(const cudaDeviceProp& prop) {
    cout << "GPU device: " << prop.name << endl;
    cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "  Total global memory: " << prop.totalGlobalMem / (1024. * 1024. * 1024.) << " GB" << endl;
    cout << "  Total constant memory: " << prop.totalConstMem / 1024. << " KB" << endl;
    cout << "  Total available static shared memory per block: " << prop.sharedMemPerBlock / 1024. << " KB" << endl;
    cout << "  Max. dynamic shared memory per block: " << prop.sharedMemPerBlockOptin / 1024. << " KB" << endl;
    cout << "  Total shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024. << " KB" << endl;
    cout << "  Number of multiprocessors: " << prop.multiProcessorCount << endl;
}

void PrintParams(const ChFsiParamsSPH_csph& params, const Counters_csph& counters) {
    cout << "Simulation parameters" << endl;

    if (params.is_uniform) 
        cout << "Problem is initially uniform, using a global particle mass" << endl;
    else
        cout << "Problem is initially not uniform, using a value of mass for each particle" << endl;
     

    switch (params.viscosity_method) {
        case ViscosityMethod_csph::NONE:
            cout << "  Viscosity treatment: None" << endl;
            break;
        case ViscosityMethod_csph::ARTIFICIAL_MONAGHAN:
            cout << "  Viscosity treatment: Artificial Monaghan";
            cout << "  (alpha coefficient: " << params.Ar_vis_alpha << ")" << endl;
            cout << "  (beta coefficient:  " << params.Ar_vis_beta << ")" << endl;
            break;
    }

    switch (params.heating_method) {
        case HeatingMethod_csph::NONE:
            cout << "  Artificial heating term: None" << endl;
            break;
        case HeatingMethod_csph::ARTIFICIAL:
            cout << "  Heating method: Artificial Conduction";
            cout << "  (g1 coefficient: " << params.Ar_heat_g1 << ")" << endl;
            cout << "  (g2 coefficient:  " << params.Ar_heat_g2 << ")" << endl;
            break;
    }

    if (params.boundary_method == BoundaryMethod_csph::ORIGINAL_ADAMI) {
        cout << "  Boundary treatment: Original Adami" << endl;
    } else if (params.boundary_method == BoundaryMethod_csph::MODIFIED_ADAMI) {
        cout << "  Boundary treatment: Modified Adami" << endl;
    }

    switch (params.kernel_type) {
        case KernelType_csph::CUBIC_SPLINE:
            cout << "  Kernel type: Cubic Spline" << endl;
            break;
        case KernelType_csph::QUINTIC_SPLINE:
            cout << "  Kernel type: Quintic Spline" << endl;
            break;
        case KernelType_csph::WENDLAND:
            cout << "  Kernel type: Wendland Quintic" << endl;
            break;
    }

    switch (params.shifting_method) {
        case ShiftingMethod::XSPH:
            cout << "  Shifting method: XSPH" << endl;
            break;
        case ShiftingMethod::NONE:
            cout << "  Shifting method: None" << endl;
            break;
    }

    switch (params.integration_scheme) {
        case IntegrationScheme_csph::EULER:
            cout << "  Integration scheme: Explicit Euler" << endl;
            break;
        case IntegrationScheme_csph::RK2:
            cout << "  Integration scheme: Runge-Kutta 2" << endl;
            break;
        case IntegrationScheme_csph::SYMPLECTIC:
            cout << "  Integration scheme: Symplectic Euler" << endl;
            break;
    }

    switch (params.rho_evolution) {
        case Rho_evolution_csph::SUMMATION:
            cout << "  Density evolution: summation equation" << endl;
            break;
        case Rho_evolution_csph::DIFFERENTIAL:
            cout << "  Density evolution: mass conservation equation" << endl;
            break;
    }

    switch (params.h_evolution) {
        case H_evolution_csph::CONSTANT:
            cout << "  Kernel radius kept constant" << endl;
            break;
        case H_evolution_csph::ADKE:
            cout << "  Kernel radius evolution: ADKE" << endl;
            break;
        case H_evolution_csph::DIFFERENTIAL:
            cout << "  Kernel radius evolution: differential equation" << endl;
            break;
    }


    cout << "  number of dimensions for the fluid problem: " << params.num_dim << endl;
    cout << "  num_neighbors: " << params.num_neighbors << endl;
    cout << "  rho0: " << params.rho0 << endl;
    cout << "  invrho0: " << params.invrho0 << endl;
    cout << "  P0: " << params.p0 << endl;
    cout << "  gamma: " << params.gamma << endl;
    cout << "  bodyForce3: " << params.bodyForce3.x << " " << params.bodyForce3.y << " " << params.bodyForce3.z << endl;
    cout << "  gravity: " << params.gravity.x << " " << params.gravity.y << " " << params.gravity.z << endl;

    cout << "  d0: " << params.d0 << endl;
    cout << "  1/d0: " << params.ood0 << endl;
    cout << "  d0_multiplier: " << params.d0_multiplier << endl;
    cout << "  h: " << params.h << endl;
    cout << "  1/h: " << params.ooh << endl;

    cout << "  num_bce_layers: " << params.num_bce_layers << endl;
    cout << "  epsMinMarkersDis: " << params.epsMinMarkersDis << endl;
    cout << "  markerMass: " << params.markerMass << endl;
    cout << "  volume0: " << params.volume0 << endl;

    cout << "  Cs0: " << params.Cs0 << endl;

    if (params.shifting_method == ShiftingMethod::XSPH) {
        cout << "  shifting_xsph_eps: " << params.shifting_xsph_eps << endl;
    }


    cout << "  density_reinit_steps active?  " << std::boolalpha << params.density_reinit_switch << endl;

    cout << "  Proximity search performed every " << params.num_proximity_search_steps << " steps" << endl;
    cout << "  dT: " << params.dT << endl;
    cout << "  use_variable_time_step: " << params.use_variable_time_step << endl;

    if (params.use_variable_time_step) {
        cout << "  C_cfl: " << params.C_cfl << endl;
        cout << "  C_force: " << params.C_force << endl;
    }

    cout << "  binSize0: " << params.binSize0 << endl;
    cout << "  boxDims: " << params.boxDims.x << " " << params.boxDims.y << " " << params.boxDims.z << endl;
    cout << "  gridSize: " << params.gridSize.x << " " << params.gridSize.y << " " << params.gridSize.z << endl;
    cout << "  cMin: " << params.cMin.x << " " << params.cMin.y << " " << params.cMin.z << endl;
    cout << "  cMax: " << params.cMax.x << " " << params.cMax.y << " " << params.cMax.z << endl;

    ////Real dt_CFL = params.Co_number * params.h / 2.0 / MaxVel;
    ////Real dt_nu = 0.2 * params.h * params.h / (params.mu0 / params.rho0);
    ////Real dt_body = 0.1 * sqrt(params.h / length(params.bodyForce3 + params.gravity));
    ////Real dt = std::min(dt_body, std::min(dt_CFL, dt_nu));

    cout << "Counters" << endl;
    cout << "  numFsiBodies:       " << counters.numFsiBodies << endl;
    cout << "  numGhostMarkers:    " << counters.numGhostMarkers << endl;
    cout << "  numHelperMarkers:   " << counters.numHelperMarkers << endl;
    cout << "  numFluidMarkers:    " << counters.numFluidMarkers << endl;
    cout << "  numBoundaryMarkers: " << counters.numBoundaryMarkers << endl;
    cout << "  numRigidMarkers:    " << counters.numRigidMarkers << endl;
    cout << "  numAllMarkers:      " << counters.numAllMarkers << endl;
    cout << "  startRigidMarkers:  " << counters.startRigidMarkers << endl;
}


void PrintRefArrays(const thrust::host_vector<int4>& referenceArray) {
    cout << "Reference array (size: " << referenceArray.size() << ")" << endl;
    for (size_t i = 0; i < referenceArray.size(); i++) {
        const int4& num = referenceArray[i];
        cout << "  " << i << ": " << num.x << " " << num.y << " " << num.z << " " << num.w << endl;
    }

    cout << endl;
}


//------------------------------------------------------------------------------------------------------------
// Add input Fsi body into the FSI system  (here fluid system part) and create the associated BCE markers
//------------------------------------------------------------------------------------------------------------

// input is pointer to FsiBody (ChFsiDefinitions), not an FsiSphBody (defined in ChFsiFluidSystemSPH class)
// System not to be already initialized
void ChFsiFluidSystemSPH_csph::OnAddFsiBody(std::shared_ptr<FsiBody> fsi_body, bool check_embedded) {
    ChAssertAlways(!m_is_initialized);  // m_is_initialized is defined in base class ChFsiFLuidSystem. Set to True if
                                        // system have been initialized.

    FsiSphBody b;  // save input from FsiBody into FsiSphBody
    b.fsi_body = fsi_body;
    b.check_embedded = check_embedded;

    CreateBCEFsiBody(fsi_body, b.bce_ids, b.bce_coords, b.bce);  // method defined just below, given input FsiBody it computes the bce markers coordinates.
    m_num_rigid_bodies++;

    m_bodies.push_back(b);
}


//// TODO FSI bodies:
////   - give control over Cartesian / polar BCE distribution (where applicable)
////   - eliminate duplicate BCE markers (from multiple volumes). Easiest if BCE created on a grid!


// Create the the local BCE coordinates, their body associations, and the initial global BCE positions for the
// given FSI rigid body. It reads the input FsiBody and creates and stores bce ids and coordinates into the three input
// vectors
void ChFsiFluidSystemSPH_csph::CreateBCEFsiBody(std::shared_ptr<FsiBody> fsi_body,    // read
                                                std::vector<int>& bce_ids,            // Body index of each bce marker
                                                std::vector<ChVector3d>& bce_coords,  // write, relative frame
                                                std::vector<ChVector3d>& bce) {       // write, global frame
    const auto& geometry = fsi_body->geometry;
    if (geometry) {  // geometry is a pointer to ChGeometry. If body doesn't have geometry associated it will be a null
                     // pointer.
        for (const auto& sphere : geometry->coll_spheres) {
            auto points = CreatePointsSphereInterior(sphere.radius, true);  // method of ChFsiFluidSystemSPH
            for (auto& p : points)
                p += sphere.pos;  // add offset = sphere center position
            bce_coords.insert(bce_coords.end(), points.begin(),
                              points.end());  // store local frame coordinates of the bce marker
        }
        for (const auto& box : geometry->coll_boxes) {
            auto points = CreatePointsBoxInterior(box.dims);
            for (auto& p : points)
                p = box.pos + box.rot.Rotate(p);
            bce_coords.insert(bce_coords.end(), points.begin(), points.end());
        }
        for (const auto& cyl : geometry->coll_cylinders) {
            auto points = CreatePointsCylinderInterior(cyl.radius, cyl.length, true);
            for (auto& p : points)
                p = cyl.pos + cyl.rot.Rotate(p);
            bce_coords.insert(bce_coords.end(), points.begin(), points.end());
        }


        // Get initial global BCE positions
        const auto& X_G_R = fsi_body->body->GetFrameRefToAbs();
        std::transform(bce_coords.begin(), bce_coords.end(), std::back_inserter(bce),
                       [&X_G_R](ChVector3d& v) { return X_G_R.TransformPointLocalToParent(v); });

        // Get local BCE coordinates relative to centroidal frame
        bce_coords.clear();
        const auto& X_G_COM = fsi_body->body->GetFrameCOMToAbs();
        std::transform(bce.begin(), bce.end(), std::back_inserter(bce_coords),
                       [&X_G_COM](ChVector3d& v) { return X_G_COM.TransformPointParentToLocal(v); });
        // Set BCE body association
        bce_ids.resize(bce_coords.size(), fsi_body->index);  // second parameter is the value at which new elements of
                                                             // resized vector (if extended) are initialized.
    }
}


//------------------------------------------------------------------------------
// Add the BCE markers for the given FSI rigid body to the underlying data manager.
// Note: BCE markers are created with zero velocities.
// Works with FsiSphBody class, not with FsiBody, so it already contains the bce markers
void ChFsiFluidSystemSPH_csph::AddBCEFsiBody(const FsiSphBody& fsisph_body) {
    const auto& fsi_body = fsisph_body.fsi_body;  // FsiBody object

    // Add BCE markers and load their local coordinates and body associations
    auto num_bce = fsisph_body.bce.size();
    for (size_t i = 0; i < num_bce; i++) {  // AddBceMarker method of FsiDataManager. That method does not update any counter
        m_data_mgr->AddBceMarker(MarkerType::BCE_RIGID, ToReal3(fsisph_body.bce[i]),
            {0, 0, 0});  // This adds the bce markers to the global vectors of all markers posRadH, valMasH, rhoPreMuH
        m_data_mgr->rigid_BCEcoords_H.push_back(ToReal3(fsisph_body.bce_coords[i]));
        m_data_mgr->rigid_BCEsolids_H.push_back(fsisph_body.bce_ids[i]);
    }

    m_fsi_bodies_bce_num.push_back((int)num_bce);

    if (m_verbose) {
        cout << "Add BCE for rigid body" << endl;
        cout << "  Num. BCE markers: " << num_bce << endl;
    }
}


//---------------------------------------------------------------------------------------
// Initialize the sph fluid system with fsi support. Some data should already be present.
//---------------------------------------------------------------------------------------

void ChFsiFluidSystemSPH_csph::Initialize(const std::vector<FsiBodyState>& body_states) {
    // create empty mesh objects
    std::vector<FsiMeshState> empty_mesh;

    // Call virtual method
    Initialize(body_states, empty_mesh, empty_mesh);
}



void ChFsiFluidSystemSPH_csph::Initialize(const std::vector<FsiBodyState>& body_states,
                                     const std::vector<FsiMeshState>& empty_mesh1,
                                     const std::vector<FsiMeshState>& empty_mesh2) {
    assert(body_states.size() == m_bodies.size());

    // Process FSI solids - load BCE data to data manager
    // Note: counters must be regenerated, as they are used as offsets for global indices
    m_num_rigid_bodies = 0;
    // add bce markers from the FsiSphBody stored in m_bodies to the marker vector in pointed FsiDataManager
    for (const auto& b : m_bodies) {  // m_bodies vector of FsiSphBody objets in the FluidSystem class
        AddBCEFsiBody(b);             // adds bces to the underlying FsiDataManager
        m_num_rigid_bodies++;
    }


    // ----------------

    // Hack to still allow time step size specified through JSON files
    if (m_paramsH->dT < 0) {
        m_paramsH->dT = GetStepSize();
    } else {
        SetStepSize(m_paramsH->dT);
    }

    // Set kernel radius factor (based on kernel type)
    switch (m_paramsH->kernel_type) {
        case KernelType_csph::CUBIC_SPLINE:
            m_paramsH->h_multiplier = 2;
            break;
        case KernelType_csph::QUINTIC_SPLINE:
            m_paramsH->h_multiplier = 3;
            break;
        case KernelType_csph::WENDLAND:
            m_paramsH->h_multiplier = 2;
            break;
    }

    // Initialize SPH particle mass and number of neighbors
    
    {  // exploring a grid centered in origin, with length [-IDX,..,IDX] along each axis.
       // This is actually a sample grid, does not traverse all tha actual grid, and does not consider actual particles
       // but computes some quantities by considering the interaction between different grid cells only
       // Process done only once and at very beginning of the problem, so it's ok to use the predefined values for h, d0 etc.
        Real sum = 0;
        int count = 0;
        int IDX = 10;
        for (int i = -IDX; i <= IDX; i++) {
            for (int j = -IDX; j <= IDX; j++) {
                for (int k = -IDX; k <= IDX; k++) {
                    // for each element in the grid we compute the "position" multiplying by d0 (initial separation of
                    // particles)
                    Real3 pos = mR3(i, j, k) * m_paramsH->d0;
                    Real W = W3h_csph(m_paramsH->kernel_type, length(pos),m_paramsH->ooh,m_paramsH->num_dim);  // compute the kernel value wrt the origin point, so W_ij with i being (0,0,0)
                    sum += W;             // accumulates W_ij
                    if (W > 0)
                        count++;  // If kernel positive (only way is distance < support radius) we register j as
                                  // "active"
                }
            }
        }
        if (m_paramsH->is_uniform && m_paramsH->markerMass == -1)      // If the problem is initially uniform and the mass has not been set:
            m_paramsH->markerMass = m_paramsH->rho0 / sum;         // compute mass as rho0/sum(W_ij) so ensuring initial density is rho0
        m_paramsH->num_neighbors = count;  // estimates the number of neighbors for each cell 
    }


    if (m_paramsH->use_init_pressure) {                                     // true if initial pressure based on height.
        size_t numParticles = m_data_mgr->sphMarkers_H->rhoPresEnH.size();  // number of all sph markers
        for (int i = 0; i < numParticles; i++) {
            double z = m_data_mgr->sphMarkers_H->posRadH[i].z;  // get z position of each marker
            double p = m_paramsH->rho0 * m_paramsH->gravity.z *
                       (z - m_paramsH->pressure_height);    // pressure0 = density0*g_z*(z - pressure_height)
            m_data_mgr->sphMarkers_H->rhoPresEnH[i].y = p;  // save the computed initial pressure
        }
    }

    // ----------------

    // This means boundaries of fluid domain (cMin/Max) have not been set by users so cMin-cMax have default values  -
    // just use an approximate domain size with no periodic sides based on the domain limits (boxDims)
    if (m_paramsH->use_default_limits) {
        m_paramsH->cMin = mR3(-2 * m_paramsH->boxDimX, -2 * m_paramsH->boxDimY, -2 * m_paramsH->boxDimZ) - 10 * mR3(m_paramsH->h);
        m_paramsH->cMax = mR3(+2 * m_paramsH->boxDimX, +2 * m_paramsH->boxDimY, +2 * m_paramsH->boxDimZ) + 10 * mR3(m_paramsH->h);
        m_paramsH->bc_type = BC_NONE;
    }
    // set parameters x/y/z_periodic depending on value of bc_type
    m_paramsH->x_periodic = m_paramsH->bc_type.x == BCType::PERIODIC;
    m_paramsH->y_periodic = m_paramsH->bc_type.y == BCType::PERIODIC;
    m_paramsH->z_periodic = m_paramsH->bc_type.z == BCType::PERIODIC;

    // Set up subdomains and grid parameters for faster neighbor particle search
    // Ok to use default h parameter as this is the initialization phase.

    m_paramsH->Apply_BC_U = false;  // option allows to set up velocity boundary conditions for sph markers
    // if a grid cell has size of about 1 or 2 kernel radius, side0 represents how many cells we have along a direction:
    int3 side0 = mI3((int)floor((m_paramsH->cMax.x - m_paramsH->cMin.x) / (m_paramsH->h_multiplier * m_paramsH->h)),
                     (int)floor((m_paramsH->cMax.y - m_paramsH->cMin.y) / (m_paramsH->h_multiplier * m_paramsH->h)),
                     (int)floor((m_paramsH->cMax.z - m_paramsH->cMin.z) / (m_paramsH->h_multiplier * m_paramsH->h)));
    // fixing number of cells (being side0) for each direction, binsize3 tells the sizes of each cell. 
    Real3 binSize3 = mR3((m_paramsH->cMax.x - m_paramsH->cMin.x) / side0.x, (m_paramsH->cMax.y - m_paramsH->cMin.y) / side0.y,
                     (m_paramsH->cMax.z - m_paramsH->cMin.z) / side0.z);

    // following lines are original code. Modification to take the maximum cell dimension possible.
    // m_paramsH->binSize0 = (binSize3.x > binSize3.y) ? binSize3.x : binSize3.y;
    // m_paramsH->binSize0 = binSize3.x;             // why redefine the same parameter as above??

    m_paramsH->binSize0 = (binSize3.x > binSize3.y) ? binSize3.x : binSize3.y;  // take maximum between x and y
    //m_paramsH->binSize0 = (m_paramsH->binSize0 > binSize3.z) ? m_paramsH->binSize0 : binSize3.z;  // maximum between z and previous value
    m_paramsH->binSize0 = binSize3.x;
    m_paramsH->boxDims = m_paramsH->cMax - m_paramsH->cMin;
    m_paramsH->delta_pressure = mR3(0);
    // fixing the length of the cell (cubic here) as being binsize0, SIDE tells how many cells along each direction 
    // of the computational domain.
    int3 SIDE = mI3(int((m_paramsH->cMax.x - m_paramsH->cMin.x) / m_paramsH->binSize0 + .1),
                    int((m_paramsH->cMax.y - m_paramsH->cMin.y) / m_paramsH->binSize0 + .1),
                    int((m_paramsH->cMax.z - m_paramsH->cMin.z) / m_paramsH->binSize0 + .1));
    Real mBinSize = m_paramsH->binSize0;
    m_paramsH->gridSize = SIDE;                                 // number of cells along each domain direction
    m_paramsH->worldOrigin = m_paramsH->cMin;                   // world origin is lower limit of domain
    m_paramsH->cellSize = mR3(mBinSize, mBinSize, mBinSize);    // dimension of the single cell (cubic)

    // Precompute grid min and max bounds considering whether we have periodic boundaries or not
    m_paramsH->minBounds = make_int3(m_paramsH->x_periodic ? INT_MIN : 0, m_paramsH->y_periodic ? INT_MIN : 0,
                                     m_paramsH->z_periodic ? INT_MIN : 0);

    m_paramsH->maxBounds = make_int3(m_paramsH->x_periodic ? INT_MAX : m_paramsH->gridSize.x - 1,
                                     m_paramsH->y_periodic ? INT_MAX : m_paramsH->gridSize.y - 1,
                                     m_paramsH->z_periodic ? INT_MAX : m_paramsH->gridSize.z - 1);


    // ----------------
    // Initialize the data manager: set reference arrays, set counters, and resize simulation arrays.
    m_data_mgr->Initialize(m_num_rigid_bodies);

    // ----------------

    // Load the initial body states
    ChDebugLog("load initial states");
    LoadSolidStates(body_states);  // given input FsiBodyStates, calls the method LoadSolidStates that stores the
                                     // states (pos,vel,force, etc) of the i-th body in the corresponding FsiBodyState_H
                                     // vector in FsiDataManager

    // Create BCE and SPH worker objects (ChFluidSystem has pointer to BceManager and FluidDynamics objects)
    // Here make unique pointers and call constructors of the classes
    m_bce_mgr = chrono_types::make_unique<BceManager_csph>(*m_data_mgr, m_verbose, m_check_errors);
    m_fluid_dynamics = chrono_types::make_unique<FluidDynamics_csph>(*m_data_mgr, *m_bce_mgr, m_verbose, m_check_errors);

    // Initialize worker objects
    m_bce_mgr->Initialize(m_fsi_bodies_bce_num);
    m_fluid_dynamics->Initialize();

    /// If active domains are not used then don't overly resize the arrays
    if (!m_paramsH->use_active_domain) {
        m_data_mgr->SetGrowthFactor(1.0f);
    }
    
    // Check if GPU is available and initialize CUDA device information
    int device;
    cudaGetDevice(&device);
    cudaCheckError();
    m_data_mgr->cudaDeviceInfo->deviceID = device;
    cudaGetDeviceProperties(&m_data_mgr->cudaDeviceInfo->deviceProp, m_data_mgr->cudaDeviceInfo->deviceID);
    cudaCheckError();

    // ----------------
    // Check fluid and Sph parameters
    CheckSPHParameters();
    CheckFluidParameters();


    if (m_verbose) {
        PrintDeviceProperties(m_data_mgr->cudaDeviceInfo->deviceProp);
        PrintParams(*m_paramsH, *m_data_mgr->countersH);
        PrintRefArrays(m_data_mgr->referenceArray);
    }

    // first value of variable step vectors is always the input step size
    m_time_steps.push_back(m_step);
    m_courant_steps.push_back(m_step);
    m_force_steps.push_back(m_step);
}



double ChFsiFluidSystemSPH_csph::GetVariableStepSize() {
    // Variable time step requires the state from the previous time step.
    // Thus, it cannot directly be used in the first time step.
    if (m_paramsH->use_variable_time_step && m_frame != 0) {
        double3 timeStep = m_fluid_dynamics->computeTimeStep();
        // save data in arrays
        m_time_steps.push_back(timeStep.x);
        m_courant_steps.push_back(timeStep.y);
        m_force_steps.push_back(timeStep.z);
        return timeStep.x;
    } else {
        return GetStepSize();
    }
}


//------------------------------------------------------------------------------
// Integrate the fluid system from "time" to "time + step"
void ChFsiFluidSystemSPH_csph::OnDoStepDynamics(double time, double step) {
    
    // Update particle activity (to mark active ones) with proper method in FluidDynamics class
    m_fluid_dynamics->UpdateActivity(m_data_mgr->sphMarkers_D);
    
    // Resize arrays if needed
    // M_FRAME DEFINED IN CHFSIFLUIDSYSTEM base class. unsigned int for the current simulation frame
    bool resize_arrays = m_fluid_dynamics->CheckActivityArrayResize();
    if (m_frame == 0 || resize_arrays) {
        m_data_mgr->ResizeArrays(m_data_mgr->countersH->numExtendedParticles);
    }

    // Perform proximity search if needed by call to FluidDynamics method (FluidDynamics has pointer to CollisionSystem
    // obj)
    bool proximity_search = m_frame % m_paramsH->num_proximity_search_steps == 0 || m_force_proximity_search;
    if (proximity_search) {
        m_fluid_dynamics->ProximitySearch();
    }
    cudaCheckError();
    // Zero-out step data (derivatives and intermediate vectors)
    // At beginning of a step it resets to zero the device vectors, no resize.
    m_data_mgr->ResetData();

    cudaCheckError();
    // Advance fluid particle states from `time` to `time+step`
    m_fluid_dynamics->DoStepDynamics(m_data_mgr->sortedSphMarkers2_D, time, step, m_paramsH->integration_scheme);
    // Dynamics advanced on sorted arrays, copy new states in the original order (copy done on device vectors)
    m_fluid_dynamics->CopySortedToOriginal(MarkerGroup::NON_SOLID, m_data_mgr->sortedSphMarkers2_D,
                                           m_data_mgr->sphMarkers_D);

    
    ChDebugLog("GPU Memory usage: " << m_data_mgr->GetCurrentGPUMemoryUsage() / 1024.0 / 1024.0 << " MB");

    // Reset flag for forcing a proximity search
    m_force_proximity_search = false;

    
}



// Additional actions taken before applying fluid forces to the solid phase.
void ChFsiFluidSystemSPH_csph::OnExchangeSolidForces() {
    m_bce_mgr->Rigid_Forces_Torques();  // method of BceManager. Computes fluid forces acting on solid bodies. results
                                        // stored in FsiDataManager D vectors.

}

// Additional actions taken after loading new solid phase states
void ChFsiFluidSystemSPH_csph::OnExchangeSolidStates() {
    if (m_num_rigid_bodies == 0)
        return;  // no solid bodies

    m_bce_mgr->UpdateBodyMarkerState();  // Update position and velocity of BCE markers on rigid solids

    // Copies solid markers states from sorted order into original order
    m_fluid_dynamics->CopySortedToOriginal(MarkerGroup::SOLID, m_data_mgr->sortedSphMarkers2_D,
                                           m_data_mgr->sphMarkers_D);
}


//------------------------------------------------------------------------------
// Utility functions to save data.
// They just call the corresponding functions in UtilsPrintSPH_compressible
//------------------------------------------------------------------------------
void ChFsiFluidSystemSPH_csph::WriteParticleFile(const std::string& filename) const {
    writeParticleFileCSV_csph(filename, *m_data_mgr);
}

void ChFsiFluidSystemSPH_csph::SaveParticleData(const std::string& dir) const {

        saveParticleDataCFD_csph(dir, m_output_level, *m_data_mgr);  // function in UtilsPrintSPH
}

void ChFsiFluidSystemSPH_csph::SaveSolidData(const std::string& dir, double time) const {
    saveSolidData_csph(dir, time, *m_data_mgr);
}

void ChFsiFluidSystemSPH_csph::SaveTimeSteps(const std::string& dir) const {
    if (m_paramsH->use_variable_time_step) {
        saveVariableTimeStep_csph(dir, m_time_steps, m_courant_steps, m_force_steps);
    } else {
        std::cout << "Variable step time not used in this problem. File will not be generated." << std::endl;
    }
}


//------------------------------------------------------------------------------
// Add Sph particles with given properties to the FSI system
//------------------------------------------------------------------------------
// Calls the method in underlying FsiDataManager. Notice it adds the particle but doesn't update any counter.
void ChFsiFluidSystemSPH_csph::AddSPHParticle(const ChVector3d& pos,
                                              double rho,
                                              double pres,
                                              double en,
                                              const ChVector3d& vel,
                                              const double h,
                                              const double mass) {
    m_data_mgr->AddSphParticle(ToReal3(pos), rho, pres, en, ToReal3(vel), h, mass);
}
// this version uses default rho, pres, en
void ChFsiFluidSystemSPH_csph::AddSPHParticle(const ChVector3d& pos,
                                              const ChVector3d& vel) {
    AddSPHParticle(pos, m_paramsH->rho0, m_paramsH->p0, m_paramsH->e0, vel, m_paramsH->h, m_paramsH->markerMass);
}


// Create SPH particles in the specified box volume.
// The SPH particles are created on a uniform grid with resolution equal to the FSI initial separation.
void ChFsiFluidSystemSPH_csph::AddBoxSPH(const ChVector3d& boxCenter, const ChVector3d& boxHalfDim) {
    // Use a chrono sampler to create a bucket of points
    utils::ChGridSampler<> sampler(m_paramsH->d0);
    std::vector<ChVector3d> points = sampler.SampleBox(boxCenter, boxHalfDim);

    // Add fluid particles from the sampler points to the FSI system
    // Added with 0 pressure and velocity
    int numPart = (int)points.size();
    for (int i = 0; i < numPart; i++) {
        AddSPHParticle(points[i], m_paramsH->rho0, 0, m_paramsH->e0,
                       ChVector3d(0),   // initial velocity
                       m_paramsH->h);  
    }
}


//------------------------------------------------------------------------------
// Add WALL BCE markers at the specified points.
// The points are assumed to be provided relative to the specified frame.
// These BCE markers are not associated with a particular FSI body and, as such, cannot be used to extract fluid
// forces and moments. If fluid reaction forces are needed, create an FSI body with the desirted geometry or list
// of BCE points and add it through the containing FSI system.
void ChFsiFluidSystemSPH_csph::AddBCEBoundary(const std::vector<ChVector3d>& points, const ChFramed& frame) {
    for (const auto& p : points)
        m_data_mgr->AddBceMarker(MarkerType::BCE_WALL, ToReal3(frame.TransformPointLocalToParent(p)), {0, 0, 0});
}

// similar as above but pass also h and mass:
void ChFsiFluidSystemSPH_csph::AddBCEBoundary(const std::vector<ChVector3d>& points,
                                              const ChFramed& frame,
                                              const double h,
                                              const double mass) {

    for (const auto& p : points) {
        m_data_mgr->AddBceMarker(MarkerType::BCE_WALL, ToReal3(frame.TransformPointLocalToParent(p)), {0, 0, 0},
                                 h, mass);
    }
}

// similar as above but pass also h and mass for each particle:
void ChFsiFluidSystemSPH_csph::AddBCEBoundary(const std::vector<ChVector3d>& points,
                                              const ChFramed& frame,
                                              const std::vector<double>& h_vec,
                                              const std::vector<double>& mass_vec) {

    assert(points.size() == h_vec.size());
    assert(points.size() == mass_vec.size());
    int i = 0;
    for (const auto& p : points) {
        m_data_mgr->AddBceMarker(MarkerType::BCE_WALL, ToReal3(frame.TransformPointLocalToParent(p)), {0, 0, 0},
                                 h_vec[i], mass_vec[i]);
        i++;
    }
}

//------------------------------------------------------------------------------
// Set of functions that create points (ChVector3d) in different volumes. Number of layers inside these volumes depend
// on system parameters The points given as output then must be added to the FSI system with other methods.
//------------------------------------------------------------------------------

const Real pi = Real(CH_PI);

// Create marker points on a rectangular plate of specified X-Y dimensions, assumed centered at the origin.
// Markers are created in a number of layers (in the negative Z direction) corresponding to system parameters.
std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsPlate(const ChVector2d& size) const {
    std::vector<ChVector3d> bce;

    Real spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Calculate actual spacing in x-y directions with proper roundings
    ChVector2d hsize = size / 2;  // half size
    int2 np = {(int)std::round(hsize.x() / spacing),
               (int)std::round(hsize.y() / spacing)};         // round returns the integral closest to the given number
    ChVector2d delta = {hsize.x() / np.x, hsize.y() / np.y};  // actual delta/spacing used in x-y after the rounding

    for (int il = 0; il < num_layers; il++) {
        for (int ix = -np.x; ix <= np.x; ix++) {
            for (int iy = -np.y; iy <= np.y; iy++) {
                bce.push_back(
                    {ix * delta.x(), iy * delta.y(), -il * spacing});  // spacing in z direction is the original one
            }
        }
    }

    return bce;
}


// Create marker points for a box container of specified dimensions.
// The box volume is assumed to be centered at the origin. The 'faces' input vector specifies which faces of the
// container are to be created: for each direction, a value of -1 indicates the face in the negative direction, a
// value of +1 indicates the face in the positive direction, and a value of 2 indicates both faces. Setting a value
// of 0 does not create container faces in that direction. Markers are created in a number of layers corresponding
// to system parameters.
// Bce markers created in layers that extend outwards.
std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsBoxContainer(const ChVector3d& size,
                                                                           const ChVector3i& faces) const {
    std::vector<ChVector3d> bce;

    Real spacing = m_paramsH->d0;
    // Buffer used in X and Y plates to extend distance in order to provide full bce cover
    // of the box corners.
    Real buffer = 2 * (m_paramsH->num_bce_layers - 1) * spacing;

    ChVector3d hsize = size / 2;
    // direction vectors associated to each face
    ChVector3d xn(-hsize.x(), 0, 0);
    ChVector3d xp(+hsize.x(), 0, 0);
    ChVector3d yn(0, -hsize.y(), 0);
    ChVector3d yp(0, +hsize.y(), 0);
    ChVector3d zn(0, 0, -hsize.z());
    ChVector3d zp(0, 0, +hsize.z());

    // Z- wall. (layers built correctly along -z direction)
    if (faces.z() == -1 || faces.z() == 2) {
        // create points over a plate of x and y dimensions along the negative z axis, number of layers from system parameters.
        auto bce1 = CreatePointsPlate({size.x(), size.y()});  // call previous function. returns vector of ChVector3d
        ChFramed X(zn, QUNIT);                                // define the actual frame direction
        // back_inserter(bce) returns an iterator that inserts element at the end of bce vector.
        std::transform(bce1.begin(), bce1.end(), std::back_inserter(bce),
                       [&X](ChVector3d& v) { return X * v; });  // transform markers coordinates. No rotation but traslation by vector zn.
    }

    // Z+ wall
    if (faces.z() == +1 || faces.z() == 2) {
        auto bce1 = CreatePointsPlate({size.x(), size.y()});
        ChFramed X(zp, QuatFromAngleX(CH_PI));
        std::transform(bce1.begin(), bce1.end(), std::back_inserter(bce), [&X](ChVector3d& v) { return X * v; });  // here also rotate points.
    }

    // X- wall
    if (faces.x() == -1 || faces.x() == 2) {
        auto bce1 = CreatePointsPlate({size.z() + buffer, size.y()});
        ChFramed X(xn, QuatFromAngleY(+CH_PI_2));
        std::transform(bce1.begin(), bce1.end(), std::back_inserter(bce), [&X](ChVector3d& v) { return X * v; });
    }

    // X+ wall
    if (faces.x() == +1 || faces.x() == 2) {
        auto bce1 = CreatePointsPlate({size.z() + buffer, size.y()});
        ChFramed X(xp, QuatFromAngleY(-CH_PI_2));
        std::transform(bce1.begin(), bce1.end(), std::back_inserter(bce), [&X](ChVector3d& v) { return X * v; });
    }

    // Y- wall
    if (faces.y() == -1 || faces.y() == 2) {
        auto bce1 = CreatePointsPlate({size.x() + buffer, size.z() + buffer});
        ChFramed X(yn, QuatFromAngleX(-CH_PI_2));
        std::transform(bce1.begin(), bce1.end(), std::back_inserter(bce), [&X](ChVector3d& v) { return X * v; });
    }

    // Y+ wall
    if (faces.y() == +1 || faces.y() == 2) {
        auto bce1 = CreatePointsPlate({size.x() + buffer, size.z() + buffer});
        ChFramed X(yp, QuatFromAngleX(+CH_PI_2));
        std::transform(bce1.begin(), bce1.end(), std::back_inserter(bce), [&X](ChVector3d& v) { return X * v; });
    }

    return bce;
}


std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsBoxInterior(const ChVector3d& size) const {
    std::vector<ChVector3d> bce;

    Real spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Decide if any direction is small enough to be filled
    ChVector3<int> np(1 + (int)std::round(size.x() / spacing),  //
                      1 + (int)std::round(size.y() / spacing),  //
                      1 + (int)std::round(size.z() / spacing));
    bool fill = np[0] <= 2 * num_layers || np[1] <= 2 * num_layers || np[2] <= 2 * num_layers;

    // Adjust spacing in each direction
    ChVector3d delta(size.x() / (np.x() - 1), size.y() / (np.y() - 1), size.z() / (np.z() - 1));

    // If any direction must be filled, the box must be filled
    if (fill) {
        for (int ix = 0; ix < np.x(); ix++) {
            double x = -size.x() / 2 + ix * delta.x();
            for (int iy = 0; iy < np.y(); iy++) {
                double y = -size.y() / 2 + iy * delta.y();
                for (int iz = 0; iz < np.z(); iz++) {
                    double z = -size.z() / 2 + iz * delta.z();
                    bce.push_back({x, y, z});
                }
            }
        }
        return bce;
    }

    // Create interior BCE layers
    for (int il = 0; il < num_layers; il++) {
        // x faces
        double xm = -size.x() / 2 + il * delta.x();
        double xp = +size.x() / 2 - il * delta.x();
        for (int iy = 0; iy < np.y(); iy++) {
            double y = -size.y() / 2 + iy * delta.y();
            for (int iz = 0; iz < np.z(); iz++) {
                double z = -size.z() / 2 + iz * delta.z();
                bce.push_back({xm, y, z});
                bce.push_back({xp, y, z});
            }
        }

        // y faces
        double ym = -size.y() / 2 + il * delta.y();
        double yp = +size.y() / 2 - il * delta.y();
        for (int iz = 0; iz < np.z(); iz++) {
            double z = -size.z() / 2 + iz * delta.z();
            for (int ix = 0; ix < np.x(); ix++) {
                double x = -size.x() / 2 + ix * delta.x();
                bce.push_back({x, ym, z});
                bce.push_back({x, yp, z});
            }
        }

        // z faces
        double zm = -size.z() / 2 + il * delta.z();
        double zp = +size.z() / 2 - il * delta.z();
        for (int ix = 0; ix < np.x(); ix++) {
            double x = -size.x() / 2 + ix * delta.x();
            for (int iy = 0; iy < np.y(); iy++) {
                double y = -size.y() / 2 + iy * delta.y();
                bce.push_back({x, y, zm});
                bce.push_back({x, y, zp});
            }
        }
    }

    return bce;
}



std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsBoxExterior(const ChVector3d& size) const {
    std::vector<ChVector3d> bce;

    Real spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Adjust spacing in each direction
    ChVector3i np(1 + (int)std::round(size.x() / spacing),  //
                  1 + (int)std::round(size.y() / spacing),  //
                  1 + (int)std::round(size.z() / spacing));
    ChVector3d delta(size.x() / (np.x() - 1), size.y() / (np.y() - 1), size.z() / (np.z() - 1));

    // Inflate box
    ChVector3i Np = np + 2 * (num_layers - 1);
    ChVector3d Size = size + 2.0 * (num_layers - 1.0) * delta;

    // Create exterior BCE layers
    for (int il = 0; il < num_layers; il++) {
        // x faces
        double xm = -Size.x() / 2 + il * delta.x();
        double xp = +Size.x() / 2 - il * delta.x();
        for (int iy = 0; iy < Np.y(); iy++) {
            double y = -Size.y() / 2 + iy * delta.y();
            for (int iz = 0; iz < Np.z(); iz++) {
                double z = -Size.z() / 2 + iz * delta.z();
                bce.push_back({xm, y, z});
                bce.push_back({xp, y, z});
            }
        }

        // y faces
        double ym = -Size.y() / 2 + il * delta.y();
        double yp = +Size.y() / 2 - il * delta.y();
        for (int iz = 0; iz < Np.z(); iz++) {
            double z = -Size.z() / 2 + iz * delta.z();
            for (int ix = 0; ix < np.x(); ix++) {
                double x = -size.x() / 2 + ix * delta.x();
                bce.push_back({x, ym, z});
                bce.push_back({x, yp, z});
            }
        }

        // z faces
        double zm = -Size.z() / 2 + il * delta.z();
        double zp = +Size.z() / 2 - il * delta.z();
        for (int ix = 0; ix < np.x(); ix++) {
            double x = -size.x() / 2 + ix * delta.x();
            for (int iy = 0; iy < np.y(); iy++) {
                double y = -size.y() / 2 + iy * delta.y();
                bce.push_back({x, y, zm});
                bce.push_back({x, y, zp});
            }
        }
    }

    return bce;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsSphereInterior(double radius, bool polar) const {
    std::vector<ChVector3d> bce;

    double spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Use polar coordinates
    if (polar) {
        double rad_in = radius - (num_layers - 1) * spacing;

        for (int ir = 0; ir < num_layers; ir++) {
            double r = rad_in + ir * spacing;
            int np_phi = (int)std::round(pi * r / spacing);
            double delta_phi = pi / np_phi;
            for (int ip = 0; ip < np_phi; ip++) {
                double phi = ip * delta_phi;
                double cphi = std::cos(phi);
                double sphi = std::sin(phi);
                double x = r * sphi;
                double y = r * sphi;
                double z = r * cphi;
                int np_th = (int)std::round(2 * pi * r * sphi / spacing);
                double delta_th = (np_th > 0) ? (2 * pi) / np_th : 1;
                for (int it = 0; it < np_th; it++) {
                    double theta = it * delta_th;
                    bce.push_back({x * std::cos(theta), y * std::sin(theta), z});
                }
            }
        }

        return bce;
    }

    // Use a Cartesian grid and accept/reject points
    int np = (int)std::round(radius / spacing);
    double delta = radius / np;

    for (int iz = 0; iz <= np; iz++) {
        double z = iz * delta;
        double rz_max = std::sqrt(radius * radius - z * z);
        double rz_min = std::max(rz_max - num_layers * delta, 0.0);
        if (iz >= np - num_layers)
            rz_min = 0;
        double rz_min2 = rz_min * rz_min;
        double rz_max2 = rz_max * rz_max;
        int nq = (int)std::round(rz_max / spacing);
        for (int ix = -nq; ix <= nq; ix++) {
            double x = ix * delta;
            for (int iy = -nq; iy <= nq; iy++) {
                double y = iy * delta;
                double r2 = x * x + y * y;
                if (r2 >= rz_min2 && r2 <= rz_max2) {
                    bce.push_back({x, y, +z});
                    bce.push_back({x, y, -z});
                }
            }
        }
    }

    return bce;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsSphereExterior(double radius, bool polar) const {
    std::vector<ChVector3d> bce;

    double spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Use polar coordinates
    if (polar) {
        for (int ir = 0; ir < num_layers; ir++) {
            double r = radius + ir * spacing;
            int np_phi = (int)std::round(pi * r / spacing);
            double delta_phi = pi / np_phi;
            for (int ip = 0; ip < np_phi; ip++) {
                double phi = ip * delta_phi;
                double cphi = std::cos(phi);
                double sphi = std::sin(phi);
                double x = r * sphi;
                double y = r * sphi;
                double z = r * cphi;
                int np_th = (int)std::round(2 * pi * r * sphi / spacing);
                double delta_th = (np_th > 0) ? (2 * pi) / np_th : 1;
                for (int it = 0; it < np_th; it++) {
                    double theta = it * delta_th;
                    bce.push_back({x * std::cos(theta), y * std::sin(theta), z});
                }
            }
        }

        return bce;
    }

    // Inflate sphere and accept/reject points on a Cartesian grid
    int np = (int)std::round(radius / spacing);
    double delta = radius / np;
    np += num_layers;
    radius += num_layers * delta;

    for (int iz = 0; iz <= np; iz++) {
        double z = iz * delta;
        double rz_max = std::sqrt(radius * radius - z * z);
        double rz_min = std::max(rz_max - num_layers * delta, 0.0);
        if (iz >= np - num_layers)
            rz_min = 0;
        double rz_min2 = rz_min * rz_min;
        double rz_max2 = radius * radius - z * z;
        for (int ix = -np; ix <= np; ix++) {
            double x = ix * delta;
            for (int iy = -np; iy <= np; iy++) {
                double y = iy * delta;
                double r2 = x * x + y * y;
                if (r2 >= rz_min2 && r2 <= rz_max2) {
                    bce.push_back({x, y, +z});
                    bce.push_back({x, y, -z});
                }
            }
        }
    }

    return bce;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsCylinderInterior(double rad, double height, bool polar) const {
    std::vector<ChVector3d> bce;

    double spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Radial direction (num divisions and adjusted spacing)
    int np_r = (int)std::round(rad / spacing);
    double delta_r = rad / np_r;

    // Axial direction (num divisions and adjusted spacing)
    int np_h = (int)std::round(height / spacing);
    double delta_h = height / np_h;

    // If the radius or height are too small, fill the entire cylinder
    bool fill = (np_r <= num_layers - 1) || (np_h <= 2 * num_layers - 1);

    // Use polar coordinates
    if (polar) {
        double rad_min = std::max(rad - (num_layers - 1) * spacing, 0.0);
        if (fill)
            rad_min = 0;
        np_r = (int)std::round((rad - rad_min) / spacing);
        delta_r = (rad - rad_min) / np_r;

        for (int ir = 0; ir <= np_r; ir++) {
            double r = rad_min + ir * delta_r;
            int np_th = std::max((int)std::round(2 * pi * r / spacing) - 1, 1);
            double delta_th = CH_2PI / np_th;
            for (int it = 0; it < np_th; it++) {
                double theta = it * delta_th;
                double x = r * cos(theta);
                double y = r * sin(theta);
                for (int iz = 0; iz <= np_h; iz++) {
                    double z = -height / 2 + iz * delta_h;
                    bce.push_back({x, y, z});
                }
            }
        }

        // Add cylinder caps (unless already filled)
        if (!fill) {
            np_r = (int)std::round(rad_min / spacing);
            delta_r = rad_min / np_r;

            for (int ir = 0; ir < np_r; ir++) {
                double r = ir * delta_r;
                int np_th = std::max((int)std::round(2 * pi * r / spacing) - 1, 1);
                double delta_th = CH_2PI / np_th;
                for (int it = 0; it < np_th; it++) {
                    double theta = it * delta_th;
                    double x = r * cos(theta);
                    double y = r * sin(theta);
                    for (int iz = 0; iz < num_layers; iz++) {
                        double z = height / 2 - iz * delta_h;
                        bce.push_back({x, y, -z});
                        bce.push_back({x, y, +z});
                    }
                }
            }
        }

        return bce;
    }

    // Use a Cartesian grid and accept/reject points
    double r_max = rad;
    double r_min = std::max(rad - num_layers * delta_r, 0.0);
    if (fill)
        r_min = 0;
    double r_max2 = r_max * r_max;
    double r_min2 = r_min * r_min;
    for (int ix = -np_r; ix <= np_r; ix++) {
        double x = ix * delta_r;
        for (int iy = -np_r; iy <= np_r; iy++) {
            double y = iy * delta_r;
            double r2 = x * x + y * y;
            if (r2 >= r_min2 && r2 <= r_max2) {
                for (int iz = 0; iz <= np_h; iz++) {
                    double z = -height / 2 + iz * delta_h;
                    bce.push_back({x, y, z});
                }
            }
            // Add cylinder caps (unless already filled)
            if (!fill && r2 < r_min2) {
                for (int iz = 0; iz < num_layers; iz++) {
                    double z = height / 2 - iz * delta_h;
                    bce.push_back({x, y, -z});
                    bce.push_back({x, y, +z});
                }
            }
        }
    }

    return bce;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsCylinderExterior(double rad, double height, bool polar) const {
    std::vector<ChVector3d> bce;

    double spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Calculate actual spacing and inflate cylinder
    int np_h = (int)std::round(height / spacing);
    double delta_h = height / np_h;
    np_h += 2 * (num_layers - 1);
    height += 2 * (num_layers - 1) * delta_h;

    // Use polar coordinates
    if (polar) {
        double rad_max = rad + num_layers * spacing;
        double rad_min = rad_max - num_layers * spacing;
        int np_r = (int)std::round((rad_max - rad_min) / spacing);
        double delta_r = (rad_max - rad_min) / np_r;

        for (int ir = 0; ir <= np_r; ir++) {
            double r = rad_min + ir * delta_r;
            int np_th = (int)std::round(2 * pi * r / spacing);
            double delta_th = (np_th > 0) ? (2 * pi) / np_th : 1;
            for (int it = 0; it < np_th; it++) {
                double theta = it * delta_th;
                double x = r * cos(theta);
                double y = r * sin(theta);
                for (int iz = 0; iz <= np_h; iz++) {
                    double z = -height / 2 + iz * delta_h;
                    bce.push_back({x, y, z});
                }
            }
        }

        // Add cylinder caps
        np_r = (int)std::round(rad_min / spacing);
        delta_r = rad_min / np_r;

        for (int ir = 0; ir < np_r; ir++) {
            double r = ir * delta_r;
            int np_th = std::max((int)std::round(2 * pi * r / spacing), 1);
            double delta_th = (2 * pi) / np_th;
            for (int it = 0; it < np_th; it++) {
                double theta = it * delta_th;
                double x = r * cos(theta);
                double y = r * sin(theta);
                for (int iz = 0; iz < num_layers; iz++) {
                    double z = -height / 2 + iz * delta_h;
                    bce.push_back({x, y, -z});
                    bce.push_back({x, y, +z});
                }
            }
        }

        return bce;
    }

    // Inflate cylinder and accept/reject points on a Cartesian grid
    int np_r = (int)std::round(rad / spacing);
    double delta_r = rad / np_r;
    np_r += num_layers;
    rad += num_layers * delta_r;

    double rad_max = rad;
    double rad_min = std::max(rad - num_layers * delta_r, 0.0);
    double r_max2 = rad_max * rad_max;
    double r_min2 = rad_min * rad_min;
    for (int ix = -np_r; ix <= np_r; ix++) {
        double x = ix * delta_r;
        for (int iy = -np_r; iy <= np_r; iy++) {
            double y = iy * delta_r;
            double r2 = x * x + y * y;
            if (r2 >= r_min2 && r2 <= r_max2) {
                for (int iz = 0; iz <= np_h; iz++) {
                    double z = -height / 2 + iz * delta_h;
                    bce.push_back({x, y, z});
                }
            }
            // Add cylinder caps
            if (r2 < r_min2) {
                for (int iz = 0; iz < num_layers; iz++) {
                    double z = -height / 2 + iz * delta_h;
                    bce.push_back({x, y, -z});
                    bce.push_back({x, y, +z});
                }
            }
        }
    }

    return bce;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsConeInterior(double rad, double height, bool polar) const {
    std::vector<ChVector3d> bce;

    double spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Calculate actual spacing
    int np_h = (int)std::round(height / spacing);
    double delta_h = height / np_h;

    // Use polar coordinates
    if (polar) {
        for (int iz = 0; iz < np_h; iz++) {
            double z = iz * delta_h;
            double rz = rad * (height - z) / height;
            double rad_out = rz;
            double rad_in = std::max(rad_out - num_layers * spacing, 0.0);
            if (iz >= np_h - num_layers)
                rad_in = 0;
            int np_r = (int)std::round((rad_out - rad_in) / spacing);
            double delta_r = (rad_out - rad_in) / np_r;
            for (int ir = 0; ir <= np_r; ir++) {
                double r = rad_in + ir * delta_r;
                int np_th = (int)std::round(2 * pi * r / spacing);
                double delta_th = (2 * pi) / np_th;
                for (int it = 0; it < np_th; it++) {
                    double theta = it * delta_th;
                    double x = r * cos(theta);
                    double y = r * sin(theta);
                    bce.push_back({x, y, z});
                }
            }
        }

        bce.push_back({0.0, 0.0, height});

        //// TODO: add cap

        return bce;
    }

    // Use a regular grid and accept/reject points
    int np_r = (int)std::round(rad / spacing);
    double delta_r = rad / np_r;

    for (int iz = 0; iz <= np_h; iz++) {
        double z = iz * delta_h;
        double rz = rad * (height - z) / height;
        double rad_out = rz;
        double rad_in = std::max(rad_out - num_layers * spacing, 0.0);
        double r_out2 = rad_out * rad_out;
        double r_in2 = rad_in * rad_in;
        for (int ix = -np_r; ix <= np_r; ix++) {
            double x = ix * delta_r;
            for (int iy = -np_r; iy <= np_r; iy++) {
                double y = iy * delta_r;
                double r2 = x * x + y * y;
                if (r2 >= r_in2 && r2 <= r_out2) {
                    bce.push_back({x, y, z});
                }
            }

            //// TODO: add cap
        }
    }

    return bce;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsConeExterior(double rad, double height, bool polar) const {
    std::vector<ChVector3d> bce;

    double spacing = m_paramsH->d0;
    int num_layers = m_paramsH->num_bce_layers;

    // Calculate actual spacing
    int np_h = (int)std::round(height / spacing);
    double delta_h = height / np_h;

    // Inflate cone
    np_h += 2 * num_layers;
    height += 2 * num_layers * delta_h;

    // Use polar coordinates
    if (polar) {
        for (int iz = 0; iz < np_h; iz++) {
            double z = iz * delta_h;
            double rz = rad * (height - z) / height;
            double rad_out = rz + num_layers * spacing;
            double rad_in = std::max(rad_out - num_layers * spacing, 0.0);
            if (iz >= np_h - num_layers)
                rad_in = 0;
            int np_r = (int)std::round((rad_out - rad_in) / spacing);
            double delta_r = (rad_out - rad_in) / np_r;
            for (int ir = 0; ir <= np_r; ir++) {
                double r = rad_in + ir * delta_r;
                int np_th = (int)std::round(2 * pi * r / spacing);
                double delta_th = (2 * pi) / np_th;
                for (int it = 0; it < np_th; it++) {
                    double theta = it * delta_th;
                    double x = r * cos(theta);
                    double y = r * sin(theta);
                    bce.push_back({x, y, z});
                }
            }
        }

        bce.push_back({0.0, 0.0, height});

        //// TODO: add cap

        return bce;
    }

    // Use a regular grid and accept/reject points
    int np_r = (int)std::round(rad / spacing);
    double delta_r = rad / np_r;

    for (int iz = 0; iz <= np_h; iz++) {
        double z = iz * delta_h;
        double rz = rad * (height - z) / height;
        double rad_out = rz + num_layers * spacing;
        double rad_in = std::max(rad_out - num_layers * spacing, 0.0);
        double r_out2 = rad_out * rad_out;
        double r_in2 = rad_in * rad_in;
        for (int ix = -np_r; ix <= np_r; ix++) {
            double x = ix * delta_r;
            for (int iy = -np_r; iy <= np_r; iy++) {
                double y = iy * delta_r;
                double r2 = x * x + y * y;
                if (r2 >= r_in2 && r2 <= r_out2) {
                    bce.push_back({x, y, z});
                }
            }

            //// TODO: add cap
        }
    }

    return bce;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsCylinderAnnulus(double rad_inner,
                                                                              double rad_outer,
                                                                              double height,
                                                                              bool polar) const {
    std::vector<ChVector3d> points;

    Real spacing = m_paramsH->d0;
    double hheight = height / 2;

    // Calculate actual spacing
    int np_h = (int)std::round(hheight / spacing);
    double delta_h = hheight / np_h;

    // Use polar coordinates
    if (polar) {
        int np_r = (int)std::round((rad_outer - rad_inner) / spacing);
        double delta_r = (rad_outer - rad_inner) / np_r;
        for (int ir = 0; ir <= np_r; ir++) {
            double r = rad_inner + ir * delta_r;
            int np_th = (int)std::round(2 * pi * r / spacing);
            double delta_th = (2 * pi) / np_th;
            for (int it = 0; it < np_th; it++) {
                double theta = it * delta_th;
                double x = r * cos(theta);
                double y = r * sin(theta);
                for (int iz = -np_h; iz <= np_h; iz++) {
                    double z = iz * delta_h;
                    points.push_back({x, y, z});
                }
            }
        }

        return points;
    }

    // Use a regular grid and accept/reject points
    int np_r = (int)std::round(rad_outer / spacing);
    double delta_r = rad_outer / np_r;

    double r_in2 = rad_inner * rad_inner;
    double r_out2 = rad_outer * rad_outer;
    for (int ix = -np_r; ix <= np_r; ix++) {
        double x = ix * delta_r;
        for (int iy = -np_r; iy <= np_r; iy++) {
            double y = iy * delta_r;
            double r2 = x * x + y * y;
            if (r2 >= r_in2 && r2 <= r_out2) {
                for (int iz = -np_h; iz <= np_h; iz++) {
                    double z = iz * delta_h;
                    points.push_back({x, y, z});
                }
            }
        }
    }

    return points;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::CreatePointsMesh(ChTriangleMeshConnected& mesh) const {
    std::vector<ChVector3d> points;

    Real spacing = m_paramsH->d0;

    // Ensure mesh if watertight
    mesh.RepairDuplicateVertexes(1e-9);
    auto bbox = mesh.GetBoundingBox();

    const double EPSI = 1e-6;

    ChVector3d ray_origin;
    for (double x = bbox.min.x(); x < bbox.max.x(); x += spacing) {
        ray_origin.x() = x + 1e-9;
        for (double y = bbox.min.y(); y < bbox.max.y(); y += spacing) {
            ray_origin.y() = y + 1e-9;
            for (double z = bbox.min.z(); z < bbox.max.z(); z += spacing) {
                ray_origin.z() = z + 1e-9;

                ChVector3d ray_dir[2] = {ChVector3d(5, 0.5, 0.25), ChVector3d(-3, 0.7, 10)};
                int intersectCounter[2] = {0, 0};

                for (unsigned int i = 0; i < mesh.m_face_v_indices.size(); ++i) {
                    auto& t_face = mesh.m_face_v_indices[i];
                    auto& v1 = mesh.m_vertices[t_face.x()];
                    auto& v2 = mesh.m_vertices[t_face.y()];
                    auto& v3 = mesh.m_vertices[t_face.z()];

                    // Find vectors for two edges sharing V1
                    auto edge1 = v2 - v1;
                    auto edge2 = v3 - v1;

                    int t_inter[2] = {0, 0};

                    for (unsigned int j = 0; j < 2; j++) {
                        // Begin calculating determinant - also used to calculate uu parameter
                        auto pvec = Vcross(ray_dir[j], edge2);
                        // if determinant is near zero, ray is parallel to plane of triangle
                        double det = Vdot(edge1, pvec);
                        // NOT CULLING
                        if (det > -EPSI && det < EPSI) {
                            t_inter[j] = 0;
                            continue;
                        }
                        double inv_det = 1.0 / det;

                        // calculate distance from V1 to ray origin
                        auto tvec = ray_origin - v1;

                        /// Calculate uu parameter and test bound
                        double uu = Vdot(tvec, pvec) * inv_det;
                        // The intersection lies outside of the triangle
                        if (uu < 0.0 || uu > 1.0) {
                            t_inter[j] = 0;
                            continue;
                        }

                        // Prepare to test vv parameter
                        auto qvec = Vcross(tvec, edge1);

                        // Calculate vv parameter and test bound
                        double vv = Vdot(ray_dir[j], qvec) * inv_det;
                        // The intersection lies outside of the triangle
                        if (vv < 0.0 || ((uu + vv) > 1.0)) {
                            t_inter[j] = 0;
                            continue;
                        }

                        double tt = Vdot(edge2, qvec) * inv_det;
                        if (tt > EPSI) {  /// ray intersection
                            t_inter[j] = 1;
                            continue;
                        }

                        // No hit, no win
                        t_inter[j] = 0;
                    }

                    intersectCounter[0] += t_inter[0];
                    intersectCounter[1] += t_inter[1];
                }

                if (((intersectCounter[0] % 2) == 1) && ((intersectCounter[1] % 2) == 1))  // inside mesh
                    points.push_back(ChVector3d(x, y, z));
            }
        }
    }

    return points;
}


//------------------------------------------------------------------------------
// Other helper functions to access various parameter of the fluid problem, stored in the data manager object
//------------------------------------------------------------------------------
double ChFsiFluidSystemSPH_csph::GetKernelLength() const {
    // default kernel length
    return m_paramsH->h;
}

double ChFsiFluidSystemSPH_csph::GetInitialSpacing() const {
    return m_paramsH->d0;
}

double ChFsiFluidSystemSPH_csph::GetMarkerMass() const {
    return m_paramsH->markerMass;
}

int ChFsiFluidSystemSPH_csph::GetNumBCELayers() const {
    return m_paramsH->num_bce_layers;
}

ChVector3d ChFsiFluidSystemSPH_csph::GetContainerDim() const {
    return ChVector3d(m_paramsH->boxDimX, m_paramsH->boxDimY, m_paramsH->boxDimZ);
}

ChAABB ChFsiFluidSystemSPH_csph::GetComputationalDomain() const {
    return ChAABB(ToChVector(m_paramsH->cMin), ToChVector(m_paramsH->cMax));
}

double ChFsiFluidSystemSPH_csph::GetBaseDensity() const {
    return m_paramsH->rho0;
}

double ChFsiFluidSystemSPH_csph::GetBasePressure() const {
    return m_paramsH->p0;
}

double ChFsiFluidSystemSPH_csph::GetBaseEnergy() const {
    return m_paramsH->e0;
}

double ChFsiFluidSystemSPH_csph::GetGamma() const {
    return m_paramsH->gamma;
}

double ChFsiFluidSystemSPH_csph::GetParticleMass() const {
    return m_paramsH->markerMass;
}

ChVector3d ChFsiFluidSystemSPH_csph::GetGravitationalAcceleration() const {
    return ChVector3d(m_paramsH->gravity.x, m_paramsH->gravity.y, m_paramsH->gravity.z);
}


ChVector3d ChFsiFluidSystemSPH_csph::GetVolumetricForce() const {
    return ChVector3d(m_paramsH->bodyForce3.x, m_paramsH->bodyForce3.y, m_paramsH->bodyForce3.z);
}

int ChFsiFluidSystemSPH_csph::GetNumProximitySearchSteps() const {
    return m_paramsH->num_proximity_search_steps;
}

int ChFsiFluidSystemSPH_csph::GetDensityReinitSteps() const {
    if (!m_paramsH->density_reinit_switch)
        return 0;
    else
        return m_paramsH->density_reinit_steps;
}

bool ChFsiFluidSystemSPH_csph::GetDensityReinitSwitch() const {
    return m_paramsH->density_reinit_switch;
}

bool ChFsiFluidSystemSPH_csph::GetUseVariableTimeStep() const {
    return m_paramsH->use_variable_time_step;
}

double2 ChFsiFluidSystemSPH_csph::GetCFLParams() const {
    return make_double2(m_paramsH->C_cfl, m_paramsH->C_force);
}

size_t ChFsiFluidSystemSPH_csph::GetNumFluidMarkers() const {
    return m_data_mgr->countersH->numFluidMarkers;
}

size_t ChFsiFluidSystemSPH_csph::GetNumRigidBodyMarkers() const {
    return m_data_mgr->countersH->numRigidMarkers;
}

size_t ChFsiFluidSystemSPH_csph::GetNumBoundaryMarkers() const {
    return m_data_mgr->countersH->numBoundaryMarkers;
}

//----------------------------------------------------------------------------------------------------------
// Helper methods to access properties and states of the particles. Returns a vector of ChVector3d with all the
// particles. These are the GetParticleSomething() methods, they actually call the GetSomething() methods defined later.
// The GetSomething() methods return Real or vectors of Real types, the GetParticleSomething() methods return ChVectors
// or vectors of ChVectors. In both cases they return properties of all particles.
//----------------------------------------------------------------------------------------------------------

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::GetParticlePositions() const {
    auto pos3 = GetPositions();  // returns a vector of Real3 type

    std::vector<ChVector3d> pos;       // we want output to be vector of ChVector3d
    for (const auto& p : pos3)         // convert all elements from Real3 to ChVector3d
        pos.push_back(ToChVector(p));  // push_back adds an element at the end of the vector

    return pos;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::GetParticleVelocities() const {
    auto vel3 = GetVelocities();

    std::vector<ChVector3d> vel;
    for (const auto& v : vel3)
        vel.push_back(ToChVector(v));

    return vel;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::GetParticleAccelerations() const {
    auto acc3 = GetAccelerations();

    std::vector<ChVector3d> acc;
    for (const auto& a : acc3)
        acc.push_back(ToChVector(a));

    return acc;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::GetParticleForces() const {
    auto frc3 = GetForces();

    std::vector<ChVector3d> frc;
    for (const auto& f : frc3)
        frc.push_back(ToChVector(f));

    return frc;
}

std::vector<ChVector3d> ChFsiFluidSystemSPH_csph::GetParticleFluidProperties() const {
    auto props3 = GetProperties();

    std::vector<ChVector3d> props;
    for (const auto& p : props3)
        props.push_back(ToChVector(p));

    return props;
}




//---------------------------------------------------------------------------------------------------------------------------------------------
// Helper functions to get particle properties. These are the GetSomething() methods. Different from previous as they
// return vectors of Reals. They call the equivalent method present in FsiDataManager. Return states for ALL
// particles/markers
//---------------------------------------------------------------------------------------------------------------------------------------------
std::vector<int> ChFsiFluidSystemSPH_csph::FindParticlesInBox(const ChFrame<>& frame, const ChVector3d& size) {
    const ChVector3d& Pos = frame.GetPos();
    ChVector3d Ax = frame.GetRotMat().GetAxisX();
    ChVector3d Ay = frame.GetRotMat().GetAxisY();
    ChVector3d Az = frame.GetRotMat().GetAxisZ();

    auto hsize = 0.5 * mR3(size.x(), size.y(), size.z());
    auto pos = mR3(Pos.x(), Pos.y(), Pos.z());
    auto ax = mR3(Ax.x(), Ax.y(), Ax.z());
    auto ay = mR3(Ay.x(), Ay.y(), Ay.z());
    auto az = mR3(Az.x(), Az.y(), Az.z());

    return m_data_mgr->FindParticlesInBox(hsize, pos, ax, ay, az);
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetPositions() const {
    return m_data_mgr->GetPositions();
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetVelocities() const {
    return m_data_mgr->GetVelocities();
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetAccelerations() const {
    return m_data_mgr->GetAccelerations();
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetForces() const {
    return m_data_mgr->GetForces();
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetProperties() const {
    return m_data_mgr->GetProperties();
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetPositions(const std::vector<int>& indices) const {
    return m_data_mgr->GetPositions(indices);
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetVelocities(const std::vector<int>& indices) const {
    return m_data_mgr->GetVelocities(indices);
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetAccelerations(const std::vector<int>& indices) const {
    return m_data_mgr->GetAccelerations(indices);
}

std::vector<Real3> ChFsiFluidSystemSPH_csph::GetForces(const std::vector<int>& indices) const {
    return m_data_mgr->GetForces(indices);
}

std::vector<Real> ChFsiFluidSystemSPH_csph::GetSound() const {
    return m_data_mgr->GetSound();
}

std::vector<Real> ChFsiFluidSystemSPH_csph::GetKernelRad() const {
    return m_data_mgr->GetKernelRad();
}

Real ChFsiFluidSystemSPH_csph::GetMaxFluidRad() const {
    return m_data_mgr->GetMaxFluidRad();
}

Real2 ChFsiFluidSystemSPH_csph::GetMaxRad() const {
    return m_data_mgr->GetMaxRad();
}


}  // end namespace compressible
}  // namespace chrono::fsi::sph