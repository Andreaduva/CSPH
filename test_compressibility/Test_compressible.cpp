// =============================================================================
// Program to test the compressibility module.
// Minimum setup to test computation of derivatives
// =============================================================================

// based on the Demo_FSI_Poiseuille_flow

// standard cpp libraries
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>

#include <filesystem>
#include <thread>
// chrono core libraries
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/assets/ChVisualSystem.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"

// chrono::fsi::sph libraries
#include "chrono_fsi/sph/ChFsiSystemSPH.h"
#include "chrono_fsi/sph/physics/SphGeneral.cuh"

#ifdef CHRONO_VSG
    #include "chrono_fsi/sph/visualization/ChFsiVisualizationVSG.h"
#endif

// new
#include "chrono_fsi/sph/math/CustomMath.cuh"
#include "chrono_thirdparty/filesystem/path.h"

// chrono::fsi::sph::compressible libraries
#include "chrono_fsi/sph_compressible/ChFsiSystemSPH_compressible.h"
#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"
#ifdef CHRONO_VSG
    #include "chrono_fsi/sph_compressible/visualization/ChFsiVisualizationVSG_compressible.h"
#endif
#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"

using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::fsi::sph;
namespace cmp = chrono::fsi::sph::compressible;



// Enable/disable run-time visualization
bool render = true;

// =============================================================================

int main(int argc, char* argv[]) {

    // create (empty) systems: multibody, fluid and ChFsiSystem
    ChSystemSMC sysMBS;
    cmp::ChFsiFluidSystemSPH_csph sysSPH;
    cmp::ChFsiSystemSPH_csph sysFSI_csph(sysMBS, sysSPH);

    std::cout << "Hello Chrono compressible!" << std::endl;
#ifdef CHRONO_FSI_USE_DOUBLE
    std::cout << "Using double as Real" << std::endl;
#else
    std::cout << "Using float as Real" << std::endl;
#endif

    double m = 1;
    double h0 = 1;
    double kern_multip = 1;

    sysSPH.SetInitialSpacing(1);
    sysSPH.SetKernelMultiplier(kern_multip);
    double gamma = 1.4;
    sysSPH.SetGamma(gamma);

    std::vector<Real> x_pos = {0.0, 0.5, 1.2};
    std::vector<Real> rho = {1.0, 0.5, 0.125};
    std::vector<Real> pres = {1.0, 0.4, 0.1};
    std::vector<Real> en = {2.5, 2, 2};
    std::vector<Real> x_vel = {0.2, 0.0, -0.1};

    // add particles
    for (int i = 0; i < rho.size(); i++)
        sysSPH.AddSPHParticle(ChVector3d(x_pos[i], 0, 0), rho[i], pres[i], en[i], ChVector3d(x_vel[i], 0, 0), h0, m);

    auto ground1 = chrono_types::make_shared<ChBody>();
    ground1->SetFixed(true);
    ground1->EnableCollision(false);
    sysMBS.AddBody(ground1);

    // Set the computational domain, add some buffers:
    ChVector3d cMin(x_pos[0] - 2*h0, -h0, -h0);
    ChVector3d cMax(x_pos.back() + 2*h0, 2* h0, 2* h0);
    sysSPH.SetComputationalDomain(ChAABB(cMin, cMax), {BCType::NONE, BCType::NONE, BCType::NONE});


    sysSPH.SetShiftingMethod(ShiftingMethod::NONE);
    sysSPH.SetBodyForce(ChVector3d(0.0));
    sysSPH.SetNumProximitySearchSteps(1);
    sysSPH.SetDensityReinitSteps(0);
    sysSPH.SetOutputLevel(OutputLevel::STATE);

    sysFSI_csph.SetVerbose(true);  // verbose true prints series of information on screen.

    sysSPH.SetRhoEvolution(cmp::Rho_evolution_csph::DIFFERENTIAL);
    sysSPH.SetKernRadEvolution(cmp::H_evolution_csph::CONSTANT);  // ADKE not working in this configuration
    sysSPH.SetADKECoeff(0.9, 0.35, kern_multip);
    sysSPH.SetViscosityMethod(cmp::ViscosityMethod_csph::NONE);
    sysSPH.SetArtificialViscosityCoefficient(0.5, 0.5);
    sysSPH.SetHeatingMethod(cmp::HeatingMethod_csph::NONE);
    sysSPH.SetArtificialHeatingCoefficient(1, 1);

    sysSPH.SetIsUniform(true);
    sysSPH.SetMarkerMass(m);

    double step_size = 1e-4;
    sysFSI_csph.SetStepSizeCFD(step_size);
    sysSPH.SetUseVariableTimeStep(false);
    sysSPH.SetIntegrationScheme(cmp::IntegrationScheme_csph::RK2);
    sysFSI_csph.Initialize();

    auto manager = sysSPH.GetDataManager();
    thrust::host_vector<Real4> rhoH = manager->sphMarkers_D->rhoPresEnD;
    std::cout << "Density particle A = " << rhoH[0].x << std::endl;
    std::cout << "Density particle B = " << rhoH[1].x << std::endl;
    std::cout << "Density particle C = " << rhoH[2].x << std::endl;

    sysFSI_csph.DoStepDynamics(step_size);

    manager = sysSPH.GetDataManager();
    thrust::host_vector<Real5> deriv = manager->derivVelRhoEnOriginalD;
    thrust::host_vector<Real4> posH_post = manager->sphMarkers_D->posRadD;
    thrust::host_vector<Real3> velH_post = manager->sphMarkers_D->velD;
    thrust::host_vector<Real4> rhoH_post = manager->sphMarkers_D->rhoPresEnD;
    thrust::host_vector<Real> soundH_post = manager->sphMarkers_D->soundD;
    Real3 posA = make_Real3(x_pos[0], 0.0, 0.0);
    Real3 posB = make_Real3(x_pos[1], 0.0, 0.0);
    Real3 posC = make_Real3(x_pos[2], 0.0, 0.0);

    Real3 grad_AB = GradW3h_CubicSpline(posA - posB, 1);
    Real3 grad_AC = GradW3h_CubicSpline(posA - posC, 1);
    Real3 grad_BC = GradW3h_CubicSpline(posB - posC, 1);

    std::cout << "Grad_AB_x = " << grad_AB.x << std::endl;
    std::cout << "Grad_AC_x = " << grad_AC.x << std::endl;
    std::cout << "Grad_BC_x = " << grad_BC.x << std::endl;


    for (int i = 0; i < x_pos.size(); i++)
    {
        std::cout << "Particle " << i + 1 << ":" << std::endl;
        std::cout << "rho_dot = " << deriv[i].w << std::endl;
        std::cout << "x_vel_dot = " << deriv[i].x << std::endl;
        std::cout << "en_dot = " << deriv[i].t << std::endl;
    }

    std::cout << "After one RK2 Step:" << std::endl;
    std::cout << std::fixed << std::showpoint << std::setprecision(8);
    for (int i = 0; i < x_pos.size(); i++) {
        std::cout << "Particle " << i + 1 << ":" << std::endl;
        std::cout << "rho = " << rhoH_post[i].x << std::endl;
        std::cout << "pres = " << rhoH_post[i].y << std::endl;
        std::cout << "en = " << rhoH_post[i].z << std::endl;
        std::cout << "x = " << posH_post[i].x << std::endl;
        std::cout << "x_vel = " << velH_post[i].x << std::endl;
        std::cout << "c_s = " << soundH_post[i] << std::endl;
    }

return 0;
}

