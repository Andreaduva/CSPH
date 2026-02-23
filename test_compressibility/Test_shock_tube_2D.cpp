// =============================================================================
// Program to test the compressibility module
// Simulates a 2D blast wave with periodic boundary conditions.
// Units are normalized.
// =============================================================================

// standard cpp libraries
#include <cassert>
#include <cstdlib>
#include <ctime>
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

// Output frequency (output csv files at selected time intervals)
bool output = true;
double out_fps = 20;

// Final simulation time
double t_end = 1;

// Enable/disable run-time visualization
bool render = true;
bool snapshots = false;
float render_fps = 100;

// =============================================================================

int main(int argc, char* argv[]) {
    // SET output directories and settings
    std::string out_path = ("TEST_OUTPUT/");
    SetChronoOutputPath(out_path);
    std::string out_dir = GetChronoOutputPath() + "shock_tube_2D";

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

    // Set fluid properties:
    double gamma = 1.4;

    // Set properties on left and right side of discontinuity
    double rho_l = 1.0;
    double p_l = 10.0;
    double e_l = p_l / (rho_l * (gamma - 1));

    double rho_r = 1.0;
    double p_r = 1.0;
    double e_r = p_r / (rho_r * (gamma -1));

    // set reference properties in fluid system:
    sysSPH.SetGamma(gamma);

    // different spacings on the two sides of the domain.
    double x_min = 0.0;
    double x_max = 1;
    double x_mid = 0.5;
    double delta_x = 0.002;
    
    double mass_l = rho_l * std::pow(delta_x, 3);
    double mass_r = rho_r * std::pow(delta_x, 3);

    int num_layer_y = 10;
    int num_layer_z = 10;
    
    double h0 = 1.2 * delta_x;
    sysSPH.SetInitialSpacing(delta_x);
    sysSPH.SetKernelMultiplier(1.2);

    // ------------------------------------------------------
    // create fluid particles:
    std::vector<Real3> pos;
    for (double x = x_min; x < x_max; x += delta_x) {  // along x
        for (int j = 0; j < num_layer_y; j++) {                    // along y
            double y = j * delta_x;
            for (int k = 0; k < num_layer_z; k++) {  // along z
                double z = k * delta_x;
                if (x <= x_mid)
                    sysSPH.AddSPHParticle(ChVector3d(x, y, z), rho_l, p_l, e_l, ChVector3d(0), h0);
                else
                    sysSPH.AddSPHParticle(ChVector3d(x, y, z), rho_r, p_r, e_r, ChVector3d(0), h0);
                pos.push_back(make_Real3(x, y, z));
            }
        }
    };

    // add bce particles on left and right of the domain:
    // create fixed body and add it to the system
    auto ground1 = chrono_types::make_shared<ChBody>();
    ground1->SetFixed(true);
    ground1->EnableCollision(false);
    sysMBS.AddBody(ground1);

    // left side bce particles
    sysSPH.SetDensity(rho_l);
    sysSPH.SetPressure(p_l);
    sysSPH.SetEnergy(e_l);
    // create particles positions
    std::vector<ChVector3d> pos_bce_l;
    double x_start_bce_l = pos[0].x - delta_x;
    for (int i = 0; i < 3; i++) {  // 3 particles on x
        double x = x_start_bce_l - i * delta_x;
        for (int j = 0; j < num_layer_y; j++) {
            double y = j * delta_x;
            for (int k = 0; k < num_layer_z; k++) {
                double z = k * delta_x;
                pos_bce_l.push_back(ChVector3d(x, y, z));
            }
        }
    }
    // add them as fixed markers:
    sysFSI_csph.AddFsiBoundary(pos_bce_l, ChFrame<>(ChVector3d(0, 0, 0), QUNIT));

    auto ground2 = chrono_types::make_shared<ChBody>();
    ground2->SetFixed(true);
    ground2->EnableCollision(false);
    sysMBS.AddBody(ground2);
    // right side bce particles
    sysSPH.SetDensity(rho_r);
    sysSPH.SetPressure(p_r);
    sysSPH.SetEnergy(e_r);
    // create particles positions
    std::vector<ChVector3d> pos_bce_r;
    double x_start_bce_r = pos.back().x + delta_x;
    for (int i = 0; i < 3; i++) {  // 3 particles on x
        double x = x_start_bce_r + i * delta_x;
        for (int j = 0; j < num_layer_y; j++) {
            double y = j * delta_x;
            for (int k = 0; k < num_layer_z; k++) {
                double z = k * delta_x;
                pos_bce_r.push_back(ChVector3d(x, y, z));
            }
        }
    }
    // add them as fixed markers:
    sysFSI_csph.AddFsiBoundary(pos_bce_r, ChFrame<>(ChVector3d(0, 0, 0), QUNIT));
    // --------------------------------------------------------
    // Set the computational domain, add some buffers:
    double W_y = num_layer_y * delta_x;
    double W_z = num_layer_z * delta_x;
    std::cout << "Left W along y = " << W_y << std::endl;
    ChVector3d cMin(x_min - 3.5*delta_x , 0, 0);
    ChVector3d cMax(x_max + 3.5*delta_x , W_y + delta_x/2, W_z + delta_x/2);
    sysSPH.SetComputationalDomain(ChAABB(cMin, cMax), {BCType::NONE, BCType::PERIODIC, BCType::PERIODIC});

    std::cout << "Particle mass on left domain is " << rho_l * std::pow(delta_x, 3) << " and on right side is "
              << rho_r * std::pow(delta_x, 3) << std::endl;

    // Set sph method properties:
    double step_size = 3e-7;

    sysSPH.SetShiftingMethod(ShiftingMethod::NONE);
    sysSPH.SetBodyForce(ChVector3d(0.0));
    sysSPH.SetNumProximitySearchSteps(1);
    sysSPH.SetDensityReinitSteps(0);
    sysSPH.SetOutputLevel(OutputLevel::STATE);
    sysFSI_csph.SetStepSizeCFD(step_size);
    sysFSI_csph.SetVerbose(true);  // verbose true prints series of information on screen.
    sysSPH.SetKernRadEvolution(cmp::H_evolution_csph::DIFFERENTIAL);  // ADKE not working in this configuration
    sysSPH.SetViscosityMethod(cmp::ViscosityMethod_csph::ARTIFICIAL_MONAGHAN);
    sysSPH.SetArtificialViscosityCoefficient(1.0, 1.0);
    sysSPH.SetADKECoeff(1, 0.5, 1.5);
    sysSPH.SetRhoEvolution(cmp::Rho_evolution_csph::DIFFERENTIAL);
    sysSPH.SetUseVariableTimeStep(false);
    sysSPH.SetIsUniform(true);
    sysSPH.SetIntegrationScheme(cmp::IntegrationScheme_csph::EULER);
    sysFSI_csph.Initialize();

    
    // Create output directories
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cerr << "Error creating directory " << out_dir << std::endl;
        return 1;
    }
    out_dir = out_dir + "/" + sysSPH.GetPhysicsProblemString() + "_" + sysSPH.GetSphIntegrationSchemeString();
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cerr << "Error creating directory " << out_dir << std::endl;
        return 1;
    }
    if (!filesystem::create_directory(filesystem::path(out_dir + "/particles"))) {
        std::cerr << "Error creating directory " << out_dir + "/particles" << std::endl;
        return 1;
    }
    auto absolute_path = std::filesystem::absolute(out_dir);
    std::cout << "Output directory created at " << absolute_path.string() << std::endl;
    if (!filesystem::create_directory(filesystem::path(out_dir + "/snapshots"))) {
        std::cerr << "Error creating directory " << out_dir + "/snapshots" << std::endl;
        return 1;
    }


    // Create a run-tme visualizer
    std::shared_ptr<ChVisualSystem> vis;




// visualization only in Release mode
#ifdef NDEBUG
    #ifdef CHRONO_VSG
    if (render) {
        // FSI plugin
        auto col_callback = chrono_types::make_shared<cmp::ParticlePressureColorCallback_csph>(0, p_r + 10, false);

        auto visFSI_csph = chrono_types::make_shared<cmp::ChFsiVisualizationVSG_csph>(&sysFSI_csph);
        visFSI_csph->EnableFluidMarkers(true);
        visFSI_csph->EnableBoundaryMarkers(true);
        visFSI_csph->EnableRigidBodyMarkers(false);
        visFSI_csph->SetSPHColorCallback(col_callback);

        // VSG visual system (attach visFSI as plugin)
        auto visVSG = chrono_types::make_shared<vsg3d::ChVisualSystemVSG>();
        visVSG->AttachPlugin(visFSI_csph);
        visVSG->AttachSystem(&sysMBS);
        visVSG->SetWindowTitle("Shock Tube 2D - SPH compressible");
        visVSG->SetWindowSize(1280, 800);
        visVSG->SetWindowPosition(100, 100);
        visVSG->AddCamera(ChVector3d(0.5, -0.5, 0.005), ChVector3d(0.5, 0, 0.005));
        visVSG->SetLightIntensity(0.9f);
        visVSG->SetLightDirection(-CH_PI_2, CH_PI / 6);
        visVSG->SetLightIntensity(1.0f);
        visVSG->AddGrid(0.05, 0.01, 20, 5, ChCoordsysd(ChVector3d(0.5, 0.1, 0), QuatFromAngleX(CH_PI_2)),
                        ChColor(0.1f, 0.1f, 0.1f));

        visVSG->Initialize();
        vis = visVSG;
    }
    #else
    render = false;
    #endif
#endif

    // Start the simulation
    double dT = sysSPH.GetStepSize();
    std::cout << "Step size = " << dT << std::endl;
    double time = 0;
    int sim_frame = 0;
    int out_frame = 0;
    int render_frame = 0;

    output = true;
    ChTimer timer;
    timer.start();
    render = false;
    t_end = 0.0005;

    #ifdef NDEBUG
    // while (render && vis->Run()) {
    //     vis->Render();
    // }
    #endif

    #ifndef NDEBUG
    while (time < t_end)
        sysFSI_csph.DoStepDynamics(dT);
    #endif

    int count = 0;
    while (time < t_end) {
        std::cout << "Iteration " << count++ << std::endl;
        std::cout << "time = " << time << std::endl;
        // Save data of the simulation
        if (output && (time >= out_frame / out_fps || count%100 == 0) ) {
            std::cout << " -- Output frame " << out_frame << " at t = " << time << std::endl;
            sysSPH.SaveParticleData(out_dir + "/particles");
            out_frame++;
        }

        // Render FSI system
        // /*
        if (render && time >= render_frame / render_fps) {
            if (!vis->Run())
                break;
            vis->Render();

            if (snapshots) {
                std::cout << " -- Snapshot frame " << render_frame << " at t = " << time << std::endl;
                std::ostringstream filename;
                filename << out_dir << "/snapshots/img_" << std::setw(5) << std::setfill('0') << render_frame + 1
                         << ".bmp";
                vis->WriteImageToFile(filename.str());
            }

            render_frame++;
        }
        // */

        // Call the FSI solver
        sysFSI_csph.DoStepDynamics(dT);

        time += dT;
        sim_frame++;
    }

    timer.stop();
    std::cout << "\nSimulation time: " << timer() << " seconds\n" << std::endl;

    // /*
    while (render && vis->Run()) {
        vis->Render();
    }
    // */

    return 0;
}