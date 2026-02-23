// =============================================================================
// Program to test the compressibility module
// Simulates a uniform flow across a channel with periodic boundary conditions.
// Units are normalized.
// =============================================================================

// based on the Demo_FSI_Poiseuille_flow

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
#include "chrono_fsi/sph_compressible/ChFsiProblemSPH_compressible.h"
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
double t_end = 10.0;

// Enable/disable run-time visualization
bool render = true;
bool snapshots = false;
float render_fps = 100;

// =============================================================================


// custom callback class to add particles having the prescribed constant velocity:
class UniformVelocityCallback : public cmp::ChFsiProblemSPH_csph::ParticlePropertiesCallback {
    virtual void set(const cmp::ChFsiFluidSystemSPH_csph& sysSPH, const ChVector3d& pos) override {
        p0 = sysSPH.GetBasePressure();
        rho0 = sysSPH.GetBaseDensity();
        e0 = sysSPH.GetBaseEnergy();
    }
};



int main(int argc, char* argv[]) {

    std::cout << "Hello Chrono compressible!" << std::endl;
#ifdef CHRONO_FSI_USE_DOUBLE
    std::cout << "Using double as Real" << std::endl;
#else
    std::cout << "Using float as Real" << std::endl;
#endif

    // SET output directories and settings
    std::string out_path = ("TEST_OUTPUT/");
    SetChronoOutputPath(out_path);
    std::string out_dir = GetChronoOutputPath() + "Uniform_flow_compressible_ChProblem";


    double initSpace0 = 0.01;  // d0 or delta_x0
    // create (empty) systems: multibody, and ChProblem
    ChSystemSMC sysMBS;
    cmp::ChFsiProblemCartesian_csph sysProb_csph(sysMBS, initSpace0);
    sysProb_csph.SetVerbose(true);
    auto& sysFSI_csph = sysProb_csph.GetSystemFSI();
    auto& sysSPH = sysProb_csph.GetFluidSystemSPH();

    // Use normalized units for this test:
    double bxDim = 0.2;
    double byDim = 0.1;
    double bzDim = 0.2;

    double h_multiplier = 1.0;
    double h0 = initSpace0 * h_multiplier;
    double rho0 = 1;
    double e0 = 1;
    double gamma = 1.4;
    double p0 = (gamma - 1) * e0 * rho0;  // 0.4

    ChVector3d bodyForce(0, 0, 0);
    ChVector3d vel0(0.1, 0, 0);
    double step_size = 1e-4;
    sysProb_csph.SetStepSizeCFD(step_size);
    sysProb_csph.SetStepsizeMBD(step_size);

    int initial_data_set = 2;
    if (initial_data_set == 2) {
        p0 = 1;
        e0 = 2.5;
        vel0.x() = 0.2;
    }

    // Set fluid properties:
    cmp::ChFsiFluidSystemSPH_csph::FluidProperties_csph fluid_props;
    fluid_props.density = rho0;
    fluid_props.pressure = p0;
    fluid_props.energy = e0;
    fluid_props.gamma = gamma;
    sysProb_csph.SetFluidProperties(fluid_props);

    auto callback_ptr = chrono_types::make_shared<UniformVelocityCallback>();
    callback_ptr->v0 = vel0;
    callback_ptr->e0 = e0;
    callback_ptr->rho0 = rho0;
    callback_ptr->p0 = p0;
    sysProb_csph.RegisterParticlePropertiesCallback(callback_ptr);

    // use the construct() method to build the particles box:
    ChVector3d fsize(bxDim, byDim, bzDim - 2 * initSpace0);
    sysProb_csph.Construct(fsize, ChVector3d(0, 0, initSpace0), BoxSide::Z_NEG | BoxSide::Z_POS);

    // Explicitely set the computational domain:
    ChVector3d cMin(-bxDim / 2 - initSpace0 / 2, -byDim / 2 - initSpace0 / 2, -5 * initSpace0);
    ChVector3d cMax(+bxDim / 2 + initSpace0 / 2, +byDim / 2 + initSpace0 / 2, bzDim + 5 * initSpace0);
    sysProb_csph.SetComputationalDomain(ChAABB(cMin, cMax), BC_ALL_PERIODIC);

    sysProb_csph.Initialize();

    // Create oputput directories
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
        auto col_callback = chrono_types::make_shared<cmp::ParticleVelocityColorCallback_csph>(0, 4);  // 0.04

        auto visFSI_csph = chrono_types::make_shared<cmp::ChFsiVisualizationVSG_csph>(&sysFSI_csph);
        visFSI_csph->EnableFluidMarkers(true);
        visFSI_csph->EnableBoundaryMarkers(true);
        visFSI_csph->EnableRigidBodyMarkers(false);
        visFSI_csph->SetSPHColorCallback(col_callback);

        // VSG visual system (attach visFSI as plugin)
        auto visVSG = chrono_types::make_shared<vsg3d::ChVisualSystemVSG>();
        visVSG->AttachPlugin(visFSI_csph);
        visVSG->AttachSystem(&sysMBS);
        visVSG->SetWindowTitle("Test compressible");
        visVSG->SetWindowSize(1280, 800);
        visVSG->SetWindowPosition(100, 100);
        visVSG->AddCamera(ChVector3d(0, -5 * byDim, 0.5 * bzDim), ChVector3d(0, 0, 0.5 * bzDim));
        visVSG->SetLightIntensity(0.9f);
        visVSG->SetLightDirection(-CH_PI_2, CH_PI / 6);

        visVSG->Initialize();
        vis = visVSG;
    }
    #else
    render = false;
    #endif
#endif

    // Start the simulation
    double time = 0;
    int sim_frame = 0;
    int out_frame = 0;
    int render_frame = 0;

// in debug mode perform only one step
#ifndef NDEBUG
    sysFSI_csph.DoStepDynamics(step_size);
#endif

    // Release mode only

#ifdef NDEBUG
    ChTimer timer;
    timer.start();

    while (time < t_end) {
        // Save data of the simulation
        if (output && time >= out_frame / out_fps) {
            std::cout << " -- Output frame " << out_frame << " at t = " << time << std::endl;
            sysSPH.SaveParticleData(out_dir + "/particles");
            out_frame++;
        }

        // Render FSI system
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

        // Call the FSI solver
        sysProb_csph.DoStepDynamics(step_size);

        time += step_size;
        sim_frame++;
    }

    timer.stop();
    std::cout << "\nSimulation time: " << timer() << " seconds\n" << std::endl;

    while (render && vis->Run()) {
        vis->Render();
    }
#endif

    return 0;
}