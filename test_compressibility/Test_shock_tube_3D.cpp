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
#include "chrono_fsi/sph_compressible/physics/SphGeneral_compressible.cuh"
#ifdef CHRONO_VSG
    #include "chrono_fsi/sph_compressible/visualization/ChFsiVisualizationVSG_compressible.h"
#endif
#include "chrono_fsi/sph_compressible/physics/FsiDataManager_compressible.cuh"


#ifdef CHRONO_POSTPROCESS
    #include "chrono_postprocess/ChGnuPlot.h"
#endif


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
    std::string out_dir = GetChronoOutputPath() + "shock_tube_3D";

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
    double p_l = 1.0;
    double e_l = 2.5;

    double rho_r = 0.125;
    double p_r = 0.1;
    double e_r = 2;

    // set reference properties in fluid system:
    sysSPH.SetGamma(gamma);

    // different spacings on the two sides of the domain.
    double x_lim_l = -0.5;
    double x_lim_r = 0.5;
    double d_l = 0.0015625;
    double d_r = 0.003125;
    //  d_r = d_l;
    int num_layer_l = 10;
    int num_layer_r = 5;
    // num_layer_l = num_layer_r;
    std::cout << "Ratio of spacing left/right is = " << d_l / d_r << std::endl;

    double h0 = 1.2 * d_l;
    sysSPH.SetInitialSpacing(d_l);
    sysSPH.SetKernelMultiplier(1.2);


    // ------------------------------------------------------
    // create fluid particles on left of domain:
    std::vector<Real3> pos_l;
    double offset_l = d_l / 2;
    int count_l = 0;
    for (double x = - d_l / 2; x > x_lim_l; x -= d_l, count_l++) { // along x
        for (int i = 0; i < num_layer_l; i++) {     // along y
            double y = i * d_l + offset_l;
            for (int k = 0; k < num_layer_l; k++) {  // along z
                double z = k * d_l + offset_l;
                sysSPH.AddSPHParticle(ChVector3d(x, y, z), rho_l, p_l, e_l, ChVector3d(0), h0);
                pos_l.push_back(make_Real3(x, y, z));
            }
        }       
    }
;
    std::cout << "In left part of domain we have " << count_l << " particles along x and " << pos_l.size() << " particles in total"
              << std::endl;

    // create fluid particles on left of domain:
    std::vector<Real3> pos_r;
    double offset_r = d_r / 2;
    int count_r = 0;
    for (double x = d_r / 2; x < x_lim_r; x += d_r, count_r++) {  // along x
        for (int i = 0; i < num_layer_r; i++) {     // along y
            double y = i * d_r + offset_r;
            for (int k = 0; k < num_layer_r; k++) {  // along z
                double z = k * d_r + offset_r;
                sysSPH.AddSPHParticle(ChVector3d(x, y, z), rho_r, p_r, e_r, ChVector3d(0), h0);
                pos_r.push_back(make_Real3(x, y, z));
            }
        }
    }
    std::cout << "In right part of domain we have " << count_r << " particles along x and " << pos_r.size()
              << " particles in total" << std::endl;
    std::cout << "Extension of y and z domain for x < 0 is: " << offset_l + pos_l.back().z << std::endl;
    std::cout << "Extension of y and z domain for x > 0 is: " << offset_r + pos_r.back().y << std::endl;
    // ------------------------------------------------------------------
    
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
    double x_start_bce_l = pos_l.back().x - d_l;
    for (int i = 0; i < 3; i++) {   // 3 particles on x
        double x = x_start_bce_l - i * d_l;
        for (int j = 0; j < num_layer_l; j++) {
            double y = j * d_l + offset_l;
            for (int k = 0; k < num_layer_l; k++) {
                double z = k * d_l + offset_l;
                pos_bce_l.push_back(ChVector3d(x, y, z));
            }
        }
    }
    // add them as fixed markers:
    sysFSI_csph.AddFsiBoundary(pos_bce_l, ChFrame<>(ChVector3d(0, 0, 0), QUNIT));
    // sysFSI_csph.AddFsiBody(ground1, pos_bce_l, ChFrame<>(ChVector3d(0, 0, 0), QUNIT), false);



    auto ground2 = chrono_types::make_shared<ChBody>();
    ground2->SetFixed(true);
    ground2->EnableCollision(false);
    // sysMBS.AddBody(ground2);
    // right side bce particles
    sysSPH.SetDensity(rho_r);
    sysSPH.SetPressure(p_r);
    sysSPH.SetEnergy(e_r);
    // create particles positions
    std::vector<ChVector3d> pos_bce_r;
    double x_start_bce_r = pos_r.back().x + d_r;
    for (int i = 0; i < 3; i++) {  // 3 particles on x
        double x = x_start_bce_r + i * d_r;
        for (int j = 0; j < num_layer_r; j++) {
            double y = j * d_r + offset_r;
            for (int k = 0; k < num_layer_r; k++) {
                double z = k * d_r + offset_r;
                pos_bce_r.push_back(ChVector3d(x, y, z));
            }
        }
    }
    // add them as fixed markers:
    sysFSI_csph.AddFsiBoundary(pos_bce_r, ChFrame<>(ChVector3d(0, 0, 0), QUNIT));
    // sysFSI_csph.AddFsiBody(ground2, pos_bce_r, ChFrame<>(ChVector3d(0, 0, 0), QUNIT), false);
    // --------------------------------------------------------
    // Set the computational domain, add some buffers:
    double W = offset_l + pos_l.back().y;
    std::cout << "Left W along y = " << W << std::endl;
    ChVector3d cMin(pos_bce_l.back().x() - h0, 0, 0);
    ChVector3d cMax(pos_bce_r.back().x() + h0, W, W);
    sysSPH.SetComputationalDomain(ChAABB(cMin, cMax), {BCType::NONE, BCType::PERIODIC, BCType::PERIODIC});


    std::cout << "Particle mass on left domain is " << rho_l * std::pow(d_l, 3) << " and on right side is "
              << rho_r * std::pow(d_r, 3) << std::endl;

    // Set sph method properties:
    double step_size = 1e-4;

    sysSPH.SetShiftingMethod(ShiftingMethod::NONE);
    sysSPH.SetBodyForce(ChVector3d(0.0));
    sysSPH.SetNumProximitySearchSteps(1);
    sysSPH.SetDensityReinitSteps(0);
    sysSPH.SetOutputLevel(OutputLevel::STATE);
    sysFSI_csph.SetStepSizeCFD(step_size);  
    sysFSI_csph.SetVerbose(true);                                             // verbose true prints series of information on screen.
    sysSPH.SetKernRadEvolution(cmp::H_evolution_csph::DIFFERENTIAL);          // ADKE not working in this configuration
    sysSPH.SetViscosityMethod(cmp::ViscosityMethod_csph::ARTIFICIAL_MONAGHAN);
    sysSPH.SetArtificialViscosityCoefficient(1.0, 1.0);
    sysSPH.SetADKECoeff(1, 0.5, 1.2);
    sysSPH.SetRhoEvolution(cmp::Rho_evolution_csph::DIFFERENTIAL);
    sysSPH.SetUseVariableTimeStep(false);


    // Also set the particle mass - unique for now
    // sysSPH.SetMarkerMass(rho_r * std::pow(d_r, 3));
    std::cout << "Left mass is: " << rho_l * std::pow(d_l, 3) << "\nright mass is: " << rho_r * std::pow(d_r, 3)
              << std::endl;
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
        auto col_callback = chrono_types::make_shared<cmp::ParticleDensityColorCallback_csph>(0, 1.2);  

        auto visFSI_csph = chrono_types::make_shared<cmp::ChFsiVisualizationVSG_csph>(&sysFSI_csph);
        visFSI_csph->EnableFluidMarkers(true);
        visFSI_csph->EnableBoundaryMarkers(true);
        visFSI_csph->EnableRigidBodyMarkers(false);
        visFSI_csph->SetSPHColorCallback(col_callback);

        // VSG visual system (attach visFSI as plugin)
        auto visVSG = chrono_types::make_shared<vsg3d::ChVisualSystemVSG>();
        visVSG->AttachPlugin(visFSI_csph);
        visVSG->AttachSystem(&sysMBS);
        visVSG->SetWindowTitle("Shock Tube 3D - SPH compressible");
        visVSG->SetWindowSize(1280, 800);
        visVSG->SetWindowPosition(100, 100);
        visVSG->AddCamera(ChVector3d(0, -0.5, 0.005), ChVector3d(0, 0, 0.005));
        visVSG->SetLightIntensity(0.9f);
        visVSG->SetLightDirection(-CH_PI_2, CH_PI / 6);
        visVSG->SetLightIntensity(1.0f);
        visVSG->AddGrid(0.01, 0.01, 100, 200, ChCoordsysd(ChVector3d(0,0.1,0), QuatFromAngleX(CH_PI_2)),
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
    

    t_end = 0.15;


        // in debug mode perform only one step
    int count = 0;
#ifndef NDEBUG
    while (time < t_end) {
        std::cout << "Iteration " << count++ << std::endl;
        sysFSI_csph.DoStepDynamics(dT);
        time += dT;
    }
#endif

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

    #ifdef CHRONO_POSTPROCESS
    std::vector<ChVector3d> pos = sysSPH.GetParticlePositions();
    std::vector<ChVector3d> rhopresen = sysSPH.GetParticleFluidProperties();

    ChVectorDynamic<> data_x(pos.size());
    ChVectorDynamic<> data_y(rhopresen.size());
    for (int i = 0; i < pos.size(); ++i) {
        double x = pos[i].x();
        double y = rhopresen[i].x();
        data_x(i) = x;
        data_y(i) = y;
    }

    postprocess::ChGnuPlot gplot(out_dir + "/density_profile.gpl");
    gplot.SetGrid();
    std::string speed_title = "Density profile - 3D";
    gplot.SetTitle(speed_title);
    gplot.SetLabelX("x");
    gplot.SetLabelY("Pressure");
    gplot.Plot(data_x, data_y, "", " with points lt -1 lw 2 lc rgb'#3333BB' ");

#endif

   // /*
    while (render && vis->Run()) {
        vis->Render();
    }
   // */
    





    return 0;
}