// =============================================================================
// Simulate a 1D shock tube scenario using a single line of particles along y = z = 0
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

#ifdef CHRONO_POSTPROCESS
    #include "chrono_postprocess/ChGnuPlot.h"
#endif


using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::fsi::sph;
namespace csph = chrono::fsi::sph::compressible;

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
    std::string out_dir = GetChronoOutputPath() + "shock_tube_1D";

    // create (empty) systems: multibody, fluid and ChFsiSystem
    ChSystemSMC sysMBS;
    csph::ChFsiFluidSystemSPH_csph sysSPH;
    csph::ChFsiSystemSPH_csph sysFSI_csph(sysMBS, sysSPH);

    std::cout << "Hello Chrono compressible!" << std::endl;
#ifdef CHRONO_FSI_USE_DOUBLE
    std::cout << "Using double as Real" << std::endl;
#else
    std::cout << "Using float as Real" << std::endl;
#endif


    // Using same exact initial conditions as the paper on Euler Equations
    // Set fluid properties:
    double gamma = 1.4;

    // Set properties on left and right side of discontinuity
    double rho_l = 1.0;
    double p_l = 1.0;
    double e_l = p_l / ( rho_l * (gamma - 1));

    double rho_r = 0.125;
    double p_r = 0.1;
    double e_r = p_r / (rho_r * (gamma - 1));

    // set reference properties in fluid system:
    sysSPH.SetGamma(gamma);

    // different spacings on the two sides of the domain.
    double x_lim_l = -0.5;
    double x_lim_r = 0.5;
    double d_l = 0.0015625;
    // double d_r = 0.003125;
    double d_r = 0.00625;

    double mass_l = rho_l * d_l;
    double mass_r = rho_r * d_r;

    int num_layer_l = 1;
    int num_layer_r = 1;

    std::cout << "Ratio of spacing left/right is = " << d_l / d_r << std::endl;

    double kern_multip = 1.2;              // between 1.1 and 1.4 for cubic spline kernel
    double h0 = kern_multip * d_r;
    sysSPH.SetInitialSpacing(d_r);
    sysSPH.SetKernelMultiplier(kern_multip);

    // ------------------------------------------------------
    // create fluid particles on left of domain:
    std::vector<Real3> pos_l;
    for (double x = -d_l / 2; x > x_lim_l; x -= d_l) {  // along x
                sysSPH.AddSPHParticle(ChVector3d(x, 0, 0), rho_l, p_l, e_l, ChVector3d(0), h0, mass_l);
                pos_l.push_back(make_Real3(x, 0, 0));
    }

    std::cout << "In left part of domain we have " << pos_l.size() << " particles along x" << std::endl;

    // create fluid particles on right of domain:
    std::vector<Real3> pos_r;
    for (double x = d_r / 2; x < x_lim_r; x += d_r) {  // along x
                sysSPH.AddSPHParticle(ChVector3d(x, 0, 0), rho_r, p_r, e_r, ChVector3d(0), h0, mass_r);
                pos_r.push_back(make_Real3(x, 0, 0));
    }
    std::cout << "In right part of domain we have " << pos_r.size() << " particles along x" << std::endl;
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
    for (int i = 0; i < 10; i++) {  // 10 particles on x
        double x = x_start_bce_l - i * d_l;
                pos_bce_l.push_back(ChVector3d(x, 0, 0));
    }

    // add them as fixed markers:
    sysFSI_csph.AddFsiBoundary(pos_bce_l, ChFrame<>(ChVector3d(0, 0, 0), QUNIT), h0, mass_l);
    // sysFSI_csph.AddFsiBody(ground1, pos_bce_l, ChFrame<>(ChVector3d(0, 0, 0), QUNIT), false);


    // right side of domain
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
    double x_start_bce_r = pos_r.back().x + d_r;
    for (int i = 0; i < 3; i++) {  // 3 particles on x
        double x = x_start_bce_r + i * d_r;
                pos_bce_r.push_back(ChVector3d(x, 0, 0));
    }
    // add them as fixed markers:
    sysFSI_csph.AddFsiBoundary(pos_bce_r, ChFrame<>(ChVector3d(0, 0, 0), QUNIT), h0, mass_r);
    // sysFSI_csph.AddFsiBody(ground2, pos_bce_r, ChFrame<>(ChVector3d(0, 0, 0), QUNIT), false);
    // --------------------------------------------------------
    // Set the computational domain, add some buffers:
    double W = pos_l.back().y;
    std::cout << "Left W along y = " << W << std::endl;
    ChVector3d cMin(pos_bce_l.back().x() - h0, -h0, -h0);
    ChVector3d cMax(pos_bce_r.back().x() + h0, 3*h0, 3*h0);
    sysSPH.SetComputationalDomain(ChAABB(cMin, cMax), {BCType::NONE, BCType::PERIODIC, BCType::PERIODIC});

    std::cout << "Particle mass on left domain is " << mass_l << " and on right side is "
              << mass_r << std::endl;
    
    // DEBUG!!
    auto manager = sysSPH.GetDataManager();
    thrust::host_vector<Real4> rhoH = manager->sphMarkers_H->rhoPresEnH;
    std::cout << "Bce particles densities are:" << std::endl;
    for (int i = 0; i < rhoH.size(); i++)
        if (rhoH[i].w >= 0)
            std::cout << rhoH[i].x << ", ";
    std::cout << std::endl;

    thrust::host_vector<Real4> posH = manager->sphMarkers_H->posRadH;
    std::cout << "Debug Positions:" << std::endl;
    std::cout << "First Particle has position: " << posH[0] << std::endl;
    std::cout << "Last particle has position: " << posH.back() << std::endl;
    std::cout << "Distance first - last = " << csph::Distance_csph(make_Real3(posH[0]), make_Real3(posH.back()), *manager->paramsH) << std::endl;

    // Set sph method properties:
   
    sysSPH.SetShiftingMethod(ShiftingMethod::NONE);
    sysSPH.SetBodyForce(ChVector3d(0.0));
    sysSPH.SetNumProximitySearchSteps(1);
    sysSPH.SetDensityReinitSteps(0);
    sysSPH.SetOutputLevel(OutputLevel::STATE);
    
    sysFSI_csph.SetVerbose(true);  // verbose true prints series of information on screen.
    
    sysSPH.SetKernelType(csph::KernelType_csph::CUBIC_SPLINE);
    sysSPH.SetRhoEvolution(csph::Rho_evolution_csph::SUMMATION);    // both types works
    sysSPH.SetKernRadEvolution(csph::H_evolution_csph::ADKE);       // for now only adke suitable for this setup
    sysSPH.SetADKECoeff(0.3, 0.5, kern_multip); 
    sysSPH.SetViscosityMethod(csph::ViscosityMethod_csph::ARTIFICIAL_MONAGHAN);
    sysSPH.SetArtificialViscosityCoefficient(1, 1);
    sysSPH.SetHeatingMethod(csph::HeatingMethod_csph::ARTIFICIAL);
    sysSPH.SetArtificialHeatingCoefficient(0.2, 0.4);
    
    sysSPH.SetIsUniform(false);
    sysSPH.SetMarkerMass(rho_l * d_l);

    double step_size = 1e-4;
    sysFSI_csph.SetStepSizeCFD(step_size);
    sysSPH.SetUseVariableTimeStep(false);
    sysSPH.SetIntegrationScheme(csph::IntegrationScheme_csph::RK2);
    sysSPH.SetNumDim(1);
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
        auto col_callback = chrono_types::make_shared<csph::ParticleDensityColorCallback_csph>(0, 1.2);

        auto visFSI_csph = chrono_types::make_shared<csph::ChFsiVisualizationVSG_csph>(&sysFSI_csph);
        visFSI_csph->EnableFluidMarkers(true);
        visFSI_csph->EnableBoundaryMarkers(true);
        visFSI_csph->EnableRigidBodyMarkers(false);
        visFSI_csph->SetSPHColorCallback(col_callback);

        // VSG visual system (attach visFSI as plugin)
        auto visVSG = chrono_types::make_shared<vsg3d::ChVisualSystemVSG>();
        visVSG->AttachPlugin(visFSI_csph);
        visVSG->AttachSystem(&sysMBS);
        visVSG->SetWindowTitle("Shock Tube 1D - SPH compressible");
        visVSG->SetWindowSize(1280, 800);
        visVSG->SetWindowPosition(100, 100);
        visVSG->AddCamera(ChVector3d(0, -0.5, 0.005), ChVector3d(0, 0, 0.005));
        visVSG->SetLightIntensity(0.9f);
        visVSG->SetLightDirection(-CH_PI_2, CH_PI / 6);
        visVSG->SetLightIntensity(1.0f);
        visVSG->AddGrid(0.01, 0.01, 100, 200, ChCoordsysd(ChVector3d(0, 0.1, 0), QuatFromAngleX(CH_PI_2)),
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

    t_end = 0.15;
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
    ChVectorDynamic<> data_rho(rhopresen.size());
    ChVectorDynamic<> data_pres(rhopresen.size());
    ChVectorDynamic<> data_en(rhopresen.size());
    for (int i = 0; i < pos.size(); ++i) {
        double x = pos[i].x();
        double rho = rhopresen[i].x();
        double pres = rhopresen[i].y();
        double en = rhopresen[i].z();
        data_x(i) = x;
        data_rho(i) = rho;
        data_pres(i) = pres;
        data_en(i) = en;
    }
    
    postprocess::ChGnuPlot rho_plot(out_dir + "/density_profile.gpl");
    rho_plot.SetGrid();
    std::string speed_title1 = "Density profile - 1D";
    rho_plot.SetTitle(speed_title1);
    rho_plot.SetLabelX("x");
    rho_plot.SetLabelY("Density");
    rho_plot.Plot(data_x, data_rho, "", " with points lt -1 lw 2 lc rgb'#3333BB' ");
    rho_plot.SetRangeX(-0.5, 0.5);
    rho_plot.SetRangeY(0, 1.2);

    postprocess::ChGnuPlot pres_plot(out_dir + "/pressure_profile.gpl");
    pres_plot.SetGrid();
    std::string speed_title2 = "Pressure profile - 1D";
    pres_plot.SetTitle(speed_title2);
    pres_plot.SetLabelX("x");
    pres_plot.SetLabelY("Pressure");
    pres_plot.Plot(data_x, data_pres, "", " with points lt -1 lw 2 lc rgb'#3333BB' ");
    pres_plot.SetRangeX(-0.5, 0.5);
    pres_plot.SetRangeY(0, 1.2);

    postprocess::ChGnuPlot en_plot(out_dir + "/energy_profile.gpl");
    en_plot.SetGrid();
    std::string speed_title3 = "Energy profile - 1D";
    en_plot.SetTitle(speed_title3);
    en_plot.SetLabelX("x");
    en_plot.SetLabelY("Energy");
    en_plot.Plot(data_x, data_en, "", " with points lt -1 lw 2 lc rgb'#3333BB' ");
    en_plot.SetRangeX(-0.5, 0.5);
    en_plot.SetRangeY(1.5, 3.2);

#endif


    // /*
    while (render && vis->Run()) {
        vis->Render();
    }
    // */


    return 0;
}