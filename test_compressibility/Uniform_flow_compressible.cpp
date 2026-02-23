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
double t_end = 10.0;

// Enable/disable run-time visualization
bool render = true;
bool snapshots = false;
float render_fps = 100;

// =============================================================================

int main(int argc, char* argv[]) {
    // SET output directories and settings
    std::string out_path = ("TEST_OUTPUT/");
    SetChronoOutputPath(out_path);
    std::string out_dir = GetChronoOutputPath() + "Uniform_flow_compressible";

    // create (empty) systems: multibody, fluid and ChFsiSystem
    ChSystemSMC sysMBS;
    cmp::ChFsiFluidSystemSPH_csph
        sysSPH;  // the constructor calls the method InitParams(), that sets m_paramsH to default values
    cmp::ChFsiSystemSPH_csph sysFSI_csph(sysMBS, sysSPH);

    std::cout << "Hello Chrono compressible!" << std::endl;

#ifdef CHRONO_FSI_USE_DOUBLE
    std::cout << "Using double as Real" << std::endl;
#else
    std::cout << "Using float as Real" << std::endl;
#endif

    // Use normalized units for this test:
    double bxDim = 0.2;
    double byDim = 0.1;
    double bzDim = 0.2;

    double initSpace0 = 0.01;  // d0 or delta_x0
    double h_multiplier = 1.2;
    double h0 = initSpace0 * h_multiplier;
    double rho0 = 1;
    double e0 = 1;
    double gamma = 1.4;
    double p0 = (gamma - 1) * e0 * rho0;  // 0.4

    int initial_data_set = 2;
    if (initial_data_set == 2) {
        p0 = 1;
        e0 = 2.5;
    }




    // Debug: Add custom uniform initial velocity
    ChVector3d vel0(0.1, 0, 0);
    if (initial_data_set == 2) {
        vel0.x() = 0.2;
    }

    sysSPH.SetDensity(rho0);
    sysSPH.SetEnergy(e0);
    sysSPH.SetPressure(p0);
    sysSPH.SetGamma(gamma);
    sysSPH.SetInitialSpacing(initSpace0);
    sysSPH.SetKernelMultiplier(h_multiplier);

    if (initial_data_set == 2) {
    }

    // Set the fluid and simulation parameters that differ from default ones
    sysSPH.SetShiftingMethod(ShiftingMethod::NONE);
    ChVector3d bodyForce(0, 0, 0);
    sysSPH.SetBodyForce(bodyForce);

    sysSPH.SetNumProximitySearchSteps(1);
    sysSPH.SetDensityReinitSteps(1000);
    sysSPH.SetOutputLevel(OutputLevel::STATE);

    // Add buffers to the computational domain. Bigger buffer along z because there will be placed bce particles.
    ChVector3d cMin(-bxDim / 2 - initSpace0 / 2, -byDim / 2 - initSpace0 / 2, -5 * initSpace0);
    ChVector3d cMax(+bxDim / 2 + initSpace0 / 2, +byDim / 2 + initSpace0 / 2, bzDim + 5 * initSpace0);
    // WorldOrigin is taken as cMin, here defined as the bottom left corner.
    // Z dimension defined as above (not with boxDim.z/2) because otherwise wouldn't be a domain corner.
    // It's user responsibility not to add particles outside the computational domain, as it will result
    // in errors during the dynamics. Furthermore they would be also saved and displayed.
    sysSPH.SetComputationalDomain(ChAABB(cMin, cMax), BC_ALL_PERIODIC);
    // Here we set cMin and cMax parameters. boxDimX (and ..Y, ..Z) are explicitely set from files or with
    // method SetContainerDim.
    // the boxDims parameter sets with Initialize() as cMin - cMax.
    // As input MUST SPECIFY either Cmin/cMax or boxDimX/Y/Z (if not set, cMin/Max computed from these).

    ChVector3d dimensions = sysSPH.GetContainerDim();
    std::cout << "Default constructor container dimensions are: " << dimensions << std::endl;

    // Create Fluid region and discretize with SPH particles
    ChVector3d boxCenter(0.0, 0.0, bzDim / 2);
    ChVector3d boxHalfDim(bxDim / 2, byDim / 2,
                          bzDim / 2 - initSpace0);  // z lower because then bce particles along z direction.
    // Use a chrono sampler to create a bucket of points
    chrono::utils::ChGridSampler<> sampler(initSpace0);
    chrono::utils::ChGenerator::PointVector points = sampler.SampleBox(boxCenter, boxHalfDim);

    // Add fluid particles from the sampler points to the FSI system
    // Following version of AddSPHParticle method adds particles with velocity set to 0 and default
    // properties taken from the dafault parameters.


    size_t numPart = points.size();
    for (int i = 0; i < numPart; i++) {
        sysSPH.AddSPHParticle(points[i], vel0);
    }

    // Fixed WALL bce markers: create Ground Body (fixed) and add it to the system.
    // Create solid region and attach BCE SPH particles
    auto ground = chrono_types::make_shared<ChBody>();
    ground->SetFixed(true);
    ground->EnableCollision(false);
    sysMBS.AddBody(ground);

    // From ground body create the bce particles. Here in the box surronding the fluid but only along + and - Z
    // direction.
    std::vector<ChVector3d> ground_bce = sysSPH.CreatePointsBoxContainer(
        ChVector3d(bxDim, byDim, bzDim), {0, 0, 2});  // create markers for box container of specified size.
    // following function adds the bce markers as rigid body markers, not as wall/boundary ones. Use other functions
    // like sysCFD.AddBceBoundary to explicitely set them as boundary particles.
    sysFSI_csph.AddFsiBody(ground, ground_bce, ChFrame<>(ChVector3d(0, 0, bzDim / 2), QUNIT), false);


    // Time step for CFD MUST BE SET.
    sysFSI_csph.SetStepSizeCFD(double(1e-3));  // Complete construction of the fluid system
    sysFSI_csph.SetVerbose(true);               // verbose true prints series of information on screen.
    sysSPH.SetKernRadEvolution(cmp::H_evolution_csph::CONSTANT);
    sysSPH.SetViscosityMethod(cmp::ViscosityMethod_csph::ARTIFICIAL_MONAGHAN);
    sysSPH.SetRhoEvolution(cmp::Rho_evolution_csph::DIFFERENTIAL);
    // sysSPH.SetADKECoeff(1, 0, 1.5);
    sysSPH.SetUseVariableTimeStep(false);
    //sysSPH.SetIsUniform(true);
    sysFSI_csph.Initialize();

    // DEBUG ---- REMEMBER TO DELETE FUNCTION -----
    auto manager = sysSPH.GetDataManager();
    std::cout << "Using density value of " << rho0 << " and energy value of " << e0 << std::endl;
    std::cout << "The pressure value is "  << sysSPH.GetBasePressure() << " (sysSPH method) or "
              << cmp::Eos_csph(rho0, e0, *manager->paramsH) << " (Eos_csph)" << std::endl;
    std::cout << "The default particle mass computed as rho*volume is " << rho0 * std::pow(initSpace0, 3) << std::endl;
    std::cout << "Instead markerMass is " << manager->paramsH->markerMass << std::endl;
    
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
        auto col_callback = chrono_types::make_shared<cmp::ParticleDensityColorCallback_csph>(0, 4);  // 0.04

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
    double dT = sysSPH.GetStepSize();
    double time = 0;
    int sim_frame = 0;
    int out_frame = 0;
    int render_frame = 0;
    t_end = 2;

    // in debug mode perform only one step
    int count = 0;
    #ifndef NDEBUG
    while (time < t_end) {
        std::cout << "Iteration " << count++ << std::endl;
        sysFSI_csph.DoStepDynamics(dT);
        time += dT;
        
    } 
    
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
        sysFSI_csph.DoStepDynamics(dT);

        time += dT;
        sim_frame++;
    }

    timer.stop();
    std::cout << "\nSimulation time: " << timer() << " seconds\n" << std::endl;

    
#ifdef CHRONO_POSTPROCESS
    std::vector<ChVector3d> pos = sysSPH.GetParticlePositions();
    std::vector<ChVector3d> rhopresen = sysSPH.GetParticleFluidProperties();
    std::vector<ChVector3d> vel = sysSPH.GetParticleVelocities();
    size_t n_fluid = sysSPH.GetNumFluidMarkers();

    ChVectorDynamic<> data_x(n_fluid);
    ChVectorDynamic<> data_y(n_fluid);
    ChVectorDynamic<> data_z(n_fluid);
    ChVectorDynamic<> data_rho(n_fluid);
    ChVectorDynamic<> data_pres(n_fluid);
    ChVectorDynamic<> data_en(n_fluid);
    ChVectorDynamic<> data_vx(n_fluid);
    ChVectorDynamic<> data_vy(n_fluid);
    ChVectorDynamic<> data_vz(n_fluid);


    for (int i = 0; i < n_fluid; ++i) {
        double x = pos[i].x();
        double y = pos[i].y();
        double z = pos[i].z();
        double rho = rhopresen[i].x();
        double pres = rhopresen[i].y();
        double en = rhopresen[i].z();

        double vx = vel[i].x();
        double vy = vel[i].y();
        double vz = vel[i].y();

        data_x(i) = x;
        data_y(i) = y;
        data_z(i) = z;
        data_rho(i) = rho;
        data_pres(i) = pres;
        data_en(i) = en;
        data_vx(i) = vx;
        data_vy(i) = vy;
        data_vz(i) = vz;
    }

    // SAVE THESE VALUES IN FILES FOR MATLAB PROCESSING
    std::string datafilename = out_dir + "/uniform_flow_final_data.dat";
    std::ofstream datafile(datafilename);
    datafile << "# x_pos y_pos z_pos  rho  pres  en  v_x  v_y  v_z  " << std::endl;
    for (size_t i = 0; i < n_fluid; i++)
        datafile << data_x(i) << " " << data_y(i) << " " << data_z(i) << " " << data_rho(i) << " " << data_pres(i) << " "
                 << data_en(i)
                 << " " << data_vx(i)
                 << " " << data_vy(i) << " " << data_vz(i) << std::endl;


    postprocess::ChGnuPlot rho_plot(out_dir + "/density_profile.gpl");
    rho_plot.SetGrid();
    std::string speed_title1 = "Density profile - 1D";
    rho_plot.SetTitle(speed_title1);
    rho_plot.SetLabelX("x");
    rho_plot.SetLabelY("Density");
    rho_plot.Plot(data_x, data_rho, "", " with points lt -1 lw 2 lc rgb'#3333BB' ");
    rho_plot.SetRangeX(-0.5, 0.5);
    // rho_plot.SetRangeY(0, 1.2);

    postprocess::ChGnuPlot pres_plot(out_dir + "/pressure_profile.gpl");
    pres_plot.SetGrid();
    std::string speed_title2 = "Pressure profile - 1D";
    pres_plot.SetTitle(speed_title2);
    pres_plot.SetLabelX("x");
    pres_plot.SetLabelY("Pressure");
    pres_plot.Plot(data_x, data_pres, "", " with points lt -1 lw 2 lc rgb'#3333BB' ");
    pres_plot.SetRangeX(-0.5, 0.5);
   //  pres_plot.SetRangeY(0, 1.2);

    postprocess::ChGnuPlot en_plot(out_dir + "/energy_profile.gpl");
    en_plot.SetGrid();
    std::string speed_title3 = "Energy profile - 1D";
    en_plot.SetTitle(speed_title3);
    en_plot.SetLabelX("x");
    en_plot.SetLabelY("Energy");
    en_plot.Plot(data_x, data_en, "", " with points lt -1 lw 2 lc rgb'#3333BB' ");
    en_plot.SetRangeX(-0.5, 0.5);
    // en_plot.SetRangeY(1.5, 3.2);

    postprocess::ChGnuPlot vel_plot(out_dir + "/vel_profile.gpl");
    en_plot.SetGrid();
    std::string speed_title4 = "Velocity profile - 1D";
    vel_plot.SetTitle(speed_title4);
    vel_plot.SetLabelX("x");
    vel_plot.SetLabelY("Velocity");
    vel_plot.Plot(data_x, data_vx, "", " with points lt -1 lw 2 lc rgb'#3333BB' ");
    //vel_plot.SetRangeX(-0.5, 0.5);
    //vel_plot.SetRangeY(1.5, 3.2);

    
    #endif

    // /*
    while (render && vis->Run()) {
        vis->Render();
    }
    // */

    #endif

    return 0;
}
