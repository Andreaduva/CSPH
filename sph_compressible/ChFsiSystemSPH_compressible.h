// =============================================================================
// Author: Andrea D'Uva - 2025
// =============================================================================
//
// Implementation of FSI system using an SPH fluid solver.
// Compressible SPH version.
// This class groups the compressible FluidSystemSPH class (by reference), the MBS 
// class (by reference), and the FsiInterfaceSPH class (by pointer) and coordinates them.
// =============================================================================

#ifndef CH_FSI_SYSTEM_SPH_COMPRESSIBLE_H
#define CH_FSI_SYSTEM_SPH_COMPRESSIBLE_H

#include "chrono_fsi/ChFsiSystem.h"

#include "chrono_fsi/sph/ChFsiFluidSystemSPH.h"
#include "chrono_fsi/sph_compressible/ChFsiFluidSystemSPH_compressible.h"

namespace chrono::fsi::sph {
namespace compressible {


// FSI system using an SPH-based fluid solver.
class CH_FSI_API ChFsiSystemSPH_csph : public ChFsiSystem {
  public:
    // Construct an FSI system to couple the specified Chrono MBS system and SPH fluid system.
    // Data exchange between the two systems is done through an FSI interface object, owned by the FSI system.
    // If 'use_generic_interface = true', the FSI system will use a generic FSI interface. Otherwise (default), use a
    // custom FSI interface which works directly with the data manager of the SPH fluid solver and thus circumvents
    // additional data movement.
    ChFsiSystemSPH_csph(ChSystem& sysMBS, ChFsiFluidSystemSPH_csph& sysSPH, bool use_generic_interface = false);
    ~ChFsiSystemSPH_csph();

    // Access the associated compressible SPH fluid system.
    ChFsiFluidSystemSPH_csph& GetFluidSystemSPH() const;

    // Allow using the AddFsiBody method from parent class
    // Needed in c++ because if in the derived class I declare a function with the same name of the base one, then the
    // base function will be hidden by name. No matter the function signature. So when calling AddFsiBody from derived
    // object, the compiler will look in the derived class functions only and throw an error if the signature doesn't
    // match, it will not go into base class methods to find a suitable function. In this case need the "using
    // BaseClass::MethodName" to explicitely tell compiler to look there as well.
    using ChFsiSystem::AddFsiBody;
    // This function signature is: std::shared_ptr<FsiBody> AddFsiBody(std::shared_ptr<ChBody> body,
    //                                                                 std::shared_ptr<ChBodyGeometry> geometry,
    //                                                                 bool check_embedded);

    // Add a rigid body to the FSI system with given set of BCE markers.
    // BCE marker points are assumed to be specified in the given frame (itself relative to the given body).
    std::shared_ptr<FsiBody> AddFsiBody(std::shared_ptr<ChBody> body,
                                        const std::vector<ChVector3d>& bce,
                                        const ChFrame<>& rel_frame,
                                        bool check_embedded);

    // Add a set of boundary BCE markers.
    // BCE marker points are assumed to be specified in the given frame (itself relative to the global frame).
    // These markers are not associated to any solid body, their type is BCE_WALL. They are simply added to the
    // sph_markers vector, not in rigid_bce or similar ones.
    void AddFsiBoundary(const std::vector<ChVector3d>& bce, const ChFrame<>& rel_frame);

    // same as above but specify a value of h and mass for the specific boundary:
    void AddFsiBoundary(const std::vector<ChVector3d>& bce, const ChFrame<>& rel_frame, const double h, const double mass);

    void AddFsiBoundary(const std::vector<ChVector3d>& bce, const ChFrame<>& rel_frame, const std::vector<double>& h_vec, const std::vector<double>& mass_vec);

  private:
    ChFsiFluidSystemSPH_csph& m_sysSPH;  //< cached SPH fluid solver
    bool m_generic_fsi_interface;

  // protected and private attributes inherited from base class:
  //  ChSystem& m_sysMBS;                               
  //  ChFsiFluidSystem& m_sysCFD;                       
  //  std::shared_ptr<ChFsiInterface> m_fsi_interface;  
  //  void AdvanceCFD(double step,double threshold);   main functions called inside DoStepDynamics.                                        
  //  void AdvanceMBS(double step,double threshold);   They integrate in time for DeltaT = step, allowing for a further hidden
                                                  ///  subdivision of the interval, but without going under the treshold value.
  //  double m_step_MBD;
  //  double m_step_CFD;  
  //  double m_time;         < current fluid dynamics simulation time (overall simulation time taken)
  //  ChTimer m_timer_step;  < timer for concurrent integration step of both MBS and Fluid systems.
  //  ChTimer m_timer_FSI;   < timer for FSI data exchange between Fluid and MBS systems.
  //  double m_timer_CFD;    < timer for fluid dynamics integration      --> set by AdvanceCFD
  //  double m_timer_MBD;    < timer for multibody dynamics integration  --> set by the AdvanceMBS function
  //  double m_RTF;          < real-time factor (simulation time / simulated time)
  //  double m_ratio_MBD;    < fraction of step simulation time for MBS integration


};




}  // end namespace compressible
}  // end namespace chrono::fsi::sph

#endif
