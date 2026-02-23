// =============================================================================
// Author: Andrea D'Uva -2025
// =============================================================================
//
// Implementation of FSI system using an SPH fluid solver.
// Version modified to consider compressible SPH
// =============================================================================

#include <iostream>
#include <algorithm>

#include "chrono/utils/ChUtils.h"

#include "chrono_fsi/sph/ChFsiSystemSPH.h"
#include "chrono_fsi/sph/ChFsiInterfaceSPH.h"

#include "chrono_fsi/sph_compressible/ChFsiSystemSPH_compressible.h"
#include "chrono_fsi/sph_compressible/ChFsiInterfaceSPH_compressible.h"


namespace chrono::fsi::sph {
namespace compressible {

// constructor. Will call base class constructor with ChSystem& and ChFsiFluidSystemSPH& (instead of the base class
// ChFsiFluidSystem). So base class will have a ChFsiFluidSystem& m_sys_CFD that is actually a reference to a derived
// object. This allows polymorphism to work. But derived class also have an additional ChFsiFluidSystemSPH& m_sysSPH
// which is correctly initialized.
ChFsiSystemSPH_csph::ChFsiSystemSPH_csph(ChSystem& sysMBS, ChFsiFluidSystemSPH_csph& sysSPH, bool use_generic_interface)
    : ChFsiSystem(sysMBS, sysSPH), m_sysSPH(sysSPH), m_generic_fsi_interface(use_generic_interface) {
    if (use_generic_interface) {
        std::cout << "Create an FSI system using a generic FSI interface" << std::endl;
        m_fsi_interface = chrono_types::make_shared<ChFsiInterfaceGeneric>(sysMBS, sysSPH);
    } else {
        std::cout << "Create an FSI system using a custom SPH FSI interface" << std::endl;
        m_fsi_interface = chrono_types::make_shared<ChFsiInterfaceSPH_csph>(sysMBS, sysSPH);
    }
}

ChFsiSystemSPH_csph::~ChFsiSystemSPH_csph() {}

ChFsiFluidSystemSPH_csph& ChFsiSystemSPH_csph::GetFluidSystemSPH() const {
    return m_sysSPH;
}

// Additional function wrt base class. This takes as input a ChBody and, instead of a ChGeometry, a set of bce markers
// and the frame they are expresses into (frame itself assumed relative to body)
std::shared_ptr<FsiBody> ChFsiSystemSPH_csph::AddFsiBody(std::shared_ptr<ChBody> body,
                                                         const std::vector<ChVector3d>& bce,
                                                         const ChFrame<>& rel_frame,
                                                         bool check_embedded) {
    // Add the FSI body with no geometry with the base class method
    auto fsi_body = ChFsiSystem::AddFsiBody(body, nullptr, check_embedded);

    // Explicitly set the BCE marker locations to the FsiSPHBody
    auto& fsisph_body = m_sysSPH.m_bodies.back();

    fsisph_body.bce_ids.resize(bce.size(), (int)fsisph_body.fsi_body->index);

    // from the input ChBody we get the transformation local_body_frame->global_frame
    ChFramed abs_frame = body->GetFrameRefToAbs() * rel_frame;  // get absolute frame for bce coordinates

    // transform bce coordinates and insert them in bce_coords and bce vectors
    // back_inserter constructs an iterator that inserts new elements at the end of the vector

    // transform bce coordinates into the body-relative frame, using rel_frame
    std::transform(bce.begin(), bce.end(), std::back_inserter(fsisph_body.bce_coords),
                   [&rel_frame](const ChVector3d& v) { return rel_frame.TransformPointLocalToParent(v); });
    // transform bce coordinates into global/absolute frame
    std::transform(bce.begin(), bce.end(), std::back_inserter(fsisph_body.bce),
                   [&abs_frame](const ChVector3d& v) { return abs_frame.TransformPointLocalToParent(v); });

    return fsi_body;
}

void ChFsiSystemSPH_csph::AddFsiBoundary(const std::vector<ChVector3d>& bce, const ChFrame<>& frame) {
    m_sysSPH.AddBCEBoundary(bce, frame);
}

void ChFsiSystemSPH_csph::AddFsiBoundary(const std::vector<ChVector3d>& bce, const ChFrame<>& frame, const double h, const double mass) {
    m_sysSPH.AddBCEBoundary(bce, frame, h, mass);
}

void ChFsiSystemSPH_csph::AddFsiBoundary(const std::vector<ChVector3d>& bce,
                                         const ChFrame<>& frame,
                                         const std::vector<double>& h_vec,
                                         const std::vector<double>& mass_vec) {
    m_sysSPH.AddBCEBoundary(bce, frame, h_vec, mass_vec);
}

}  // end namespace compressible
}  // end namespace chrono::fsi::sph
