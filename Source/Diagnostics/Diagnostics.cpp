#include "Diagnostics.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_RealBox.H>

#include "../Code.H"
#include "../Input/GeometryProperties/GeometryProperties.H"
#include "../Input/MacroscopicProperties/MacroscopicProperties.H"
#include "../Output/Output.H"
#include "../Utils/CodeUtils/CodeUtil.H"
#include "../Utils/SelectWarpXUtils/WarpXUtil.H"

#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB2_IndexSpace_STL.H>
#include <AMReX_EBSupport.H>
#endif

using namespace amrex;

c_Diagnostics::c_Diagnostics()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_Diagnostics() "
                      "Constructor************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    ReadDiagnostics();

#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t}************************c_Diagnostics() "
                      "Constructor************************\n";
#endif
}

c_Diagnostics::~c_Diagnostics()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_Diagnostics() "
                      "Destructor************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t}************************c_Diagnostics() "
                      "Destructor************************\n";
#endif
}

void c_Diagnostics::ReadDiagnostics()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_Diagnostics()::"
                      "ReadDiagnostics()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t\t";
#endif

    amrex::ParmParse pp_diag("diag");

    // setting default values
    specify_using_eb = 0;

    amrex::Print() << "\n##### DIAGNOSTICS PROPERTIES #####\n\n";
    pp_diag.query("specify_using_eb", specify_using_eb);
    amrex::Print() << "##### diag.specify_using_eb: " << specify_using_eb
                   << "\n";

#ifdef AMREX_USE_EB
    if (specify_using_eb)
    {
        ReadEBDiagnostics();
    }
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_Diagnostics()::"
                      "ReadDiagnostics()************************\n";
#endif
}

void c_Diagnostics::InitData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_Diagnostics()::"
                      "InitData()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t\t";
#endif

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();

#ifdef AMREX_USE_EB
    if (specify_using_eb)
    {
        SetGeometry(&rGprop.geom, &rGprop.ba, &rGprop.dm, specify_using_eb);
        CreateFactory();
    }
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_Diagnostics()::"
                      "InitData()************************\n";
#endif
}

void c_Diagnostics::ComputeAndWriteDiagnostics(int step, amrex::Real time)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t\t{************************c_Diagnostics()::"
           "ComputeAndWriteDiagnostics()************************\n";
    amrex::Print() << "\t\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

#ifdef AMREX_USE_EB
    if (specify_using_eb) ComputeAndWriteEBDiagnostics(step, time);
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t\t}************************c_Diagnostics()::"
                      "ComputeAndWriteDiagnostics()************************\n";
#endif
}
