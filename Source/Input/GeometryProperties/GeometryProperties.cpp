#include "GeometryProperties.H"

#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
//#include "Code.H"

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_RealBox.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBSupport.H>
#endif

using namespace amrex;

c_GeometryProperties::c_GeometryProperties()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_GeometryProperties "
                      "Constructor()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    ReadData();

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_GeometryProperties "
                      "Constructor()************************\n";
#endif
}

void c_GeometryProperties::ReadData()
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t{************************c_GeometryProperties::"
           "ReadData()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    ParseBasicDomainInput();

#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t}************************c_GeometryProperties::"
           "ReadData()************************\n";
#endif
}

void c_GeometryProperties::InitData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_GeometryProperties::"
                      "InitData()************************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    InitializeBoxArrayAndDistributionMap();

#ifdef AMREX_USE_EB
    if (embedded_boundary_flag) pEB->BuildGeometry(&geom, &ba, &dm);
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_GeometryProperties::"
                      "InitData()************************\n";
#endif
}

void c_GeometryProperties::ParseBasicDomainInput()
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t{************************c_GeometryProperties::"
           "ParseBasicDomainInput()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t";
#endif

    amrex::Vector<int> num_cell;
    amrex::Vector<amrex::Real> prob_min(AMREX_SPACEDIM);
    amrex::Vector<amrex::Real> prob_max(AMREX_SPACEDIM);
    amrex::Vector<amrex::Real> mg{AMREX_D_DECL(128, 128, 128)};  // default
                                                                 // values
    amrex::Vector<amrex::Real> bf{AMREX_D_DECL(8, 8, 8)};
    amrex::Vector<amrex::Real> periodicity{AMREX_D_DECL(0, 0, 0)};
    std::string coord_sys_str = "cartesian";
    coord_sys = amrex::CoordSys::cartesian;  // default
    embedded_boundary_flag = 0;

    amrex::ParmParse pp_domain("domain");

    getArrWithParser(pp_domain, "prob_lo", prob_min, 0, AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(prob_lo.size() == AMREX_SPACEDIM);

    getArrWithParser(pp_domain, "prob_hi", prob_max, 0, AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(prob_hi.size() == AMREX_SPACEDIM);

    getArrWithParser(pp_domain, "n_cell", num_cell, 0, AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(n_cell.size() == AMREX_SPACEDIM);

    queryArrWithParser(pp_domain, "max_grid_size", mg, 0, AMREX_SPACEDIM);

    queryArrWithParser(pp_domain, "blocking_factor", bf, 0, AMREX_SPACEDIM);

    pp_domain.queryarr("is_periodic", periodicity);

    pp_domain.query("coord_sys", coord_sys_str);

    pp_domain.query("embedded_boundary", embedded_boundary_flag);

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        n_cell[i] = num_cell[i];
        prob_lo[i] = prob_min[i];
        prob_hi[i] = prob_max[i];
        max_grid_size[i] = mg[i];
        blocking_factor[i] = bf[i];
        is_periodic[i] = periodicity[i];
        dx[i] = (prob_hi[i] - prob_lo[i]) / n_cell[i];
    }
    if (coord_sys_str == "cartesian")
    {
        coord_sys = amrex::CoordSys::cartesian;
    }
    else if (coord_sys_str == "radial")
    {
        coord_sys = amrex::CoordSys::RZ;
    }

    amrex::Print() << "\n##### GEOMETRY PROPERTIES #####\n\n";
    amrex::Print() << "##### n_cell: ";
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        amrex::Print() << n_cell[i] << "  ";
    amrex::Print() << "\n";
    amrex::Print() << "##### prob_lo: ";
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        amrex::Print() << prob_lo[i] << "  ";
    amrex::Print() << "\n";
    amrex::Print() << "##### prob_hi: ";
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        amrex::Print() << prob_hi[i] << "  ";
    amrex::Print() << "\n";
    amrex::Print() << "##### dx,dy,dz: ";
    for (int i = 0; i < AMREX_SPACEDIM; ++i) amrex::Print() << dx[i] << "  ";
    amrex::Print() << "\n";
    amrex::Print() << "##### max_grid_size: ";
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        amrex::Print() << max_grid_size[i] << "  ";
    amrex::Print() << "\n";
    amrex::Print() << "##### blocking_factor: ";
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        amrex::Print() << blocking_factor[i] << "  ";
    amrex::Print() << "\n";
    amrex::Print() << "##### is_periodic: ";
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        amrex::Print() << is_periodic[i] << "  ";
    amrex::Print() << "\n";
    amrex::Print() << "##### coord_sys: " << coord_sys << "\n";
    amrex::Print() << "##### embedded_boundary_flag: " << embedded_boundary_flag
                   << "\n";

#ifdef AMREX_USE_EB
    if (embedded_boundary_flag)
        pEB = std::make_unique<c_EmbeddedBoundaries>(&dx);
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t}************************c_GeometryProperties::"
                      "ParseBasicDomainInput()************************\n";
#endif
}

void c_GeometryProperties::InitializeBoxArrayAndDistributionMap()
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t{************************c_GeometryProperties::"
           "InitializeBoxArrayAndDistributionMap()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));  // domain low
    amrex::IntVect dom_hi(AMREX_D_DECL(n_cell[0] - 1, n_cell[1] - 1,
                                       n_cell[2] - 1));  // domain high

    amrex::Box domain(dom_lo,
                      dom_hi);  // Make a single box that is the entire domain

    ba.define(
        domain);  // initialize the boxarray 'ba' from the single box 'domain'

    ba.maxSize(max_grid_size);  // break up ba into chunks no larger than
                                // 'max_grid_size' along a direction

    amrex::RealBox real_box({AMREX_D_DECL(prob_lo[0], prob_lo[1], prob_lo[2])},
                            {AMREX_D_DECL(prob_hi[0], prob_hi[1],
                                          prob_hi[2])});  // physical domain

    geom.define(domain, real_box, coord_sys,
                is_periodic);  // define the geom object

    dm.define(ba);

#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t}************************c_GeometryProperties::"
           "InitializeBoxArrayAndDistributionMap()************************\n";
#endif
}

bool c_GeometryProperties::Is_Point_Inside_Physical_Domain(
    const amrex::Real *pos)
{
    // auto& rCode = c_Code::GetInstance();
    // auto& rGprop = rCode.get_GeometryProperties();
    // const auto& PL = rGprop.get_ProbLo();
    // const auto& PH = rGprop.get_ProbHi();

    const amrex::Real eps = 1e-14;

    for (int j = 0; j < AMREX_SPACEDIM; ++j)
    {
        if ((pos[j] - eps) < prob_lo[j] || (pos[j] + eps) > prob_hi[j])
        {
            return false;
        }
    }
    return true;
}

bool c_GeometryProperties::Is_Position_Inside_BoxArray(const amrex::Real *pos)
{
    amrex::Real lx = (pos[0] - prob_lo[0] - dx[0] * 0.5) / dx[0];
    amrex::Real ly = (pos[1] - prob_lo[1] - dx[1] * 0.5) / dx[1];
    amrex::Real lz = (pos[2] - prob_lo[2] - dx[2] * 0.5) / dx[2];

    int i = static_cast<int>(amrex::Math::floor(lx));
    int j = static_cast<int>(amrex::Math::floor(ly));
    int k = static_cast<int>(amrex::Math::floor(lz));

    return ba.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)));
}

int c_GeometryProperties::Get_BoxID(const amrex::Real *pos)
{
    amrex::Real lx = (pos[0] - prob_lo[0] - dx[0] * 0.5) / dx[0];
    amrex::Real ly = (pos[1] - prob_lo[1] - dx[1] * 0.5) / dx[1];
    amrex::Real lz = (pos[2] - prob_lo[2] - dx[2] * 0.5) / dx[2];

    int i = static_cast<int>(amrex::Math::floor(lx));
    int j = static_cast<int>(amrex::Math::floor(ly));
    int k = static_cast<int>(amrex::Math::floor(lz));

    IntVect iv = amrex::IntVect(AMREX_D_DECL(i, j, k));

    std::vector<std::pair<int, Box>> isects;

    bool findfirst = true;
    ba.intersections(Box(iv, iv), isects, findfirst,
                     amrex::IntVect::TheZeroVector());

    int box = isects.empty() ? -1 : isects[0].first;

    return box;
}
