#include "MLMG.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_RealBox.H>

#include "../../Utils/CodeUtils/CodeUtil.H"
#include "../../Utils/SelectWarpXUtils/TextMsg.H"
#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Code.H"
#include "Input/BoundaryConditions/BoundaryConditions.H"
#include "Input/GeometryProperties/GeometryProperties.H"
#include "Input/MacroscopicProperties/MacroscopicProperties.H"

#ifdef AMREX_USE_HYPRE
#include <AMReX_Hypre.H>
#endif

using namespace amrex;

c_MLMGSolver::c_MLMGSolver()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_MLMGSolver "
                      "Constructor()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    ReadData();

    all_homogeneous_boundaries = true;
    some_functionbased_inhomogeneous_boundaries = false;
    some_constant_inhomogeneous_boundaries = false;

    AssignLinOpBCTypeToBoundaries();

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_MLMGSolver "
                      "Constructor()************************\n";
#endif
}

c_MLMGSolver::~c_MLMGSolver()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_MLMGSolver "
                      "Destructor()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    alpha = nullptr;
    beta = nullptr;
    soln = nullptr;
    rhs = nullptr;
    robin_a = nullptr;
    robin_b = nullptr;
    robin_f = nullptr;

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_MLMGSolver "
                      "Destructor()************************\n";
#endif
}

void c_MLMGSolver::ReadData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t{************************c_MLMGSolver::"
                      "ReadData()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    ParmParse pp_mlmg("mlmg");
    pp_mlmg.get("ascalar", ascalar);
    pp_mlmg.get("bscalar", bscalar);

    if (queryWithParser(pp_mlmg, "set_verbose", set_verbose))
    {
    }
    else
    {
        set_verbose = 0;
        std::stringstream warnMsg;
        warnMsg << "MLMG parameter 'set_verbose'"
                << " is not specified in the input file. The default value of "
                << set_verbose << " is used.";
        c_Code::GetInstance().RecordWarning("MLMG properties", warnMsg.str());
    }

    if (queryWithParser(pp_mlmg, "max_order", max_order))
    {
    }
    else
    {
        max_order = 2;
        std::stringstream warnMsg;
        warnMsg << "MLMG parameter 'max_order'"
                << " is not specified in the input file. The default value of "
                << max_order << " is used.";
        c_Code::GetInstance().RecordWarning("MLMG properties", warnMsg.str());
    }

    if (queryWithParser(pp_mlmg, "relative_tolerance", relative_tolerance))
    {
    }
    else
    {
        relative_tolerance = 1.0e-10;
        std::stringstream warnMsg;
        warnMsg << "MLMG parameter 'relative_tolerance'"
                << " is not specified in the input file. The default value of "
                << relative_tolerance << " is used.";
        c_Code::GetInstance().RecordWarning("MLMG properties", warnMsg.str());
    }

    if (queryWithParser(pp_mlmg, "absolute_tolerance", absolute_tolerance))
    {
    }
    else
    {
        absolute_tolerance = 0.;
        std::stringstream warnMsg;
        warnMsg << "MLMG parameter 'absolute_tolerance'"
                << " is not specified in the input file. The default value of "
                << absolute_tolerance << " is used.";
        c_Code::GetInstance().RecordWarning("MLMG properties", warnMsg.str());
    }

    auto &rCode = c_Code::GetInstance();
    auto &rMprop = rCode.get_MacroscopicProperties();
    std::map<std::string, int>::iterator it_Mprop;

    pp_mlmg.get("alpha", alpha_str);

    // Assert that the multifab name is specified through
    // 'macroscopic.fields_to_define' in the input file.
    it_Mprop = rMprop.map_param_all.find(alpha_str);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        it_Mprop != rMprop.map_param_all.end(),
        "'" + alpha_str +
            "' is not specified through 'macroscopic.fields_to_define' in the "
            "input file.");

    pp_mlmg.get("beta", beta_str);

    it_Mprop = rMprop.map_param_all.find(beta_str);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        it_Mprop != rMprop.map_param_all.end(),
        beta_str +
            " is not specified through 'macroscopic.fields_to_define' "
            "in the input file.");

    pp_mlmg.get("soln", soln_str);

    it_Mprop = rMprop.map_param_all.find(soln_str);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        it_Mprop != rMprop.map_param_all.end(),
        soln_str +
            " is not specified through 'macroscopic.fields_to_define' "
            "in the input file.");

    pp_mlmg.get("rhs", rhs_str);

    it_Mprop = rMprop.map_param_all.find(rhs_str);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        it_Mprop != rMprop.map_param_all.end(),
        rhs_str +
            " is not specified through 'macroscopic.fields_to_define' in "
            "the input file.");

    auto &rBC = rCode.get_BoundaryConditions();
    auto some_robin_boundaries = rBC.some_robin_boundaries;

    if (some_robin_boundaries)
    {
        pp_mlmg.get("robin_a", robin_a_str);
        it_Mprop = rMprop.map_param_all.find(robin_a_str);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            it_Mprop != rMprop.map_param_all.end(),
            robin_a_str +
                " is not specified through "
                "'macroscopic.fields_to_define' in the input file.");

        pp_mlmg.get("robin_b", robin_b_str);
        it_Mprop = rMprop.map_param_all.find(robin_b_str);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            it_Mprop != rMprop.map_param_all.end(),
            robin_b_str +
                " is not specified through "
                "'macroscopic.fields_to_define' in the input file.");

        pp_mlmg.get("robin_f", robin_f_str);
        it_Mprop = rMprop.map_param_all.find(robin_f_str);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            it_Mprop != rMprop.map_param_all.end(),
            robin_f_str +
                " is not specified through "
                "'macroscopic.fields_to_define' in the input file.");
    }

    amrex::Print() << "\n##### MLMG Solver #####\n\n";
    amrex::Print() << "##### ascalar: " << ascalar << "\n";
    amrex::Print() << "##### bscalar: " << bscalar << "\n";
    amrex::Print() << "##### set_verbose: " << set_verbose << "\n";
    amrex::Print() << "##### max_order: " << max_order << "\n";
    amrex::Print() << "##### relative_tolerance: " << relative_tolerance
                   << "\n";
    amrex::Print() << "##### absolute_tolerance: " << absolute_tolerance
                   << "\n";
    amrex::Print() << "##### alpha is denoted by: " << alpha_str << "\n";
    amrex::Print() << "##### beta is denoted by: " << beta_str << "\n";
    amrex::Print() << "##### soln is denoted by: " << soln_str << "\n";
    amrex::Print() << "##### rhs is denoted by: " << rhs_str << "\n";
    if (some_robin_boundaries)
    {
        amrex::Print() << "##### robin_a is denoted by: " << robin_a_str
                       << "\n";
        amrex::Print() << "##### robin_b is denoted by: " << robin_b_str
                       << "\n";
        amrex::Print() << "##### robin_f is denoted by: " << robin_f_str
                       << "\n";
    }
#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t}************************c_MLMGSolver::ReadData("
                      ")************************\n";
#endif
}

void c_MLMGSolver::AssignLinOpBCTypeToBoundaries()
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t{************************c_MLMGSolver::"
           "AssignLinOpBCTypeToBoundaries()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t";
#endif
    /*
        LinOpBCType::interior:
        LinOpBCType::Dirichlet:
        LinOpBCType::Neumann:
        LinOpBCType::reflect_odd:
        LinOpBCType::Marshak:
        LinOpBCType::SanchezPomraning:
        LinOpBCType::inflow:
        LinOpBCType::inhomogNeumann:
        LinOpBCType::Robin:
        LinOpBCType::Periodic:
    */

    auto &rCode = c_Code::GetInstance();
    auto &rBC = rCode.get_BoundaryConditions();
    auto &map_boundary_type = rBC.map_boundary_type;
    auto &bcType_2d = rBC.bcType_2d;
    auto &map_bcAny_2d = rBC.map_bcAny_2d;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < AMREX_SPACEDIM; ++j)
        {
            switch (map_boundary_type[bcType_2d[i][j]])
            {
                case s_BoundaryConditions::dir:
                {
                    LinOpBCType_2d[i][j] = LinOpBCType::Dirichlet;

                    if (map_bcAny_2d[i][j] == "inhomogeneous_constant")
                    {
                        all_homogeneous_boundaries = false;
                        some_constant_inhomogeneous_boundaries = true;
                    }
                    if (map_bcAny_2d[i][j] == "inhomogeneous_function")
                    {
                        all_homogeneous_boundaries = false;
                        some_functionbased_inhomogeneous_boundaries = true;
                    }
                    break;
                }
                case s_BoundaryConditions::neu:
                {
                    if (map_bcAny_2d[i][j] == "homogeneous")
                    {
                        LinOpBCType_2d[i][j] = LinOpBCType::Neumann;
                    }
                    else if (map_bcAny_2d[i][j] == "inhomogeneous_constant")
                    {
                        LinOpBCType_2d[i][j] = LinOpBCType::inhomogNeumann;
                        all_homogeneous_boundaries = false;
                        some_constant_inhomogeneous_boundaries = true;
                    }
                    else if (map_bcAny_2d[i][j] == "inhomogeneous_function")
                    {
                        LinOpBCType_2d[i][j] = LinOpBCType::inhomogNeumann;
                        all_homogeneous_boundaries = false;
                        some_functionbased_inhomogeneous_boundaries = true;
                    }
                    break;
                }
                case s_BoundaryConditions::per:
                {
                    LinOpBCType_2d[i][j] = LinOpBCType::Periodic;
                    break;
                }
                case s_BoundaryConditions::rob:
                {
                    LinOpBCType_2d[i][j] = LinOpBCType::Robin;
                    all_homogeneous_boundaries = false;
                    some_functionbased_inhomogeneous_boundaries = true;
                    break;
                }
                case s_BoundaryConditions::ref:
                {
                    LinOpBCType_2d[i][j] = LinOpBCType::reflect_odd;
                    break;
                }
            }
        }
    }

#ifdef PRINT_LOW
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < AMREX_SPACEDIM; ++j)
        {
            amrex::Print() << prt << "side: " << i << " direction: " << j
                           << " LinOpBCType: " << LinOpBCType_2d[i][j]
                           << " type: " << map_bcAny_2d[i][j] << "\n";
        }
    }
#endif
#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t}************************c_MLMGSolver::"
           "AssignLinOpBCTypeToBoundaries()************************\n";
#endif
}

void c_MLMGSolver::InitData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_MLMGSolver::InitData("
                      ")************************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    Set_MLMG_CellCentered_Multifabs();

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();

#ifdef AMREX_USE_EB
    if (rGprop.is_eb_enabled())
    {
        Setup_MLEBABecLaplacian_ForPoissonEqn();
    }
#endif
    if (!rGprop.is_eb_enabled())
    {
        Setup_MLABecLaplacian_ForPoissonEqn();
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_MLMGSolver::InitData()***"
                      "*********************\n";
#endif
}

void c_MLMGSolver::Set_MLMG_CellCentered_Multifabs()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_MLMGSolver::Set_"
                      "MLMG_CellCentered_Multifabs()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    auto &rCode = c_Code::GetInstance();
    auto &rMprop = rCode.get_MacroscopicProperties();

    alpha = rMprop.get_p_mf(alpha_str);
    beta = rMprop.get_p_mf(beta_str);
    soln = rMprop.get_p_mf(soln_str);
    rhs = rMprop.get_p_mf(rhs_str);

    auto &rBC = rCode.get_BoundaryConditions();

    if (rBC.some_robin_boundaries)
    {
        robin_a = rMprop.get_p_mf(robin_a_str);
        robin_b = rMprop.get_p_mf(robin_b_str);
        robin_f = rMprop.get_p_mf(robin_f_str);
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_MLMGSolver::Set_MLMG_"
                      "CellCentered_Multifabs()************************\n";
#endif
}

#ifdef AMREX_USE_EB
void c_MLMGSolver::Setup_MLEBABecLaplacian_ForPoissonEqn()
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t{************************c_MLMGSolver::Setup_"
           "MLEBABecLaplacian_ForPoissonEqn()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    auto &ba = rGprop.ba;
    auto &dm = rGprop.dm;
    auto &geom = rGprop.geom;

    p_mlebabec = std::make_unique<amrex::MLEBABecLap>();
    p_mlebabec->define({geom}, {ba}, {dm}, info,
                       {&*rGprop.pEB->p_factory_union});

    // Force singular system to be solvable
    p_mlebabec->setEnforceSingularSolvable(false);

    // set order of stencil
    p_mlebabec->setMaxOrder(max_order);

    // assign domain boundary conditions to the solver
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    p_mlebabec->setDomainBC(LinOpBCType_2d[0], LinOpBCType_2d[1]);

    [[maybe_unused]] auto &rBC = rCode.get_BoundaryConditions();
    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries

    int amrlev = 0;  // refers to the setcoarsest level of the solve

    if (some_constant_inhomogeneous_boundaries)
    {
        Fill_Constant_Inhomogeneous_Boundaries();
    }
    if (some_functionbased_inhomogeneous_boundaries)
    {
        Fill_FunctionBased_Inhomogeneous_Boundaries();
        // Note that previously in c_BoundaryCondition constructor, it has been
        // asserted that the use of robin is not supported with embedded
        // boundaries.
    }
    soln->FillBoundary(geom.periodicity());

    p_mlebabec->setLevelBC(amrlev, soln);

    // set scaling factors
    p_mlebabec->setScalars(ascalar, bscalar);

    // set alpha, and beta_fc coefficients
    p_mlebabec->setACoeffs(amrlev, *alpha);

    // beta_fc =  std::make_unique<amrex::FArrayBox,AMREX_SPACEDIM>;
    // beta_fc is a face-centered multifab
    AMREX_D_TERM(beta_fc[0].define(convert(ba, IntVect(AMREX_D_DECL(1, 0, 0))),
                                   dm, 1, 0);
                 ,
                 beta_fc[1].define(convert(ba, IntVect(AMREX_D_DECL(0, 1, 0))),
                                   dm, 1, 0);
                 ,
                 beta_fc[2].define(convert(ba, IntVect(AMREX_D_DECL(0, 0, 1))),
                                   dm, 1, 0););

    beta->FillBoundary();

    // converts from cell-centered permittivity to face-center and store in
    // beta_fc
    Multifab_Manipulation::AverageCellCenteredMultiFabToCellFaces(*beta,
                                                                  beta_fc);

    p_mlebabec->setBCoeffs(amrlev, amrex::GetArrOfConstPtrs(beta_fc));

    if (rGprop.pEB->specify_inhomogeneous_dirichlet == 0)
    {
        // p_mlebabec->setEBHomogDirichlet(amrlev,
        // *rGprop.pEB->p_surf_beta_union);
        p_mlebabec->setEBHomogDirichlet(amrlev, *beta);
    }
    else
    {
        // p_mlebabec->setEBDirichlet(amrlev, *rGprop.pEB->p_surf_soln_union,
        // *rGprop.pEB->p_surf_beta_union);
        p_mlebabec->setEBDirichlet(amrlev, *rGprop.pEB->p_surf_soln_union,
                                   *beta);
    }

    pMLMG = std::make_unique<MLMG>(*p_mlebabec);

    pMLMG->setVerbose(set_verbose);

#ifdef AMREX_USE_HYPRE
    pMLMG->setBottomSolver(MLMG::BottomSolver::hypre);
    pMLMG->setHypreInterface(Hypre::Interface::ij);
#endif

#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t}************************c_MLMGSolver::Setup_"
           "MLEBABecLaplacian_ForPoissonEqn()************************\n";
#endif
}

#endif

void c_MLMGSolver::Setup_MLABecLaplacian_ForPoissonEqn()
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t{************************c_MLMGSolver::Setup_"
           "MLABecLaplacian_ForPoissonEqn()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    auto &ba = rGprop.ba;
    auto &dm = rGprop.dm;
    auto &geom = rGprop.geom;

    p_mlabec = std::make_unique<amrex::MLABecLaplacian>();
    p_mlabec->define({geom}, {ba}, {dm}, info);

    // Force singular system to be solvable
    p_mlabec->setEnforceSingularSolvable(false);

    // set order of stencil
    p_mlabec->setMaxOrder(max_order);

    // assign domain boundary conditions to the solver
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    p_mlabec->setDomainBC(LinOpBCType_2d[0], LinOpBCType_2d[1]);

    auto &rBC = rCode.get_BoundaryConditions();
    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries

    int amrlev = 0;  // refers to the setcoarsest level of the solve

    if (some_constant_inhomogeneous_boundaries)
    {
        Fill_Constant_Inhomogeneous_Boundaries();
    }
    if (some_functionbased_inhomogeneous_boundaries)
    {
        Fill_FunctionBased_Inhomogeneous_Boundaries();
    }
    soln->FillBoundary(geom.periodicity());

    if (rBC.some_robin_boundaries)
    {
        robin_a->FillBoundary(geom.periodicity());
        robin_b->FillBoundary(geom.periodicity());
        robin_f->FillBoundary(geom.periodicity());
        p_mlabec->setLevelBC(amrlev, soln, robin_a, robin_b, robin_f);
    }
    else
    {
        p_mlabec->setLevelBC(amrlev, soln);
    }

    // set scaling factors
    p_mlabec->setScalars(ascalar, bscalar);

    // set alpha, and beta_fc coefficients
    p_mlabec->setACoeffs(amrlev, *alpha);

    // beta_fc =  std::make_unique<amrex::FArrayBox,AMREX_SPACEDIM>;
    // beta_fc is a face-centered multifab

    AMREX_D_TERM(beta_fc[0].define(convert(ba, IntVect(AMREX_D_DECL(1, 0, 0))),
                                   dm, 1, 0);
                 ,
                 beta_fc[1].define(convert(ba, IntVect(AMREX_D_DECL(0, 1, 0))),
                                   dm, 1, 0);
                 ,
                 beta_fc[2].define(convert(ba, IntVect(AMREX_D_DECL(0, 0, 1))),
                                   dm, 1, 0););

    beta->FillBoundary(geom.periodicity());

    Multifab_Manipulation::AverageCellCenteredMultiFabToCellFaces(
        *beta, beta_fc);  // converts from cell-centered permittivity to
                          // face-center and store in beta_fc

    p_mlabec->setBCoeffs(amrlev, amrex::GetArrOfConstPtrs(beta_fc));

    pMLMG = std::make_unique<MLMG>(*p_mlabec);

    pMLMG->setVerbose(set_verbose);

#ifdef AMREX_USE_HYPRE
    pMLMG->setBottomSolver(MLMG::BottomSolver::hypre);
    pMLMG->setHypreInterface(Hypre::Interface::ij);
#endif

#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t}************************c_MLMGSolver::Setup_MLABecLaplacian_"
           "ForPoissonEqn()************************\n";
#endif
}

void c_MLMGSolver::UpdateBoundaryConditions(bool update_surface_soln)
{
    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    auto &rBC = rCode.get_BoundaryConditions();
    auto &geom = rGprop.geom;
    int amrlev = 0;

#ifdef AMREX_USE_EB
    if (rGprop.is_eb_enabled())
    {
        if (some_constant_inhomogeneous_boundaries)
        {
            Fill_Constant_Inhomogeneous_Boundaries();
        }
        if (some_functionbased_inhomogeneous_boundaries)
        {
            Fill_FunctionBased_Inhomogeneous_Boundaries();
        }
        soln->FillBoundary(geom.periodicity());

        if (rBC.some_robin_boundaries)
        {
            robin_a->FillBoundary(geom.periodicity());
            robin_b->FillBoundary(geom.periodicity());
            robin_f->FillBoundary(geom.periodicity());
            p_mlebabec->setLevelBC(amrlev, soln, robin_a, robin_b, robin_f);
        }
        else
        {
            p_mlebabec->setLevelBC(amrlev, soln);
        }

        if (update_surface_soln)
        {
            const amrex::Real time = rCode.get_time();

            rGprop.pEB->Update_SurfaceSolution(time);

            if (rGprop.pEB->specify_inhomogeneous_dirichlet != 0)
            {
                p_mlebabec->setEBDirichlet(amrlev,
                                           *rGprop.pEB->p_surf_soln_union,
                                           *beta);
            }
        }
    }
#endif
    if (!rGprop.is_eb_enabled())
    {
        if (some_constant_inhomogeneous_boundaries)
        {
            Fill_Constant_Inhomogeneous_Boundaries();
        }
        if (some_functionbased_inhomogeneous_boundaries)
        {
            Fill_FunctionBased_Inhomogeneous_Boundaries();
        }
        soln->FillBoundary(geom.periodicity());

        if (rBC.some_robin_boundaries)
        {
            robin_a->FillBoundary(geom.periodicity());
            robin_b->FillBoundary(geom.periodicity());
            robin_f->FillBoundary(geom.periodicity());
            p_mlabec->setLevelBC(amrlev, soln, robin_a, robin_b, robin_f);
        }
        else
        {
            p_mlabec->setLevelBC(amrlev, soln);
        }
    }
}

void c_MLMGSolver::Fill_Constant_Inhomogeneous_Boundaries()
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t{************************c_MLMGSolver::Fill_Constant_"
           "Inhomogeneous_Boundaries()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t";
#endif

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    Box const &domain = rGprop.geom.Domain();

    auto &rBC = rCode.get_BoundaryConditions();
    auto &bcAny_2d = rBC.bcAny_2d;
    auto &map_bcAny_2d = rBC.map_bcAny_2d;

    std::vector<int> dir_inhomo_const_lo;
    std::string value = "inhomogeneous_constant";
    bool found_lo = findByValue(dir_inhomo_const_lo, map_bcAny_2d[0], value);

#ifdef PRINT_LOW
    if (found_lo)
    {
        amrex::Print() << "\n"
                       << prt << "Low directions with value `" << value
                       << "' are:\n";
        for (auto dir : dir_inhomo_const_lo)
            amrex::Print() << prt << "direction: " << dir << " boundary value: "
                           << std::any_cast<amrex::Real>(bcAny_2d[0][dir])
                           << "\n";
    }
#endif

    std::vector<int> dir_inhomo_const_hi;
    bool found_hi = findByValue(dir_inhomo_const_hi, map_bcAny_2d[1], value);

#ifdef PRINT_LOW
    if (found_hi)
    {
        amrex::Print() << "\n"
                       << prt << "High directions with value `" << value
                       << "' are:\n";
        for (auto dir : dir_inhomo_const_hi)
            amrex::Print() << prt << "direction: " << dir << " boundary value: "
                           << std::any_cast<amrex::Real>(bcAny_2d[1][dir])
                           << "\n";
    }
#endif

    int len = 1;
    for (MFIter mfi(*soln, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto &soln_arr = soln->array(mfi);

        const auto &bx = mfi.tilebox();

        if (found_lo)
        {
            for (auto dir : dir_inhomo_const_lo)
            {
                if (bx.smallEnd(dir) == domain.smallEnd(dir))
                {
                    auto value = std::any_cast<amrex::Real>(bcAny_2d[0][dir]);
                    Box const &bxlo = amrex::adjCellLo(bx, dir, len);
                    amrex::ParallelFor(bxlo,
                                       [=] AMREX_GPU_DEVICE(int i, int j,
                                                            int k) noexcept
                                       { soln_arr(i, j, k) = value; });
                }
            }
        }
        if (found_hi)
        {
            for (auto dir : dir_inhomo_const_hi)
            {
                if (bx.bigEnd(dir) == domain.bigEnd(dir))
                {
                    auto value = std::any_cast<amrex::Real>(bcAny_2d[1][dir]);
                    Box const &bxhi = amrex::adjCellHi(bx, dir, len);
                    amrex::ParallelFor(bxhi,
                                       [=] AMREX_GPU_DEVICE(int i, int j,
                                                            int k) noexcept
                                       { soln_arr(i, j, k) = value; });
                }
            }
        }
    }

#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t}************************c_MLMGSolver::Fill_Constant_"
           "Inhomogeneous_Boundaries()************************\n";
#endif
}

void c_MLMGSolver::Set_map_BoundarySoln_Name_Value(
    const int id, const std::vector<int> &dir_inhomo_func)
{
    auto &rCode = c_Code::GetInstance();
    const amrex::Real time = rCode.get_time();
    auto &rBC = rCode.get_BoundaryConditions();
    auto &bcAny_2d = rBC.bcAny_2d;
    auto &map_robin_coeff = rBC.map_robin_coeff;

    for (auto dir : dir_inhomo_func)  // looping over boundaries of type
                                      // inhomogeneous_function
    {
        if (LinOpBCType_2d[id][dir] == LinOpBCType::Robin)
        {
            std::string main_str =
                std::any_cast<std::string>(bcAny_2d[id][dir]);
            for (auto it_rob : map_robin_coeff)
            {
                // loop over parser names with robin subscripts e.g. Zmin_a,
                // Zmin_b, Zmin_f
                std::string macro_str = main_str + it_rob.first;

                auto pParser = rBC.get_p_parser(macro_str);
#ifdef TIME_DEPENDENT
                const auto &macro_parser = pParser->compile<4>();
#else
                const auto &macro_parser = pParser->compile<3>();
#endif
                rBC.map_BoundarySoln_Name_Value[macro_str] =
                    macro_parser(0., 0., 0., time);
            }
        }
        else  // it is dirichlet or neumann inhomogeneous function boundary
        {
            std::string macro_str =
                std::any_cast<std::string>(bcAny_2d[id][dir]);
            auto pParser = rBC.get_p_parser(macro_str);
#ifdef TIME_DEPENDENT
            const auto &macro_parser = pParser->compile<4>();
#else
            const auto &macro_parser = pParser->compile<3>();
#endif
            rBC.map_BoundarySoln_Name_Value[macro_str] =
                macro_parser(0., 0., 0., time);
        }
    }
}

void c_MLMGSolver::Fill_FunctionBased_Inhomogeneous_Boundaries()
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t{************************c_MLMGSolver::Fill_"
           "FunctionBased_Inhomogeneous_Boundaries()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t";
#endif

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    Box const &domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto &real_box = rGprop.geom.ProbDomain();
    const auto iv = soln->ixType().toIntVect();

    auto &rBC = rCode.get_BoundaryConditions();
    auto &bcAny_2d = rBC.bcAny_2d;
    auto &map_bcAny_2d = rBC.map_bcAny_2d;
    auto &map_robin_coeff = rBC.map_robin_coeff;

    std::vector<int> dir_inhomo_func_lo;
    std::string value = "inhomogeneous_function";
    bool found_lo = findByValue(dir_inhomo_func_lo, map_bcAny_2d[0], value);
    const amrex::Real time = rCode.get_time();

#ifdef PRINT_LOW
    if (found_lo)
    {
        amrex::Print() << "\n"
                       << prt << "Low directions with value `" << value
                       << "' are:\n";
        for (auto dir : dir_inhomo_func_lo)
            amrex::Print() << prt << "direction: " << dir << " boundary value: "
                           << std::any_cast<std::string>(bcAny_2d[0][dir])
                           << "\n";
    }
#endif

    std::vector<int> dir_inhomo_func_hi;
    bool found_hi = findByValue(dir_inhomo_func_hi, map_bcAny_2d[1], value);

#ifdef PRINT_LOW
    if (found_hi)
    {
        amrex::Print() << "\n"
                       << prt << "High directions with value `" << value
                       << "' are:\n";
        for (auto dir : dir_inhomo_func_hi)
            amrex::Print() << prt << "direction: " << dir << " boundary value: "
                           << std::any_cast<std::string>(bcAny_2d[1][dir])
                           << "\n";
    }
#endif

    // setting map_BoundarySoln_Name_Value for all procs, even those that are
    // not adjacent to boundary
    if (found_lo)
    {
        Set_map_BoundarySoln_Name_Value(0, dir_inhomo_func_lo);
    }
    if (found_hi)
    {
        Set_map_BoundarySoln_Name_Value(1, dir_inhomo_func_hi);
    }

    // filling function boundaries
    for (MFIter mfi(*soln, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto &soln_arr = soln->array(mfi);
        const auto &bx = mfi.tilebox();
        /*for low sides*/
        if (found_lo)
        {
            for (auto dir : dir_inhomo_func_lo)  // looping over boundaries of
                                                 // type inhomogeneous_function
            {
                if (bx.smallEnd(dir) ==
                    domain.smallEnd(dir))  // work with a box that adjacent to
                                           // the domain boundary
                {
                    Box const &bxlo = amrex::adjCellLo(bx, dir);

                    if (LinOpBCType_2d[0][dir] ==
                        LinOpBCType::Robin)  // if the boundary is robin then it
                                             // is treated differently
                    {
                        const auto &robin_a_arr = robin_a->array(mfi);
                        const auto &robin_b_arr = robin_b->array(mfi);
                        const auto &robin_f_arr = robin_f->array(mfi);

                        std::string main_str =
                            std::any_cast<std::string>(bcAny_2d[0][dir]);
                        for (auto it_rob :
                             map_robin_coeff)  // loop over parser names with
                                               // robin subscripts e.g. Zmin_a,
                                               // Zmin_b, Zmin_f
                        {
                            std::string macro_str = main_str + it_rob.first;

                            auto pParser = rBC.get_p_parser(macro_str);
#ifdef TIME_DEPENDENT
                            const auto &macro_parser = pParser->compile<4>();
#else
                            const auto &macro_parser = pParser->compile<3>();
#endif

                            switch (it_rob.second)
                            {
                                case s_RobinCoefficient::_a:
                                {
                                    amrex::ParallelFor(
                                        bxlo,
                                        [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) noexcept
                                        {
#ifdef TIME_DEPENDENT
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_4vars(
                                                    i, j, k, time, dx, real_box,
                                                    iv, macro_parser,
                                                    robin_a_arr);
#else
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_3vars(
                                                    i, j, k, dx, real_box, iv,
                                                    macro_parser, robin_a_arr);
#endif
                                        });
                                    break;
                                }
                                case s_RobinCoefficient::_b:
                                {
                                    amrex::ParallelFor(
                                        bxlo,
                                        [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) noexcept
                                        {
#ifdef TIME_DEPENDENT
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_4vars(
                                                    i, j, k, time, dx, real_box,
                                                    iv, macro_parser,
                                                    robin_b_arr);
#else
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_3vars(
                                                    i, j, k, dx, real_box, iv,
                                                    macro_parser, robin_b_arr);
#endif
                                        });
                                    break;
                                }
                                case s_RobinCoefficient::_f:
                                {
                                    amrex::ParallelFor(
                                        bxlo,
                                        [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) noexcept
                                        {
#ifdef TIME_DEPENDENT
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_4vars(
                                                    i, j, k, time, dx, real_box,
                                                    iv, macro_parser,
                                                    robin_f_arr);
#else
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_3vars(
                                                    i, j, k, dx, real_box, iv,
                                                    macro_parser, robin_f_arr);
#endif
                                        });
                                    break;
                                }
                            }
                        }
                    }
                    else  // it is dirichlet or neumann inhomogeneous function
                          // boundary
                    {
                        std::string macro_str =
                            std::any_cast<std::string>(bcAny_2d[0][dir]);

                        auto pParser = rBC.get_p_parser(macro_str);
#ifdef TIME_DEPENDENT
                        const auto &macro_parser = pParser->compile<4>();
#else
                        const auto &macro_parser = pParser->compile<3>();
#endif

                        amrex::ParallelFor(
                            bxlo,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                            {
#ifdef TIME_DEPENDENT
                                Multifab_Manipulation::
                                    ConvertParserIntoMultiFab_4vars(
                                        i, j, k, time, dx, real_box, iv,
                                        macro_parser, soln_arr);
#else
                                Multifab_Manipulation::
                                    ConvertParserIntoMultiFab_3vars(
                                        i, j, k, dx, real_box, iv, macro_parser,
                                        soln_arr);
#endif
                            });
                    }
                }
            }
        }
        /*for hi sides*/
        if (found_hi)
        {
            for (auto dir : dir_inhomo_func_hi)  // looping over boundaries of
                                                 // type inhomogeneous_function
            {
                if (bx.bigEnd(dir) ==
                    domain.bigEnd(dir))  // work with a box that adjacent to the
                                         // domain boundary
                {
                    Box const &bxhi = amrex::adjCellHi(bx, dir);

                    if (LinOpBCType_2d[1][dir] ==
                        LinOpBCType::Robin)  // if the boundary is robin then it
                                             // is treated differently
                    {
                        const auto &robin_a_arr = robin_a->array(mfi);
                        const auto &robin_b_arr = robin_b->array(mfi);
                        const auto &robin_f_arr = robin_f->array(mfi);

                        std::string main_str =
                            std::any_cast<std::string>(bcAny_2d[1][dir]);
                        for (auto it_rob :
                             map_robin_coeff)  // loop over parser names with
                                               // robin subscripts e.g. Zmin_a,
                                               // Zmin_b, Zmin_f
                        {
                            std::string macro_str = main_str + it_rob.first;

                            auto pParser = rBC.get_p_parser(macro_str);
#ifdef TIME_DEPENDENT
                            const auto &macro_parser = pParser->compile<4>();
#else
                            const auto &macro_parser = pParser->compile<3>();
#endif

                            switch (it_rob.second)
                            {
                                case s_RobinCoefficient::_a:
                                {
                                    amrex::ParallelFor(
                                        bxhi,
                                        [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) noexcept
                                        {
#ifdef TIME_DEPENDENT
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_4vars(
                                                    i, j, k, time, dx, real_box,
                                                    iv, macro_parser,
                                                    robin_a_arr);
#else
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_3vars(
                                                    i, j, k, dx, real_box, iv,
                                                    macro_parser, robin_a_arr);
#endif
                                        });
                                    break;
                                }
                                case s_RobinCoefficient::_b:
                                {
                                    amrex::ParallelFor(
                                        bxhi,
                                        [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) noexcept
                                        {
#ifdef TIME_DEPENDENT
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_4vars(
                                                    i, j, k, time, dx, real_box,
                                                    iv, macro_parser,
                                                    robin_b_arr);
#else
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_3vars(
                                                    i, j, k, dx, real_box, iv,
                                                    macro_parser, robin_b_arr);
#endif
                                        });
                                    break;
                                }
                                case s_RobinCoefficient::_f:
                                {
                                    amrex::ParallelFor(
                                        bxhi,
                                        [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) noexcept
                                        {
#ifdef TIME_DEPENDENT
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_4vars(
                                                    i, j, k, time, dx, real_box,
                                                    iv, macro_parser,
                                                    robin_f_arr);
#else
                                            Multifab_Manipulation::
                                                ConvertParserIntoMultiFab_3vars(
                                                    i, j, k, dx, real_box, iv,
                                                    macro_parser, robin_f_arr);
#endif
                                        });
                                    break;
                                }
                            }
                        }
                    }
                    else  // it is dirichlet or neumann inhomogeneous function
                          // boundary
                    {
                        std::string macro_str =
                            std::any_cast<std::string>(bcAny_2d[1][dir]);

                        auto pParser = rBC.get_p_parser(macro_str);
#ifdef TIME_DEPENDENT
                        const auto &macro_parser = pParser->compile<4>();
#else
                        const auto &macro_parser = pParser->compile<3>();
#endif

                        amrex::ParallelFor(
                            bxhi,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                            {
#ifdef TIME_DEPENDENT
                                Multifab_Manipulation::
                                    ConvertParserIntoMultiFab_4vars(
                                        i, j, k, time, dx, real_box, iv,
                                        macro_parser, soln_arr);
#else
                                Multifab_Manipulation::
                                    ConvertParserIntoMultiFab_3vars(
                                        i, j, k, dx, real_box, iv, macro_parser,
                                        soln_arr);
#endif
                            });
                    }
                }
            }
        }
    }

#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t}************************c_MLMGSolver::Fill_"
           "FunctionBased_Inhomogeneous_Boundaries()************************\n";
#endif
}

amrex::Real c_MLMGSolver::Solve_PoissonEqn()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_MLMGSolver::Solve_"
                      "PoissonEqn()************************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    amrex::Real mlmg_solve_beg_step = amrex::second();

    pMLMG->solve({soln}, {rhs}, relative_tolerance, absolute_tolerance);

    amrex::Real mlmg_solve_time = amrex::second() - mlmg_solve_beg_step;

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();

    soln->FillBoundary(rGprop.geom.periodicity());

    return mlmg_solve_time;
#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_MLMGSolver::Solve_"
                      "PoissonEqn()************************\n";
#endif
}

void c_MLMGSolver::Compute_vecField(
    std::array<amrex::MultiFab, AMREX_SPACEDIM> &vecField)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_MLMGSolver::"
                      "Compute_vecField(*)************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    /** First the gradient of phi is computed then it is multiplied by one. */
    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    auto &ba = rGprop.ba;
    auto &dm = rGprop.dm;

    amrex::Vector<Array<MultiFab, AMREX_SPACEDIM>> gradPhi;
    gradPhi.resize(1);
    // TD<decltype(gradPhi)> gradPhi_type;
    // //amrex::Vector<std::array<amrex::MultiFab, 3> > TD<decltype(gradPhi[0])>
    // gradPhi_0_type; //std::array<amrex::MultiFab, 3>&
    // TD<decltype(gradPhi[0][0])> gradPhi_00_type; //amrex::MultiFab&
    // TD<decltype(gradPhi[0][0][0])> gradPhi_000_type; //amrex::FArrayBox&
    // TD<decltype(gradPhi[0][0][0].array())> gradPhi_000array_type;
    // //amrex::Array4<double> TD<decltype(vecField)> vecField_type;
    // //std::array<amrex::MultiFab, 3>& TD<decltype(vecField[0])>
    // vecField_0_type; //amrex::MultiFab& TD<decltype(vecField[0][0])>
    // vecField_00_type;//amrex::FArrayBox&

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        gradPhi[0][idim].define(
            amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1,
            0);

        vecField[idim].define(amrex::convert(ba,
                                             IntVect::TheDimensionVector(idim)),
                              dm, 1, 0);
    }
    pMLMG->getGradSolution(amrex::GetVecOfArrOfPtrs(gradPhi));

    /*Saxpy does E[x] += -1*gradPhi[0][x], where initial value of E[x]=0*/
    AMREX_D_TERM(MultiFab::Saxpy(vecField[0], -1.0, gradPhi[0][0], 0, 0, 1, 0);
                 ,
                 MultiFab::Saxpy(vecField[1], -1.0, gradPhi[0][1], 0, 0, 1, 0);
                 , MultiFab::Saxpy(vecField[2], -1.0, gradPhi[0][2], 0, 0, 1,
                                   0););

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_MLMGSolver::Compute_"
                      "vecField(*)************************\n";
#endif
}

void c_MLMGSolver::Compute_vecFlux(
    std::array<amrex::MultiFab, AMREX_SPACEDIM> &vecFlux)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_MLMGSolver::"
                      "Compute_vecFlux(*)************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    /** Flux (-beta*grad phi)  is computed. */

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    auto &ba = rGprop.ba;
    auto &dm = rGprop.dm;
    amrex::Vector<Array<MultiFab, AMREX_SPACEDIM>> minus_epsGradPhi;
    minus_epsGradPhi.resize(1);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        minus_epsGradPhi[0][idim].define(
            amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1,
            0);
        vecFlux[idim].define(
            amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1,
            0);
    }
    pMLMG->getFluxes(amrex::GetVecOfArrOfPtrs(minus_epsGradPhi));

    /*Copy copies minus_epsGradPhi[0][x] multifab to vecFlux[x] multifab */
    AMREX_D_TERM(MultiFab::Copy(vecFlux[0], minus_epsGradPhi[0][0], 0, 0, 1, 0);
                 ,
                 MultiFab::Copy(vecFlux[1], minus_epsGradPhi[0][1], 0, 0, 1, 0);
                 , MultiFab::Copy(vecFlux[2], minus_epsGradPhi[0][2], 0, 0, 1,
                                  0););

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_MLMGSolver::Compute_"
                      "vecFlux(*)************************\n";
#endif
}
