#include "Code.H"

#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Diagnostics/Diagnostics.H"
#include "Input/BoundaryConditions/BoundaryConditions.H"
#include "Input/GeometryProperties/GeometryProperties.H"
#include "Input/MacroscopicProperties/MacroscopicProperties.H"
#include "Output/Output.H"
#include "PostProcessor/PostProcessor.H"
#include "Solver/Electrostatics/MLMG.H"
#include "Utils/SelectWarpXUtils/MsgLogger/MsgLogger.H"
#include "Utils/SelectWarpXUtils/WarnManager.H"
#include "Utils/SelectWarpXUtils/WarpXProfilerWrapper.H"
#include "Utils/SelectWarpXUtils/WarpXUtil.H"
#ifdef USE_TRANSPORT
#include "Solver/Transport/Transport.H"
#endif

c_Code *c_Code::m_instance = nullptr;
#ifdef AMREX_USE_GPU
bool c_Code::do_device_synchronize = true;
#else
bool c_Code::do_device_synchronize = false;
#endif

c_Code &c_Code::GetInstance()
{
    if (!m_instance)
    {
        m_instance = new c_Code();
    }
    return *m_instance;
}

void c_Code::ResetInstance()
{
    delete m_instance;
    m_instance = nullptr;
}

c_Code::c_Code()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************c_Code "
                      "Constructor()************************\n";
#endif
    m_instance = this;
    m_p_warn_manager = std::make_unique<Utils::WarnManager>();

    ReadData();

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************c_Code "
                      "Constructor()************************\n";
#endif
}

c_Code::~c_Code()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************c_Code "
                      "Destructor()************************\n";
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************c_Code "
                      "Destructor()************************\n";
#endif
}

void c_Code::RecordWarning(std::string topic, std::string text,
                           WarnPriority priority)
{
    WARPX_PROFILE("WarpX::RecordWarning");

    auto msg_priority = Utils::MsgLogger::Priority::high;
    if (priority == WarnPriority::low)
        msg_priority = Utils::MsgLogger::Priority::low;
    else if (priority == WarnPriority::medium)
        msg_priority = Utils::MsgLogger::Priority::medium;

    if (m_always_warn_immediately)
    {
        amrex::Warning(
            "!!!!!! WARNING: [" +
            std::string(Utils::MsgLogger::PriorityToString(msg_priority)) +
            "][" + topic + "] " + text);
    }

#ifdef AMREX_USE_OMP
#pragma omp critical
#endif
    {
        m_p_warn_manager->record_warning(topic, text, msg_priority);
    }
}

void c_Code::PrintLocalWarnings(const std::string &when)
{
    WARPX_PROFILE("WarpX::PrintLocalWarnings");
    const std::string warn_string =
        m_p_warn_manager->print_local_warnings(when);
    amrex::AllPrint() << warn_string;
}

void c_Code::PrintGlobalWarnings(const std::string &when)
{
    WARPX_PROFILE("WarpX::PrintGlobalWarnings");
    const std::string warn_string =
        m_p_warn_manager->print_global_warnings(when);
    amrex::Print() << warn_string;
}

void c_Code::ReadData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_Code::ReadData()*****"
                      "*******************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif
    // amrex::ParmParse pp_const("my_constants");
    // amrex::Real R_CN;
    // queryWithParser(pp_const,"R_CN", R_CN);
    // amrex::Print() << "my_constants.R_CN: " << R_CN << "\n";

    amrex::ParmParse pp;

#ifdef TIME_DEPENDENT
    queryWithParser(pp, "timestep", m_timestep);
    queryWithParser(pp, "steps", m_total_steps);
    pp.query("restart", m_flag_restart);

    amrex::Print() << "##### timestep: " << m_timestep << "\n";
    amrex::Print() << "##### steps: " << m_total_steps << "\n";
    amrex::Print() << "##### flag restart: " << m_flag_restart << "\n";

    if (m_flag_restart)
    {
        getWithParser(pp, "restart_step", m_restart_step);
        amrex::Print() << "##### restart_step: " << m_restart_step << "\n";
    }

#endif
    use_electrostatic = 0;
    pp.query("use_electrostatic", use_electrostatic);
    amrex::Print() << "##### use_electrostatic: " << use_electrostatic << "\n";

    if (use_electrostatic)
    {
        m_pGeometryProperties = std::make_unique<c_GeometryProperties>();

        m_pBoundaryConditions = std::make_unique<c_BoundaryConditions>();

        m_pMacroscopicProperties = std::make_unique<c_MacroscopicProperties>();
    }

    use_transport = 0;
    pp.query("use_transport", use_transport);
    amrex::Print() << "##### use_transport: " << use_transport << "\n";

#ifdef USE_TRANSPORT
    if (use_transport)
        m_pTransportSolver = std::make_unique<c_TransportSolver>();
#endif

    use_diagnostics = 0;

    if (use_electrostatic)
    {
        m_pMLMGSolver = std::make_unique<c_MLMGSolver>();

        m_pPostProcessor = std::make_unique<c_PostProcessor>();

        pp.query("use_diagnostics", use_diagnostics);
        amrex::Print() << "##### use_diagnostics: " << use_diagnostics << "\n";
        if (use_diagnostics) m_pDiagnostics = std::make_unique<c_Diagnostics>();

        m_pOutput = std::make_unique<c_Output>();
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_Code::ReadData()*********"
                      "***************\n";
#endif
}

void c_Code::InitData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************c_Code::InitData()*******"
                      "*****************\n";
    amrex::Print() << "\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    if (use_electrostatic)
    {
        m_pGeometryProperties->InitData();

        if (use_diagnostics) m_pDiagnostics->InitData();

        m_pMacroscopicProperties->InitData();
    }

#ifdef USE_TRANSPORT
    if (use_transport) m_pTransportSolver->InitData();
#endif

    if (use_electrostatic)
    {
        m_pMLMGSolver->InitData();

        m_pPostProcessor->InitData();

        m_pOutput->InitData();

        if (m_pOutput->m_write_after_init)
        {
            m_pOutput->WriteOutput_AfterInit();
        }
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************c_Code::InitData()***********"
                      "*************\n";
#endif
}

void c_Code::EstimateOfRequiredMemory()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************c_Code::"
                      "EstimateOfRequiredMemory()************************\n";
    amrex::Print() << "\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif
    amrex::Print() << "\n";
    std::string prt = "\t";

    auto &rGprop = *m_pGeometryProperties;
    auto &rMprop = *m_pMacroscopicProperties;
    auto &rBC = *m_pBoundaryConditions;
    auto &rPost = *m_pPostProcessor;
    auto &rOutput = *m_pOutput;
    auto const &n_cell_array = rGprop.get_NumCells();

    amrex::Print() << prt << "Rough estimation of required memory: \n";
    int num_cells = 1;
    //    num_cells = rGprop.n_cell[0]*rGprop.n_cell[1]*rGprop.n_cell[2];
    for (auto i : n_cell_array) num_cells *= i;

    amrex::Print() << prt << "Number of cells: " << num_cells << "\n";
    amrex::Real ratio = num_cells / pow(512, 3);
    auto Gprop_mem = ratio * 1;  // GB
    amrex::Print() << prt << "To defined geometry: " << Gprop_mem << " GBs\n";

    auto num_Mprop_mf = rMprop.num_params;
    auto Mprop_mem = ratio * num_Mprop_mf;  // GB
    amrex::Print() << prt << "To create " << num_Mprop_mf
                   << " macroscopic multifabs: " << Mprop_mem << " GBs\n";

    auto num_PostPro_mf =
        rPost.num_mf_params + rPost.num_arraymf_params * AMREX_SPACEDIM;
    auto PostPro_mem = num_PostPro_mf * ratio;
    amrex::Print() << prt << "To create " << num_PostPro_mf
                   << " post-processed multifabs: " << PostPro_mem << " GBs\n";

    auto MLMG_extra_mf_mem =
        ratio * 1;  // GB (to copy cell centered beta into nodal)
    auto MLMG_setup_mem = 0.;
    auto MLMG_solve_mem = 0.;
    // if(!rBC.some_robin_boundaries)
    //{
    MLMG_setup_mem = ratio * 7;  // GB
    MLMG_solve_mem = ratio * 6;  // GB
    //}
    // else
    //{
    //    MLMG_setup_mem = ratio*7; //GB
    //    MLMG_solve_mem = ratio*6; //GB
    //}
    auto MLMG_PostPro_solve_mem = 0.0;

    if (rPost.map_param_arraymf.find("vecField") !=
        rPost.map_param_arraymf.end())
    {
        MLMG_PostPro_solve_mem += ratio * 3;
    }
    if (rPost.map_param_arraymf.find("vecFlux") !=
        rPost.map_param_arraymf.end())
    {
        MLMG_PostPro_solve_mem += ratio * 6;
    }
    amrex::Print() << prt << "To copy beta into nodal mf in MLMG: "
                   << MLMG_extra_mf_mem << " GBs\n";
    amrex::Print() << prt
                   << "To setup MLMG [internal memory]: " << MLMG_setup_mem
                   << " GBs\n";
    amrex::Print()
        << prt
        << "To postprocess some quantities using MLMG [internal memory]: "
        << MLMG_PostPro_solve_mem << " GBs\n";

    auto num_Output_mf = rOutput.m_num_params_plot_single_level;
    // auto& write_after_init = rOutput.m_write_after_init;
    // if(write_after_init) {
    //    num_Output_mf =
    // }
    auto Output_mem = ratio * num_Output_mf;
    amrex::Print() << prt << "To copy " << num_Output_mf
                   << " multifabs while outputting single level plot file: "
                   << Output_mem << " GBs\n";

    auto Total_mem = Gprop_mem + Mprop_mem + PostPro_mem + MLMG_extra_mf_mem +
                     MLMG_setup_mem + MLMG_solve_mem + MLMG_PostPro_solve_mem +
                     Output_mem;

    amrex::Print() << prt << "Total memory: " << Total_mem << " GBs\n";
    auto Free_mem = 32 - Total_mem;

    if (Free_mem < 0)
    {
        amrex::Print() << prt
                       << "More than 1 GPU is required because total required "
                          "global memory is greater than 32 GBs! \n";
    }
    else
    {
        amrex::Print() << prt
                       << "Free global memory if 1 GPU is used: " << Free_mem
                       << " GBs\n";
    }
    amrex::Print() << "\n";

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************c_Code::"
                      "EstimateOfRequiredMemory()************************\n";
#endif
}

void c_Code::Cleanup()
{
#ifdef USE_TRANSPORT
    if (use_transport) m_pTransportSolver->Cleanup();
#endif
}

amrex::Real c_Code::Solve_OnlyElectrostatics()
{
    bool update_surface_soln_flag = true;
    m_pMLMGSolver->UpdateBoundaryConditions(update_surface_soln_flag);

    auto mlmg_solve_time = m_pMLMGSolver->Solve_PoissonEqn();
    amrex::Print() << "    mlmg_solve_time: " << std::setw(15)
                   << mlmg_solve_time << "\n";
    return mlmg_solve_time;
}

void c_Code::Solve_PostProcess_Output()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************c_Code::Solve_"
                      "PostProcess_Output()************************\n";
    amrex::Print() << "\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    amrex::Real avg_mlmg_solve_time = 0.;
    for (int step = m_restart_step; step < m_total_steps; ++step)
    {
        auto time = set_time(step);
        amrex::Print() << "step: " << std::setw(5) << step << std::setw(7)
                       << "     time: " << std::setw(5) << time << std::setw(15)
                       << "\n";

#ifdef USE_TRANSPORT
        if (use_transport) m_pTransportSolver->Solve(step, time);
        if (use_electrostatic)
        {
            if (!use_transport)
                avg_mlmg_solve_time += Solve_OnlyElectrostatics();
            m_pPostProcessor->Compute();
        }
#else
        if (use_electrostatic)
        {
            avg_mlmg_solve_time += Solve_OnlyElectrostatics();
            m_pPostProcessor->Compute();
        }
#endif

        amrex::Print() << "use diagnostics: " << use_diagnostics << "\n";
        if (use_diagnostics)
            m_pDiagnostics->ComputeAndWriteDiagnostics(step, time);

        if (use_electrostatic) m_pOutput->WriteOutput(step, time);
    }

    if (use_electrostatic)
    {
        avg_mlmg_solve_time = avg_mlmg_solve_time / m_total_steps;
        amrex::Print() << "avg_mlmg_solve_time: " << avg_mlmg_solve_time
                       << " calculated over " << m_total_steps
                       << " total steps.\n";
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************c_Code::Solve()**************"
                      "**********\n";
#endif
}
