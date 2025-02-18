#include "Transport.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>

#include "../../Code.H"
#include "../../Input/BoundaryConditions/BoundaryConditions.H"
#include "../../Input/GeometryProperties/GeometryProperties.H"
#include "../../Input/MacroscopicProperties/MacroscopicProperties.H"
#include "../../Utils/CodeUtils/CodeUtil.H"
#include "../../Utils/SelectWarpXUtils/TextMsg.H"
#include "../../Utils/SelectWarpXUtils/WarpXConst.H"
#include "../Electrostatics/MLMG.H"
#include "../Output/Output.H"
#include "../PostProcessor/PostProcessor.H"
#include "Transport_Table_ReadWrite.H"

using namespace amrex;

enum class s_NS_Type : int
{
    CNT,
    Graphene,
    Silicon,
    AtomicChain
};
enum class s_Algorithm_Type : int
{
    broyden_second,
    simple_mixing
};

const std::map<std::string, s_NS_Type> c_TransportSolver::map_NSType_enum = {
    {"carbon nanotube", s_NS_Type::CNT},
    {"Carbon Nanotube", s_NS_Type::CNT},
    {"CNT", s_NS_Type::CNT},
    {"graphene", s_NS_Type::Graphene},
    {"Graphene", s_NS_Type::Graphene},
    {"silicon", s_NS_Type::Silicon},
    {"Silicon", s_NS_Type::Silicon},
    {"atomic_chain", s_NS_Type::AtomicChain},
    {"Atomic_Chain", s_NS_Type::AtomicChain}
};

const std::map<std::string, s_Algorithm_Type>
    c_TransportSolver::map_AlgorithmType = {
        {"broyden_second", s_Algorithm_Type::broyden_second},
        {"Broyden_second", s_Algorithm_Type::broyden_second},
        {"simple_mixing", s_Algorithm_Type::simple_mixing}};

c_TransportSolver::c_TransportSolver() { ReadData(); }

void c_TransportSolver::Cleanup()
{
#ifdef BROYDEN_PARALLEL
    if (use_selfconsistent_potential) Free_MPIDerivedDataTypes();
#endif
}

void c_TransportSolver::ReadData()
{
    amrex::Print() << "\n##### Transport Solver #####\n\n";

    amrex::ParmParse pp_transport("transport");

    Read_NSNames(pp_transport);

    Read_NSTypes(pp_transport);

    // ComplexType EmU = E - U;
    auto &rCode = c_Code::GetInstance();
    if (rCode.use_electrostatic)
    {
        Read_GatherAndDepositFields(pp_transport);
    }

    Read_SelfConsistencyInput(pp_transport);
}

void c_TransportSolver::Read_NSNames(amrex::ParmParse &pp)
{
    amrex::Vector<std::string> temp_vec;
    bool NS_names_specified = pp.queryarr("NS_names", temp_vec);

    if (NS_names_specified)
    {
        int c = 0;
        for (auto it : temp_vec)
        {
            /*handling duplicates*/
            if (std::find(vec_NS_names.begin(), vec_NS_names.end(), it) ==
                vec_NS_names.end())
            {
                vec_NS_names.push_back(it);
                ++c;
            }
            else
            {
                amrex::Abort("NS_name " + it + " is duplicated!");
            }
        }
        temp_vec.clear();

        NS_num = vec_NS_names.size();
        amrex::Print() << "#####* Number of nanostructures, NS_num: " << NS_num
                       << "\n";
    }
    else
    {
        auto NS_num_isDefined = queryWithParser(pp, "NS_num", NS_num);

        if (NS_num == 0)
            amrex::Abort(
                "NS_num=0!, Either specify vector of NS_names or NS_num!");

        if (NS_num_isDefined)
        {
            for (int n = 0; n < NS_num; ++n)
            {
                vec_NS_names.push_back("NS_" + std::to_string(n + 1));
            }
        }
    }
    amrex::Print() << "#####* Number of nanostructures, NS_num: " << NS_num
                   << "\n";
    amrex::Print() << "#####* Names: ";
    for (auto name : vec_NS_names) amrex::Print() << name << "\n";
    amrex::Print() << "\n";
}

void c_TransportSolver::Read_NSTypes(amrex::ParmParse &pp)
{
    flag_isDefined_NS_type_default =
        pp.query("NS_type_default", NS_type_default);

    if (!flag_isDefined_NS_type_default)
    {
        for (auto name : vec_NS_names)
        {
            amrex::ParmParse pp_ns(name);
            std::string type_str;
            pp_ns.get("type", type_str);
            map_NSNameToTypeStr[name] = type_str;
            amrex::Print() << "##### name & type: " << name << "  " << type_str
                           << "\n";
        }
    }
}

void c_TransportSolver::InitData()
{
    amrex::Real negf_init_time = amrex::second();

    amrex::Print() << "\n##### TRANSPORT PROPERTIES #####\n\n";

    amrex::ParmParse pp_transport("transport");
    Read_ControlFlags(pp_transport);

    Set_NEGFFolderDirectories();
    Create_NEGFFolderDirectories();

    auto &rCode = c_Code::GetInstance();

    if (rCode.use_electrostatic)
    {
        auto &rGprop = rCode.get_GeometryProperties();
        _geom = &rGprop.geom;
        _dm = &rGprop.dm;
        _ba = &rGprop.ba;

        Assert_GatherAndDepositFields();

        if (!flag_isDefined_InitialDepositValue) Define_InitialDepositValue();
    }
    num_field_sites_all_NS = Instantiate_Materials();

    if (rCode.use_electrostatic) Group_ChargeDepositedByAllNS();

    if (use_selfconsistent_potential) Set_Broyden_Parallel();
    else Define_Only_ChargeDensityOut();
}

void c_TransportSolver::Read_ControlFlags(amrex::ParmParse &pp)
{
    pp.query("use_selfconsistent_potential", use_selfconsistent_potential);
    pp.query("use_negf", use_negf);
    amrex::Print() << "##### transport.use_selfconsistent_potential: "
                   << use_selfconsistent_potential << "\n";
    amrex::Print() << "##### transport.use_negf: " << use_negf << "\n";
}

void c_TransportSolver::Set_NEGFFolderDirectories()
{
    amrex::ParmParse pp_plot("plot");
    std::string foldername_str = "output";
    pp_plot.query("folder_name", foldername_str);
    CreateDirectory(foldername_str);

    negf_foldername_str = foldername_str + "/negf";
    common_foldername_str = negf_foldername_str + "/transport_common";
}

void c_TransportSolver::Create_NEGFFolderDirectories()
{
    CreateDirectory(negf_foldername_str);
    CreateDirectory(common_foldername_str);
}

void c_TransportSolver::Read_GatherAndDepositFields(amrex::ParmParse &pp)
{
    pp.get("NS_gather_field", NS_gather_field_str);
    amrex::Print() << "##### transport.NS_gather_field: " << NS_gather_field_str
                   << "\n";

    pp.get("NS_deposit_field", NS_deposit_field_str);
    amrex::Print() << "##### transport.NS_deposit_field: "
                   << NS_deposit_field_str << "\n";
}

void c_TransportSolver::Assert_GatherAndDepositFields()
{
    if (Evaluate_TypeOf_MacroStr(NS_gather_field_str) != 0)
    {
        amrex::Abort("NS_gather_field " + NS_gather_field_str +
                     " not defined in Mprop.");
    }
    if (Evaluate_TypeOf_MacroStr(NS_deposit_field_str) != 0)
    {
        amrex::Abort("NS_deposit_field " + NS_deposit_field_str +
                     " not defined in Mprop.");
    }
}

void c_TransportSolver::Define_InitialDepositValue()
{
    auto dxi = _geom->InvCellSizeArray();
    amrex::Real inv_vol = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);
    NS_initial_deposit_value = PhysConst::q_e * inv_vol;
}

void c_TransportSolver::Read_SelfConsistencyInput(amrex::ParmParse &pp)
{
    flag_isDefined_InitialDepositValue =
        queryWithParser(pp, "NS_initial_deposit_value",
                        NS_initial_deposit_value);
    amrex::Print() << "##### transport.NS_initial_deposit_value: "
                   << NS_initial_deposit_value << "\n";

    queryWithParser(pp, "Broyden_fraction", Broyden_Original_Fraction);
    amrex::Print() << "##### Broyden_fraction: " << Broyden_Original_Fraction
                   << "\n";

    queryWithParser(pp, "Broyden_max_norm", Broyden_max_norm);
    amrex::Print() << "##### Broyden_max_norm: " << Broyden_max_norm << "\n";

    pp.query("Broyden_norm_type", Broyden_Norm_Type);
    amrex::Print() << "##### Broyden_norm_type: " << Broyden_Norm_Type << "\n";

    Broyden_Threshold_MaxStep = 200;
    pp.query("Broyden_threshold_maxstep", Broyden_Threshold_MaxStep);
    amrex::Print() << "##### Broyden_Threshold_MaxStep: "
                   << Broyden_Threshold_MaxStep << "\n";

    pp.query("selfconsistency_algorithm", Algorithm_Type);
    amrex::Print() << "##### selfconsistency_algorithm: " << Algorithm_Type
                   << "\n";

    pp.query("reset_with_previous_charge_distribution",
             flag_reset_with_previous_charge_distribution);
    amrex::Print() << "##### reset_with_previous_charge_distribution: "
                   << flag_reset_with_previous_charge_distribution << "\n";
}

std::string c_TransportSolver::Get_NS_type_str(const std::string &name)
{
    return flag_isDefined_NS_type_default ? NS_type_default
                                          : map_NSNameToTypeStr.at(name);
}

template <typename NSType>
void c_TransportSolver::Create_Nanostructure(const std::string &name,
                                             const int NS_id_counter)
{
    if constexpr (RequiresParticleContainer<NSType>::value)
    {
        vp_NS.push_back(std::make_unique<c_Nanostructure<NSType>>(
            *_geom, *_dm, *_ba, name, NS_id_counter, NS_gather_field_str,
            NS_deposit_field_str, NS_initial_deposit_value, use_negf,
            negf_foldername_str));
    }
    else
    {
        // call the constructor for false specialization
        vp_NS.push_back(std::make_unique<c_Nanostructure<NSType>>(
            name, NS_id_counter, NS_initial_deposit_value,
            use_negf, negf_foldername_str));
    }    
}

int c_TransportSolver::Instantiate_Materials()
{
    amrex::Vector<int> NS_field_sites_cumulative(1, 0);
    int NS_id_counter = 0;

    for (auto name : vec_NS_names)
    {
        amrex::Print() << "##### Instantiating material: " << name << "\n";

        int field_sites = 0;
        switch (c_TransportSolver::map_NSType_enum.at(Get_NS_type_str(name)))
        {
            case s_NS_Type::CNT:
            {
                Create_Nanostructure<c_CNT>(name, NS_id_counter);
                break;
            }
            case s_NS_Type::Graphene:
            {
                Create_Nanostructure<c_Graphene>(name, NS_id_counter);
                amrex::Abort("NS_type graphene is not yet defined.");
                break;
            }
            case s_NS_Type::Silicon:
            {
                amrex::Abort("NS_type silicon is not yet defined.");
                break;
            }
            case s_NS_Type::AtomicChain:
            {
                Create_Nanostructure<c_AtomicChain>(name, NS_id_counter);
                break;
            }
            default:
            {
                amrex::Abort("NS_type " + Get_NS_type_str(name) +
                             " is not supported.");
            }
        }

        field_sites = vp_NS.back()->Get_NumFieldSites();

        int cumulative_sites = NS_field_sites_cumulative.back() + field_sites;
        NS_field_sites_cumulative.push_back(cumulative_sites);

        NS_id_counter++;
    }
    vec_biases.resize(NS_id_counter, std::make_pair(0, 0));

    amrex::Print() << "NS_field_sites_cumulative: \n";
    for (auto offset : NS_field_sites_cumulative)
    {
        amrex::Print() << " " << offset << "\n";
    }

    return NS_field_sites_cumulative.back();
}

void c_TransportSolver::Group_ChargeDepositedByAllNS()
{
    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    auto &rMprop = rCode.get_MacroscopicProperties();

    amrex::MultiFab *p_mf_deposit = rMprop.get_p_mf(NS_deposit_field_str);
    p_mf_deposit->FillBoundary(rGprop.geom.periodicity());
}

void c_TransportSolver::Set_CommonStepFolder(const int step)
{
    common_step_folder_str = amrex::Concatenate(common_foldername_str + "/step",
                                                step, negf_plt_name_digits);

    amrex::Print() << "common_step_folder_str: " << common_step_folder_str
                   << "\n";

    CreateDirectory(common_step_folder_str);
    /*eg. output/negf/common/step0001 for step 1*/
}

void c_TransportSolver::Solve(const int step, const amrex::Real time)
{
    auto &rCode = c_Code::GetInstance();
    auto &rMprop = rCode.get_MacroscopicProperties();
    auto &rMLMG = rCode.get_MLMGSolver();
    auto &rOutput = rCode.get_Output();
    auto &rPostPro = rCode.get_PostProcessor();

    m_iter = 0;
    long long total_intg_pts_in_all_iter = 0;
    m_step = step;

    for (int c = 0; c < vp_NS.size(); ++c)
    {
        vp_NS[c]->Set_StepFilenames(m_step);
        Set_CommonStepFolder(m_step);
    }

    amrex::Real time_counter[7] = {0., 0., 0., 0., 0., 0., 0.};
    amrex::Real total_time_counter_diff[7] = {0., 0., 0., 0., 0., 0., 0.};

    if (rCode.use_electrostatic)
    {
        BL_PROFILE_VAR("Part1_to_6_sum", part1_to_6_sum_counter);

        bool flag_update_terminal_bias = true;
        do
        {
            amrex::Print() << "\n\n##### Self-Consistent Iteration: " << m_iter
                           << " #####\n";
            if (Broyden_Step > Broyden_Threshold_MaxStep)
            {
                amrex::Abort(
                    "Broyden_Step has exceeded the Broyden_Threshold_MaxStep!");
            }

            // Part 1: Electrostatics
            time_counter[0] = amrex::second();

            rMprop.ReInitializeMacroparam(NS_gather_field_str);
            rMLMG.UpdateBoundaryConditions(flag_update_terminal_bias);

            auto mlmg_solve_time = rMLMG.Solve_PoissonEqn();
            rPostPro.Compute();
            // rOutput.WriteOutput(m_iter+100, time);

            time_counter[1] = amrex::second();

            // Part 2: Gather
            for (int c = 0; c < vp_NS.size(); ++c)
            {
                vp_NS[c]->Set_IterationFilenames(m_iter);
                vp_NS[c]->Gather_PotentialAtAtoms();
            }
            time_counter[2] = amrex::second();

            // Part 3: NEGF
            int total_intg_pts_in_this_iter = 0;
            for (int c = 0; c < vp_NS.size(); ++c)
            {
#ifdef AMREX_USE_GPU
                total_intg_pts_in_this_iter +=
                    vp_NS[c]->Solve_NEGF(d_n_curr_out_data, m_iter,
                                         flag_update_terminal_bias);
#else
                total_intg_pts_in_this_iter +=
                    vp_NS[c]->Solve_NEGF(h_n_curr_out_data, m_iter,
                                         flag_update_terminal_bias);
#endif
                if (flag_update_terminal_bias)
                    vec_biases[c] = vp_NS[c]->Get_Biases();
            }
            total_intg_pts_in_all_iter += total_intg_pts_in_this_iter;
            time_counter[3] = amrex::second();

            // Part 4: Self-consistency
            Perform_SelfConsistencyAlgorithm();

            time_counter[4] = amrex::second();

            // Part 5: Deposit
            rMprop.ReInitializeMacroparam(NS_deposit_field_str);
            rMprop.Deposit_AllExternalChargeDensitySources();

            for (int c = 0; c < vp_NS.size(); ++c)
            {
                Copy_BroydenPredictedChargeToHost(
                    vp_NS[c]->Get_NanostructureID());

                vp_NS[c]->Copy_BroydenPredictedChargeToNanostructure(
                    h_n_curr_in_data);

                vp_NS[c]->Deposit_ChargeDensityToMesh();
            }
            Group_ChargeDepositedByAllNS();

            time_counter[5] = amrex::second();

            // Part 6: PostProcess: compute/write DOS, current, other data
            for (int c = 0; c < vp_NS.size(); ++c)
            {
                bool isIter = true;
                vp_NS[c]->PostProcess(isIter, m_iter);

                if (vp_NS[c]->Get_Flag_WriteAtIter())
                    Copy_DataToBeWrittenToHost(vp_NS[c]->Get_NanostructureID());

                vp_NS[c]->Write_Output(isIter, m_iter, m_iter,
                                       static_cast<amrex::Real>(
                                           total_intg_pts_in_this_iter),
                                       h_n_curr_out_data, h_Norm_data);
            }
            time_counter[6] = amrex::second();

            flag_update_terminal_bias = false;
            m_iter += 1;

            total_time_counter_diff[0] += time_counter[1] - time_counter[0];
            total_time_counter_diff[1] += time_counter[2] - time_counter[1];
            total_time_counter_diff[2] += time_counter[3] - time_counter[2];
            total_time_counter_diff[3] += time_counter[4] - time_counter[3];
            total_time_counter_diff[4] += time_counter[5] - time_counter[4];
            total_time_counter_diff[5] += time_counter[6] - time_counter[5];
            total_time_counter_diff[6] += (time_counter[3] - time_counter[2]) /
                                          total_intg_pts_in_this_iter;

            amrex::Print() << " Times for: \n";
            amrex::Print() << " Electrostatics:   "
                           << time_counter[1] - time_counter[0] << "\n";
            amrex::Print() << " Gathering field:  "
                           << time_counter[2] - time_counter[1] << "\n";
            amrex::Print() << " NEGF:             "
                           << time_counter[3] - time_counter[2] << "\n";
            amrex::Print() << " NEGF (per intg pt.), intg_pts:"
                           << (time_counter[3] - time_counter[2]) /
                                  total_intg_pts_in_this_iter
                           << "  " << total_intg_pts_in_this_iter << "\n";
            amrex::Print() << " Self-consistency: "
                           << time_counter[4] - time_counter[3] << "\n";
            amrex::Print() << " Deposit charge:   "
                           << time_counter[5] - time_counter[4] << "\n";
            amrex::Print() << " Writing at iter:  "
                           << time_counter[6] - time_counter[5] << "\n";
            amrex::Print() << " Total time (write excluded):   "
                           << time_counter[5] - time_counter[0] << "\n";

        } while (Broyden_Norm > Broyden_max_norm);

        BL_PROFILE_VAR_STOP(part1_to_6_sum_counter);

        Obtain_maximum_time(total_time_counter_diff);

        // Part 7: Postprocess
        amrex::Real time_for_current = amrex::second();
        for (int c = 0; c < vp_NS.size(); ++c)
        {
            bool isIter = false;
            vp_NS[c]->PostProcess(isIter, m_step);

            Copy_DataToBeWrittenToHost(vp_NS[c]->Get_NanostructureID());

            vp_NS[c]->Write_Output(
                isIter, m_step, m_iter,
                static_cast<amrex::Real>(total_intg_pts_in_all_iter) / m_iter,
                h_n_curr_out_data, h_Norm_data);
        }
        amrex::Print() << "Time for postprocessing dos/current/writing data:   "
                       << amrex::second() - time_for_current << "\n";

        Reset_ForNextBiasStep();
    }
    else  // if not using electrostatics
    {
        m_iter=1;
        bool flag_update_terminal_bias=false;

        // Part 1: Solve NEGF
        for (int c = 0; c < vp_NS.size(); ++c)
        {
#ifdef AMREX_USE_GPU
            total_intg_pts_in_all_iter += vp_NS[c]->Solve_NEGF(d_n_curr_out_data, 
                    m_iter, flag_update_terminal_bias);
#else
            total_intg_pts_in_all_iter += vp_NS[c]->Solve_NEGF(h_n_curr_out_data, 
                    m_iter, flag_update_terminal_bias);
#endif
        }

        // Part 2: Postprocess
        amrex::Real time_for_postpro = amrex::second();
        for (int c = 0; c < vp_NS.size(); ++c)
        {
            bool isIter = false;
            vp_NS[c]->PostProcess(isIter, m_step);

            Copy_DataToBeWrittenToHost(vp_NS[c]->Get_NanostructureID());

            vp_NS[c]->Write_Output(
                isIter, m_step, m_iter,
                static_cast<amrex::Real>(total_intg_pts_in_all_iter) / m_iter,
                h_n_curr_out_data, h_Norm_data);
        }
        amrex::Print() << "Time for postprocessing dos/current/writing data:   "
                       << amrex::second() - time_for_postpro << "\n";

        Reset_ForNextBiasStep();
    }
}

void c_TransportSolver::Copy_BroydenPredictedChargeToHost(int NS_id)
{
#ifdef AMREX_USE_GPU
    int begin = site_size_loc_cumulative[NS_id];
    int end = site_size_loc_cumulative[NS_id + 1];
    int site_size_loc = end - begin;

    h_n_curr_in_data.resize({0}, {site_size_loc}, The_Pinned_Arena());

    auto const &h_n_curr_in = h_n_curr_in_data.table();
    auto const &d_n_curr_in = d_n_curr_in_data.table();

    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, d_n_curr_in.p + begin,
                          d_n_curr_in.p + end, h_n_curr_in.p);
    amrex::Gpu::streamSynchronize();
#endif
}

void c_TransportSolver::Copy_DataToBeWrittenToHost(int NS_id)
{
#ifdef BROYDEN_SKIP_GPU_OPTIMIZATION
    auto const &h_n_curr_out = h_n_curr_out_data.table();
    auto const &h_Norm = h_Norm_data.table();
#else
    /*only select data need to be copied for multiple NS*/

    int begin = site_size_loc_cumulative[NS_id];
    int end = site_size_loc_cumulative[NS_id + 1];
    int site_size_loc = end - begin;

    amrex::Print() << "Create_Global: begin/end/site_size_loc: " << begin << " "
                   << end << " " << site_size_loc << "\n";

    h_n_curr_out_data.resize({0}, {site_size_loc}, The_Pinned_Arena());
    h_Norm_data.resize({0}, {site_size_loc}, The_Pinned_Arena());

    auto const &h_n_curr_out = h_n_curr_out_data.table();
    auto const &h_Norm = h_Norm_data.table();

    auto const &d_n_curr_out = d_n_curr_out_data.const_table();
    auto const &d_Norm = d_Norm_data.const_table();

    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, d_n_curr_out.p + begin,
                          d_n_curr_out.p + end, h_n_curr_out.p);

    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, d_Norm.p + begin,
                          d_Norm.p + end, h_Norm.p);

    amrex::Gpu::streamSynchronize();

    amrex::Print() << "h_n_curr_out(0): " << h_n_curr_out(0) << "\n";
    amrex::Print() << "h_Norm(0): " << h_Norm(0) << "\n";
#endif
}

void c_TransportSolver::Perform_SelfConsistencyAlgorithm()
{
    switch (c_TransportSolver::map_AlgorithmType.at(Algorithm_Type))
    {
        case s_Algorithm_Type::broyden_second:
        {
#ifdef BROYDEN_SKIP_GPU_OPTIMIZATION
            Execute_Broyden_Modified_Second_Algorithm_Parallel_SkipGPU();
#else
            Execute_Broyden_Modified_Second_Algorithm_Parallel();
#endif
            break;
        }
        case s_Algorithm_Type::simple_mixing:
        {
            amrex::Abort(
                "At present, the `simple mixing' algorithm exists with only\
                         serial implementation (BROYDEN_PARALLEL=FALSE).");
            break;
        }
        default:
        {
            amrex::Abort("selfconsistency_algorithm, " + Algorithm_Type +
                         ", is not yet defined.");
        }
    }
}

void c_TransportSolver::Obtain_maximum_time(
    amrex::Real const *total_time_counter_diff)
{
    const int num_var = 7;
    amrex::Real total_max_time_for_current_step[num_var] = {0., 0., 0., 0.,
                                                            0., 0., 0.};

    MPI_Reduce(total_time_counter_diff, total_max_time_for_current_step,
               num_var, MPI_DOUBLE, MPI_MAX,
               ParallelDescriptor::IOProcessorNumber(),
               ParallelDescriptor::Communicator());

    if (ParallelDescriptor::IOProcessor())
    {
        amrex::Real avg_curr[num_var] = {
            total_max_time_for_current_step[0] / m_iter,
            total_max_time_for_current_step[1] / m_iter,
            total_max_time_for_current_step[2] / m_iter,
            total_max_time_for_current_step[3] / m_iter,
            total_max_time_for_current_step[4] / m_iter,
            total_max_time_for_current_step[5] / m_iter,
            total_max_time_for_current_step[6] / m_iter};
        amrex::Print() << "\nIterations in this step: " << m_iter << "\n";

        amrex::Print() << "Avg. max times for: \n ";
        amrex::Print() << " Electrostatics:   " << avg_curr[0] << "\n";

        amrex::Print() << " Gather field:     " << avg_curr[1] << "\n";

        amrex::Print() << " NEGF:             " << avg_curr[2] << "\n";

        amrex::Print() << " NEGF (per intg pt):    " << avg_curr[6] << "\n";

        amrex::Print() << " Self-consistency: " << avg_curr[3] << "\n";

        amrex::Print() << " Deposit:          " << avg_curr[4] << "\n";

        amrex::Print() << " Write at iter:    " << avg_curr[5] << "\n";

        amrex::Print() << " Total time (write excluded): "
                       << avg_curr[0] + avg_curr[1] + avg_curr[2] +
                              avg_curr[3] + avg_curr[4]
                       << "\n";

        for (int i = 0; i < num_var; ++i)
        {
            total_max_time_across_all_steps[i] +=
                total_max_time_for_current_step[i];
        }

        total_iter += m_iter;

        amrex::Real avg_all[num_var] = {
            total_max_time_across_all_steps[0] / total_iter,
            total_max_time_across_all_steps[1] / total_iter,
            total_max_time_across_all_steps[2] / total_iter,
            total_max_time_across_all_steps[3] / total_iter,
            total_max_time_across_all_steps[4] / total_iter,
            total_max_time_across_all_steps[5] / total_iter,
            total_max_time_across_all_steps[6] / total_iter};

        amrex::Real avg_total =
            avg_all[0] + avg_all[1] + avg_all[2] + avg_all[3] + avg_all[4];

        amrex::Print() << "\nTotal iterations so far: " << total_iter << "\n";
        amrex::Print() << "Avg. time over all steps for: \n";
        amrex::Print() << " Electrostatics:   " << avg_all[0] << std::setw(15)
                       << (avg_all[0] / avg_total) * 100 << " %"
                       << "\n";
        amrex::Print() << " Gather field:     " << avg_all[1] << std::setw(15)
                       << (avg_all[1] / avg_total) * 100 << " %"
                       << "\n";
        amrex::Print() << " NEGF:             " << avg_all[2] << std::setw(15)
                       << (avg_all[2] / avg_total) * 100 << " %"
                       << "\n";
        amrex::Print() << " NEGF (per intg pt):" << avg_all[6] << "\n";
        amrex::Print() << " Self-Consistency: " << avg_all[3] << std::setw(15)
                       << (avg_all[3] / avg_total) * 100 << " %"
                       << "\n";
        amrex::Print() << " Deposit:          " << avg_all[4] << std::setw(15)
                       << (avg_all[4] / avg_total) * 100 << " %"
                       << "\n";

        amrex::Print() << " Write at iter:    " << avg_all[5] << "\n";

        amrex::Print() << " Total time (write excluded): " << avg_total << "\n";
    }
}

void c_TransportSolver::Reset_ForNextBiasStep()
{
    auto &rCode = c_Code::GetInstance();
    if (rCode.use_electrostatic)
    {
        Reset_Broyden_Parallel();
    }
    MPI_Barrier(ParallelDescriptor::Communicator());
}


void c_TransportSolver::Define_CumulativeInfo_ForAllNS()
{
    /* Each process executes this function.
     *
     * Create the following:
     * my_rank
     * total_proc
     * site_size_loc_all_NS
     * site_size_loc_cumulative vector of size equal to all nanostructure.
     */
    total_proc = amrex::ParallelDescriptor::NProcs();
    my_rank = amrex::ParallelDescriptor::MyProc();

    site_size_loc_cumulative.resize(vp_NS.size() + 1);
    site_size_loc_cumulative[0] = 0;

    for (int c = 0; c < vp_NS.size(); ++c)
    {
        vp_NS[c]->Set_NumFieldSites_Local_NSOffset(site_size_loc_cumulative[c]);

        site_size_loc_cumulative[c + 1] =
            site_size_loc_cumulative[c] + vp_NS[c]->Get_NumFieldSites_Local();
    }
    site_size_loc_all_NS = site_size_loc_cumulative[vp_NS.size()];
}


void c_TransportSolver::Define_Only_ChargeDensityOut()
{
    Define_CumulativeInfo_ForAllNS();

#ifdef AMREX_USE_GPU
    d_n_curr_out_data.resize({0}, {site_size_loc_all_NS}, The_Arena());
    d_Norm_data.resize({0}, {site_size_loc_all_NS}, The_Arena());

    auto const &n_curr_out = d_n_curr_out_data.table();
    auto const &Norm = d_Norm_data.table();

    amrex::ParallelFor(site_size_loc_all_NS,
                       [=] AMREX_GPU_DEVICE(int site) noexcept
                       {
                           n_curr_out(site) = 0.;
                           Norm(site) = 0.;
                       });
    amrex::Gpu::streamSynchronize();
#else
    h_n_curr_out_data.resize({0}, {site_size_loc_all_NS}, The_Pinned_Arena());
    h_Norm_data.resize({0}, {site_size_loc_all_NS}, The_Pinned_Arena());
    
    SetVal_RealTable1D(h_n_curr_out_data, 0.);
    SetVal_RealTable1D(h_Norm_data, 0.);
#endif
}


void c_TransportSolver::SetVal_RealTable1D(RealTable1D &Tab1D_data,
                                           amrex::Real val)
{
    auto tlo = Tab1D_data.lo();
    auto thi = Tab1D_data.hi();

    auto const &Tab1D = Tab1D_data.table();

    for (int i = tlo[0]; i < thi[0]; ++i)
    {
        Tab1D(i) = val;
    }
}

void c_TransportSolver::SetVal_RealTable2D(RealTable2D &Tab2D_data,
                                           amrex::Real val)
{
    auto tlo = Tab2D_data.lo();
    auto thi = Tab2D_data.hi();

    auto const &Tab2D = Tab2D_data.table();

    for (int i = tlo[0]; i < thi[0]; ++i)
    {
        for (int j = tlo[1]; j < thi[1];
             ++j)  // slow moving index. printing slow
        {
            Tab2D(i, j) = val;
        }
    }
}
