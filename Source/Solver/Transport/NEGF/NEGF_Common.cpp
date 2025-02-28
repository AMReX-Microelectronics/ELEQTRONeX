#include "NEGF_Common.H"

#include "../../../Code.H"
#include "../../../Input/BoundaryConditions/BoundaryConditions.H"
#include "../../../Input/GeometryProperties/GeometryProperties.H"
#include "../../../Input/MacroscopicProperties/MacroscopicProperties.H"
#include "../../Utils/CodeUtils/CodeUtil.H"
#include "../../Utils/SelectWarpXUtils/TextMsg.H"
#include "../../Utils/SelectWarpXUtils/WarpXConst.H"
#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Matrix_Block_Util.H"

/*Explicit specializations*/
template class c_NEGF_Common<ComplexType[BLOCK_SIZE]>;  // of c_CNT
template class c_NEGF_Common<
    ComplexType[BLOCK_SIZE][BLOCK_SIZE]>;  // c_Graphene

const std::map<std::string, AngleType> map_strToAngleType = {
    {"d", AngleType::Degrees},
    {"D", AngleType::Degrees},
    {"r", AngleType::Radians},
    {"R", AngleType::Radians},
};

const std::map<std::string, AxisType> map_strToAxisType = {
    {"x", AxisType::X}, {"X", AxisType::X}, {"y", AxisType::Y},
    {"Y", AxisType::Y}, {"z", AxisType::Z}, {"Z", AxisType::Z}};

const std::map<std::string, Gate_Terminal_Type> map_S_GateTerminalType = {
    {"eb", Gate_Terminal_Type::EB},
    {"EB", Gate_Terminal_Type::EB},
    {"boundary", Gate_Terminal_Type::Boundary},
    {"Boundary", Gate_Terminal_Type::Boundary}};

template <typename T>
void c_NEGF_Common<T>::Set_KeyParams(const std::string &name_str,
                                     const int &id_counter,
                                     const amrex::Real &initial_deposit_value)
{
    name = name_str;
    NS_Id = id_counter;
    initial_charge = initial_deposit_value;

    num_proc = amrex::ParallelDescriptor::NProcs();
    my_rank = amrex::ParallelDescriptor::MyProc();
}

template <typename T>
void c_NEGF_Common<T>::Define_FoldersAndFiles(
    const std::string &negf_foldername_str)
{
    step_foldername_str = negf_foldername_str + "/" + name;
    /*eg. output/negf/cnt for nanostructure named cnt */

    step_filename_prefix_str = step_foldername_str + "/step";
    /*eg. output/negf/cnt/step */

    if (ParallelDescriptor::IOProcessor())
    {
        CreateDirectory(step_foldername_str);

        current_filename_str = step_foldername_str + "/I.dat";
    }

    Define_FileHeaderForCurrent();
}

template <typename T>
void c_NEGF_Common<T>::Define_FileHeaderForCurrent()
{
    if (ParallelDescriptor::IOProcessor())
    {
        outfile_I.open(current_filename_str.c_str(), std::ios::app);
        outfile_I << "'step', 'Vds' , 'Vgs', ";
        for (int k = 0; k < NUM_CONTACTS; ++k)
        {
            outfile_I << ", 'I at contact_" << k + 1 << "',";
        }
        outfile_I << "'Avg_intg_pts', 'Total_Iter', 'Broyden_Fraction', \
                      'Broyden_Scalar', 'Conductance / (2q^2/h)'"
                  << "\n";
        outfile_I.close();
    }
}

template <typename T>
void c_NEGF_Common<T>::Set_StepFilenameString(const int step)
{
    step_filename_str = amrex::Concatenate(step_filename_prefix_str, step,
                                           negf_plt_name_digits);
    /*eg. output/negf/cnt/step0001 for step 1*/
    amrex::Print() << "step_filename_str: " << step_filename_str << "\n";

    if (write_at_iter)
    {
        iter_foldername_str = step_filename_str + "_iter/";
        /*eg. output/negf/cnt/step0001_iter/ */
        amrex::Print() << "iter_foldername_str: " << iter_foldername_str
                       << "\n";

        CreateDirectory(iter_foldername_str);

        iter_filename_prefix_str = iter_foldername_str + "iter";
        /*eg. output/negf/cnt/step0001_iter/iter */
        amrex::Print() << "iter_filename_prefix_str: "
                       << iter_filename_prefix_str << "\n";
    }
}

template <typename T>
void c_NEGF_Common<T>::Set_IterationFilenameString(const int iter)
{
    if (write_at_iter)
    {
        iter_filename_str = amrex::Concatenate(iter_filename_prefix_str, iter,
                                               negf_plt_name_digits);
        /*eg. output/negf/cnt/step0001_iter/iter0001  for iteration 1*/
        // amrex::Print() << " iter_filename_str: " << iter_filename_str <<
        // "\n";

        if (flag_write_integrand_main)
        {
            if (iter == 0 or iter % write_integrand_interval == 0)
            {
                flag_write_integrand_iter = true;
                amrex::Print() << " setting flag_write_integrand_iter to: "
                               << flag_write_integrand_iter << "\n";
            }
            else
            {
                flag_write_integrand_iter = false;
                if (iter == 1)
                {
                    amrex::Print() << " setting flag_write_integrand_iter to: "
                                   << flag_write_integrand_iter << "\n";
                }
            }
        }
    }
}

template <typename T>
void c_NEGF_Common<T>::Define_MPISendCountAndDisp()
{
    if (ParallelDescriptor::IOProcessor())
    {
        MPI_send_count.resize(num_proc);
        MPI_send_disp.resize(num_proc);

        std::fill(MPI_send_count.begin(), MPI_send_count.end(), 0);
        std::fill(MPI_send_disp.begin(), MPI_send_disp.end(), 0);
    }
    MPI_Gather(&(min_local_site_id), 1, MPI_INT, MPI_send_disp.data(), 1,
               MPI_INT, ParallelDescriptor::IOProcessorNumber(),
               ParallelDescriptor::Communicator());

    MPI_Gather(&(num_local_field_sites), 1, MPI_INT, MPI_send_count.data(), 1,
               MPI_INT, ParallelDescriptor::IOProcessorNumber(),
               ParallelDescriptor::Communicator());

    // if(ParallelDescriptor::IOProcessor())
    //{
    //     for(int p=0; p < num_proc; ++p)
    //     {
    //         amrex::Print() << " process/MPI_send_disp/count: " << p << " "
    //                                                            <<
    //                                                            MPI_send_disp[p]
    //                                                            << " "
    //                                                            <<
    //                                                            MPI_send_count[p]
    //                                                            << "\n";
    //     }
    // }
    // amrex::Abort("Manually stopping for debugging");
}

template <typename T>
void c_NEGF_Common<T>::Initialize_ChargeAtFieldSites()
{
    if (flag_initialize_charge_distribution)
    {
        h_n_curr_in_glo_data.resize({0}, {num_field_sites}, The_Pinned_Arena());
        auto const &h_n_curr_in_glo = h_n_curr_in_glo_data.table();

        if (ParallelDescriptor::IOProcessor()) /*&*/
        {
            Read_Table1D(num_field_sites, h_n_curr_in_glo_data,
                         charge_distribution_filename);
        }

        ParallelDescriptor::Bcast(&h_n_curr_in_glo(0), Hsize_glo,
                                  ParallelDescriptor::IOProcessorNumber());

        // h_n_curr_in_glo exists temporarily until Broyden's algorithm is
        // initialized with the input charge in Set_Broyden_Parallel. It is
        // deallocated in Fetch_InputLocalCharge_FromNanostructure(*).

        // We use h_n_curr_in_loc for depositing charge to mesh.
        auto const &h_n_curr_in_loc = h_n_curr_in_loc_data.table();
        MPI_Scatterv(&h_n_curr_in_glo(0), MPI_send_count.data(),
                     MPI_send_disp.data(), MPI_DOUBLE, &h_n_curr_in_loc(0),
                     num_local_field_sites, MPI_DOUBLE,
                     ParallelDescriptor::IOProcessorNumber(),
                     ParallelDescriptor::Communicator());

        // if(ParallelDescriptor::IOProcessor())  /*&*/
        //{
        //    // bool full_match = true;
        //    // for(int i=0; i < num_local_field_sites; ++i)
        //    // {
        //    //     if(h_n_curr_in_loc(i) !=
        //    h_n_curr_in_glo(i+min_local_site_id)
        //    )

        //   //         full_match = false;
        //   //         amrex::Abort("n_curr_in_loc doesn't match with glo: "
        //   //                 + std::to_string(i) + " " +
        //   std::to_string(h_n_curr_in_loc(i)) + " "
        //   //                 +
        //   std::to_string(h_n_curr_in_glo(i+min_local_site_id)));
        //   //     }
        //   // }
        //   // std::cout << "process: " << my_rank << " full_match: " <<
        //   full_match << "\n";
        //}
    }
    else
    {
        h_n_curr_in_glo_data.resize({0}, {Hsize_glo}, The_Pinned_Arena());
        SetVal_Table1D(h_n_curr_in_glo_data, initial_charge);
        SetVal_Table1D(h_n_curr_in_loc_data, initial_charge);
    }
}

template <typename T>
template <typename TableType>
void c_NEGF_Common<T>::Read_Table1D(int assert_size, TableType &Tab1D_data,
                                    std::string filename)
{
    amrex::Print() << "Reading Table1D. filename: " << filename << "\n";

    std::ifstream infile;
    infile.open(filename.c_str());

    if (infile.fail())
    {
        amrex::Abort("Failed to read file " + filename);
    }
    else
    {
        auto const &Tab1D = Tab1D_data.table();
        auto thi = Tab1D_data.hi();
        auto tlo = Tab1D_data.lo();

        int filesize = 0;
        std::string line;
        while (infile.peek() != EOF)
        {
            std::getline(infile, line);
            filesize++;
        }
        amrex::Print() << "filesize: " << filesize << "\n";

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            filesize - 1 == assert_size,
            "Assert size, " + std::to_string(assert_size) +
                ", is not equal to the filesize-1, " +
                std::to_string(filesize - 1) + " !");

        infile.seekg(0, std::ios_base::beg);

        std::getline(infile, line);
        amrex::Print() << "file header: " << line << "\n";

        amrex::Real position, value;
        for (int l = tlo[0]; l < thi[0]; ++l)
        {
            infile >> position >> value;
            Tab1D(l) = value;
            // amrex::Print() << "position/value: " << position << "    " <<
            // Tab1D(l) << "\n";
        }
        infile.close();
    }
}

template <typename T>
void c_NEGF_Common<T>::Read_Unitcells(amrex::ParmParse &pp_ns)
{
    queryWithParser(pp_ns, "num_unitcells", num_unitcells);
}

template <typename T>
void c_NEGF_Common<T>::Read_MaterialOrientationParams(amrex::ParmParse &pp_ns)
{
    /*translation*/
    amrex::Vector<amrex::Real> vec_offset(AMREX_SPACEDIM, 0.);
    auto offset_isDefined =
        queryArrWithParser(pp_ns, "offset", vec_offset, 0, AMREX_SPACEDIM);

    if (offset_isDefined) offset = vecToArr(vec_offset);

    /*rotation*/
    if (p_rotInputParams == nullptr)
        p_rotInputParams = std::make_unique<RotationInputParams>();

    amrex::Vector<amrex::Real> vec_rotation_angles(AMREX_SPACEDIM, 0);
    auto angles_isDefined =
        queryArrWithParser(pp_ns, "rotation_angles", vec_rotation_angles, 0,
                           AMREX_SPACEDIM);

    if (angles_isDefined) p_rotInputParams->angles = vec_rotation_angles;

    std::string angle_type_str = "";
    auto angleType_isDefined =
        pp_ns.query("rotation_angle_type", angle_type_str);

    if (angleType_isDefined)
    {
        p_rotInputParams->angle_type = map_strToAngleType.at(angle_type_str);
    }

    amrex::Vector<std::string> vec_rot_order_str = {};
    auto rotationOrder_isDefined =
        pp_ns.queryarr("rotation_order", vec_rot_order_str);

    if (rotationOrder_isDefined)
    {
        p_rotInputParams->rotation_order.clear();

        for (auto axis_str : vec_rot_order_str)
        {
            if (map_strToAxisType.find(axis_str) != map_strToAxisType.end())
            {
                p_rotInputParams->rotation_order.push_back(
                    map_strToAxisType.at(axis_str));
            }
        }
    }
}

template <typename T>
void c_NEGF_Common<T>::Read_FieldAveragingParams(amrex::ParmParse &pp_ns)
{
    std::string avg_type_str = "all";
    pp_ns.query("field_averaging_type", avg_type_str);
    if (avg_type_str == "all" or avg_type_str == "All" or avg_type_str == "ALL")
    {
        avg_type = s_AVG_Type::ALL;
    }
    if (avg_type_str == "specific" or avg_type_str == "Specific" or
        avg_type_str == "SPECIFIC")
    {
        avg_type = s_AVG_Type::SPECIFIC;
        pp_ns.queryarr("atom_indices_for_averaging", vec_avg_indices);
    }
}

template <typename T>
void c_NEGF_Common<T>::set_Contact_Electrochemical_Potential(
    const amrex::Vector<amrex::Real> &ep)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ep.size() == NUM_CONTACTS,
                                     "ep.size != NUM_CONTACTS. It is: " +
                                         std::to_string(ep.size()));

    for (int k = 0; k < ep.size(); ++k)
    {
        Contact_Electrochemical_Potential[k] = ep[k];
    }
    flag_EC_potential_updated = true;
}

template <typename T>
void c_NEGF_Common<T>::Read_TerminalParams(amrex::ParmParse &pp_ns)
{
    pp_ns.query("gate_string", Gate_String);

    pp_ns.query("gate_terminal_type", gate_terminal_type_str);

    if (map_S_GateTerminalType.find(gate_terminal_type_str) !=
        map_S_GateTerminalType.end())
    {
        gate_terminal_type = map_S_GateTerminalType.at(gate_terminal_type_str);
    }
    else
    {
        amrex::Abort("gate_terminal_type: " + gate_terminal_type_str +
                     " is invalid. Valid type: EB or Boundary.");
    }
    queryWithParser(pp_ns, "contact_Fermi_level", E_f);

    pp_ns.query("contact_mu_specified", flag_contact_mu_specified);

    if (flag_contact_mu_specified)
    {
        amrex::Vector<amrex::Real> vec_mu;
        auto is_specified =
            queryArrWithParser(pp_ns, "contact_mu", vec_mu, 0, NUM_CONTACTS);
        if (is_specified)
        {
            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                Contact_Electrochemical_Potential[c] = vec_mu[c];
            }
            flag_EC_potential_updated = true;
        }
    }
    else
    {
        amrex::Vector<std::string> temp_vec;
        auto is_specified = pp_ns.queryarr("contact_parser_string", temp_vec);
        if (is_specified)
        {
            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                Contact_Parser_String[c] = temp_vec[c];
            }
        }
    }

    amrex::Vector<amrex::Real> vec_Temperature;
    auto is_specified = queryArrWithParser(pp_ns, "contact_T", vec_Temperature,
                                           0, NUM_CONTACTS);
    if (is_specified)
    {
        for (int c = 0; c < NUM_CONTACTS; ++c)
        {
            Contact_Temperature[c] = vec_Temperature[c];
        }
    }
}

template <typename T>
void c_NEGF_Common<T>::Read_PotentialProfileParams(amrex::ParmParse &pp_ns)
{
    pp_ns.query("impose_potential", flag_impose_potential);
    if (flag_impose_potential)
    {
        pp_ns.query("potential_profile_type", potential_profile_type_str);
    }
}

template <typename T>
void c_NEGF_Common<T>::Read_EqContourIntgPts(amrex::ParmParse &pp_ns)
{
    auto is_specified_eq = queryArrWithParser(pp_ns, "eq_integration_pts",
                                              eq_integration_pts, 0, 3);
}

template <typename T>
void c_NEGF_Common<T>::Read_DOSParams(amrex::ParmParse &pp_ns)
{
    pp_ns.query("flag_compute_DOS", flag_compute_DOS);
    pp_ns.query("flag_write_LDOS", flag_write_LDOS);
    pp_ns.query("flag_write_LDOS_iter", flag_write_LDOS_iter);
    pp_ns.query("write_LDOS_iter_period", write_LDOS_iter_period);
}

template <typename T>
void c_NEGF_Common<T>::Read_FlatbandDOSParams(amrex::ParmParse &pp_ns)
{
    pp_ns.query("flag_compute_flatband_dos", flag_compute_flatband_dos);

    queryWithParser(pp_ns, "flatband_dos_integration_pts",
                    flatband_dos_integration_pts);
    auto is_specified_dos_limit =
        queryArrWithParser(pp_ns, "flatband_dos_integration_limits",
                           flatband_dos_integration_limits, 0, 2);
}

template <typename T>
void c_NEGF_Common<T>::Read_NonEqPathParams(amrex::ParmParse &pp_ns)
{
    auto flag_num_noneq_paths =
        queryWithParser(pp_ns, "num_noneq_paths", num_noneq_paths);

    if (num_noneq_paths > 1)
    {
        auto flag_noneq_percent_intercuts =
            queryArrWithParser(pp_ns, "noneq_percent_intercuts",
                               noneq_percent_intercuts, 0, num_noneq_paths - 1);
        if (!flag_noneq_percent_intercuts)
        {
            noneq_percent_intercuts.resize(num_noneq_paths - 1);
            amrex::Real uniform_percent_val = 100. / num_noneq_paths;
            for (int c = 0; c < num_noneq_paths - 1; ++c)
            {
                noneq_percent_intercuts[c] = (c + 1) * uniform_percent_val;
            }
        }
    }

    auto flag_noneq_intg_pts =
        queryArrWithParser(pp_ns, "noneq_integration_pts",
                           noneq_integration_pts, 0, num_noneq_paths);
}

template <typename T>
void c_NEGF_Common<T>::Read_IntegrationParams(amrex::ParmParse &pp_ns)
{
    queryWithParser(pp_ns, "E_valence_min", E_valence_min);
    queryWithParser(pp_ns, "E_pole_max", E_pole_max);

    amrex::Real imag = 1e-8;
    auto is_imag_specified = queryWithParser(pp_ns, "E_zPlus_imag", imag);
    if (is_imag_specified)
    {
        ComplexType val(0., imag);
        E_zPlus = val;
    }

    amrex::Vector<amrex::Real> vec_FermiTailFactors;
    auto is_specified = queryArrWithParser(pp_ns, "Fermi_tail_factors",
                                           vec_FermiTailFactors, 0, 2);
    if (is_specified)
    {
        Fermi_tail_factor_lower = vec_FermiTailFactors[0];
        Fermi_tail_factor_upper = vec_FermiTailFactors[1];
    }
}

template <typename T>
void c_NEGF_Common<T>::Read_AdaptiveIntegrationParams(amrex::ParmParse &pp_ns)
{
    pp_ns.query("flag_adaptive_integration_limits",
                flag_adaptive_integration_limits);
    if (flag_adaptive_integration_limits)
    {
        pp_ns.query("integrand_correction_interval",
                    integrand_correction_interval);

        auto flag_kT_window_around_singularity =
            queryArrWithParser(pp_ns, "kT_window_around_singularity",
                               kT_window_around_singularity, 0, 2);

        flag_noneq_integration_pts_density =
            queryArrWithParser(pp_ns, "noneq_integration_pts_density",
                               noneq_integration_pts_density, 0,
                               num_noneq_paths);
    }
}

template <typename T>
void c_NEGF_Common<T>::Read_IntegrandWritingParams(amrex::ParmParse &pp_ns)
{
    pp_ns.query("flag_write_integrand", flag_write_integrand_main);
    if (flag_write_integrand_main)
    {
        pp_ns.query("write_integrand_interval", write_integrand_interval);
    }
}

template <typename T>
void c_NEGF_Common<T>::Read_WritingRelatedFlags(amrex::ParmParse &pp_ns)
{
    queryWithParser(pp_ns, "write_at_iter", write_at_iter);

    pp_ns.query("flag_write_charge_components", flag_write_charge_components);

    Read_IntegrandWritingParams(pp_ns);
}

template <typename T>
void c_NEGF_Common<T>::Read_AtomLocationAndChargeDistributionFilename(
    amrex::ParmParse &pp_ns)
{
    // filename for atom location
    pp_ns.query("read_atom_filename", read_atom_filename);

    pp_ns.query("initialize_charge_distribution",
                flag_initialize_charge_distribution);
    if (flag_initialize_charge_distribution)
    {
        amrex::ParmParse pp;
        int flag_restart = 0;
        pp.query("restart", flag_restart);

        if (flag_restart)
        {
            int restart_step = 0;
            getWithParser(pp, "restart_step", restart_step);
            std::string step_filename_str =
                amrex::Concatenate(step_filename_prefix_str, restart_step - 1,
                                   negf_plt_name_digits);
            /*eg. output/negf/cnt/step0007_Qout.dat for step 1*/
            charge_distribution_filename = step_filename_str + "_Qout.dat";
            pp_ns.query("charge_distribution_filename",
                        charge_distribution_filename);
        }
        else
        {
            std::string cd_filename = "";
            std::string read_negf_foldername_str = "";
            int read_step = -1;

            auto cd_filename_isDefined =
                pp_ns.query("charge_distribution_filename", cd_filename);
            auto read_folder_isDefined =
                pp_ns.query("read_negf_foldername", read_negf_foldername_str);
            auto read_step_isDefined = pp_ns.query("read_step", read_step);

            if (cd_filename_isDefined)
            {
                charge_distribution_filename = cd_filename;
            }
            else if (read_folder_isDefined and read_step_isDefined)
            {
                std::string read_filename_prefix_str =
                    read_negf_foldername_str + "/" + name + "/step";
                /*eg. output/negf/cnt/step */

                std::string read_filename_str =
                    amrex::Concatenate(read_filename_prefix_str, read_step - 1,
                                       negf_plt_name_digits);

                charge_distribution_filename = read_filename_str + "_Qout.dat";
            }
        }
    }
}

template <typename T>
void c_NEGF_Common<T>::Read_RecursiveOptimizationParams(amrex::ParmParse &pp_ns)
{
    queryWithParser(pp_ns, "num_recursive_parts", num_recursive_parts);
}

template <typename T>
void c_NEGF_Common<T>::Read_DecimationTechniqueParams(amrex::ParmParse &pp_ns)
{
    queryWithParser(pp_ns, "use_decimation", use_decimation);
    queryWithParser(pp_ns, "decimation_max_iter", decimation_max_iter);
    queryWithParser(pp_ns, "decimation_rel_error", decimation_rel_error);
    queryWithParser(pp_ns, "decimation_layers", decimation_layers);

    amrex::Print() << "##### use_decimation: " << use_decimation << "\n";
    amrex::Print() << "##### decimation_max_iter: " << decimation_max_iter
                   << "\n";
    amrex::Print() << "##### decimation_rel_error: " << decimation_rel_error
                   << "\n";
    amrex::Print() << "##### decimation_layers: " << decimation_layers << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Assert_Reads()
{
    amrex::Print() << "##### Assert Reads\n";
    // qeuryWithParser(pp_ns, "num_unitcells", num_unitcells);
    // pp_ns.queryarr("atom_indices_for_averaging", vec_avg_indices);
    // if(flag_contact_mu_specified)
    //{
    //     amrex::Vector<amrex::Real> vec_mu;
    //     queryArrWithParser(pp_ns, "contact_mu", vec_mu, 0, NUM_CONTACTS);
    //     for(int c=0; c<NUM_CONTACTS; ++c)
    //     {
    //         Contact_Electrochemical_Potential[c] = vec_mu[c];
    //     }
    //     flag_EC_potential_updated = true;
    //     vec_mu.clear();
    // }
    // else
    //{
    //     amrex::Vector<std::string> temp_vec;
    //     pp_ns.queryarr("contact_parser_string", temp_vec);
    //     for(int c=0; c<NUM_CONTACTS; ++c)
    //     {
    //         Contact_Parser_String[c] = temp_vec[c];
    //     }
    //     temp_vec.clear();
    // }
    // if(!flag_restart) {
    //         pp_ns.query("charge_distribution_filename",
    //         charge_distribution_filename);
    // }
}

template <typename T>
void c_NEGF_Common<T>::Read_CommonData(amrex::ParmParse &pp)
{
    Read_Unitcells(pp);
    Read_MaterialOrientationParams(pp);
    Read_FieldAveragingParams(pp);
    Read_TerminalParams(pp);
    Read_PotentialProfileParams(pp);
    Read_EqContourIntgPts(pp);
    Read_DOSParams(pp);
    Read_FlatbandDOSParams(pp);
    Read_NonEqPathParams(pp);
    Read_IntegrationParams(pp);
    Read_AdaptiveIntegrationParams(pp);
    Read_WritingRelatedFlags(pp);
    Read_AtomLocationAndChargeDistributionFilename(pp);
    Read_RecursiveOptimizationParams(pp);
    Read_DecimationTechniqueParams(pp);
}

template <typename T>
void c_NEGF_Common<T>::Read_NanostructureProperties()
{
    amrex::ParmParse pp_ns_default("NS_default");
    amrex::ParmParse pp_ns(name);
    amrex::ParmParse *pp = &pp_ns_default;

    amrex::Print() << "##### Reading ParmParse NS_Default\n";
    for (int i = 0; i < 2; ++i)
    {
        if (i == 1)
        {
            pp = &pp_ns;
            amrex::Print() << "##### Reading ParmParse: " << name << "\n";
        }
        Read_CommonData(*pp);
    }
    Assert_Reads();

    Read_MaterialSpecificNanostructureProperties();
}

template <typename T>
void c_NEGF_Common<T>::Set_MaterialParameters()
{
    Set_MaterialSpecificParameters();

    num_atoms = Compute_NumAtoms();
    num_atoms_per_unitcell = Compute_AtomsPerUnitcell();
    num_field_sites = Compute_NumFieldSites();
    num_atoms_per_field_site = Compute_NumAtomsPerFieldSite();
    primary_transport_dir = Set_PrimaryTransportDir();
    average_field_flag = Set_AverageFieldFlag();
    offDiag_repeatBlkSize = get_offDiag_repeatBlkSize();

    if (average_field_flag) num_atoms_to_avg_over = Compute_NumAtomsToAvgOver();

    Set_BlockDegeneracyVector(block_degen_vec);

    Assert_KeyParameters();

    Set_RotationMatrix();
    Set_BlockDegeneracyGPUVector();
    Define_GPUVectorOfAvgIndices();
    Set_Arrays_OfSize_NumFieldSites();
}

template <typename T>
void c_NEGF_Common<T>::Set_RotationMatrix()
{
    if (p_rotInputParams && !p_rotInputParams->angles.empty())
        p_rotator =
            std::make_unique<c_RotationMatrix>(std::move(p_rotInputParams));
}

template <typename T>
void c_NEGF_Common<T>::Set_BlockDegeneracyGPUVector()
{
#if AMREX_USE_GPU
    block_degen_gpuvec.resize(block_degen_vec.size());

    amrex::Gpu::copy(amrex::Gpu::hostToDevice, block_degen_vec.begin(),
                     block_degen_vec.end(), block_degen_gpuvec.begin());
#endif
}

template <typename T>
void c_NEGF_Common<T>::Assert_KeyParameters()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(num_field_sites > 0,
                                     "Assert: num_field_sites > 0");
}

template <typename T>
void c_NEGF_Common<T>::Print_ReadData()
{
    Print_KeyParams();
    Print_MaterialParams();
    Print_MaterialOrientationParams();
    Print_FieldAveragingParams();
    Print_TerminalParams();
    Print_PotentialProfileParams();
    Print_EqContourIntgPts();
    Print_DOSParams();
    Print_FlatbandDOSParams();
    Print_NonEqPathParams();
    Print_IntegrationParams();
    Print_AdaptiveIntegrationParams();
    Print_WritingRelatedFlags();
    Print_AtomLocationAndChargeDistributionFilename();
    Print_RecursiveOptimizationParams();

    Print_MaterialSpecificReadData();
}

template <typename T>
void c_NEGF_Common<T>::Print_KeyParams()
{
    amrex::Print() << "##### name: " << name << "\n";
    amrex::Print() << "##### NS_Id: " << NS_Id << "\n";
    amrex::Print() << "##### initial_charge: " << initial_charge << "\n";
    amrex::Print() << "##### num_proc: " << num_proc << "\n";
    amrex::Print() << "##### my_rank: " << my_rank << "\n";
    amrex::Print() << "step_filename_prefix_str: " << step_filename_prefix_str
                   << "\n";
    amrex::Print() << "current_filename_str: " << current_filename_str << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_MaterialParams()
{
    amrex::Print() << "\n";
    amrex::Print() << "##### num_unitcells: " << num_unitcells << "\n";
    amrex::Print() << "#####* num_atoms: " << num_atoms << "\n";
    amrex::Print() << "#####* num_atoms_per_unitcell: "
                   << num_atoms_per_unitcell << "\n";
    amrex::Print() << "#####* num_unitcells: " << num_unitcells << "\n";
    amrex::Print() << "#####* num_field_sites: " << num_field_sites << "\n";
    amrex::Print() << "#####* average_field_flag: " << average_field_flag
                   << "\n";
    amrex::Print() << "#####* num_atoms_per_field_site: "
                   << num_atoms_per_field_site << "\n";
    amrex::Print() << "#####* num_atoms_to_avg_over: " << num_atoms_to_avg_over
                   << "\n";
    amrex::Print() << "#####* primary_transport_dir: " << primary_transport_dir
                   << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_MaterialOrientationParams()
{
    amrex::Print() << "##### offset: ";
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        amrex::Print() << offset[i] << "  ";
    amrex::Print() << "\n";

    if (p_rotator != nullptr) p_rotator->Print_RotationParams();
}

template <typename T>
void c_NEGF_Common<T>::Print_FieldAveragingParams()
{
    if (avg_type == s_AVG_Type::ALL)
    {
        amrex::Print() << "##### field_averaging_type: ALL\n";
    }
    else if (avg_type == s_AVG_Type::SPECIFIC)
    {
        amrex::Print() << "##### field_averaging_type: SPECIFIC\n";
        amrex::Print() << "##### atom_indices_for_averaging: \n";
        for (int i = 0; i < vec_avg_indices.size(); ++i)
        {
            amrex::Print() << vec_avg_indices[i] << "  ";
        }
    }
}

template <typename T>
void c_NEGF_Common<T>::Print_TerminalParams()
{
    amrex::Print() << "##### gate_string: " << Gate_String << "\n";

    amrex::Print() << "##### gate_terminal_type: " << gate_terminal_type_str
                   << "\n";

    amrex::Print() << "##### contact_Fermi_level, E_f / [eV]: " << E_f << "\n";

    amrex::Print() << "##### contact_mu_specified: "
                   << flag_contact_mu_specified << "\n";
    if (flag_contact_mu_specified)
    {
        amrex::Print()
            << "##### Contact_Electrochemical_Potentials, mu / [eV]: \n";
        for (int c = 0; c < NUM_CONTACTS; ++c)
        {
            amrex::Print() << "#####   contact, mu: " << c << "  "
                           << Contact_Electrochemical_Potential[c] << "\n";
        }
    }
    else
    {
        amrex::Print() << "##### Contact_Parser_String: \n";
        for (int c = 0; c < NUM_CONTACTS; ++c)
        {
            amrex::Print() << "#####   contact, mu: " << c << "  "
                           << Contact_Parser_String[c] << "\n";
        }
    }

    amrex::Print() << "##### Contact_Temperatures, T / [K]: \n";
    for (int c = 0; c < NUM_CONTACTS; ++c)
    {
        amrex::Print() << "#####   contact, T: " << c << "  "
                       << Contact_Temperature[c] << "\n";
    }
}

template <typename T>
void c_NEGF_Common<T>::Print_PotentialProfileParams()
{
    amrex::Print() << "##### flag_impose_potential: " << flag_impose_potential
                   << "\n";
    amrex::Print() << "##### potential_profile_type: "
                   << potential_profile_type_str << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_EqContourIntgPts()
{
    amrex::Print()
        << "##### Equilibrium contour integration points, eq_integration_pts: ";
    for (int c = 0; c < 3; ++c)
    {
        amrex::Print() << eq_integration_pts[c] << "  ";
    }
    amrex::Print() << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_DOSParams()
{
    amrex::Print() << "##### flag_compute_DOS: " << flag_compute_DOS << "\n";

    amrex::Print() << "##### flag_write_LDOS: " << flag_write_LDOS << "\n";

    amrex::Print() << "##### flag_write_LDOS_iter: " << flag_write_LDOS_iter
                   << "\n";

    amrex::Print() << "##### write_LDOS_iter_period: " << write_LDOS_iter_period
                   << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_FlatbandDOSParams()
{
    amrex::Print() << "##### flag_compute_flatband_dos: "
                   << flag_compute_flatband_dos << "\n";
    amrex::Print() << "##### flatband_dos_integration_pts: "
                   << flatband_dos_integration_pts << "\n";
    amrex::Print() << "##### flatband_dos_integration_limits: ";
    amrex::Print() << flatband_dos_integration_limits[0] << "  "
                   << flatband_dos_integration_limits[1] << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_NonEqPathParams()
{
    amrex::Print() << "##### num_noneq_paths: " << num_noneq_paths << "\n";
    if (num_noneq_paths > 1)
    {
        amrex::Print() << "##### Nonequilibrium path intercuts in percentage, "
                          "noneq_percent_intercuts: ";
        for (int c = 0; c < num_noneq_paths - 1; ++c)
        {
            amrex::Print() << noneq_percent_intercuts[c] << "  ";
        }
        amrex::Print() << "\n";
    }
    amrex::Print() << "##### noneq_integration_pts: ";
    total_noneq_integration_pts = 0;
    for (int c = 0; c < num_noneq_paths; ++c)
    {
        amrex::Print() << noneq_integration_pts[c] << "  ";
        total_noneq_integration_pts += noneq_integration_pts[c];
    }
    amrex::Print() << "\n";
    amrex::Print() << "#####* Total nonequilibrium contour integration points: "
                   << total_noneq_integration_pts << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_WritingRelatedFlags()
{
    amrex::Print() << "##### write_at_iter: " << write_at_iter << "\n";
    amrex::Print() << "##### flag_write_charge_components: "
                   << flag_write_charge_components << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_IntegrationParams()
{
    amrex::Print() << "##### valence_band_lower_limit, E_valence_min (eV): "
                   << E_valence_min << "\n";
    amrex::Print() << "##### pole_energy_upper_limit, E_pole_max (eV): "
                   << E_pole_max << "\n";
    amrex::Print() << "##### tiny distance between real axis and "
                      "nonequilibrium integration line, E_zPlus_imag: "
                   << E_zPlus << "\n";
    amrex::Print() << "##### Fermi tail factors (lower, upper): "
                   << Fermi_tail_factor_lower << " " << Fermi_tail_factor_upper
                   << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_AdaptiveIntegrationParams()
{
    amrex::Print() << "##### flag_adaptive_integration_limits: "
                   << flag_adaptive_integration_limits << "\n";
    if (flag_adaptive_integration_limits)
    {
        amrex::Print() << "##### integrand_correction_interval: "
                       << integrand_correction_interval << "\n";
        amrex::Print() << "##### kT_window_around_singularity: "
                       << "\n";
        for (int i = 0; i < 2; ++i)
        {
            amrex::Print() << kT_window_around_singularity[i] << "  ";
        }
        amrex::Print() << "\n";
        if (flag_noneq_integration_pts_density)
        {
            amrex::Print() << "##### noneq_integration_pts_density specified: ";
            for (int c = 0; c < num_noneq_paths; ++c)
            {
                amrex::Print() << noneq_integration_pts_density[c] << "  ";
            }
            amrex::Print() << "\n";
        }
    }
}

template <typename T>
void c_NEGF_Common<T>::Print_IntegrandWritingParams()
{
    amrex::Print() << "##### flag_write_integrand: "
                   << flag_write_integrand_main << "\n";
    if (flag_write_integrand_main)
    {
        amrex::Print() << "##### write_integrand_interval: "
                       << write_integrand_interval << "\n";
    }
}

template <typename T>
void c_NEGF_Common<T>::Print_AtomLocationAndChargeDistributionFilename()
{
    if (!read_atom_filename.empty())
        amrex::Print() << "##### read_atom_filename: " << read_atom_filename
                       << "\n";

    amrex::Print() << "##### flag_initialize_charge_distribution: "
                   << flag_initialize_charge_distribution << "\n";
    if (flag_initialize_charge_distribution)
        amrex::Print() << "##### charge_distribution_filename: "
                       << charge_distribution_filename << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Print_RecursiveOptimizationParams()
{
    amrex::Print() << "##### num_recursive_parts: " << num_recursive_parts
                   << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Define_GPUVectorOfAvgIndices()
{
#if AMREX_USE_GPU
    if (avg_type == s_AVG_Type::SPECIFIC)
    {
        gpuvec_avg_indices.resize(vec_avg_indices.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, vec_avg_indices.begin(),
                         vec_avg_indices.end(), gpuvec_avg_indices.begin());
    }
#endif
}

template <typename T>
void c_NEGF_Common<T>::Set_Arrays_OfSize_NumFieldSites()
{
    if (ParallelDescriptor::IOProcessor())
    {
        h_PTD_glo_vec.resize(num_field_sites);
    }
}

template <typename T>
void c_NEGF_Common<T>::Generate_AtomLocations(amrex::Vector<s_Position3D> &pos)
{
    /*First code atom locations for particular material in the specialization*/
    /*Then call this function and apply global offset specified by the user*/
    /*Also, specify PrimaryTransportDirection PTD array*/
    /*amrex::Print() << "PTD is specified in nm! \n";*/

    for (int l = 0; l < num_field_sites; ++l)
    {
        h_PTD_glo_vec[l] =
            pos[l * num_atoms_per_field_site].dir[primary_transport_dir] /
            1.e-9;
        // amrex::Print() << l << "  " << h_PTD_glo_vec(l) << "\n";
    }

    for (int i = 0; i < num_atoms; ++i)
    {
        for (int j = 0; j < AMREX_SPACEDIM; ++j)
        {
            pos[i].dir[j] += offset[j];
        }

        if (p_rotator != nullptr)
            p_rotator->RotateContainer(pos[i].dir, offset);
    }
}

template <typename T>
void c_NEGF_Common<T>::Define_PotentialProfile()
{
    auto const &h_U_loc = h_U_loc_data.table();

    switch (map_PotentialProfile[potential_profile_type_str])
    {
        case s_Potential_Profile_Type::CONSTANT:
        {
            amrex::ParmParse pp_ns(name);
            amrex::Real V_const = 0;
            queryWithParser(pp_ns, "applied_voltage", V_const);
            amrex::Print()
                << "#####* For Constant Potential Profile, applied voltage: "
                << V_const << "\n";

            for (int l = 0; l < blkCol_size_loc; ++l)
            {
                h_U_loc(l) = -V_const;
            }
            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                U_contact[c] = -V_const;
            }
            break;
        }
        case s_Potential_Profile_Type::LINEAR:
        {
            amrex::ParmParse pp_ns(name);
            amrex::Vector<amrex::Real> vec_V;
            getArrWithParser(pp_ns, "applied_voltage_limits", vec_V, 0,
                             NUM_CONTACTS);

            amrex::Print() << "#####* For Linear Potential Profile, "
                              "applied_voltage_limits: "
                           << vec_V[0] << "  " << vec_V[1] << "\n";
            for (int l = 0; l < blkCol_size_loc; ++l)
            {
                int gid = vec_blkCol_gids[l];
                h_U_loc(l) = -vec_V[0] + (static_cast<amrex::Real>(gid) /
                                          (num_field_sites - 1.)) *
                                             (vec_V[0] - vec_V[1]);
            }
            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                U_contact[c] =
                    -vec_V[0] +
                    (static_cast<amrex::Real>(global_contact_index[c]) /
                     (num_field_sites - 1.)) *
                        (vec_V[0] - vec_V[1]);
            }
            break;
        }
        case s_Potential_Profile_Type::POINT_CHARGE:
        {
            // amrex::Array<amrex::Real,2> QD_loc = {0., 1}; //1nm away in z
            // for (int l=0; l<NSType::num_field_sites; ++l)
            //{
            //     amrex::Real r = sqrt(pow((PTD[l] - QD_loc[0]),2.) +
            //     pow(QD_loc[1],2))*1e-9; NSType::Potential[l]   =
            //     -1*(1./(4.*MathConst::pi*PhysConst::ep0*1.)*(PhysConst::q_e/r));
            // }
            break;
        }
    }
}

template <typename T>
void c_NEGF_Common<T>::Define_MatrixPartition()
{
    Hsize_glo = get_Hsize();
    amrex::Print() << "\nHsize_glo: " << Hsize_glo << "\n";
    Hsize_recur_part = ceil(Hsize_glo / num_recursive_parts);
    amrex::Print()
        << "#####* Hsize_recur_part = ceil(Hsize_glo/num_recursive_parts): "
        << Hsize_recur_part << "\n";

    bool flag_fixed_blk_size = false;
    const int THRESHOLD_BLKCOL_SIZE = 40000; /*matrix size*/

    if (flag_fixed_blk_size)
        amrex::Print() << "max_blkCol_perProc is fixed by user\n";
    else
    {
        amrex::Print() << "max_blkCol_perProc is computed at run-time\n";

        max_blkCol_perProc =
            ceil(static_cast<amrex::Real>(Hsize_glo) / num_proc);
        if (max_blkCol_perProc > THRESHOLD_BLKCOL_SIZE)
        {
            max_blkCol_perProc = THRESHOLD_BLKCOL_SIZE;
            /*assert that use larger number of procs*/
        }
    }
    amrex::Print() << "max block columns per proc: " << max_blkCol_perProc
                   << "\n";

    num_proc_with_blkCol =
        ceil(static_cast<amrex::Real>(Hsize_glo) / max_blkCol_perProc);
    amrex::Print() << "number of procs with block columns: "
                   << num_proc_with_blkCol << "\n";
    /*if num_proc_with_blk >= num_proc, assert.*/

    vec_cumu_blkCol_size.resize(num_proc_with_blkCol + 1);
    vec_cumu_blkCol_size[0] = 0;
    for (int p = 1; p < num_proc_with_blkCol; ++p)
    {
        vec_cumu_blkCol_size[p] =
            vec_cumu_blkCol_size[p - 1] + max_blkCol_perProc;
        /*All proc except the last one contains max_blkCol_perProc number of
         * column blks.*/
    }
    vec_cumu_blkCol_size[num_proc_with_blkCol] = Hsize_glo;

    blkCol_size_loc = 0;
    if (my_rank < num_proc_with_blkCol)
    {
        int blk_gid = my_rank;
        blkCol_size_loc =
            vec_cumu_blkCol_size[blk_gid + 1] - vec_cumu_blkCol_size[blk_gid];

        /*check later:setting vec_blkCol_gids may not be necessary*/
        for (int c = 0; c < blkCol_size_loc; ++c)
        {
            int col_gid = vec_cumu_blkCol_size[blk_gid] + c;
            vec_blkCol_gids.push_back(col_gid);
        }
    }

    MPI_recv_count.resize(num_proc);
    MPI_recv_disp.resize(num_proc);

    for (int p = 0; p < num_proc; ++p)
    {
        if (p < num_proc_with_blkCol)
        {
            MPI_recv_count[p] =
                vec_cumu_blkCol_size[p + 1] - vec_cumu_blkCol_size[p];
            MPI_recv_disp[p] = vec_cumu_blkCol_size[p];
        }
        else
        {
            MPI_recv_count[p] = 0;
            MPI_recv_disp[p] = 0;
        }
        // amrex::Print() << "p,recv, disp: " << p << "  "
        //                << MPI_recv_count[p] << "  "
        //                << MPI_recv_disp[p] << "\n";
    }

    Define_MPI_BlkType();
}

template <typename T>
void c_NEGF_Common<T>::Initialize_NEGF_Params(
    const std::string &name_str, const int &id_counter,
    const amrex::Real &initial_deposit_value,
    const std::string &negf_foldername_str)
{
    Set_KeyParams(name_str, id_counter, initial_deposit_value);

    Define_FoldersAndFiles(negf_foldername_str);

    Read_NanostructureProperties();

    Set_MaterialParameters();

    Print_ReadData();

    Define_MatrixPartition();
}

template <typename T>
void c_NEGF_Common<T>::Initialize_NEGF(const std::string common_foldername_str,
                                       const bool _use_electrostatic)
{
    Allocate_Arrays();

    Construct_Hamiltonian();

    if (use_decimation) Construct_ContactHamiltonian();

    Define_ContactInfo();

    amrex::Print() << "#####* Initially defining energy limits:\n";
    Define_EnergyLimits();

    Define_IntegrationPaths();

    if (flag_compute_flatband_dos)
    {
        bool flag_write_LDOS = false;

        std::string dos_dir = step_foldername_str + "/DOS_flatband";

        if (ParallelDescriptor::IOProcessor()) CreateDirectory(dos_dir);

        Compute_DensityOfStates(dos_dir, flag_write_LDOS);
    }

    Compute_Rho0();

    if (!_use_electrostatic)
    {
        Define_PotentialProfile();
    }
}

template <typename T>
void c_NEGF_Common<T>::Solve_NEGF(RealTable1D &n_curr_out_data, const int iter,
                                  const bool flag_update_terminal_bias)
{
    if (flag_update_terminal_bias) Set_TerminalBiasesAndContactPotential();

    Add_PotentialToHamiltonian();

    if (flag_adaptive_integration_limits and
        iter % integrand_correction_interval == 0)
    {
        flag_correct_integration_limits = true;
        amrex::Print() << "\n setting flag_correct_integration_limits: "
                       << flag_correct_integration_limits << "\n";
    }
    else
    {
        flag_correct_integration_limits = false;
    }

    if (flag_EC_potential_updated)
    {
        Update_ContactElectrochemicalPotential();
        Define_EnergyLimits();
    }
    if (flag_EC_potential_updated or flag_correct_integration_limits)
    {
        Update_IntegrationPaths();
        flag_EC_potential_updated = false;
    }

    Compute_InducedCharge(n_curr_out_data);
}

template <typename T>
void c_NEGF_Common<T>::Write_Data_General(const bool isIter,
                                          const int iter_or_step,
                                          const int total_iterations_in_step,
                                          const amrex::Real avg_intg_pts,
                                          RealTable1D const &h_n_curr_out_data,
                                          RealTable1D const &h_Norm_data)
{
    // Note: n_curr_out_glo is output from negf and input to broyden.
    // n_curr_in is the broyden predicted charge for next iteration.
    // NEGF->n_curr_out -> Broyden->n_curr_in -> Electrostatics -> NEGF.
    std::string write_filename;
    bool to_write = false;

    if (isIter)
    {
        write_filename = get_iter_filename();
        if (get_flag_write_at_iter())
        {
            to_write = true;
        }
    }
    else
    {
        write_filename = get_step_filename();
        to_write = true;
    }

    if (to_write)
    {
        auto const &h_n_curr_out = h_n_curr_out_data.table();
        auto const &h_Norm = h_Norm_data.table();

        RealTable1D n_curr_out_glo_data;
        RealTable1D Norm_glo_data;

        if (ParallelDescriptor::IOProcessor())
        {
            const int Hsize = get_num_field_sites();
            n_curr_out_glo_data.resize({0}, {Hsize}, The_Pinned_Arena());
            Norm_glo_data.resize({0}, {Hsize}, The_Pinned_Arena());
        }
        auto const &n_curr_out_glo = n_curr_out_glo_data.table();
        auto const &Norm_glo = Norm_glo_data.table();

        MPI_Gatherv(&h_n_curr_out(0), MPI_recv_count[my_rank], MPI_DOUBLE,
                    &n_curr_out_glo(0), MPI_recv_count.data(),
                    MPI_recv_disp.data(), MPI_DOUBLE,
                    ParallelDescriptor::IOProcessorNumber(),
                    ParallelDescriptor::Communicator());

        MPI_Gatherv(&h_Norm(0), MPI_recv_count[my_rank], MPI_DOUBLE,
                    &Norm_glo(0), MPI_recv_count.data(), MPI_recv_disp.data(),
                    MPI_DOUBLE, ParallelDescriptor::IOProcessorNumber(),
                    ParallelDescriptor::Communicator());

        Write_Data(write_filename, n_curr_out_glo_data, Norm_glo_data);
    }

    if (!isIter)
    {
        Write_Current(iter_or_step, total_iterations_in_step, avg_intg_pts);
    }
}

template <typename T>
void c_NEGF_Common<T>::Write_Data(const std::string filename_prefix,
                                  const RealTable1D &n_curr_out_data,
                                  const RealTable1D &Norm_data)
{
    Write_PotentialAtSites(filename_prefix);
    if (ParallelDescriptor::IOProcessor())
    {
        Write_InducedCharge(filename_prefix, n_curr_out_data);
        Write_ChargeNorm(filename_prefix, Norm_data);
    }
}

template <typename T>
void c_NEGF_Common<T>::Allocate_ArraysForHamiltonian()
{
    ComplexType zero(0., 0.);

    h_minusHa_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_minusHa_loc_data, zero);

    h_Hb_loc_data.resize({0}, {offDiag_repeatBlkSize}, The_Pinned_Arena());
    SetVal_Table1D(h_Hb_loc_data, zero);

    h_Hc_loc_data.resize({0}, {offDiag_repeatBlkSize}, The_Pinned_Arena());
    SetVal_Table1D(h_Hc_loc_data, zero);
}

template <typename T>
void c_NEGF_Common<T>::Allocate_ArraysForContactHamiltonian()
{
    /* H_left  = | Alpha0  Beta4                                  |
     *           | Beta4^D Alpha4  Beta3                          |
     *           |         Beta3^D Alpha3  Beta2                  |
     *           |                 Beta2^D Alpha2  Beta1          |
     *           |                         Beta1^D Alpha1  Beta0  |
     *           |                                 Beta0^D Alpha0 |
     * Alpha, Beta size: decimation layers
     */

    ComplexType zero(0., 0.);
    h_HcontactAlpha_loc_data.resize({0}, {decimation_layers - 1},
                                    The_Pinned_Arena());
    SetVal_Table1D(h_HcontactAlpha_loc_data, zero);

    h_HcontactBeta_loc_data.resize({0}, {decimation_layers - 1},
                                   The_Pinned_Arena());
    SetVal_Table1D(h_HcontactBeta_loc_data, zero);
}

template <typename T>
void c_NEGF_Common<T>::Compute_CondensedHamiltonian(CondensedHamiltonian &CondH,
                                                    const ComplexType EmU)
{
    auto &Xi_s = CondH.Xi_s;
    auto &Xi = CondH.Xi;
    auto &Pi = CondH.Pi;

    int P = decimation_layers;

    BlkTable1D H_tilde_data({0}, {P - 1});
    auto const &H_tilde = H_tilde_data.table();

    auto const &HcontactAlpha = h_HcontactAlpha_loc_data.table();
    auto const &HcontactBeta = h_HcontactBeta_loc_data.table();

    MatrixBlock<T> EmUI;
    EmUI.SetDiag(EmU);  // This is (E-U)I
    auto temp = EmUI - HcontactAlpha(P - 1);
    H_tilde(P - 1) =
        temp.Inverse();  // This is (P-1) element of [(E-U)I - contact_alpha]^-1

    auto H_tilde_kP = H_tilde(P - 1);  // This is H_tilde(P-1)(P-1)

    for (int k = P - 2; k >= 1; --k)
    {
        auto temp1 =
            EmUI - HcontactAlpha(k) -
            HcontactBeta(k) * H_tilde(k + 1) * HcontactBeta(k).Dagger();

        H_tilde(k) = temp1.Inverse();

        H_tilde_kP = H_tilde(k) * HcontactBeta(k) * H_tilde_kP;
    }
    // here H_tilde_kP is H_tilde_1P

    MatrixBlock<T> C_tilde_kk = H_tilde(1);  //==H_tilde_11 = C_tilde_11

    for (int k = 2; k < P; ++k)
    {
        C_tilde_kk = H_tilde(k) + H_tilde(k) * HcontactBeta(k - 1).Dagger() *
                                      C_tilde_kk * HcontactBeta(k - 1) *
                                      H_tilde(k);
    }
    /* Here, C_tilde_kk = C_tilde_(P-1)(P-1)
     * and,  C_tilde_1P = H_tilde_1P = H_tilde_kP
     *       C_tilde_11 = H_tilde(1)
     */

    Xi_s = HcontactAlpha(0) +
           HcontactBeta(0) * H_tilde(1) * HcontactBeta(0).Dagger();
    Xi = Xi_s + HcontactBeta(P - 1) * C_tilde_kk * HcontactBeta(P - 1).Dagger();
    Pi = HcontactBeta(0) * H_tilde_kP * HcontactBeta(P - 1);

    /* For P=2, as an example,
     *
     * Xi_s = Hb(0) * H_tilde(1) * Hb(0).Dagger();
     * Xi   = Xi_s + Hb(1) * C_tilde_kk * Hb(1).Dagger();
     * Pi   = Hb(0) * H_tilde_kP * Hb(1);
     *
     * H_tilde(1) = 1/(E-U)
     * Hb(0)      = beta
     * Hb(1)      = gamma
     * C_tilde_kk = H_tilde(1)
     * H_tilde_kP = H_tilde(1)
     *
     */

    /*
      amrex::Print() << " EmU: " << EmU  << "\n";
      amrex::Print() << "Xi_s: " << Xi_s << "\n";
      amrex::Print() << "  Xi: " << Xi   << "\n";
      amrex::Print() << "  Pi: " << Pi   << "\n";
      amrex::Print() << " H_tilde(1): " << H_tilde(1) << "\n";
      amrex::Print() << " C_tilde_kk: " << C_tilde_kk << "\n";
      amrex::Print() << " H_tilde_kP: " << H_tilde_kP << "\n";
    */
}

template <typename T>
void c_NEGF_Common<T>::Allocate_ArraysForLeadSpecificQuantities()
{
    ComplexType zero(0., 0.);

    h_tau_glo_data.resize({0}, {NUM_CONTACTS}, The_Pinned_Arena());
    SetVal_Table1D(h_tau_glo_data, zero);
}

template <typename T>
void c_NEGF_Common<T>::Allocate_ArraysForGreensAndSpectralFunction()
{
    ComplexType zero(0., 0.);

#if AMREX_USE_GPU
#ifdef COMPUTE_GREENS_FUNCTION_OFFDIAG_ELEMS
    d_GR_loc_data.resize({0, 0}, {Hsize_glo, blkCol_size_loc}, The_Arena());
#else
    d_GR_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
#endif
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
    d_A_loc_data.resize({0, 0}, {Hsize_glo, blkCol_size_loc}, The_Arena());
#else
    d_A_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
#endif
    Initialize_GPUArraysForGreensAndSpectralFunctionToZero();
#else
#ifdef COMPUTE_GREENS_FUNCTION_OFFDIAG_ELEMS
    h_GR_loc_data.resize({0, 0}, {Hsize_glo, blkCol_size_loc}, The_Arena());
    SetVal_Table2D(h_GR_loc_data, zero);
#else
    h_GR_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_GR_loc_data, zero);
#endif
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
    h_A_loc_data.resize({0, 0}, {Hsize_glo, blkCol_size_loc}, The_Arena());
    SetVal_Table2D(h_A_loc_data, zero);
#else
    h_A_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_A_loc_data, zero);
#endif
#endif
}

template <typename T>
void c_NEGF_Common<T>::Initialize_GPUArraysForChargeComputationToZero()
{
#if AMREX_USE_GPU
    auto const &Rho0_loc = d_Rho0_loc_data.table();
    auto const &RhoEq_loc = d_RhoEq_loc_data.table();
    auto const &RhoNonEq_loc = d_RhoNonEq_loc_data.table();
    auto const &GR_atPoles_loc = d_GR_atPoles_loc_data.table();
    amrex::ParallelFor(blkCol_size_loc,
                       [=] AMREX_GPU_DEVICE(int n) noexcept
                       {
                           Rho0_loc(n) = 0.;
                           RhoEq_loc(n) = 0.;
                           RhoNonEq_loc(n) = 0.;
                           GR_atPoles_loc(n) = 0.;
                       });
#endif
}

template <typename T>
void c_NEGF_Common<T>::Initialize_GPUArraysForGreensAndSpectralFunctionToZero()
{
#if AMREX_USE_GPU
    auto const &GR_loc = d_GR_loc_data.table();
    auto const &A_loc = d_A_loc_data.table();
    amrex::ParallelFor(blkCol_size_loc,
                       [=] AMREX_GPU_DEVICE(int n) noexcept
                       {
#ifdef COMPUTE_GREENS_FUNCTION_OFFDIAG_ELEMS
                           for (int m = 0; m < Hsize; ++m)
                           {
                               GR_loc(m, n) = 0.;
                           }
#else
        GR_loc(n) = 0.;
#endif
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                           for (int m = 0; m < Hsize; ++m)
                           {
                               A_loc(m, n) = 0.;
                           }
#else
        A_loc(n) = 0.;
#endif
                       });
#endif
}

template <typename T>
void c_NEGF_Common<T>::Allocate_ArraysForChargeAndCurrent()
{
    ComplexType zero(0., 0.);
#if AMREX_USE_GPU
    d_Rho0_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
    d_RhoEq_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
    d_RhoNonEq_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
    d_GR_atPoles_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());

    Initialize_GPUArraysForChargeComputationToZero();
#else
    h_Rho0_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_Rho0_loc_data, zero);

    h_RhoEq_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_RhoEq_loc_data, zero);

    h_RhoNonEq_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_RhoNonEq_loc_data, zero);

    h_GR_atPoles_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_GR_atPoles_loc_data, zero);
#endif

    h_Current_loc_data.resize({0}, {NUM_CONTACTS}, The_Pinned_Arena());
    SetVal_Table1D(h_Current_loc_data, 0.);
}

template <typename T>
void c_NEGF_Common<T>::Allocate_ArrayForPotential()
{
    h_U_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_U_loc_data, 0.);
}

template <typename T>
void c_NEGF_Common<T>::Allocate_ArrayForEnergy()
{
    h_E_RealPath_data.resize({0}, {NUM_ENERGY_PTS_REAL}, The_Pinned_Arena());
}

template <typename T>
void c_NEGF_Common<T>::Allocate_Arrays()
{
    Allocate_ArraysForHamiltonian();
    if (use_decimation) Allocate_ArraysForContactHamiltonian();
    Allocate_ArraysForLeadSpecificQuantities();
    Allocate_ArraysForGreensAndSpectralFunction();
    Allocate_ArraysForChargeAndCurrent();
    Allocate_ArrayForPotential();
    Allocate_ArrayForEnergy();
}

template <typename T>
void c_NEGF_Common<T>::Add_PotentialToHamiltonian()
{
    auto const &h_minusHa = h_minusHa_loc_data.table();
    auto const &h_U_loc = h_U_loc_data.table();

    for (int c = 0; c < blkCol_size_loc; ++c)
    {
        h_minusHa(c) =
            -1 * h_U_loc(c); /*Note: Ha = H0a (=0) + U, so -Ha = -U */
    }
}

template <typename T>
void c_NEGF_Common<T>::Update_ContactElectrochemicalPotential()
{
    // amrex::Print() <<  "Updated contact electrochemical potential: \n";
    for (int c = 0; c < NUM_CONTACTS; ++c)
    {
        mu_contact[c] = Contact_Electrochemical_Potential[c];
        // amrex::Print() << "  contact, mu: " <<  c << " " << mu_contact[c] <<
        // "\n";
    }
    // std::cout << " proc, mu_contact, U_contact: " << my_rank << "; "
    //                                     << mu_contact[0] << " " <<
    //                                     mu_contact[1] << "; "
    //                                     << U_contact[0] << " "  <<
    //                                     U_contact[1] << "\n";
    // MPI_Barrier(ParallelDescriptor::Communicator());
}

template <typename T>
void c_NEGF_Common<T>::Define_EnergyLimits()
{
    for (int c = 0; c < NUM_CONTACTS; ++c)
    {
        kT_contact[c] = PhysConst::kb_eVperK *
                        Contact_Temperature[c]; /*set Temp in the input*/
    }

    mu_min = mu_contact[0];
    mu_max = mu_contact[0];
    kT_min = kT_contact[0];
    kT_max = kT_contact[0];

    flag_noneq_exists = false;

    for (int c = 1; c < NUM_CONTACTS; ++c)
    {
        if (mu_min > mu_contact[c])
        {
            mu_min = mu_contact[c];
        }
        if (mu_max < mu_contact[c])
        {
            mu_max = mu_contact[c];
        }
        if (kT_min > kT_contact[c])
        {
            kT_min = kT_contact[c];
        }
        if (kT_max < kT_contact[c])
        {
            kT_max = kT_contact[c];
        }
    }
    if (fabs(mu_min - mu_max) > 1e-8) flag_noneq_exists = true;
    if (fabs(kT_min - kT_max) > 0.01) flag_noneq_exists = true;

    // flag_noneq_exists = true;
    // ComplexType val(0.,1e-8);
    // E_zPlus = val;
    E_contour_left = E_valence_min + E_zPlus; /*set in the input*/
    E_rightmost = mu_max + Fermi_tail_factor_upper * kT_max + E_zPlus;

    int num_poles = int((E_pole_max - MathConst::pi * kT_max) /
                            (2. * MathConst::pi * kT_max) +
                        1);

    if (flag_noneq_exists)
    {
        amrex::Print() << "\n Nonequilibrium exists!\n";
        E_contour_right = mu_min - Fermi_tail_factor_lower * kT_max + E_zPlus;
        num_enclosed_poles = 0;

        ComplexType val2(E_contour_right.real(),
                         2 * num_poles * MathConst::pi * kT_max);
        E_zeta = val2;

        ComplexType val3(E_contour_right.real() - 14 * kT_max, E_zeta.imag());
        E_eta = val3;
    }
    else
    {
        amrex::Print() << "\n Nonequilibrium doesn't exist!\n";
        E_contour_right = E_rightmost;
        num_enclosed_poles = num_poles;
        E_poles_vec.resize(num_enclosed_poles);

        for (int p = 0; p < num_enclosed_poles; ++p)
        {
            ComplexType pole(mu_min, MathConst::pi * kT_max * (2 * p + 1));
            E_poles_vec[p] = pole;
        }
        ComplexType val2(E_contour_right.real(),
                         2 * num_poles * MathConst::pi * kT_max);
        E_zeta = val2;
        amrex::Real Fermi_tail_factor_max =
            std::max(Fermi_tail_factor_lower, Fermi_tail_factor_upper);
        ComplexType val3(mu_min - Fermi_tail_factor_max * kT_max,
                         E_zeta.imag());
        E_eta = val3;
    }

    amrex::Print() << " U_contact: ";
    for (int c = 0; c < NUM_CONTACTS; ++c)
    {
        amrex::Print() << U_contact[c] << " ";
    }
    amrex::Print() << "\n";
    amrex::Print() << " E_f: " << E_f << "\n";
    amrex::Print() << " mu_min/max: " << mu_min << " " << mu_max << "\n";
    amrex::Print() << " kT_min/max: " << kT_min << " " << kT_max << "\n";
    amrex::Print() << " E_zPlus: " << E_zPlus << "\n";
    amrex::Print() << " E_contour_left/E_contour_right/E_rightmost: "
                   << E_contour_left << "  " << E_contour_right << "  "
                   << E_rightmost << "\n";
    amrex::Print() << " E_pole_max: " << E_pole_max
                   << ", number of poles: " << num_enclosed_poles
                   << ", pikT: " << MathConst::pi * kT_max << "\n";
    amrex::Print() << " E_zeta: " << E_zeta << "\n";
    amrex::Print() << " E_eta: " << E_eta << "\n";
}

template <typename T>
ComplexType c_NEGF_Common<T>::FermiFunction(ComplexType E_minus_Mu,
                                            const amrex::Real kT)
{
    ComplexType one(1., 0.);
    return one / (exp(E_minus_Mu / kT) + one);
}

template <typename T>
void c_NEGF_Common<T>::Define_IntegrationPaths()
{
    /* Define_ContourPath_Rho0 */
    ContourPath_Rho0.resize(3);
    ContourPath_Rho0[0].Define_GaussLegendrePoints(E_zPlus, E_zeta, 30, 0);
    ContourPath_Rho0[1].Define_GaussLegendrePoints(E_zeta, E_eta, 30, 0);
    ContourPath_Rho0[2].Define_GaussLegendrePoints(E_eta, E_contour_left, 30,
                                                   1);

    /* Define_ContourPath_DOS */
    if (flag_compute_flatband_dos)
    {
        ContourPath_DOS.resize(1);
        ComplexType min(flatband_dos_integration_limits[0], E_zPlus.imag());
        ComplexType max(flatband_dos_integration_limits[1], E_zPlus.imag());
        ContourPath_DOS[0].Define_GaussLegendrePoints(
            min, max, flatband_dos_integration_pts, 0);
    }
}

template <typename T>
int c_NEGF_Common<T>::get_Total_NonEq_Integration_Pts() const
{
    return std::accumulate(noneq_integration_pts.begin(),
                           noneq_integration_pts.end(), 0);
}

template <typename T>
int c_NEGF_Common<T>::get_Total_Integration_Pts() const
{
    int intg_pts = 0;
    if (flag_noneq_exists)
    {
        intg_pts = get_Total_NonEq_Integration_Pts();
    }
    else
    {
        for (auto &path : ContourPath_RhoEq)
        {
            intg_pts += path.num_pts;
        }
    }
    return intg_pts;
}

// template<typename T>
// amrex::Real
// c_NEGF_Common<T>::FermiFunction_real(const amrex::Real E_minus_Mu, const
// amrex::Real kT)
//{
//     ComplexType one(1., 0.);
//     return one / (exp(E_minus_Mu / kT) + one);
// }
//
// template<typename T>
// amrex::Real
// c_NEGF_Common<T>::Derivative
//     ( amrex::Real (*f) (const std::vector<amrex::Real>& E_vec, const
//     amrex::Real kT),
//       const std::vector<amrex::Real>& v,
//       double epsilon=1e-8)
//{
//     int size = v.size();
//     std::vector<amrex::Real> derivative(size, 0.);
//     std::vector<amrex::Real> perturbation(size, epsilon);
//
//     amrex::Real fPlus
//     for(int i=0; i<size; ++i)
//     {
//         derivative[i] = (fPlus - fMinus) / (2*epsilon);
//     }
//     return derivative;
// }

template <typename T>
amrex::Real c_NEGF_Common<T>::Compute_Conductance(
    const amrex::Vector<ComplexType> E_vec,
    const RealTable1D &Transmission_data, RealTable1D &Conductance_data)
{
    amrex::Real sum_conductance = 0.;
    auto const &Tran = Transmission_data.const_table();
    auto const &Cond = Conductance_data.table();
    /* G_quantum = 2*q^2/h.
     * This quantum of conductance, G_quantum, includes spin degeneracy
     * The subband degeneracy factor is taken into account in calculation of
     * transmission. T(E) in the calculation formula is not transmission
     * probability but transmission. For e.g. for (17,0) nanotube, the first
     * subband is double degenerate. So if we use 1 mode for calculation, then
     * is extra factor would be 2. We take G_quantum=1, i.e., outputted results
     * are already normalized by (2q^2/h);
     */
    amrex::Real G_quantum = 1.;

    std::vector<amrex::Real> dFdE(E_vec.size(), 0.);
    amrex::Real deltaE = E_vec[0].real() - E_vec[1].real();

    amrex::Real denom_inv = 1. / (2 * deltaE);

    amrex::Real f_0 =
        FermiFunction(E_vec[0] - mu_contact[0], kT_contact[0]).real();
    amrex::Real f_1 =
        FermiFunction(E_vec[1] - mu_contact[0], kT_contact[0]).real();
    amrex::Real f_2 =
        FermiFunction(E_vec[2] - mu_contact[0], kT_contact[0]).real();
    dFdE[0] = (-3 * f_0 + 4 * f_1 - f_2) * denom_inv;

    int n = E_vec.size() - 1;
    amrex::Real f_n =
        FermiFunction(E_vec[n] - mu_contact[0], kT_contact[0]).real();
    amrex::Real f_nm1 =
        FermiFunction(E_vec[n - 1] - mu_contact[0], kT_contact[0]).real();
    amrex::Real f_nm2 =
        FermiFunction(E_vec[n - 2] - mu_contact[0], kT_contact[0]).real();
    dFdE[n] = (3 * f_n - 4 * f_nm1 + f_nm2) * denom_inv;

    for (int i = 1; i < n; ++i)
    {
        amrex::Real F_im1 =
            FermiFunction(E_vec[i - 1] - mu_contact[0], kT_contact[0]).real();
        amrex::Real F_ip1 =
            FermiFunction(E_vec[i + 1] - mu_contact[0], kT_contact[0]).real();
        dFdE[i] = (F_ip1 - F_im1) * denom_inv;
    }

    for (int i = 0; i < E_vec.size(); ++i)
    {
        Cond(i) = -G_quantum * Tran(i) * dFdE[i];
        sum_conductance += Cond(i) * deltaE;
    }
    return sum_conductance;
}

template <typename T>
void c_NEGF_Common<T>::Write_Eql_Characteristics(
    const amrex::Vector<ComplexType> E_vec, const RealTable1D &DOS_data,
    const RealTable1D &Transmission_data, const RealTable1D &Conductance_data,
    std::string filename)
{
    amrex::Print()
        << "Writing Fermi Functions, Transmission, Conductance as E. E.size(): "
        << E_vec.size() << "\n";
    std::ofstream outfile;
    outfile.open(filename.c_str());

    auto const &DOS = DOS_data.const_table();
    auto const &Tran = Transmission_data.const_table();
    auto const &Cond = Conductance_data.const_table();
    auto thi_dos = DOS_data.hi();
    auto thi_tran = Transmission_data.hi();
    auto thi_cond = Conductance_data.hi();

    outfile << " E_r, Fermi_Contact1, Fermi_Contact2, DOS_r, Transmission_r, "
               "Conductance_r \n";
    if (E_vec.size() != thi_dos[0] || E_vec.size() != thi_tran[0] ||
        E_vec.size() != thi_cond[0])
    {
        outfile << "Mismatch in the size of Vec size: " << E_vec.size()
                << " and Table1D_data for DOS, transmission, or conductance: "
                << thi_dos[0] << ", " << thi_tran[0] << ", " << thi_cond[0]
                << "\n";
    }
    else
    {
        for (int i = 0; i < E_vec.size(); ++i)
        {
            outfile
                << std::setw(20) << E_vec[i].real() << std::setw(20)
                << FermiFunction(E_vec[i] - mu_contact[0], kT_contact[0]).real()
                << std::setw(20)
                << FermiFunction(E_vec[i] - mu_contact[1], kT_contact[1]).real()
                << std::setw(35) << DOS(i) << std::setw(35) << Tran(i)
                << std::setw(35) << Cond(i) << "\n";
        }
    }
    outfile.close();
}

template <typename T>
template <typename TableType>
void c_NEGF_Common<T>::Write_Integrand(const amrex::Vector<ComplexType> &E_vec,
                                       const TableType &Arr_Channel_data,
                                       const TableType &Arr_Source_data,
                                       const TableType &Arr_Drain_data,
                                       std::string filename)
{
    amrex::Print() << "Writing integrand in file: " << filename << "\n";
    std::ofstream outfile;
    outfile.open(filename.c_str());
    outfile << "(E_r-mu_0)/kT Intg_Channel Intg_Source Intg_Drain |F1-F2| E_r"
            << "\n";

    auto const &Channel = Arr_Channel_data.const_table();
    auto const &Source = Arr_Source_data.const_table();
    auto const &Drain = Arr_Drain_data.const_table();
    auto thi = Arr_Channel_data.hi();

    if (E_vec.size() == thi[0])
    {
        for (int e = 0; e < thi[0]; ++e)
        {
            amrex::Real Fermi_Diff = std::fabs(
                FermiFunction(E_vec[e] - mu_contact[0], kT_contact[0]).real() -
                FermiFunction(E_vec[e] - mu_contact[1], kT_contact[1]).real());
            outfile << std::setprecision(15) << std::setw(25)
                    << (E_vec[e].real() - mu_contact[0]) / kT_contact[0]
                    << std::setw(25) << Channel(e) << std::setw(25) << Source(e)
                    << std::setw(25) << Drain(e) << std::setw(25) << Fermi_Diff
                    << std::setw(25) << E_vec[e].real() << "\n";
        }
    }
    else
    {
        outfile << "In Write_Integrand: Mismatch in the size of Vec and "
                   "Table1D_data!"
                << "\n";
    }
    outfile.close();
}

template <typename T>
void c_NEGF_Common<T>::Generate_NonEq_Paths()
{
    ContourPath_RhoNonEq.clear();
    ContourPath_RhoNonEq.resize(num_noneq_paths);
    amrex::Vector<ComplexType> noneq_path_min(num_noneq_paths);
    amrex::Vector<ComplexType> noneq_path_max(num_noneq_paths);

    noneq_path_min[0] = E_contour_right;
    noneq_path_max[num_noneq_paths - 1] = E_rightmost;
    ComplexType noneq_path_length = E_rightmost - E_contour_right;

    for (int p = 0; p < num_noneq_paths - 1; ++p)
    {
        noneq_path_max[p] =
            E_contour_right +
            (noneq_percent_intercuts[p] / 100.) * noneq_path_length;
    }
    for (int p = 1; p < num_noneq_paths; ++p)
    {
        noneq_path_min[p] = noneq_path_max[p - 1];
    }

    amrex::Print() << "\n creating noneq paths:\n";
    for (int p = 0; p < num_noneq_paths; ++p)
    {
        amrex::Print() << "\npath_id: " << p << " " << noneq_path_min[p].real()
                       << " to " << noneq_path_max[p].real()
                       << " Pts: " << noneq_integration_pts[p] << "\n";
        amrex::Print() << "  i.e. (E-mu_0)/kT[0]: "
                       << (noneq_path_min[p].real() - mu_contact[0]) /
                              kT_contact[0]
                       << " to "
                       << (noneq_path_max[p].real() - mu_contact[0]) /
                              kT_contact[0]
                       << "\n";

        ContourPath_RhoNonEq[p].Define_GaussLegendrePoints(
            noneq_path_min[p], noneq_path_max[p], noneq_integration_pts[p], 0);
    }
    amrex::Print() << " total noneq integration pts: "
                   << get_Total_NonEq_Integration_Pts() << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Find_NonEq_Percent_Intercuts_Adaptively()
{
    amrex::Print() << "\n Finding NonEq Percent Intercuts Adaptively: \n";
    amrex::Real E_star =
        (E_at_max_noneq_integrand - mu_contact[0]) / kT_contact[0];
    amrex::Real E_min =
        (E_contour_right.real() - mu_contact[0]) / kT_contact[0];
    amrex::Real E_max = (E_rightmost.real() - mu_contact[0]) / kT_contact[0];

    if (num_noneq_paths == 3)
    {
        amrex::Print() << "\n original noneq_integration_pts: ";
        for (int c = 0; c < num_noneq_paths; ++c)
        {
            amrex::Print() << noneq_integration_pts[c] << "  ";
        }
        if (!flag_noneq_integration_pts_density)
        {
            noneq_integration_pts_density.resize(num_noneq_paths);

            noneq_integration_pts_density[0] =
                noneq_integration_pts[0] /
                ((noneq_percent_intercuts[0] / 100.) * (E_max - E_min));

            noneq_integration_pts_density[1] =
                noneq_integration_pts[1] /
                (((noneq_percent_intercuts[1] - noneq_percent_intercuts[0]) /
                  100.) *
                 (E_max - E_min));

            noneq_integration_pts_density[2] =
                noneq_integration_pts[2] /
                (((100. - noneq_percent_intercuts[1]) / 100.) *
                 (E_max - E_min));
            flag_noneq_integration_pts_density = true;
        }
        amrex::Print() << "\n noneq_integration_pts_density (per kT): ";
        for (int c = 0; c < num_noneq_paths; ++c)
        {
            amrex::Print() << noneq_integration_pts_density[c] << "  ";
        }

        amrex::Real E_lower =
            std::max(E_star - kT_window_around_singularity[0], E_min);
        amrex::Real E_upper =
            std::min(E_star + kT_window_around_singularity[1], E_max);

        amrex::Print() << "\nE_min/max/star/lower/upper: " << E_min << " "
                       << E_max << " " << E_star << " " << E_lower << " "
                       << E_upper << "\n";
        if (fabs(E_min - E_lower) < 1e-8)
        {
            amrex::Print() << " Note: E_min == E_lower: First path has 0 pts\n";
        }

        noneq_percent_intercuts[0] = 100 * (E_lower - E_min) / (E_max - E_min);
        noneq_percent_intercuts[1] = 100 * (E_upper - E_min) / (E_max - E_min);

        noneq_integration_pts[0] =
            round(((E_lower - E_min) * noneq_integration_pts_density[0]) / 2) *
            2;
        noneq_integration_pts[1] =
            round(((E_upper - E_lower) * noneq_integration_pts_density[1]) /
                  2) *
            2;
        noneq_integration_pts[2] =
            round(((E_max - E_upper) * noneq_integration_pts_density[2]) / 2) *
            2;
    }
    else if (num_noneq_paths == 2)
    {
        if (!flag_noneq_integration_pts_density)
        {
            noneq_integration_pts_density.resize(num_noneq_paths);
            noneq_integration_pts_density[0] =
                noneq_integration_pts[0] /
                ((noneq_percent_intercuts[0] / 100.) * (E_max - E_min));

            noneq_integration_pts_density[1] =
                noneq_integration_pts[1] /
                (((100 - noneq_percent_intercuts[0]) / 100.) * (E_max - E_min));

            flag_noneq_integration_pts_density = true;
        }
        amrex::Print() << "\n noneq_integration_pts_density (per kT): ";
        for (int c = 0; c < num_noneq_paths; ++c)
        {
            amrex::Print() << noneq_integration_pts_density[c] << "  ";
        }
        amrex::Real E_upper = E_star + kT_window_around_singularity[1];
        amrex::Print() << "\nE_min/max/star/upper: " << E_min << " " << E_max
                       << " " << E_star << " " << E_upper << "\n";

        noneq_percent_intercuts[0] = 100 * (E_upper - E_min) / (E_max - E_min);

        noneq_integration_pts[0] =
            round(((E_upper - E_min) * noneq_integration_pts_density[0]) / 2) *
            2;
        noneq_integration_pts[1] =
            round(((E_max - E_upper) * noneq_integration_pts_density[1]) / 2) *
            2;
    }

    total_noneq_integration_pts = 0;
    for (int c = 0; c < num_noneq_paths; ++c)
    {
        total_noneq_integration_pts += noneq_integration_pts[c];
    }

    amrex::Print() << "\n Updated noneq_percent_intercuts: ";
    for (int c = 0; c < num_noneq_paths - 1; ++c)
    {
        amrex::Print() << noneq_percent_intercuts[c] << "  ";
    }
    amrex::Print() << "\n";
    amrex::Print() << "\n Updated noneq_integration_pts: ";
    for (int c = 0; c < num_noneq_paths; ++c)
    {
        amrex::Print() << noneq_integration_pts[c] << "  ";
    }
    amrex::Print() << "\n";
    amrex::Print() << " total_noneq_integration_pts: "
                   << total_noneq_integration_pts << "\n";
}

template <typename T>
void c_NEGF_Common<T>::Update_IntegrationPaths()
{
    ContourPath_RhoEq.clear();
    ContourPath_DOS.clear();

    /* Define_ContourPath_RhoEq */
    ContourPath_RhoEq.resize(3);
    ContourPath_RhoEq[0].Define_GaussLegendrePoints(E_contour_right, E_zeta,
                                                    eq_integration_pts[0], 0);
    ContourPath_RhoEq[1].Define_GaussLegendrePoints(E_zeta, E_eta,
                                                    eq_integration_pts[1], 0);
    ContourPath_RhoEq[2].Define_GaussLegendrePoints(E_eta, E_contour_left,
                                                    eq_integration_pts[2], 1);

    /* Define_ContourPath_RhoNonEq */
    if (flag_noneq_exists)
    {
        Generate_NonEq_Paths();
        if (flag_correct_integration_limits)
        {
            Compute_RhoNonEq();
            Find_NonEq_Percent_Intercuts_Adaptively();
            Generate_NonEq_Paths();
        }
        ContourPath_DOS =
            ContourPath_RhoNonEq;  // works through copy assignment operator

        // amrex::Print() << "Printing Noneq path: \n";
        // int path_counter=0;
        // for(auto& path: ContourPath_RhoNonEq)
        //{
        //     amrex::Print() << "path: " << path_counter << " num_pts: " <<
        //     path.num_pts << "\n"; for(int e=0; e< path.num_pts; ++e) {
        //         amrex::Print() << e
        //                        << std::setw(15) << path.E_vec[e]
        //                        << std::setw(15) << path.weight_vec[e]
        //                        << std::setw(15) << path.mul_factor_vec[e] <<
        //                        "\n";
        //     }
        //     ++path_counter;
        // }
        // amrex::Print() << "Printing DOS path: \n";
        // path_counter=0;
        // for(auto& path: ContourPath_DOS)
        //{
        //     amrex::Print() << "path: " << path_counter << " num_pts: " <<
        //     path.num_pts << "\n"; for(int e=0; e< path.num_pts; ++e) {
        //         amrex::Print() << e
        //                        << std::setw(15) << path.E_vec[e]
        //                        << std::setw(15) << path.weight_vec[e]
        //                        << std::setw(15) << path.mul_factor_vec[e] <<
        //                        "\n";
        //     }
        //     ++path_counter;
        // }
    }
    else
    {
        ContourPath_DOS.resize(1);
        ContourPath_DOS[0].Define_GaussLegendrePoints(
            E_eta.real(), E_rightmost, flatband_dos_integration_pts, 0);
    }
}

template <typename T>
void c_NEGF_Common<T>::Allocate_TemporaryArraysForGFComputation()
{
    ComplexType zero(0., 0.);
    h_Alpha_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_Alpha_loc_data, zero);

    h_Alpha_glo_data.resize({0}, {Hsize_glo}, The_Pinned_Arena());
    SetVal_Table1D(h_Alpha_glo_data, zero);

    h_Xtil_glo_data.resize({0}, {Hsize_glo}, The_Pinned_Arena());
    SetVal_Table1D(h_Xtil_glo_data, zero);

    h_Ytil_glo_data.resize({0}, {Hsize_glo}, The_Pinned_Arena());
    SetVal_Table1D(h_Ytil_glo_data, zero);

    h_X_glo_data.resize({0}, {Hsize_glo}, The_Pinned_Arena());
    SetVal_Table1D(h_X_glo_data, zero);

    h_Y_glo_data.resize({0}, {Hsize_glo}, The_Pinned_Arena());
    SetVal_Table1D(h_Y_glo_data, zero);

    h_X_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_X_loc_data, zero);

    h_Y_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
    SetVal_Table1D(h_Y_loc_data, zero);

    h_Sigma_contact_data.resize({0}, {NUM_CONTACTS}, The_Pinned_Arena());
    SetVal_Table1D(h_Sigma_contact_data, zero);

    h_Fermi_contact_data.resize({0}, {NUM_CONTACTS}, The_Pinned_Arena());
    SetVal_Table1D(h_Fermi_contact_data, zero);

    h_Alpha_contact_data.resize({0}, {NUM_CONTACTS}, The_Pinned_Arena());
    SetVal_Table1D(h_Alpha_contact_data, zero);

    h_X_contact_data.resize({0}, {NUM_CONTACTS}, The_Pinned_Arena());
    SetVal_Table1D(h_X_contact_data, zero);

    h_Y_contact_data.resize({0}, {NUM_CONTACTS}, The_Pinned_Arena());
    SetVal_Table1D(h_Y_contact_data, zero);

    h_Trace_r.resize(num_traces);
    h_Trace_i.resize(num_traces);

#ifdef AMREX_USE_GPU
    d_Alpha_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
    d_X_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
    d_Y_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
    d_Xtil_glo_data.resize({0}, {Hsize_glo}, The_Arena());
    d_Ytil_glo_data.resize({0}, {Hsize_glo}, The_Arena());
    d_Sigma_contact_data.resize({0}, {NUM_CONTACTS}, The_Arena());
    d_Fermi_contact_data.resize({0}, {NUM_CONTACTS}, The_Arena());

    d_Alpha_contact_data.resize({0}, {NUM_CONTACTS}, The_Arena());
    d_X_contact_data.resize({0}, {NUM_CONTACTS}, The_Arena());
    d_Y_contact_data.resize({0}, {NUM_CONTACTS}, The_Arena());

    d_Trace_r.resize(num_traces);
    d_Trace_i.resize(num_traces);
#endif
}

template <typename T>
void c_NEGF_Common<T>::Deallocate_TemporaryArraysForGFComputation()
{
    h_Alpha_glo_data.clear();
    h_X_glo_data.clear();
    h_Y_glo_data.clear();

    h_Alpha_loc_data.clear();
    h_Xtil_glo_data.clear();
    h_Ytil_glo_data.clear();
    h_X_loc_data.clear();
    h_Y_loc_data.clear();
    h_Sigma_contact_data.clear();
    h_Fermi_contact_data.clear();

    h_Alpha_contact_data.clear();
    h_X_contact_data.clear();
    h_Y_contact_data.clear();

    h_Trace_r.clear();
    h_Trace_i.clear();

#ifdef AMREX_USE_GPU
    d_Alpha_loc_data.clear();
    d_X_loc_data.clear();
    d_Y_loc_data.clear();
    d_Xtil_glo_data.clear();
    d_Ytil_glo_data.clear();
    d_Sigma_contact_data.clear();
    d_Fermi_contact_data.clear();

    d_Alpha_contact_data.clear();
    d_X_contact_data.clear();
    d_Y_contact_data.clear();

    d_Trace_r.clear();
    d_Trace_i.clear();
#endif
}

template <typename T>
void c_NEGF_Common<T>::Compute_DensityOfStates_General(bool flag_isIter,
                                                       int iter_or_step)
{
    if (flag_isIter)
    {
        if (flag_write_LDOS_iter and
            (iter_or_step + 1) % write_LDOS_iter_period == 0)
        {
            std::string iter_dos_foldername_str =
                amrex::Concatenate(iter_foldername_str + "/DOS_iter",
                                   iter_or_step, negf_plt_name_digits);

            CreateDirectory(iter_dos_foldername_str);
            // e.g.: /negf/cnt/step0001_iter/LDOS_iter0002/

            Compute_DensityOfStates(iter_dos_foldername_str,
                                    flag_write_LDOS_iter);
        }
    }
    else if (flag_compute_DOS)
    {
        std::string dos_step_foldername_str =
            amrex::Concatenate(step_foldername_str + "/DOS_step", iter_or_step,
                               negf_plt_name_digits);

        CreateDirectory(dos_step_foldername_str);
        // e.g.: /negf/cnt/DOS_step0001/

        Compute_DensityOfStates(dos_step_foldername_str, flag_write_LDOS);
    }
}

template <typename T>
void c_NEGF_Common<T>::Compute_DensityOfStates(std::string dos_foldername,
                                               bool flag_write_LDOS)
{
    int E_total_pts = 0;
    for (auto &path : ContourPath_DOS)
    {
        E_total_pts += path.num_pts;
    }
    h_DOS_loc_data.resize({0}, {E_total_pts}, The_Pinned_Arena());
    SetVal_Table1D(h_DOS_loc_data, 0.);

    h_Transmission_loc_data.resize({0}, {E_total_pts}, The_Pinned_Arena());
    SetVal_Table1D(h_Transmission_loc_data, 0.);

    RealTable1D h_LDOS_loc_data;
    RealTable1D h_LDOS_glo_data;

    if (flag_write_LDOS)
    {
        h_LDOS_loc_data.resize({0}, {blkCol_size_loc}, The_Pinned_Arena());
        if (ParallelDescriptor::IOProcessor())
        {
            h_LDOS_glo_data.resize({0}, {Hsize_glo}, The_Pinned_Arena());
        }
    }

    auto const &h_minusHa_loc = h_minusHa_loc_data.table();
    auto const &h_Hb_loc = h_Hb_loc_data.table();
    auto const &h_Hc_loc = h_Hc_loc_data.table();
    auto const &h_tau = h_tau_glo_data.table();
    auto const &h_DOS_loc = h_DOS_loc_data.table();
    auto const &h_Transmission_loc = h_Transmission_loc_data.table();

    Allocate_TemporaryArraysForGFComputation();

    auto const &h_Alpha_loc = h_Alpha_loc_data.table();
    auto const &h_Alpha_glo = h_Alpha_glo_data.table();
    auto const &h_Xtil_glo = h_Xtil_glo_data.table();
    auto const &h_Ytil_glo = h_Ytil_glo_data.table();
    auto const &h_X_glo = h_X_glo_data.table();
    auto const &h_Y_glo = h_Y_glo_data.table();
    auto const &h_X_loc = h_X_loc_data.table();
    auto const &h_Y_loc = h_Y_loc_data.table();
    auto const &h_Alpha_contact = h_Alpha_contact_data.table();
    auto const &h_Y_contact = h_Y_contact_data.table();
    auto const &h_X_contact = h_X_contact_data.table();
    auto const &h_Sigma_contact = h_Sigma_contact_data.table();
    auto const &h_LDOS_loc = h_LDOS_loc_data.table();
    auto const &h_LDOS_glo = h_LDOS_glo_data.table();

#ifdef AMREX_USE_GPU
    auto const &GR_loc = d_GR_loc_data.table();
    auto const &A_loc = d_A_loc_data.table();
    /*constant references*/
    auto const &Alpha = d_Alpha_loc_data.const_table();
    auto const &Xtil_glo = d_Xtil_glo_data.const_table();
    auto const &d_Xtil_glo = d_Xtil_glo_data.table();
    auto const &Ytil_glo = d_Ytil_glo_data.const_table();
    auto const &d_Ytil_glo = d_Ytil_glo_data.table();
    auto const &X = d_X_loc_data.const_table();
    auto const &d_X_loc = d_X_loc_data.table();
    auto const &Y = d_Y_loc_data.const_table();
    auto const &d_Y_loc = d_Y_loc_data.table();

    auto const &Alpha_contact = d_Alpha_contact_data.const_table();
    auto const &X_contact = d_X_contact_data.const_table();
    auto const &Y_contact = d_Y_contact_data.const_table();
    auto const &Sigma_contact = d_Sigma_contact_data.const_table();

    auto *trace_r = d_Trace_r.dataPtr();
    auto *trace_i = d_Trace_i.dataPtr();
    auto &degen_vec = block_degen_gpuvec;

    RealTable1D d_LDOS_loc_data;
    if (flag_write_LDOS)
    {
        d_LDOS_loc_data.resize({0}, {blkCol_size_loc}, The_Arena());
    }
    auto const &LDOS_loc = d_LDOS_loc_data.table();
#else
    auto const &GR_loc = h_GR_loc_data.table();
    auto const &A_loc = h_A_loc_data.table();
    /*constant references*/
    auto const &Alpha = h_Alpha_loc_data.const_table();
    auto const &Xtil_glo = h_Xtil_glo_data.const_table();

    auto const &Ytil_glo = h_Ytil_glo_data.const_table();
    auto const &X = h_X_loc_data.const_table();
    auto const &Y = h_Y_loc_data.const_table();

    auto const &Alpha_contact = h_Alpha_contact_data.const_table();
    auto const &X_contact = h_X_contact_data.const_table();
    auto const &Y_contact = h_Y_contact_data.const_table();
    auto const &Sigma_contact = h_Sigma_contact_data.const_table();

    auto *trace_r = h_Trace_r.dataPtr();
    auto *trace_i = h_Trace_i.dataPtr();
    auto &degen_vec = block_degen_vec;
    auto const &LDOS_loc = h_LDOS_loc_data.table();
#endif

    int e_prev_pts = 0;
    for (int p = 0; p < ContourPath_DOS.size(); ++p)
    {
        for (int e = 0; e < ContourPath_DOS[p].num_pts; ++e)
        {
            int e_glo = e_prev_pts + e;
            ComplexType E = ContourPath_DOS[p].E_vec[e];
            ComplexType weight = ContourPath_DOS[p].weight_vec[e];
            ComplexType mul_factor = ContourPath_DOS[p].mul_factor_vec[e];

            for (int n = 0; n < blkCol_size_loc; ++n)
            {
                h_Alpha_loc(n) = E + h_minusHa_loc(n);
                /*+ because h_minusHa is defined previously as -(H0+U)*/
            }

            get_Sigma_at_contacts(h_Sigma_contact_data, E);
#ifdef AMREX_USE_GPU
            d_Sigma_contact_data.copy(h_Sigma_contact_data);
#endif

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n_glo = global_contact_index[c];

                if (n_glo >= vec_cumu_blkCol_size[my_rank] &&
                    n_glo < vec_cumu_blkCol_size[my_rank + 1])
                {
                    int n = n_glo - vec_cumu_blkCol_size[my_rank];
                    h_Alpha_loc(n) = h_Alpha_loc(n) - h_Sigma_contact(c);
                }
            }
#ifdef AMREX_USE_GPU
            d_Alpha_loc_data.copy(h_Alpha_loc_data);
#endif

            /*MPI_Allgather*/
            MPI_Allgatherv(&h_Alpha_loc(0), blkCol_size_loc, MPI_BlkType,
                           &h_Alpha_glo(0), MPI_recv_count.data(),
                           MPI_recv_disp.data(), MPI_BlkType,
                           ParallelDescriptor::Communicator());

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n_glo = global_contact_index[c];
                h_Alpha_contact(c) = h_Alpha_glo(n_glo);
            }
#ifdef AMREX_USE_GPU
            d_Alpha_contact_data.copy(h_Alpha_contact_data);
#endif

            h_Y_glo(0) = 0;
            h_X_glo(Hsize_glo - 1) = 0;
            for (int section = 0; section < num_recursive_parts; ++section)
            {
                for (int n = std::max(1, Hsize_recur_part * section);
                     n < std::min(Hsize_recur_part * (section + 1), Hsize_glo);
                     ++n)
                {
                    int p = (n - 1) % offDiag_repeatBlkSize;
                    h_Ytil_glo(n) =
                        h_Hc_loc(p) / (h_Alpha_glo(n - 1) - h_Y_glo(n - 1));
                    h_Y_glo(n) = h_Hb_loc(p) * h_Ytil_glo(n);
                }
                for (int n = std::min(Hsize_glo - 2,
                                      Hsize_recur_part *
                                          (num_recursive_parts - section));
                     n >=
                     Hsize_recur_part * (num_recursive_parts - section - 1);
                     n--)
                {
                    int p = n % offDiag_repeatBlkSize;
                    h_Xtil_glo(n) =
                        h_Hb_loc(p) / (h_Alpha_glo(n + 1) - h_X_glo(n + 1));
                    h_X_glo(n) = h_Hc_loc(p) * h_Xtil_glo(n);
                }

#ifdef AMREX_USE_GPU
                int Ytil_begin = section * Hsize_recur_part;
                int Ytil_end =
                    std::min(Hsize_recur_part * (section + 1), Hsize_glo);

                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_Ytil_glo.p + Ytil_begin,
                                      h_Ytil_glo.p + Ytil_end,
                                      d_Ytil_glo.p + Ytil_begin);

                int Xtil_begin =
                    Hsize_recur_part * (num_recursive_parts - section - 1);
                int Xtil_end =
                    std::min(Hsize_glo, Hsize_recur_part *
                                            (num_recursive_parts - section));

                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_Xtil_glo.p + Xtil_begin,
                                      h_Xtil_glo.p + Xtil_end,
                                      d_Xtil_glo.p + Xtil_begin);
#endif
            }

#ifdef AMREX_USE_GPU
            int X_begin = vec_cumu_blkCol_size[my_rank];
            int X_end = X_begin + blkCol_size_loc;
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_X_glo.p + X_begin,
                                  h_X_glo.p + X_end, d_X_loc.p);

            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Y_glo.p + X_begin,
                                  h_Y_glo.p + X_end, d_Y_loc.p);
#else
            for (int c = 0; c < blkCol_size_loc; ++c)
            {
                int n = c + vec_cumu_blkCol_size[my_rank];
                h_Y_loc(c) = h_Y_glo(n);
                h_X_loc(c) = h_X_glo(n);
            }
#endif

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n = global_contact_index[c];
                h_Y_contact(c) = h_Y_glo(n);
                h_X_contact(c) = h_X_glo(n);
            }
#ifdef AMREX_USE_GPU
            d_X_contact_data.copy(h_X_contact_data);
            d_Y_contact_data.copy(h_Y_contact_data);
#endif

            for (int t = 0; t < num_traces; ++t)
            {
                h_Trace_r[t] = 0.;
                h_Trace_i[t] = 0.;
            }
#ifdef AMREX_USE_GPU
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Trace_r.begin(),
                                  h_Trace_r.end(), d_Trace_r.begin());
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Trace_i.begin(),
                                  h_Trace_i.end(), d_Trace_i.begin());
            amrex::Gpu::streamSynchronize();
#endif

            /*following is for lambda capture*/
            int cumulative_columns = vec_cumu_blkCol_size[my_rank];
            int Hsize = Hsize_glo;
            auto &GC_ID = global_contact_index;
            auto &CT_ID = contact_transmission_index;
            auto *degen_vec_ptr = degen_vec.dataPtr();
            bool gpu_flag_write_LDOS = flag_write_LDOS;

            amrex::ParallelFor(
                blkCol_size_loc,
                [=] AMREX_GPU_DEVICE(int n) noexcept
                {
                    int n_glo = n + cumulative_columns; /*global column number*/
                    ComplexType one(1., 0.);
                    ComplexType minus_one(-1., 0.);
                    ComplexType imag(0., 1.);

#ifdef COMPUTE_GREENS_FUNCTION_OFFDIAG_ELEMS
                    GR_loc(n_glo, n) = one / (Alpha(n) - X(n) - Y(n));

                    for (int m = n_glo; m > 0; m--)
                    {
                        GR_loc(m - 1, n) = -1 * Ytil_glo(m) * GR_loc(m, n);
                    }
                    for (int m = n_glo; m < Hsize - 1; ++m)
                    {
                        GR_loc(m + 1, n) = -1 * Xtil_glo(m) * GR_loc(m, n);
                    }
#else
                    GR_loc(n) = one / (Alpha(n) - X(n) - Y(n));
#endif

                    MatrixBlock<T> A_tk[NUM_CONTACTS];
                    MatrixBlock<T> Gamma[NUM_CONTACTS];
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                    for (int m = 0; m < Hsize; ++m)
                    {
                        A_loc(m, n) = 0.;
                    }
#else
                    A_loc(n) = 0.;
#endif
                    for (int k = 0; k < NUM_CONTACTS; ++k)
                    {
                        int k_glo = GC_ID[k];
                        MatrixBlock<T> G_contact_kk =
                            one /
                            (Alpha_contact(k) - X_contact(k) - Y_contact(k));

                        MatrixBlock<T> temp = G_contact_kk;
                        for (int m = k_glo; m < n_glo; ++m)
                        {
                            temp = -1 * Xtil_glo(m) * temp;
                        }
                        for (int m = k_glo; m > n_glo; m--)
                        {
                            temp = -1 * Ytil_glo(m) * temp;
                        }
                        MatrixBlock<T> G_contact_nk = temp;

                        Gamma[k] = imag * (Sigma_contact(k) -
                                           Sigma_contact(k).Dagger());

                        MatrixBlock<T> A_nn =
                            G_contact_nk * Gamma[k] * G_contact_nk.Dagger();
                        MatrixBlock<T> A_kn =
                            G_contact_kk * Gamma[k] * G_contact_nk.Dagger();
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                        A_loc(k_glo, n) = A_loc(k_glo, n) + A_kn;
                        for (int m = k_glo + 1; m < Hsize; ++m)
                        {
                            A_kn = -1 * Xtil_glo(m - 1) * A_kn;
                            A_loc(m, n) = A_loc(m, n) + A_kn;
                        }
                        for (int m = k_glo - 1; m >= 0; m--)
                        {
                            A_kn = -1 * Ytil_glo(m + 1) * A_kn;
                            A_loc(m, n) = A_loc(m, n) + A_kn;
                        }
                        A_tk[k] = 0.;
                        if (n_glo == CT_ID[k])
                        {
                            A_tk[k] = A_kn;
                        }
#else
                        for (int m = k_glo + 1; m < Hsize; ++m)
                        {
                            A_kn = -1 * Xtil_glo(m - 1) * A_kn;
                        }
                        for (int m = k_glo - 1; m >= 0; m--)
                        {
                            A_kn = -1 * Ytil_glo(m + 1) * A_kn;
                        }
                        A_tk[k] = 0.;
                        if (n_glo == CT_ID[k])
                        {
                            A_tk[k] = A_kn;
                        }
                        A_loc(n) = A_loc(n) + A_nn;
#endif
                    }

                /*LDOS*/

#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                    ComplexType val =
                        A_loc(n_glo, n).DiagDotSum(degen_vec_ptr) /
                        (2. * MathConst::pi);
#else
                    ComplexType val = A_loc(n).DiagDotSum(degen_vec_ptr) /
                                      (2. * MathConst::pi);
#endif

                    if (gpu_flag_write_LDOS) LDOS_loc(n) = val.real();

                    amrex::HostDevice::Atomic::Add(&(trace_r[0]), val.real());
                    amrex::HostDevice::Atomic::Add(&(trace_i[0]), val.imag());

                    /*Transmission*/
                    auto T12 = Gamma[0] * A_tk[1];

                    ComplexType T12_blksum = T12.DiagDotSum(degen_vec_ptr);

                    amrex::HostDevice::Atomic::Add(&(trace_r[1]),
                                                   T12_blksum.real());
                    amrex::HostDevice::Atomic::Add(&(trace_i[1]),
                                                   T12_blksum.imag());
                });

#ifdef AMREX_USE_GPU
            amrex::Gpu::streamSynchronize();
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_Trace_r.begin(),
                             d_Trace_r.end(), h_Trace_r.begin());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_Trace_i.begin(),
                             d_Trace_i.end(), h_Trace_i.begin());
#endif

            for (int t = 0; t < num_traces; ++t)
            {
                amrex::ParallelDescriptor::ReduceRealSum(h_Trace_r[t]);
                amrex::ParallelDescriptor::ReduceRealSum(h_Trace_i[t]);
            }

            h_DOS_loc(e_glo) =
                spin_degen * h_Trace_r[0] / num_atoms_per_unitcell;
            h_Transmission_loc(e_glo) = h_Trace_r[1];

            if (flag_write_LDOS)
            {
#ifdef AMREX_USE_GPU
                h_LDOS_loc_data.copy(d_LDOS_loc_data);
                amrex::Gpu::streamSynchronize();
#endif

                MPI_Gatherv(&h_LDOS_loc(0), blkCol_size_loc, MPI_DOUBLE,
                            &h_LDOS_glo(0), MPI_recv_count.data(),
                            MPI_recv_disp.data(), MPI_DOUBLE,
                            ParallelDescriptor::IOProcessorNumber(),
                            ParallelDescriptor::Communicator());

                std::string spatialdos_filename =
                    dos_foldername + "/Ept_" + std::to_string(e_glo) + ".dat";

                Write_Table1D(h_PTD_glo_vec, h_LDOS_glo_data,
                              spatialdos_filename,
                              "PTD LDOS_r at E=" + std::to_string(E.real()));
            }
        }
        e_prev_pts += ContourPath_DOS[p].num_pts;
    }

    amrex::Vector<ComplexType> E_total_vec(E_total_pts);
    e_prev_pts = 0;
    for (auto &path : ContourPath_DOS)
    {
        for (int e = 0; e < path.num_pts; ++e)
        {
            int e_glo = e_prev_pts + e;
            E_total_vec[e_glo] = path.E_vec[e].real();
        }
        e_prev_pts += path.num_pts;
    }

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        RealTable1D h_Conductance_loc_data({0}, {E_total_pts},
                                           The_Pinned_Arena());
        total_conductance =
            Compute_Conductance(E_total_vec, h_Transmission_loc_data,
                                h_Conductance_loc_data);

        amrex::Print() << "Total conductance: " << total_conductance << "\n";

        Write_Eql_Characteristics(E_total_vec, h_DOS_loc_data,
                                  h_Transmission_loc_data,
                                  h_Conductance_loc_data,
                                  dos_foldername + "/transport_char.dat");
    }

    Deallocate_TemporaryArraysForGFComputation();

    // if(e==0)
    //{
    //     h_GR_loc_data.copy(d_GR_loc_data); //copy from cpu to gpu
    //     h_A_loc_data.copy(d_A_loc_data); //copy from cpu to gpu

    //    amrex::Gpu::streamSynchronize();

    //    amrex::Print() << "Printing GR_loc: \n";
    //    Print_Table2D_loc(h_GR_loc_data);

    //    amrex::Print() << "Printing A_loc: \n";
    //    Print_Table2D_loc(h_A_loc_data);

    //}
}

template <typename T>
void c_NEGF_Common<T>::Compute_InducedCharge(RealTable1D &n_curr_out_data)
{
    // amrex::Real proc_times[2] = {0.,0.};
    // amrex::Real max_times[2] = {0.,0.};

    // amrex::Real eq_time_begin = amrex::second();
    Compute_RhoEq();
    // proc_times[0] = amrex::second() - eq_time_begin;

    // amrex::Real noneq_time_begin = amrex::second();
    if (flag_noneq_exists)
    {
        Compute_RhoNonEq();
    }
    // proc_times[1] = amrex::second() - noneq_time_begin;

    // MPI_Reduce(proc_times,
    //            max_times,
    //            2,
    //            MPI_DOUBLE,
    //            MPI_MAX,
    //            ParallelDescriptor::IOProcessorNumber(),
    //            ParallelDescriptor::Communicator());

    // if(ParallelDescriptor::IOProcessor())
    //{
    //     amrex::Print() << "Max times for Compute_RhoEq and RhoNonEq: "
    //                    << std::setw(20) << proc_times[0]
    //                    << std::setw(20) << proc_times[1] << "\n";
    // }

#ifdef AMREX_USE_GPU
    auto const &Rho0_loc = d_Rho0_loc_data.const_table();
    auto const &RhoEq_loc = d_RhoEq_loc_data.const_table();
    auto const &RhoNonEq_loc = d_RhoNonEq_loc_data.const_table();
#else
    auto const &Rho0_loc = h_Rho0_loc_data.const_table();
    auto const &RhoEq_loc = h_RhoEq_loc_data.const_table();
    auto const &RhoNonEq_loc = h_RhoNonEq_loc_data.const_table();
#endif
    auto const &RhoInduced_loc = n_curr_out_data.table();

    int field_sites_NS_offset = num_field_sites_loc_NS_offset;
    amrex::ParallelFor(blkCol_size_loc,
                       [=] AMREX_GPU_DEVICE(int n) noexcept
                       {
                           RhoInduced_loc(n + field_sites_NS_offset) =
                               (RhoEq_loc(n).DiagSum().imag() +
                                RhoNonEq_loc(n).DiagSum().real() -
                                Rho0_loc(n).DiagSum().imag());
                       });

#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif

    if (flag_write_charge_components)
    {
/*Printing individual components for debugging*/
#ifdef AMREX_USE_GPU
        RealTable1D d_Rho0_Imag_loc_data({0}, {blkCol_size_loc}, The_Arena());
        RealTable1D d_RhoEq_Imag_loc_data({0}, {blkCol_size_loc}, The_Arena());
        RealTable1D d_RhoNonEq_Real_loc_data({0}, {blkCol_size_loc},
                                             The_Arena());

        auto const &d_Rho0_Imag_loc = d_Rho0_Imag_loc_data.table();
        auto const &d_RhoEq_Imag_loc = d_RhoEq_Imag_loc_data.table();
        auto const &d_RhoNonEq_Real_loc = d_RhoNonEq_Real_loc_data.table();

        amrex::ParallelFor(blkCol_size_loc,
                           [=] AMREX_GPU_DEVICE(int n) noexcept
                           {
                               d_Rho0_Imag_loc(n) =
                                   Rho0_loc(n).DiagSum().imag();
                               d_RhoEq_Imag_loc(n) =
                                   RhoEq_loc(n).DiagSum().imag();
                               d_RhoNonEq_Real_loc(n) =
                                   RhoNonEq_loc(n).DiagSum().real();
                           });
        amrex::Gpu::streamSynchronize();
#endif

#ifdef AMREX_USE_GPU
        RealTable1D h_Rho0_loc_data({0}, {blkCol_size_loc}, The_Pinned_Arena());
        RealTable1D h_RhoEq_loc_data({0}, {blkCol_size_loc},
                                     The_Pinned_Arena());
        RealTable1D h_RhoNonEq_loc_data({0}, {blkCol_size_loc},
                                        The_Pinned_Arena());
        RealTable1D h_RhoInduced_loc_data({0}, {blkCol_size_loc},
                                          The_Pinned_Arena());
        auto const &h_RhoInduced_loc = h_RhoInduced_loc_data.const_table();
#else
        auto const &h_RhoInduced_loc = n_curr_out_data.const_table();
#endif

        auto const &h_Rho0_loc = h_Rho0_loc_data.const_table();
        auto const &h_RhoEq_loc = h_RhoEq_loc_data.const_table();
        auto const &h_RhoNonEq_loc = h_RhoNonEq_loc_data.const_table();

#ifdef AMREX_USE_GPU
        h_Rho0_loc_data.copy(d_Rho0_Imag_loc_data);
        h_RhoEq_loc_data.copy(d_RhoEq_Imag_loc_data);
        h_RhoNonEq_loc_data.copy(d_RhoNonEq_Real_loc_data);
        h_RhoInduced_loc_data.copy(n_curr_out_data);
#endif

        MPI_Barrier(ParallelDescriptor::Communicator());

        RealTable1D h_Rho0_data({0}, {Hsize_glo}, The_Pinned_Arena());
        RealTable1D h_RhoEq_data({0}, {Hsize_glo}, The_Pinned_Arena());
        RealTable1D h_RhoNonEq_data({0}, {Hsize_glo}, The_Pinned_Arena());
        RealTable1D h_RhoInduced_data({0}, {Hsize_glo}, The_Pinned_Arena());

        auto const &h_Rho0 = h_Rho0_data.table();
        auto const &h_RhoEq = h_RhoEq_data.table();
        auto const &h_RhoNonEq = h_RhoNonEq_data.table();
        auto const &h_RhoInduced = h_RhoInduced_data.table();

        MPI_Gatherv(&h_Rho0_loc(0), blkCol_size_loc, MPI_DOUBLE, &h_Rho0(0),
                    MPI_recv_count.data(), MPI_recv_disp.data(), MPI_DOUBLE,
                    ParallelDescriptor::IOProcessorNumber(),
                    ParallelDescriptor::Communicator());

        MPI_Gatherv(&h_RhoEq_loc(0), blkCol_size_loc, MPI_DOUBLE, &h_RhoEq(0),
                    MPI_recv_count.data(), MPI_recv_disp.data(), MPI_DOUBLE,
                    ParallelDescriptor::IOProcessorNumber(),
                    ParallelDescriptor::Communicator());

        MPI_Gatherv(&h_RhoNonEq_loc(0), blkCol_size_loc, MPI_DOUBLE,
                    &h_RhoNonEq(0), MPI_recv_count.data(), MPI_recv_disp.data(),
                    MPI_DOUBLE, ParallelDescriptor::IOProcessorNumber(),
                    ParallelDescriptor::Communicator());

        MPI_Gatherv(&h_RhoInduced_loc(0), blkCol_size_loc, MPI_DOUBLE,
                    &h_RhoInduced(0), MPI_recv_count.data(),
                    MPI_recv_disp.data(), MPI_DOUBLE,
                    ParallelDescriptor::IOProcessorNumber(),
                    ParallelDescriptor::Communicator());

        if (ParallelDescriptor::IOProcessor())
        {
            Write_ChargeComponents(iter_filename_str + "_chargeComp.dat",
                                   h_RhoEq_data, h_RhoNonEq_data, h_Rho0_data,
                                   h_RhoInduced_data);
        }
    }
}

template <typename T>
template <typename TableType>
void c_NEGF_Common<T>::Write_ChargeComponents(
    std::string filename, const TableType &h_RhoEq_data,
    const TableType &h_RhoNonEq_data, const TableType &h_Rho0_data,
    const TableType &h_RhoInduced_data)
{
    auto const &h_RhoEq = h_RhoEq_data.const_table();
    auto const &h_RhoNonEq = h_RhoNonEq_data.const_table();
    auto const &h_Rho0 = h_Rho0_data.const_table();
    auto const &h_RhoInduced = h_RhoInduced_data.const_table();

    amrex::Print() << "Writing charge components in file: " << filename << "\n";
    std::ofstream outfile;
    outfile.open(filename.c_str());

    outfile << "axis, RhoEq, RhoNonEq, Rho0, Rho_Induced"
            << "\n";
    for (int n = 0; n < Hsize_glo; ++n)
    {
        outfile << h_PTD_glo_vec[n] << std::setw(20) << h_RhoEq(n)
                << std::setw(20) << h_RhoNonEq(n) << std::setw(20) << h_Rho0(n)
                << std::setw(20) << h_RhoInduced(n) << "\n";
    }
    outfile.close();
}

template <typename T>
void c_NEGF_Common<T>::Copy_LocalChargeFromNanostructure(
    RealTable1D &container_data, const int NS_offset)
{
    auto const &h_n_curr_in_glo = h_n_curr_in_glo_data.table();
    auto const &container = container_data.table();

    int disp = MPI_recv_disp[my_rank];
    int data_size = MPI_recv_count[my_rank];

    for (int i = 0; i < data_size; ++i)
    {
        int gid = disp + i;
        container(i + NS_offset) = h_n_curr_in_glo(gid);
    }
    h_n_curr_in_glo_data.clear();
}

template <typename T>
void c_NEGF_Common<T>::Write_PotentialAtSites(const std::string filename_prefix)
{
    /* (?) may need to be changed for multiple nanotubes */
    RealTable1D h_U_glo_data;

    std::ofstream outfile;
    std::string filename = filename_prefix + "_U.dat";

    if (ParallelDescriptor::IOProcessor())
    {
        int size = num_field_sites;
        outfile.open(filename);
        h_U_glo_data.resize({0}, {size}, The_Pinned_Arena());
    }
    auto const &h_U_glo = h_U_glo_data.table();
    auto const &h_U_loc = h_U_loc_data.table();

    MPI_Gatherv(&h_U_loc(0), blkCol_size_loc, MPI_DOUBLE, &h_U_glo(0),
                MPI_recv_count.data(), MPI_recv_disp.data(), MPI_DOUBLE,
                ParallelDescriptor::IOProcessorNumber(),
                ParallelDescriptor::Communicator());

    if (ParallelDescriptor::IOProcessor())
    {
        amrex::Print() << " Root Writing " << filename << "\n";
        for (int l = 0; l < num_field_sites; ++l)
        {
            outfile << l << std::setw(35) << h_PTD_glo_vec[l] << std::setw(35)
                    << h_U_glo(l) << "\n";
        }

        h_U_glo_data.clear();
        outfile.close();
    }
}

template <typename T>
void c_NEGF_Common<T>::Write_InputInducedCharge(
    const std::string filename_prefix, const RealTable1D &n_curr_in_data)
{
    if (ParallelDescriptor::IOProcessor())
    {
        std::string filename = filename_prefix + "_Qin.dat";

        Write_Table1D(
            h_PTD_glo_vec, n_curr_in_data, filename.c_str(),
            "'axial location / (nm)', 'Induced charge per site / (e)'");
    }
}

template <typename T>
void c_NEGF_Common<T>::Write_InducedCharge(const std::string filename_prefix,
                                           const RealTable1D &n_curr_out_data)
{
    std::string filename = filename_prefix + "_Qout.dat";

    Write_Table1D(h_PTD_glo_vec, n_curr_out_data, filename.c_str(),
                  "'axial location / (nm)', 'Induced charge per site / (e)'");
}

template <typename T>
void c_NEGF_Common<T>::Write_ChargeNorm(const std::string filename_prefix,
                                        const RealTable1D &Norm_data)
{
    std::string filename = filename_prefix + "_norm.dat";

    Write_Table1D(h_PTD_glo_vec, Norm_data, filename.c_str(),
                  "'axial location / (nm)', 'norm");
}

template <typename T>
void c_NEGF_Common<T>::Compute_RhoNonEq()
{
    auto const &h_minusHa_loc = h_minusHa_loc_data.table();
    auto const &h_Hb_loc = h_Hb_loc_data.table();
    auto const &h_Hc_loc = h_Hc_loc_data.table();
    auto const &h_tau = h_tau_glo_data.table();

    Allocate_TemporaryArraysForGFComputation();

    auto const &h_Alpha_loc = h_Alpha_loc_data.table();
    auto const &h_Alpha_glo = h_Alpha_glo_data.table();
    auto const &h_Xtil_glo = h_Xtil_glo_data.table();
    auto const &h_Ytil_glo = h_Ytil_glo_data.table();
    auto const &h_X_glo = h_X_glo_data.table();
    auto const &h_Y_glo = h_Y_glo_data.table();
    auto const &h_X_loc = h_X_loc_data.table();
    auto const &h_Y_loc = h_Y_loc_data.table();
    auto const &h_Alpha_contact = h_Alpha_contact_data.table();
    auto const &h_Y_contact = h_Y_contact_data.table();
    auto const &h_X_contact = h_X_contact_data.table();
    auto const &h_Sigma_contact = h_Sigma_contact_data.table();
    auto const &h_Fermi_contact = h_Fermi_contact_data.table();

#ifdef AMREX_USE_GPU
    auto const &RhoNonEq_loc = d_RhoNonEq_loc_data.table();
    amrex::ParallelFor(blkCol_size_loc, [=] AMREX_GPU_DEVICE(int n) noexcept
                       { RhoNonEq_loc(n) = 0.; });
    auto const &GR_loc = d_GR_loc_data.table();
    auto const &A_loc = d_A_loc_data.table();
    /*constant references*/
    auto const &Alpha = d_Alpha_loc_data.const_table();
    auto const &Xtil_glo = d_Xtil_glo_data.const_table();
    auto const &d_Xtil_glo = d_Xtil_glo_data.table();
    auto const &Ytil_glo = d_Ytil_glo_data.const_table();
    auto const &d_Ytil_glo = d_Ytil_glo_data.table();
    auto const &X = d_X_loc_data.const_table();
    auto const &d_X_loc = d_X_loc_data.table();
    auto const &Y = d_Y_loc_data.const_table();
    auto const &d_Y_loc = d_Y_loc_data.table();

    auto const &Alpha_contact = d_Alpha_contact_data.const_table();
    auto const &X_contact = d_X_contact_data.const_table();
    auto const &Y_contact = d_Y_contact_data.const_table();
    auto const &Sigma_contact = d_Sigma_contact_data.const_table();
    auto const &Fermi_contact = d_Fermi_contact_data.const_table();

    auto &degen_vec = block_degen_gpuvec;

#else
    ComplexType zero(0., 0.);
    SetVal_Table1D(h_RhoNonEq_loc_data, zero);
    auto const &RhoNonEq_loc = h_RhoNonEq_loc_data.table();
    auto const &GR_loc = h_GR_loc_data.table();
    auto const &A_loc = h_A_loc_data.table();
    /*constant references*/
    auto const &Alpha = h_Alpha_loc_data.const_table();
    auto const &Xtil_glo = h_Xtil_glo_data.const_table();
    auto const &Ytil_glo = h_Ytil_glo_data.const_table();
    auto const &X = h_X_loc_data.const_table();
    auto const &Y = h_Y_loc_data.const_table();

    auto const &Alpha_contact = h_Alpha_contact_data.const_table();
    auto const &X_contact = h_X_contact_data.const_table();
    auto const &Y_contact = h_Y_contact_data.const_table();
    auto const &Sigma_contact = h_Sigma_contact_data.const_table();
    auto const &Fermi_contact = h_Fermi_contact_data.const_table();

    auto &degen_vec = block_degen_vec;

#endif
    bool flag_compute_integrand = false;
    if (flag_write_integrand_iter or flag_correct_integration_limits)
    {
        flag_compute_integrand = true;
        amrex::Print() << "\n setting flag_compute_integrand: "
                       << flag_compute_integrand << "\n";
    }

    if (flag_compute_integrand)
    {
        h_NonEq_Integrand_data.resize({0}, {total_noneq_integration_pts},
                                      The_Pinned_Arena());
        h_NonEq_Integrand_Source_data.resize({0}, {total_noneq_integration_pts},
                                             The_Pinned_Arena());

        h_NonEq_Integrand_Drain_data.resize({0}, {total_noneq_integration_pts},
                                            The_Pinned_Arena());
#ifdef AMREX_USE_GPU
        d_NonEq_Integrand_data.resize({0}, {total_noneq_integration_pts},
                                      The_Arena());
        d_NonEq_Integrand_Source_data.resize({0}, {total_noneq_integration_pts},
                                             The_Arena());
        d_NonEq_Integrand_Drain_data.resize({0}, {total_noneq_integration_pts},
                                            The_Arena());
#endif
    }
    auto const &h_NonEq_Integrand = h_NonEq_Integrand_data.table();
    auto const &h_NonEq_Integrand_Source =
        h_NonEq_Integrand_Source_data.table();
    auto const &h_NonEq_Integrand_Drain = h_NonEq_Integrand_Drain_data.table();
#ifdef AMREX_USE_GPU
    auto const &NonEq_Integrand = d_NonEq_Integrand_data.table();
    auto const &NonEq_Integrand_Source = d_NonEq_Integrand_Source_data.table();
    auto const &NonEq_Integrand_Drain = d_NonEq_Integrand_Drain_data.table();
#else
    auto const &NonEq_Integrand = h_NonEq_Integrand_data.table();
    auto const &NonEq_Integrand_Source = h_NonEq_Integrand_Source_data.table();
    auto const &NonEq_Integrand_Drain = h_NonEq_Integrand_Drain_data.table();
#endif
    if (flag_compute_integrand)
    {
        amrex::ParallelFor(total_noneq_integration_pts,
                           [=] AMREX_GPU_DEVICE(int e) noexcept
                           {
                               NonEq_Integrand(e) = 0.;
                               NonEq_Integrand_Source(e) = 0.;
                               NonEq_Integrand_Drain(e) = 0.;
                           });
    }

    int e_prev = 0;
    for (int p = 0; p < ContourPath_RhoNonEq.size(); ++p)
    {
        for (int e = 0; e < ContourPath_RhoNonEq[p].num_pts; ++e)
        {
            ComplexType E = ContourPath_RhoNonEq[p].E_vec[e];
            ComplexType weight = ContourPath_RhoNonEq[p].weight_vec[e];
            ComplexType mul_factor = ContourPath_RhoNonEq[p].mul_factor_vec[e];

            for (int n = 0; n < blkCol_size_loc; ++n)
            {
                h_Alpha_loc(n) = E + h_minusHa_loc(n);
                /*+ because h_minusHa is defined previously as -(H0+U)*/
            }

            get_Sigma_at_contacts(h_Sigma_contact_data, E);
#ifdef AMREX_USE_GPU
            d_Sigma_contact_data.copy(h_Sigma_contact_data);
#endif

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n_glo = global_contact_index[c];
                int n = n_glo - vec_cumu_blkCol_size[my_rank];

                if (n_glo >= vec_cumu_blkCol_size[my_rank] &&
                    n_glo < vec_cumu_blkCol_size[my_rank + 1])
                {
                    h_Alpha_loc(n) = h_Alpha_loc(n) - h_Sigma_contact(c);
                }
                h_Fermi_contact(c) =
                    FermiFunction(E - mu_contact[c], kT_contact[c]);
            }
#ifdef AMREX_USE_GPU
            d_Fermi_contact_data.copy(h_Fermi_contact_data);
            d_Alpha_loc_data.copy(h_Alpha_loc_data);
#endif

            /*MPI_Allgather*/
            MPI_Allgatherv(&h_Alpha_loc(0), blkCol_size_loc, MPI_BlkType,
                           &h_Alpha_glo(0), MPI_recv_count.data(),
                           MPI_recv_disp.data(), MPI_BlkType,
                           ParallelDescriptor::Communicator());

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n_glo = global_contact_index[c];
                h_Alpha_contact(c) = h_Alpha_glo(n_glo);
            }
#ifdef AMREX_USE_GPU
            d_Alpha_contact_data.copy(h_Alpha_contact_data);
#endif

            h_Y_glo(0) = 0;
            h_X_glo(Hsize_glo - 1) = 0;
            for (int section = 0; section < num_recursive_parts; ++section)
            {
                for (int n = std::max(1, Hsize_recur_part * section);
                     n < std::min(Hsize_recur_part * (section + 1), Hsize_glo);
                     ++n)
                {
                    int p = (n - 1) % offDiag_repeatBlkSize;
                    h_Ytil_glo(n) =
                        h_Hc_loc(p) / (h_Alpha_glo(n - 1) - h_Y_glo(n - 1));
                    h_Y_glo(n) = h_Hb_loc(p) * h_Ytil_glo(n);
                }
                for (int n = std::min(Hsize_glo - 2,
                                      Hsize_recur_part *
                                          (num_recursive_parts - section));
                     n >=
                     Hsize_recur_part * (num_recursive_parts - section - 1);
                     n--)
                {
                    int p = n % offDiag_repeatBlkSize;
                    h_Xtil_glo(n) =
                        h_Hb_loc(p) / (h_Alpha_glo(n + 1) - h_X_glo(n + 1));
                    h_X_glo(n) = h_Hc_loc(p) * h_Xtil_glo(n);
                }

#ifdef AMREX_USE_GPU
                int Ytil_begin = section * Hsize_recur_part;
                int Ytil_end =
                    std::min(Hsize_recur_part * (section + 1), Hsize_glo);

                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_Ytil_glo.p + Ytil_begin,
                                      h_Ytil_glo.p + Ytil_end,
                                      d_Ytil_glo.p + Ytil_begin);

                int Xtil_begin =
                    Hsize_recur_part * (num_recursive_parts - section - 1);
                int Xtil_end =
                    std::min(Hsize_glo, Hsize_recur_part *
                                            (num_recursive_parts - section));

                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_Xtil_glo.p + Xtil_begin,
                                      h_Xtil_glo.p + Xtil_end,
                                      d_Xtil_glo.p + Xtil_begin);
#endif
            }

#ifdef AMREX_USE_GPU
            int X_begin = vec_cumu_blkCol_size[my_rank];
            int X_end = X_begin + blkCol_size_loc;
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_X_glo.p + X_begin,
                                  h_X_glo.p + X_end, d_X_loc.p);

            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Y_glo.p + X_begin,
                                  h_Y_glo.p + X_end, d_Y_loc.p);
#else
            for (int c = 0; c < blkCol_size_loc; ++c)
            {
                int n = c + vec_cumu_blkCol_size[my_rank];
                h_Y_loc(c) = h_Y_glo(n);
                h_X_loc(c) = h_X_glo(n);
            }
#endif

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n = global_contact_index[c];
                h_Y_contact(c) = h_Y_glo(n);
                h_X_contact(c) = h_X_glo(n);
            }
#ifdef AMREX_USE_GPU
            d_X_contact_data.copy(h_X_contact_data);
            d_Y_contact_data.copy(h_Y_contact_data);
            amrex::Gpu::streamSynchronize();
#endif

            /*following is for lambda capture*/
            int cumulative_columns = vec_cumu_blkCol_size[my_rank];
            int Hsize = Hsize_glo;
            auto &GC_ID = global_contact_index;
            auto *degen_vec_ptr = degen_vec.dataPtr();

            amrex::Real const_multiplier =
                -1 * spin_degen / (2 * MathConst::pi);
            bool compute_integrand = flag_compute_integrand;
            int e_glo = e + e_prev;

            amrex::ParallelFor(
                blkCol_size_loc,
                [=] AMREX_GPU_DEVICE(int n) noexcept
                {
                    int n_glo = n + cumulative_columns; /*global column number*/
                    ComplexType one(1., 0.);

#ifdef COMPUTE_GREENS_FUNCTION_OFFDIAG_ELEMS
                    GR_loc(n_glo, n) = one / (Alpha(n) - X(n) - Y(n));
                    for (int m = n_glo; m > 0; m--)
                    {
                        GR_loc(m - 1, n) = -1 * Ytil_glo(m) * GR_loc(m, n);
                    }
                    for (int m = n_glo; m < Hsize - 1; ++m)
                    {
                        GR_loc(m + 1, n) = -1 * Xtil_glo(m) * GR_loc(m, n);
                    }
#else
                    GR_loc(n) = one / (Alpha(n) - X(n) - Y(n));
#endif

                    MatrixBlock<T> A_tk[NUM_CONTACTS];
                    MatrixBlock<T> Gamma[NUM_CONTACTS];
                    MatrixBlock<T> AnF_sum;
                    AnF_sum = 0.;
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                    for (int m = 0; m < Hsize; ++m)
                    {
                        A_loc(m, n) = 0.;
                    }
#else
                    A_loc(n) = 0.;
#endif
                    ComplexType imag(0., 1.);
                    for (int k = 0; k < NUM_CONTACTS; ++k)
                    {
                        int k_glo = GC_ID[k];
                        MatrixBlock<T> G_contact_kk =
                            one /
                            (Alpha_contact(k) - X_contact(k) - Y_contact(k));

                        MatrixBlock<T> temp = G_contact_kk;
                        for (int m = k_glo; m < n_glo; ++m)
                        {
                            temp = -1 * Xtil_glo(m) * temp;
                        }
                        for (int m = k_glo; m > n_glo; m--)
                        {
                            temp = -1 * Ytil_glo(m) * temp;
                        }
                        MatrixBlock<T> G_contact_nk = temp;

                        Gamma[k] = imag * (Sigma_contact(k) -
                                           Sigma_contact(k).Dagger());

                        MatrixBlock<T> A_nn =
                            G_contact_nk * Gamma[k] * G_contact_nk.Dagger();

#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                        MatrixBlock<T> A_kn =
                            G_contact_kk * Gamma[k] * G_contact_nk.Dagger();

                        A_loc(k_glo, n) = A_loc(k_glo, n) + A_kn;
                        for (int m = k_glo + 1; m < Hsize; ++m)
                        {
                            A_kn = -1 * Xtil_glo(m - 1) * A_kn;
                            A_loc(m, n) = A_loc(m, n) + A_kn;
                        }
                        for (int m = k_glo - 1; m >= 0; m--)
                        {
                            A_kn = -1 * Ytil_glo(m + 1) * A_kn;
                            A_loc(m, n) = A_loc(m, n) + A_kn;
                        }
                        A_tk[k] = 0.;
                        if (n_glo == CT_ID[k])
                        {
                            A_tk[k] = A_kn;
                        }
#else
                        A_loc(n) = A_loc(n) + A_nn;
#endif
                        AnF_sum = AnF_sum + A_nn * Fermi_contact(k);
                    }

                    /*RhoNonEq*/
                    MatrixBlock<T> RhoNonEq_n =
                        const_multiplier * AnF_sum * weight * mul_factor;
                    RhoNonEq_loc(n) =
                        RhoNonEq_loc(n) + RhoNonEq_n.DiagMult(degen_vec_ptr);

                    if (compute_integrand)
                    {
                        if (n_glo == int(Hsize / 2))
                        {
                            MatrixBlock<T> Intermed =
                                const_multiplier * AnF_sum;
                            NonEq_Integrand(e_glo) =
                                Intermed.DiagMult(degen_vec_ptr)
                                    .DiagSum()
                                    .real();
                        }
                        else if (n_glo == int(Hsize / 4))
                        {
                            MatrixBlock<T> Intermed =
                                const_multiplier * AnF_sum;
                            NonEq_Integrand_Source(e_glo) =
                                Intermed.DiagMult(degen_vec_ptr)
                                    .DiagSum()
                                    .real();
                        }
                        else if (n_glo == int(Hsize * 3 / 4))
                        {
                            MatrixBlock<T> Intermed =
                                const_multiplier * AnF_sum;
                            NonEq_Integrand_Drain(e_glo) =
                                Intermed.DiagMult(degen_vec_ptr)
                                    .DiagSum()
                                    .real();
                        }
                    }
                });
#ifdef AMREX_USE_GPU
            amrex::Gpu::streamSynchronize();
#endif
        }
        e_prev += ContourPath_RhoNonEq[p].num_pts;
    }

    if (flag_compute_integrand)
    {
#ifdef AMREX_USE_GPU
        h_NonEq_Integrand_data.copy(d_NonEq_Integrand_data);
        h_NonEq_Integrand_Source_data.copy(d_NonEq_Integrand_Source_data);
        h_NonEq_Integrand_Drain_data.copy(d_NonEq_Integrand_Drain_data);
        amrex::Gpu::streamSynchronize();
#endif

        MPI_Allreduce(MPI_IN_PLACE, &(h_NonEq_Integrand(0)),
                      total_noneq_integration_pts, MPI_DOUBLE, MPI_SUM,
                      ParallelDescriptor::Communicator());

        MPI_Allreduce(MPI_IN_PLACE, &(h_NonEq_Integrand_Source(0)),
                      total_noneq_integration_pts, MPI_DOUBLE, MPI_SUM,
                      ParallelDescriptor::Communicator());

        MPI_Allreduce(MPI_IN_PLACE, &(h_NonEq_Integrand_Drain(0)),
                      total_noneq_integration_pts, MPI_DOUBLE, MPI_SUM,
                      ParallelDescriptor::Communicator());

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            amrex::Vector<ComplexType> E_total_vec(total_noneq_integration_pts);
            int e_prev_pts = 0;
            amrex::Real max_noneq_integrand = 0;
            for (auto &path : ContourPath_RhoNonEq)
            {
                for (int e = 0; e < path.num_pts; ++e)
                {
                    int e_glo = e_prev_pts + e;
                    E_total_vec[e_glo] = path.E_vec[e];

                    if (max_noneq_integrand < fabs(h_NonEq_Integrand(e_glo)))
                    {
                        max_noneq_integrand = fabs(h_NonEq_Integrand(e_glo));
                        E_at_max_noneq_integrand = E_total_vec[e_glo].real();
                    }
                }
                e_prev_pts += path.num_pts;
            }
            amrex::Print() << "\n Abs. value of max integrand: "
                           << max_noneq_integrand
                           << " at E (eV): " << E_at_max_noneq_integrand
                           << " i.e., (E-mu_0)/kT_0: "
                           << (E_at_max_noneq_integrand - mu_contact[0]) /
                                  kT_contact[0]
                           << "\n";

            if (flag_write_integrand_iter)
            {
                Write_Integrand(E_total_vec, h_NonEq_Integrand_data,
                                h_NonEq_Integrand_Source_data,
                                h_NonEq_Integrand_Drain_data,
                                iter_filename_str + "_integrand.dat");
            }
            E_total_vec.clear();
        }

        ParallelDescriptor::Bcast(&E_at_max_noneq_integrand, 1,
                                  ParallelDescriptor::IOProcessorNumber());

        h_NonEq_Integrand_data.clear();
        h_NonEq_Integrand_Source_data.clear();
        h_NonEq_Integrand_Drain_data.clear();
#ifdef AMREX_USE_GPU
        d_NonEq_Integrand_data.clear();
        d_NonEq_Integrand_Source_data.clear();
        d_NonEq_Integrand_Drain_data.clear();
#endif
    }

    Deallocate_TemporaryArraysForGFComputation();
}

template <typename T>
void c_NEGF_Common<T>::Compute_RhoEq()
{
    auto const &h_minusHa_loc = h_minusHa_loc_data.table();
    auto const &h_Hb_loc = h_Hb_loc_data.table();
    auto const &h_Hc_loc = h_Hc_loc_data.table();
    auto const &h_tau = h_tau_glo_data.table();

    Allocate_TemporaryArraysForGFComputation();

    auto const &h_Alpha_loc = h_Alpha_loc_data.table();
    auto const &h_Alpha_glo = h_Alpha_glo_data.table();
    auto const &h_Xtil_glo = h_Xtil_glo_data.table();
    auto const &h_Ytil_glo = h_Ytil_glo_data.table();
    auto const &h_X_glo = h_X_glo_data.table();
    auto const &h_Y_glo = h_Y_glo_data.table();
    auto const &h_X_loc = h_X_loc_data.table();
    auto const &h_Y_loc = h_Y_loc_data.table();
    auto const &h_Sigma_contact = h_Sigma_contact_data.table();

#ifdef AMREX_USE_GPU
    auto const &RhoEq_loc = d_RhoEq_loc_data.table();
    amrex::ParallelFor(blkCol_size_loc, [=] AMREX_GPU_DEVICE(int n) noexcept
                       { RhoEq_loc(n) = 0.; });
    /*constant references*/
    auto const &Alpha = d_Alpha_loc_data.const_table();
    auto const &Xtil_glo = d_Xtil_glo_data.const_table();
    auto const &d_Xtil_glo = d_Xtil_glo_data.table();
    auto const &Ytil_glo = d_Ytil_glo_data.const_table();
    auto const &d_Ytil_glo = d_Ytil_glo_data.table();
    auto const &X = d_X_loc_data.const_table();
    auto const &d_X_loc = d_X_loc_data.table();
    auto const &Y = d_Y_loc_data.const_table();
    auto const &d_Y_loc = d_Y_loc_data.table();
    auto const &Sigma_contact = d_Sigma_contact_data.const_table();
    auto &degen_vec = block_degen_gpuvec;
#else
    ComplexType zero(0., 0.);
    SetVal_Table1D(h_RhoEq_loc_data, zero);
    auto const &RhoEq_loc = h_RhoEq_loc_data.table();
    /*constant references*/
    auto const &Alpha = h_Alpha_loc_data.const_table();
    auto const &Xtil_glo = h_Xtil_glo_data.const_table();
    auto const &Ytil_glo = h_Ytil_glo_data.const_table();
    auto const &X = h_X_loc_data.const_table();
    auto const &Y = h_Y_loc_data.const_table();
    auto const &Sigma_contact = h_Sigma_contact_data.const_table();
    auto &degen_vec = block_degen_vec;
#endif

    for (int p = 0; p < ContourPath_RhoEq.size(); ++p)
    {
        for (int e = 0; e < ContourPath_RhoEq[p].num_pts; ++e)
        {
            ComplexType E = ContourPath_RhoEq[p].E_vec[e];
            ComplexType weight = ContourPath_RhoEq[p].weight_vec[e];
            ComplexType mul_factor = ContourPath_RhoEq[p].mul_factor_vec[e];

            for (int n = 0; n < blkCol_size_loc; ++n)
            {
                h_Alpha_loc(n) = E + h_minusHa_loc(n);
                /*+ because -Ha = -H0a (=0) -U = -U as defined previously*/
            }

            get_Sigma_at_contacts(h_Sigma_contact_data, E);

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n_glo = global_contact_index[c];

                if (n_glo >= vec_cumu_blkCol_size[my_rank] &&
                    n_glo < vec_cumu_blkCol_size[my_rank + 1])
                {
                    int n = n_glo - vec_cumu_blkCol_size[my_rank];
                    h_Alpha_loc(n) = h_Alpha_loc(n) - h_Sigma_contact(c);
                }
            }
#ifdef AMREX_USE_GPU
            d_Alpha_loc_data.copy(h_Alpha_loc_data);
#endif

            /*MPI_Allgather*/
            MPI_Allgatherv(&h_Alpha_loc(0), blkCol_size_loc, MPI_BlkType,
                           &h_Alpha_glo(0), MPI_recv_count.data(),
                           MPI_recv_disp.data(), MPI_BlkType,
                           ParallelDescriptor::Communicator());

            h_Y_glo(0) = 0;
            h_X_glo(Hsize_glo - 1) = 0;
            for (int section = 0; section < num_recursive_parts; ++section)
            {
                for (int n = std::max(1, Hsize_recur_part * section);
                     n < std::min(Hsize_recur_part * (section + 1), Hsize_glo);
                     ++n)
                {
                    int p = (n - 1) % offDiag_repeatBlkSize;
                    h_Ytil_glo(n) =
                        h_Hc_loc(p) / (h_Alpha_glo(n - 1) - h_Y_glo(n - 1));
                    h_Y_glo(n) = h_Hb_loc(p) * h_Ytil_glo(n);
                }
                for (int n = std::min(Hsize_glo - 2,
                                      Hsize_recur_part *
                                          (num_recursive_parts - section));
                     n >=
                     Hsize_recur_part * (num_recursive_parts - section - 1);
                     n--)
                {
                    int p = n % offDiag_repeatBlkSize;
                    h_Xtil_glo(n) =
                        h_Hb_loc(p) / (h_Alpha_glo(n + 1) - h_X_glo(n + 1));
                    h_X_glo(n) = h_Hc_loc(p) * h_Xtil_glo(n);
                }

#ifdef AMREX_USE_GPU
                int Ytil_begin = section * Hsize_recur_part;
                int Ytil_end =
                    std::min(Hsize_recur_part * (section + 1), Hsize_glo);

                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_Ytil_glo.p + Ytil_begin,
                                      h_Ytil_glo.p + Ytil_end,
                                      d_Ytil_glo.p + Ytil_begin);

                int Xtil_begin =
                    Hsize_recur_part * (num_recursive_parts - section - 1);
                int Xtil_end =
                    std::min(Hsize_glo, Hsize_recur_part *
                                            (num_recursive_parts - section));

                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_Xtil_glo.p + Xtil_begin,
                                      h_Xtil_glo.p + Xtil_end,
                                      d_Xtil_glo.p + Xtil_begin);
#endif
            }

#ifdef AMREX_USE_GPU
            int X_begin = vec_cumu_blkCol_size[my_rank];
            int X_end = X_begin + blkCol_size_loc;
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_X_glo.p + X_begin,
                                  h_X_glo.p + X_end, d_X_loc.p);

            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Y_glo.p + X_begin,
                                  h_Y_glo.p + X_end, d_Y_loc.p);
            amrex::Gpu::streamSynchronize();
#else
            for (int c = 0; c < blkCol_size_loc; ++c)
            {
                int n = c + vec_cumu_blkCol_size[my_rank];
                h_Y_loc(c) = h_Y_glo(n);
                h_X_loc(c) = h_X_glo(n);
            }
#endif

            /*following is for lambda capture*/

            int cumulative_columns = vec_cumu_blkCol_size[my_rank];
            auto *degen_vec_ptr = degen_vec.dataPtr();
            ComplexType nF_eq = FermiFunction(E - mu_min, kT_min);

            amrex::Real const_multiplier = -1. * spin_degen / MathConst::pi;

            amrex::ParallelFor(blkCol_size_loc,
                               [=] AMREX_GPU_DEVICE(int n) noexcept
                               {
                                   ComplexType one(1., 0.);
                                   MatrixBlock<T> G_nn =
                                       one / (Alpha(n) - X(n) - Y(n));
                                   MatrixBlock<T> RhoEq_n = const_multiplier *
                                                            G_nn * weight *
                                                            mul_factor * nF_eq;
                                   RhoEq_loc(n) =
                                       RhoEq_loc(n) +
                                       RhoEq_n.DiagMult(degen_vec_ptr);
                               });
#ifdef AMREX_USE_GPU
            amrex::Gpu::streamSynchronize();
#endif
        } /*Energy loop*/
    }     /*Path loop*/

    Deallocate_TemporaryArraysForGFComputation();

    if (!flag_noneq_exists)
    {
#ifdef AMREX_USE_GPU
        auto const &GR_atPoles_loc = d_GR_atPoles_loc_data.const_table();
#else
        auto const &GR_atPoles_loc = h_GR_atPoles_loc_data.const_table();
#endif

        Compute_GR_atPoles();

        amrex::ParallelFor(blkCol_size_loc,
                           [=] AMREX_GPU_DEVICE(int n) noexcept {
                               RhoEq_loc(n) = RhoEq_loc(n) + GR_atPoles_loc(n);
                           });
#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
    }
}

template <typename T>
void c_NEGF_Common<T>::Compute_GR_atPoles()
{
    auto const &h_minusHa_loc = h_minusHa_loc_data.table();
    auto const &h_Hb_loc = h_Hb_loc_data.table();
    auto const &h_Hc_loc = h_Hc_loc_data.table();
    auto const &h_tau = h_tau_glo_data.table();

    Allocate_TemporaryArraysForGFComputation();
    auto const &h_Alpha_loc = h_Alpha_loc_data.table();
    auto const &h_Alpha_glo = h_Alpha_glo_data.table();
    auto const &h_Xtil_glo = h_Xtil_glo_data.table();
    auto const &h_Ytil_glo = h_Ytil_glo_data.table();
    auto const &h_X_glo = h_X_glo_data.table();
    auto const &h_Y_glo = h_Y_glo_data.table();
    auto const &h_X_loc = h_X_loc_data.table();
    auto const &h_Y_loc = h_Y_loc_data.table();
    auto const &h_Sigma_contact = h_Sigma_contact_data.table();

#ifdef AMREX_USE_GPU
    auto const &GR_atPoles_loc = d_GR_atPoles_loc_data.table();
    amrex::ParallelFor(blkCol_size_loc, [=] AMREX_GPU_DEVICE(int n) noexcept
                       { GR_atPoles_loc(n) = 0.; });

    /*constant references*/
    auto const &Alpha = d_Alpha_loc_data.const_table();
    auto const &Xtil_glo = d_Xtil_glo_data.const_table();
    auto const &d_Xtil_glo = d_Xtil_glo_data.table();
    auto const &Ytil_glo = d_Ytil_glo_data.const_table();
    auto const &d_Ytil_glo = d_Ytil_glo_data.table();
    auto const &X = d_X_loc_data.const_table();
    auto const &d_X_loc = d_X_loc_data.table();
    auto const &Y = d_Y_loc_data.const_table();
    auto const &d_Y_loc = d_Y_loc_data.table();
    auto const &Sigma_contact = d_Sigma_contact_data.const_table();
    auto &degen_vec = block_degen_gpuvec;
#else
    ComplexType zero(0., 0.);
    SetVal_Table1D(h_GR_atPoles_loc_data, zero);
    auto const &GR_atPoles_loc = h_GR_atPoles_loc_data.table();
    /*constant references*/
    auto const &Alpha = h_Alpha_loc_data.const_table();
    auto const &Xtil_glo = h_Xtil_glo_data.const_table();
    auto const &Ytil_glo = h_Ytil_glo_data.const_table();
    auto const &X = h_X_loc_data.const_table();
    auto const &Y = h_Y_loc_data.const_table();
    auto const &Sigma_contact = h_Sigma_contact_data.const_table();
    auto &degen_vec = block_degen_vec;
#endif

    for (int e = 0; e < E_poles_vec.size(); ++e)
    {
        ComplexType E = E_poles_vec[e];

        for (int n = 0; n < blkCol_size_loc; ++n)
        {
            h_Alpha_loc(n) = E + h_minusHa_loc(n);
            /*+ because h_minusHa is defined previously as -(H0+U)*/
        }
        get_Sigma_at_contacts(h_Sigma_contact_data, E);

        for (int c = 0; c < NUM_CONTACTS; ++c)
        {
            int n_glo = global_contact_index[c];

            if (n_glo >= vec_cumu_blkCol_size[my_rank] &&
                n_glo < vec_cumu_blkCol_size[my_rank + 1])
            {
                int n = n_glo - vec_cumu_blkCol_size[my_rank];
                h_Alpha_loc(n) = h_Alpha_loc(n) - h_Sigma_contact(c);
            }
        }

#ifdef AMREX_USE_GPU
        d_Alpha_loc_data.copy(h_Alpha_loc_data);
#endif

        /*MPI_Allgather*/
        MPI_Allgatherv(&h_Alpha_loc(0), blkCol_size_loc, MPI_BlkType,
                       &h_Alpha_glo(0), MPI_recv_count.data(),
                       MPI_recv_disp.data(), MPI_BlkType,
                       ParallelDescriptor::Communicator());

        h_Y_glo(0) = 0;
        h_X_glo(Hsize_glo - 1) = 0;
        for (int section = 0; section < num_recursive_parts; ++section)
        {
            for (int n = std::max(1, Hsize_recur_part * section);
                 n < std::min(Hsize_recur_part * (section + 1), Hsize_glo); ++n)
            {
                int p = (n - 1) % offDiag_repeatBlkSize;
                h_Ytil_glo(n) =
                    h_Hc_loc(p) / (h_Alpha_glo(n - 1) - h_Y_glo(n - 1));
                h_Y_glo(n) = h_Hb_loc(p) * h_Ytil_glo(n);
            }
            for (int n = std::min(Hsize_glo - 2,
                                  Hsize_recur_part *
                                      (num_recursive_parts - section));
                 n >= Hsize_recur_part * (num_recursive_parts - section - 1);
                 n--)
            {
                int p = n % offDiag_repeatBlkSize;
                h_Xtil_glo(n) =
                    h_Hb_loc(p) / (h_Alpha_glo(n + 1) - h_X_glo(n + 1));
                h_X_glo(n) = h_Hc_loc(p) * h_Xtil_glo(n);
            }

#ifdef AMREX_USE_GPU
            int Ytil_begin = section * Hsize_recur_part;
            int Ytil_end =
                std::min(Hsize_recur_part * (section + 1), Hsize_glo);

            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                  h_Ytil_glo.p + Ytil_begin,
                                  h_Ytil_glo.p + Ytil_end,
                                  d_Ytil_glo.p + Ytil_begin);

            int Xtil_begin =
                Hsize_recur_part * (num_recursive_parts - section - 1);
            int Xtil_end =
                std::min(Hsize_glo,
                         Hsize_recur_part * (num_recursive_parts - section));

            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                  h_Xtil_glo.p + Xtil_begin,
                                  h_Xtil_glo.p + Xtil_end,
                                  d_Xtil_glo.p + Xtil_begin);
#endif
        }

#ifdef AMREX_USE_GPU
        int X_begin = vec_cumu_blkCol_size[my_rank];
        int X_end = X_begin + blkCol_size_loc;
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_X_glo.p + X_begin,
                              h_X_glo.p + X_end, d_X_loc.p);

        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Y_glo.p + X_begin,
                              h_Y_glo.p + X_end, d_Y_loc.p);
        amrex::Gpu::streamSynchronize();
#else
        for (int c = 0; c < blkCol_size_loc; ++c)
        {
            int n = c + vec_cumu_blkCol_size[my_rank];
            h_Y_loc(c) = h_Y_glo(n);
            h_X_loc(c) = h_X_glo(n);
        }
#endif

        /*following is for lambda capture*/
        int cumulative_columns = vec_cumu_blkCol_size[my_rank];

        ComplexType pole_const(0., -2 * kT_min * spin_degen);
        auto *degen_vec_ptr = degen_vec.dataPtr();

        amrex::ParallelFor(blkCol_size_loc,
                           [=] AMREX_GPU_DEVICE(int n) noexcept
                           {
                               ComplexType one(1., 0.);
                               MatrixBlock<T> G_nn =
                                   one / (Alpha(n) - X(n) - Y(n));
                               MatrixBlock<T> GR_atPoles_n = pole_const * G_nn;

                               GR_atPoles_loc(n) =
                                   GR_atPoles_loc(n) +
                                   GR_atPoles_n.DiagMult(degen_vec_ptr);
                           });
#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
    }
    Deallocate_TemporaryArraysForGFComputation();
}

template <typename T>
void c_NEGF_Common<T>::Compute_Rho0()
{
    auto const &h_minusHa_loc = h_minusHa_loc_data.table();
    auto const &h_Hb_loc = h_Hb_loc_data.table();
    auto const &h_Hc_loc = h_Hc_loc_data.table();
    auto const &h_tau = h_tau_glo_data.table();

    Allocate_TemporaryArraysForGFComputation();

    auto const &h_Alpha_loc = h_Alpha_loc_data.table();
    auto const &h_Alpha_glo = h_Alpha_glo_data.table();
    auto const &h_Xtil_glo = h_Xtil_glo_data.table();
    auto const &h_Ytil_glo = h_Ytil_glo_data.table();
    auto const &h_X_glo = h_X_glo_data.table();
    auto const &h_Y_glo = h_Y_glo_data.table();
    auto const &h_X_loc = h_X_loc_data.table();
    auto const &h_Y_loc = h_Y_loc_data.table();
    auto const &h_Sigma_contact = h_Sigma_contact_data.table();

#ifdef AMREX_USE_GPU
    auto const &Rho0_loc = d_Rho0_loc_data.table();
    amrex::ParallelFor(blkCol_size_loc, [=] AMREX_GPU_DEVICE(int n) noexcept
                       { Rho0_loc(n) = 0.; });
    /*constant references*/
    auto const &Alpha = d_Alpha_loc_data.const_table();
    auto const &Xtil_glo = d_Xtil_glo_data.const_table();
    auto const &Ytil_glo = d_Ytil_glo_data.const_table();
    auto const &X = d_X_loc_data.const_table();
    auto const &Y = d_Y_loc_data.const_table();
    auto const &Sigma_contact = d_Sigma_contact_data.const_table();
    auto &degen_vec = block_degen_gpuvec;
#else
    ComplexType zero(0., 0.);
    SetVal_Table1D(h_Rho0_loc_data, zero);
    auto const &Rho0_loc = h_Rho0_loc_data.table();
    /*constant references*/
    auto const &Alpha = h_Alpha_loc_data.const_table();
    auto const &Xtil_glo = h_Xtil_glo_data.const_table();
    auto const &Ytil_glo = h_Ytil_glo_data.const_table();
    auto const &X = h_X_loc_data.const_table();
    auto const &Y = h_Y_loc_data.const_table();
    auto const &Sigma_contact = h_Sigma_contact_data.const_table();
    auto &degen_vec = block_degen_vec;
#endif

    for (int p = 0; p < ContourPath_Rho0.size(); ++p)
    {
        for (int e = 0; e < ContourPath_Rho0[p].num_pts; ++e)
        {
            ComplexType E = ContourPath_Rho0[p].E_vec[e];
            ComplexType weight = ContourPath_Rho0[p].weight_vec[e];
            ComplexType mul_factor = ContourPath_Rho0[p].mul_factor_vec[e];

            for (int n = 0; n < blkCol_size_loc; ++n)
            {
                h_Alpha_loc(n) = E + h_minusHa_loc(n);
                /*+ because h_minusHa is defined previously as -(H0+U)*/
            }

            get_Sigma_at_contacts(h_Sigma_contact_data, E);

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n_glo = global_contact_index[c];

                if (n_glo >= vec_cumu_blkCol_size[my_rank] &&
                    n_glo < vec_cumu_blkCol_size[my_rank + 1])
                {
                    int n = n_glo - vec_cumu_blkCol_size[my_rank];
                    h_Alpha_loc(n) = h_Alpha_loc(n) - h_Sigma_contact(c);
                }
            }

            /*MPI_Allgather*/
            MPI_Allgatherv(&h_Alpha_loc(0), blkCol_size_loc, MPI_BlkType,
                           &h_Alpha_glo(0), MPI_recv_count.data(),
                           MPI_recv_disp.data(), MPI_BlkType,
                           ParallelDescriptor::Communicator());

            h_Y_glo(0) = 0;
            for (int n = 1; n < Hsize_glo; ++n)
            {
                int p = (n - 1) % offDiag_repeatBlkSize;
                h_Ytil_glo(n) =
                    h_Hc_loc(p) / (h_Alpha_glo(n - 1) - h_Y_glo(n - 1));
                h_Y_glo(n) = h_Hb_loc(p) * h_Ytil_glo(n);
            }

            h_X_glo(Hsize_glo - 1) = 0;
            for (int n = Hsize_glo - 2; n > -1; n--)
            {
                int p = n % offDiag_repeatBlkSize;
                h_Xtil_glo(n) =
                    h_Hb_loc(p) / (h_Alpha_glo(n + 1) - h_X_glo(n + 1));
                h_X_glo(n) = h_Hc_loc(p) * h_Xtil_glo(n);
            }

            for (int c = 0; c < blkCol_size_loc; ++c)
            {
                int n = c + vec_cumu_blkCol_size[my_rank];
                h_Y_loc(c) = h_Y_glo(n);
                h_X_loc(c) = h_X_glo(n);
            }

#ifdef AMREX_USE_GPU
            d_Alpha_loc_data.copy(h_Alpha_loc_data);
            d_X_loc_data.copy(h_X_loc_data);
            d_Y_loc_data.copy(h_Y_loc_data);
            amrex::Gpu::streamSynchronize();
#endif

            /*following is for lambda capture*/
            int cumulative_columns = vec_cumu_blkCol_size[my_rank];
            auto *degen_vec_ptr = degen_vec.dataPtr();
            amrex::Real const_multiplier = -1 * spin_degen / (MathConst::pi);

            amrex::ParallelFor(blkCol_size_loc,
                               [=] AMREX_GPU_DEVICE(int n) noexcept
                               {
                                   ComplexType one(1., 0.);
                                   MatrixBlock<T> G_nn =
                                       one / (Alpha(n) - X(n) - Y(n));
                                   MatrixBlock<T> Rho0_n = const_multiplier *
                                                           G_nn * weight *
                                                           mul_factor;
                                   Rho0_loc(n) = Rho0_loc(n) +
                                                 Rho0_n.DiagMult(degen_vec_ptr);
                               });
#ifdef AMREX_USE_GPU
            amrex::Gpu::streamSynchronize();
#endif
        } /*Energy loop*/
    }     /*Path loop*/

    Deallocate_TemporaryArraysForGFComputation();
}

template <typename T>
void c_NEGF_Common<T>::DecimationTechnique(MatrixBlock<T> &gr,
                                           const ComplexType EmU)
{
    CondensedHamiltonian CondH;
    Compute_CondensedHamiltonian(CondH, EmU);

    MatrixBlock<T> mu, nu, gamma, zeta;

    // initialize these blocks
    {
        MatrixBlock<T> EmUI;
        EmUI.SetDiag(EmU);

        auto mu0 = EmUI - CondH.Xi_s;
        auto nu0 = EmUI - CondH.Xi;
        auto gamma0 = CondH.Pi;

        auto nu0_inverse = nu0.Inverse();
        auto gamma0_dagger = gamma0.Dagger();

        MatrixBlock<T> temp1 = gamma0 * nu0_inverse;
        MatrixBlock<T> temp2 = temp1 * gamma0_dagger;
        MatrixBlock<T> temp3 = gamma0_dagger * nu0_inverse;

        mu = mu0 - temp2;
        nu = nu0 - temp2 - temp3 * gamma0;
        gamma = temp1 * gamma0;
        zeta = temp3 * gamma0_dagger;
    }

    amrex::Real error = 1;
    int i = 0;

    while (error > decimation_rel_error)
    {
        auto mu_old = mu;

        auto nu_inverse = nu.Inverse();
        auto temp1 = zeta * nu_inverse * gamma;
        auto temp2 = gamma * nu_inverse * zeta;

        mu = mu - temp1;
        nu = nu - temp1 - temp2;
        gamma = gamma * nu_inverse * gamma;
        zeta = zeta * nu_inverse * zeta;

        auto sum = mu + mu_old;
        auto error_mat = (mu - mu_old) * sum.Inverse();

        error = sqrt(amrex::norm(error_mat.FrobeniusNorm()));
        if (i > decimation_max_iter) break;

        i++;
    }
    // amrex::Print() << "decimation iterations: " << i << " error: " << error
    // << "\n";

    gr = mu.Inverse();
}

template <typename T>
void c_NEGF_Common<T>::Compute_SurfaceGreensFunction(MatrixBlock<T> &gr,
                                                     const ComplexType EmU)
{
    DecimationTechnique(gr, EmU);
    // amrex::Print() << "Using decimation, gr: " << gr << "\n";
}

template <typename T>
void c_NEGF_Common<T>::get_Sigma_at_contacts(BlkTable1D &h_Sigma_contact_data,
                                             ComplexType E)
{
    auto const &h_tau = h_tau_glo_data.table();
    auto const &h_Sigma = h_Sigma_contact_data.table();

    for (std::size_t c = 0; c < NUM_CONTACTS; ++c)
    {
        MatrixBlock<T> gr;
        Compute_SurfaceGreensFunction(gr, E - U_contact[c]);
        h_Sigma(c) = h_tau(c) * gr * h_tau(c).Dagger();
    }
}

// template<typename T>
// AMREX_GPU_HOST_DEVICE
// MatrixBlock<T>
// c_NEGF_Common<T>::get_Gamma(const MatrixBlock<T>& Sigma)
//{
//     //ComplexType imag(0., 1.);
//     //return imag*(Sigma - Sigma.Dagger());
//     return Sigma;
// }

template <typename T>
void c_NEGF_Common<T>::Write_BlkTable1D_asaf_E(
    const amrex::Vector<ComplexType> &E_vec, const BlkTable1D &Arr_data,
    std::string filename, std::string header)
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        // amrex::Print() << "Root Writing " << filename << "\n";
        std::ofstream outfile;
        outfile.open(filename.c_str());

        auto const &Arr = Arr_data.const_table();
        auto thi = Arr_data.hi();

        outfile << header << "\n";
        if (E_vec.size() == thi[0])
        {
            for (int e = 0; e < thi[0]; ++e)
            {
                outfile << std::setw(10) << E_vec[e] << std::setw(15) << Arr(e)
                        << "\n";
            }
        }
        else
        {
            outfile << "Mismatch in the size of E_vec and Arr_data!"
                    << "\n";
        }
        outfile.close();
    }
}

template <typename T>
template <typename VectorType, typename TableType>
void c_NEGF_Common<T>::Write_Table1D(const amrex::Vector<VectorType> &Vec,
                                     const TableType &Arr_data,
                                     std::string filename, std::string header)
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        // amrex::Print() << "\nRoot Writing " << filename << "\n";
        std::ofstream outfile;
        outfile.open(filename.c_str());

        auto const &Arr = Arr_data.const_table();
        auto thi = Arr_data.hi();

        outfile << header << "\n";

        if (Vec.size() == thi[0])
        {
            for (int e = 0; e < Vec.size(); ++e)
            {
                outfile << std::setprecision(15) << std::setw(35) << Vec[e]
                        << std::setw(35) << Arr(e) << "\n";
            }
        }
        else
        {
            outfile << "Mismatch in the size of Vec size: " << Vec.size()
                    << " and Table1D_data: " << thi[0] << "\n";
        }

        outfile.close();
    }
}

template <typename T>
void c_NEGF_Common<T>::Write_BlkTable2D_asaf_E(const BlkTable2D &Arr_data,
                                               std::string filename,
                                               std::string header)
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        // amrex::Print() << "\n Root Writing " << filename << "\n";
        std::ofstream outfile;
        outfile.open(filename.c_str());

        auto const &E = h_E_RealPath_data.const_table();
        auto thi = h_E_RealPath_data.hi();
        auto const &Arr = Arr_data.const_table();

        outfile << header << "\n";
        for (int e = 0; e < thi[0]; ++e)
        {
            outfile << std::setw(10) << E(e) << std::setw(15) << Arr(0, e)
                    << std::setw(15) << Arr(1, e) << "\n";
        }

        outfile.close();
    }
}

template <typename T>
template <typename U>
void c_NEGF_Common<T>::Print_Table1D_loc(const U &Tab1D_data)
{
    auto tlo = Tab1D_data.lo();
    auto thi = Tab1D_data.hi();

    auto const &Tab1D = Tab1D_data.table();

    for (int i = tlo[0]; i < thi[0]; ++i)
    {
        std::cout << std::setw(15) << std::setprecision(2) << Tab1D(i);
        std::cout << "\n";
    }
}

template <typename T>
template <typename U>
void c_NEGF_Common<T>::Print_Table2D_loc(const U &Tab2D_data)
{
    auto tlo = Tab2D_data.lo();
    auto thi = Tab2D_data.hi();

    auto const &Tab2D = Tab2D_data.table();

    std::cout << "[\n";
    for (int i = tlo[0]; i < thi[0]; ++i)
    {
        for (int j = tlo[1]; j < thi[1];
             ++j)  // slow moving index. printing slow
        {
            std::cout << std::setw(5) << std::setprecision(2) << Tab2D(i, j);
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

template <typename T>
void c_NEGF_Common<T>::Compute_Current()
{
    SetVal_Table1D(h_Current_loc_data, 0.);

    auto const &h_minusHa_loc = h_minusHa_loc_data.table();
    auto const &h_Hb_loc = h_Hb_loc_data.table();
    auto const &h_Hc_loc = h_Hc_loc_data.table();
    auto const &h_tau = h_tau_glo_data.table();
    auto const &h_Current_loc = h_Current_loc_data.table();

    Allocate_TemporaryArraysForGFComputation();

    auto const &h_Alpha_loc = h_Alpha_loc_data.table();
    auto const &h_Alpha_glo = h_Alpha_glo_data.table();
    auto const &h_Xtil_glo = h_Xtil_glo_data.table();
    auto const &h_Ytil_glo = h_Ytil_glo_data.table();
    auto const &h_X_glo = h_X_glo_data.table();
    auto const &h_Y_glo = h_Y_glo_data.table();
    auto const &h_X_loc = h_X_loc_data.table();
    auto const &h_Y_loc = h_Y_loc_data.table();
    auto const &h_Alpha_contact = h_Alpha_contact_data.table();
    auto const &h_Y_contact = h_Y_contact_data.table();
    auto const &h_X_contact = h_X_contact_data.table();
    auto const &h_Sigma_contact = h_Sigma_contact_data.table();
    auto const &h_Fermi_contact = h_Fermi_contact_data.table();

#ifdef AMREX_USE_GPU
    RealTable1D d_Current_loc_data({0}, {NUM_CONTACTS}, The_Arena());
    auto const &Current_loc = d_Current_loc_data.table();
    d_Current_loc_data.copy(h_Current_loc_data);

    auto const &GR_loc = d_GR_loc_data.table();
    auto const &A_loc = d_A_loc_data.table();
    /*constant references*/
    auto const &Alpha = d_Alpha_loc_data.const_table();
    auto const &Xtil_glo = d_Xtil_glo_data.const_table();
    auto const &d_Xtil_glo = d_Xtil_glo_data.table();
    auto const &Ytil_glo = d_Ytil_glo_data.const_table();
    auto const &d_Ytil_glo = d_Ytil_glo_data.table();
    auto const &X = d_X_loc_data.const_table();
    auto const &d_X_loc = d_X_loc_data.table();
    auto const &Y = d_Y_loc_data.const_table();
    auto const &d_Y_loc = d_Y_loc_data.table();

    auto const &Alpha_contact = d_Alpha_contact_data.const_table();
    auto const &X_contact = d_X_contact_data.const_table();
    auto const &Y_contact = d_Y_contact_data.const_table();
    auto const &Sigma_contact = d_Sigma_contact_data.const_table();
    auto const &Fermi_contact = d_Fermi_contact_data.const_table();

    auto *trace_r = d_Trace_r.dataPtr();
    auto *trace_i = d_Trace_i.dataPtr();
    auto &degen_vec = block_degen_gpuvec;

#else
    auto const &Current_loc = h_Current_loc_data.table();

    auto const &GR_loc = h_GR_loc_data.table();
    auto const &A_loc = h_A_loc_data.table();
    /*constant references*/
    auto const &Alpha = h_Alpha_loc_data.const_table();
    auto const &Xtil_glo = h_Xtil_glo_data.const_table();
    auto const &Ytil_glo = h_Ytil_glo_data.const_table();
    auto const &X = h_X_loc_data.const_table();
    auto const &Y = h_Y_loc_data.const_table();

    auto const &Alpha_contact = h_Alpha_contact_data.const_table();
    auto const &X_contact = h_X_contact_data.const_table();
    auto const &Y_contact = h_Y_contact_data.const_table();
    auto const &Sigma_contact = h_Sigma_contact_data.const_table();
    auto const &Fermi_contact = h_Fermi_contact_data.const_table();

    auto *trace_r = h_Trace_r.dataPtr();
    auto *trace_i = h_Trace_i.dataPtr();
    auto &degen_vec = block_degen_vec;
#endif

    if (flag_write_integrand_main)
    {
        h_NonEq_Integrand_data.resize({0}, {total_noneq_integration_pts},
                                      The_Pinned_Arena());
        h_NonEq_Integrand_Source_data.resize({0}, {total_noneq_integration_pts},
                                             The_Pinned_Arena());
        h_NonEq_Integrand_Drain_data.resize({0}, {total_noneq_integration_pts},
                                            The_Pinned_Arena());
#ifdef AMREX_USE_GPU
        d_NonEq_Integrand_data.resize({0}, {total_noneq_integration_pts},
                                      The_Arena());
        d_NonEq_Integrand_Source_data.resize({0}, {total_noneq_integration_pts},
                                             The_Arena());
        d_NonEq_Integrand_Drain_data.resize({0}, {total_noneq_integration_pts},
                                            The_Arena());
#endif
    }
    auto const &h_NonEq_Integrand = h_NonEq_Integrand_data.table();
    auto const &h_NonEq_Integrand_Source =
        h_NonEq_Integrand_Source_data.table();
    auto const &h_NonEq_Integrand_Drain = h_NonEq_Integrand_Drain_data.table();
#ifdef AMREX_USE_GPU
    auto const &NonEq_Integrand = d_NonEq_Integrand_data.table();
    auto const &NonEq_Integrand_Source = d_NonEq_Integrand_Source_data.table();
    auto const &NonEq_Integrand_Drain = d_NonEq_Integrand_Drain_data.table();
#else
    auto const &NonEq_Integrand = h_NonEq_Integrand_data.table();
    auto const &NonEq_Integrand_Source = h_NonEq_Integrand_Source_data.table();
    auto const &NonEq_Integrand_Drain = h_NonEq_Integrand_Drain_data.table();
#endif
    if (flag_write_integrand_main)
    {
        amrex::ParallelFor(total_noneq_integration_pts,
                           [=] AMREX_GPU_DEVICE(int e) noexcept
                           {
                               NonEq_Integrand(e) = 0.;
                               NonEq_Integrand_Source(e) = 0.;
                               NonEq_Integrand_Drain(e) = 0.;
                           });
    }

    int e_prev = 0;
    for (int p = 0; p < ContourPath_RhoNonEq.size(); ++p)
    {
        for (int e = 0; e < ContourPath_RhoNonEq[p].num_pts; ++e)
        {
            ComplexType E = ContourPath_RhoNonEq[p].E_vec[e];
            ComplexType weight = ContourPath_RhoNonEq[p].weight_vec[e];
            ComplexType mul_factor = ContourPath_RhoNonEq[p].mul_factor_vec[e];

            for (int n = 0; n < blkCol_size_loc; ++n)
            {
                h_Alpha_loc(n) = E + h_minusHa_loc(n);
                /*+ because h_minusHa is defined previously as -(H0+U)*/
            }

            get_Sigma_at_contacts(h_Sigma_contact_data, E);
#ifdef AMREX_USE_GPU
            d_Sigma_contact_data.copy(h_Sigma_contact_data);
#endif

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n_glo = global_contact_index[c];
                int n = n_glo - vec_cumu_blkCol_size[my_rank];

                if (n_glo >= vec_cumu_blkCol_size[my_rank] &&
                    n_glo < vec_cumu_blkCol_size[my_rank + 1])
                {
                    h_Alpha_loc(n) = h_Alpha_loc(n) - h_Sigma_contact(c);
                }
                h_Fermi_contact(c) =
                    FermiFunction(E - mu_contact[c], kT_contact[c]);
            }
#ifdef AMREX_USE_GPU
            d_Fermi_contact_data.copy(h_Fermi_contact_data);
            d_Alpha_loc_data.copy(h_Alpha_loc_data);
#endif

            /*MPI_Allgather*/
            MPI_Allgatherv(&h_Alpha_loc(0), blkCol_size_loc, MPI_BlkType,
                           &h_Alpha_glo(0), MPI_recv_count.data(),
                           MPI_recv_disp.data(), MPI_BlkType,
                           ParallelDescriptor::Communicator());

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n_glo = global_contact_index[c];
                h_Alpha_contact(c) = h_Alpha_glo(n_glo);
            }
#ifdef AMREX_USE_GPU
            d_Alpha_contact_data.copy(h_Alpha_contact_data);
#endif

            h_Y_glo(0) = 0;
            h_X_glo(Hsize_glo - 1) = 0;
            for (int section = 0; section < num_recursive_parts; ++section)
            {
                for (int n = std::max(1, Hsize_recur_part * section);
                     n < std::min(Hsize_recur_part * (section + 1), Hsize_glo);
                     ++n)
                {
                    int p = (n - 1) % offDiag_repeatBlkSize;
                    h_Ytil_glo(n) =
                        h_Hc_loc(p) / (h_Alpha_glo(n - 1) - h_Y_glo(n - 1));
                    h_Y_glo(n) = h_Hb_loc(p) * h_Ytil_glo(n);
                }
                for (int n = std::min(Hsize_glo - 2,
                                      Hsize_recur_part *
                                          (num_recursive_parts - section));
                     n >=
                     Hsize_recur_part * (num_recursive_parts - section - 1);
                     n--)
                {
                    int p = n % offDiag_repeatBlkSize;
                    h_Xtil_glo(n) =
                        h_Hb_loc(p) / (h_Alpha_glo(n + 1) - h_X_glo(n + 1));
                    h_X_glo(n) = h_Hc_loc(p) * h_Xtil_glo(n);
                }

#ifdef AMREX_USE_GPU
                int Ytil_begin = section * Hsize_recur_part;
                int Ytil_end =
                    std::min(Hsize_recur_part * (section + 1), Hsize_glo);

                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_Ytil_glo.p + Ytil_begin,
                                      h_Ytil_glo.p + Ytil_end,
                                      d_Ytil_glo.p + Ytil_begin);

                int Xtil_begin =
                    Hsize_recur_part * (num_recursive_parts - section - 1);
                int Xtil_end =
                    std::min(Hsize_glo, Hsize_recur_part *
                                            (num_recursive_parts - section));

                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_Xtil_glo.p + Xtil_begin,
                                      h_Xtil_glo.p + Xtil_end,
                                      d_Xtil_glo.p + Xtil_begin);
#endif
            }

#ifdef AMREX_USE_GPU
            int X_begin = vec_cumu_blkCol_size[my_rank];
            int X_end = X_begin + blkCol_size_loc;
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_X_glo.p + X_begin,
                                  h_X_glo.p + X_end, d_X_loc.p);

            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Y_glo.p + X_begin,
                                  h_Y_glo.p + X_end, d_Y_loc.p);
#else
            for (int c = 0; c < blkCol_size_loc; ++c)
            {
                int n = c + vec_cumu_blkCol_size[my_rank];
                h_Y_loc(c) = h_Y_glo(n);
                h_X_loc(c) = h_X_glo(n);
            }
#endif

            for (int c = 0; c < NUM_CONTACTS; ++c)
            {
                int n = global_contact_index[c];
                h_Y_contact(c) = h_Y_glo(n);
                h_X_contact(c) = h_X_glo(n);
            }
#ifdef AMREX_USE_GPU
            d_X_contact_data.copy(h_X_contact_data);
            d_Y_contact_data.copy(h_Y_contact_data);
            amrex::Gpu::streamSynchronize();
#endif

            /*following is for lambda capture*/
            int cumulative_columns = vec_cumu_blkCol_size[my_rank];
            int Hsize = Hsize_glo;
            auto &GC_ID = global_contact_index;
            auto *degen_vec_ptr = degen_vec.dataPtr();

            amrex::Real const_multiplier =
                spin_degen * (PhysConst::q_e) / (PhysConst::h_eVperHz);
            bool write_integrand = flag_write_integrand_main;
            int e_glo = e + e_prev;

            amrex::ParallelFor(
                blkCol_size_loc,
                [=] AMREX_GPU_DEVICE(int n) noexcept
                {
                    int n_glo = n + cumulative_columns; /*global column number*/
                    ComplexType one(1., 0.);
                    ComplexType minus_one(-1., 0.);
                    ComplexType imag(0., 1.);

#ifdef COMPUTE_GREENS_FUNCTION_OFFDIAG_ELEMS
                    GR_loc(n_glo, n) = one / (Alpha(n) - X(n) - Y(n));
                    for (int m = n_glo; m > 0; m--)
                    {
                        GR_loc(m - 1, n) = -1 * Ytil_glo(m) * GR_loc(m, n);
                    }
                    for (int m = n_glo; m < Hsize - 1; ++m)
                    {
                        GR_loc(m + 1, n) = -1 * Xtil_glo(m) * GR_loc(m, n);
                    }
#else
                    GR_loc(n) = one / (Alpha(n) - X(n) - Y(n));
#endif

                    MatrixBlock<T> A_tk[NUM_CONTACTS];
                    MatrixBlock<T> Gamma[NUM_CONTACTS];
                    MatrixBlock<T> Gn_nn;
                    Gn_nn = 0.;
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                    for (int m = 0; m < Hsize; ++m)
                    {
                        A_loc(m, n) = 0.;
                    }
#else
                    A_loc(n) = 0.;
#endif
                    for (int k = 0; k < NUM_CONTACTS; ++k)
                    {
                        int k_glo = GC_ID[k];
                        MatrixBlock<T> G_contact_kk =
                            one /
                            (Alpha_contact(k) - X_contact(k) - Y_contact(k));

                        MatrixBlock<T> temp = G_contact_kk;
                        for (int m = k_glo; m < n_glo; ++m)
                        {
                            temp = -1 * Xtil_glo(m) * temp;
                        }
                        for (int m = k_glo; m > n_glo; m--)
                        {
                            temp = -1 * Ytil_glo(m) * temp;
                        }
                        MatrixBlock<T> G_contact_nk = temp;

                        Gamma[k] = imag * (Sigma_contact(k) -
                                           Sigma_contact(k).Dagger());

                        MatrixBlock<T> A_nn =
                            G_contact_nk * Gamma[k] * G_contact_nk.Dagger();
                        MatrixBlock<T> A_kn =
                            G_contact_kk * Gamma[k] * G_contact_nk.Dagger();
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                        A_loc(k_glo, n) = A_loc(k_glo, n) + A_kn;
                        for (int m = k_glo + 1; m < Hsize; ++m)
                        {
                            A_kn = -1 * Xtil_glo(m - 1) * A_kn;
                            A_loc(m, n) = A_loc(m, n) + A_kn;
                        }
                        for (int m = k_glo - 1; m >= 0; m--)
                        {
                            A_kn = -1 * Ytil_glo(m + 1) * A_kn;
                            A_loc(m, n) = A_loc(m, n) + A_kn;
                        }

                        A_tk[k] = 0.;
                        if (n_glo == CT_ID[k])
                        {
                            A_tk[k] = A_kn;
                        }
#else
                        A_loc(n) = A_loc(n) + A_nn;
#endif
                        Gn_nn = Gn_nn + A_nn * Fermi_contact(k);
                    }

                    /*Current calculation*/
                    for (int k = 0; k < NUM_CONTACTS; ++k)
                    {
                        if (n_glo == GC_ID[k])
                        {
#ifdef COMPUTE_SPECTRAL_FUNCTION_OFFDIAG_ELEMS
                            MatrixBlock<T> IF =
                                Gamma[k] * Fermi_contact(k) * A_loc(n_glo, n) -
                                Gamma[k] * Gn_nn;
#else
                            MatrixBlock<T> IF =
                                Gamma[k] * Fermi_contact(k) * A_loc(n) -
                                Gamma[k] * Gn_nn;
#endif

                            MatrixBlock<T> Current_atE =
                                const_multiplier * IF.DiagMult(degen_vec_ptr) *
                                weight * mul_factor;
                            /*integrating*/
                            Current_loc(k) =
                                Current_loc(k) + Current_atE.DiagSum().real();
                        }
                    }

                    if (write_integrand)
                    {
                        if (n_glo == int(Hsize / 2))
                        {
                            MatrixBlock<T> Intermed = const_multiplier * Gn_nn;
                            NonEq_Integrand(e_glo) =
                                Intermed.DiagMult(degen_vec_ptr)
                                    .DiagSum()
                                    .real();
                        }
                        else if (n_glo == 0)
                        {
                            MatrixBlock<T> Intermed = const_multiplier * Gn_nn;
                            NonEq_Integrand_Source(e_glo) =
                                Intermed.DiagMult(degen_vec_ptr)
                                    .DiagSum()
                                    .real();
                        }
                        else if (n_glo == Hsize - 1)
                        {
                            MatrixBlock<T> Intermed = const_multiplier * Gn_nn;
                            NonEq_Integrand_Drain(e_glo) =
                                Intermed.DiagMult(degen_vec_ptr)
                                    .DiagSum()
                                    .real();
                        }
                    }
                });

#ifdef AMREX_USE_GPU
            amrex::Gpu::streamSynchronize();
#endif
        }
        e_prev += ContourPath_RhoNonEq[p].num_pts;
    }
#ifdef AMREX_USE_GPU
    h_Current_loc_data.copy(d_Current_loc_data);
    amrex::Gpu::streamSynchronize();
#endif

    for (int k = 0; k < NUM_CONTACTS; ++k)
    {
        amrex::ParallelDescriptor::ReduceRealSum(h_Current_loc(k));
    }
    if (ParallelDescriptor::IOProcessor())
    {
        amrex::Print() << "\nCurrent: \n";
        for (int k = 0; k < NUM_CONTACTS; ++k)
        {
            amrex::Print() << " contact, total current: " << k
                           << std::setprecision(5) << std::setw(15)
                           << h_Current_loc(k) << "\n";
        }
    }

    if (flag_write_integrand_main)
    {
#ifdef AMREX_USE_GPU
        h_NonEq_Integrand_data.copy(d_NonEq_Integrand_data);
        h_NonEq_Integrand_Source_data.copy(d_NonEq_Integrand_Source_data);
        h_NonEq_Integrand_Drain_data.copy(d_NonEq_Integrand_Drain_data);
        amrex::Gpu::streamSynchronize();
#endif
        MPI_Allreduce(MPI_IN_PLACE, &(h_NonEq_Integrand(0)),
                      total_noneq_integration_pts, MPI_DOUBLE, MPI_SUM,
                      ParallelDescriptor::Communicator());

        MPI_Allreduce(MPI_IN_PLACE, &(h_NonEq_Integrand_Source(0)),
                      total_noneq_integration_pts, MPI_DOUBLE, MPI_SUM,
                      ParallelDescriptor::Communicator());

        MPI_Allreduce(MPI_IN_PLACE, &(h_NonEq_Integrand_Drain(0)),
                      total_noneq_integration_pts, MPI_DOUBLE, MPI_SUM,
                      ParallelDescriptor::Communicator());

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            amrex::Vector<ComplexType> E_total_vec;
            E_total_vec.resize(total_noneq_integration_pts);
            int e_prev_pts = 0;
            for (auto &path : ContourPath_RhoNonEq)
            {
                for (int e = 0; e < path.num_pts; ++e)
                {
                    int e_glo = e_prev_pts + e;
                    E_total_vec[e_glo] = path.E_vec[e].real();
                }
                e_prev_pts += path.num_pts;
            }
            Write_Integrand(E_total_vec, h_NonEq_Integrand_data,
                            h_NonEq_Integrand_Source_data,
                            h_NonEq_Integrand_Drain_data,
                            step_filename_str + "_integrand.dat");

            amrex::Real max_noneq_integrand = 0;
            for (int e = 0; e < E_total_vec.size(); ++e)
            {
                if (max_noneq_integrand < fabs(h_NonEq_Integrand(e)))
                {
                    max_noneq_integrand = fabs(h_NonEq_Integrand(e));
                    E_at_max_noneq_integrand = E_total_vec[e].real();
                }
            }

            amrex::Print() << "\n Abs. value of max integrand: "
                           << max_noneq_integrand
                           << " at E (eV): " << E_at_max_noneq_integrand
                           << " i.e., (E-mu_0)/kT_0: "
                           << (E_at_max_noneq_integrand - mu_contact[0]) /
                                  kT_contact[0]
                           << "\n";
        }
        ParallelDescriptor::Bcast(&E_at_max_noneq_integrand, 1,
                                  ParallelDescriptor::IOProcessorNumber());

        h_NonEq_Integrand_data.clear();
        h_NonEq_Integrand_Source_data.clear();
        h_NonEq_Integrand_Drain_data.clear();
#ifdef AMREX_USE_GPU
        d_NonEq_Integrand_data.clear();
        d_NonEq_Integrand_Source_data.clear();
        d_NonEq_Integrand_Drain_data.clear();
#endif
    }

    Deallocate_TemporaryArraysForGFComputation();
}

template <typename T>
void c_NEGF_Common<T>::Write_Current(const int step,
                                     const int total_iterations_in_step,
                                     const amrex::Real avg_intg_pts)
{
    if (ParallelDescriptor::IOProcessor())
    {
        amrex::Print() << "Root writing current\n";
        auto const &h_Current_loc = h_Current_loc_data.table();

        outfile_I.open(current_filename_str.c_str(), std::ios_base::app);

        outfile_I << std::setw(10) << step << std::setw(15) << Vds
                  << std::setw(15) << Vgs;
        for (int k = 0; k < NUM_CONTACTS; ++k)
        {
            outfile_I << std::setw(20) << h_Current_loc(k);
        }
        outfile_I << std::setw(10) << avg_intg_pts << std::setw(10)
                  << total_iterations_in_step << std::setw(20)
                  << total_conductance << "\n";
        outfile_I.close();
    }
}

template <typename T>
void c_NEGF_Common<T>::Set_TerminalBiasesAndContactPotential()
{
    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    auto &rBC = rCode.get_BoundaryConditions();

    if (!flag_contact_mu_specified)
    {
        amrex::Real V_contact[NUM_CONTACTS] = {0., 0.};
        amrex::Vector<amrex::Real> ep(NUM_CONTACTS, 0);

        const auto *ps = get_Contact_Parser_String();
        for (int k = 0; k < NUM_CONTACTS; ++k)
        {
            V_contact[k] = rGprop.pEB->Read_SurfSoln(ps[k]);

            amrex::Print() << "\n Updated terminal voltage: " << k << "  "
                           << V_contact[k] << " V\n";

            ep[k] = get_Fermi_level() - V_contact[k];
        }
        set_Contact_Electrochemical_Potential(ep);

        Vds = V_contact[1] - V_contact[0];

        if (gate_terminal_type == Gate_Terminal_Type::EB)
        {
            Vgs = rGprop.pEB->Read_SurfSoln(get_Gate_String()) - V_contact[0];
        }
        else if (gate_terminal_type == Gate_Terminal_Type::Boundary)
        {
            Vgs = rBC.Read_BoundarySoln(get_Gate_String()) - V_contact[0];
        }
    }
    else
    {
        amrex::Real ep_s = get_Source_Electrochemical_Potential();
        amrex::Real ep_d = get_Drain_Electrochemical_Potential();

        Vds = ep_s - ep_d;

        amrex::Real GV = 0.;
        if (gate_terminal_type == Gate_Terminal_Type::EB)
        {
            GV = rGprop.pEB->Read_SurfSoln(get_Gate_String());
        }
        else if (gate_terminal_type == Gate_Terminal_Type::Boundary)
        {
            GV = rBC.Read_BoundarySoln(get_Gate_String());
        }

        amrex::Print() << "\n Updated gate voltage: " << GV << " V\n";
        Vgs = GV - (get_Fermi_level() - ep_s);
    }
    amrex::Print() << " Vds: " << Vds << " V, Vgs: " << Vgs << " V\n";
}

template <typename T>
void c_NEGF_Common<T>::Copy_BroydenPredictedCharge(
    const RealTable1D &h_n_curr_in_data)
{
    auto const &h_n_curr_in = h_n_curr_in_data.table();

    RealTable1D n_curr_in_glo_data;

    if (ParallelDescriptor::IOProcessor())
    {
        n_curr_in_glo_data.resize({0}, {num_field_sites}, The_Pinned_Arena());
    }
    auto const &n_curr_in_glo = n_curr_in_glo_data.table();

    MPI_Gatherv(&h_n_curr_in(0), MPI_recv_count[my_rank], MPI_DOUBLE,
                &n_curr_in_glo(0), MPI_recv_count.data(), MPI_recv_disp.data(),
                MPI_DOUBLE, ParallelDescriptor::IOProcessorNumber(),
                ParallelDescriptor::Communicator());

    auto const &h_n_curr_in_loc = h_n_curr_in_loc_data.table();

    MPI_Scatterv(&n_curr_in_glo(0), MPI_send_count.data(), MPI_send_disp.data(),
                 MPI_DOUBLE, &h_n_curr_in_loc(0), num_local_field_sites,
                 MPI_DOUBLE, ParallelDescriptor::IOProcessorNumber(),
                 ParallelDescriptor::Communicator());
    // amrex::Print() << "h_n_curr_in_loc in CopyToNS: \n";
    // amrex::Print() << "MPI_send_count/disp: " << MPI_send_count[0] << " " <<
    // MPI_send_disp[0] << "\n"; if (ParallelDescriptor::IOProcessor())
    //{
    //     for(int n=0; n < 5; ++n) {
    //        amrex::Print() << n << "  " <<  h_n_curr_in_loc(n) << "\n";
    //     }
    // }
    // MPI_Barrier(ParallelDescriptor::Communicator());

    // write out charge
    if (write_at_iter)
    {
        Write_InputInducedCharge(iter_filename_str, n_curr_in_glo_data);
    }
}
