#include "Output.H"

#include <AMReX_ParmParse.H>

#include "../Utils/CodeUtils/CodeUtil.H"
#include "../Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Code.H"
#include "Input/GeometryProperties/GeometryProperties.H"
#include "Input/MacroscopicProperties/MacroscopicProperties.H"
#include "PostProcessor/PostProcessor.H"

using namespace amrex;

c_Output::c_Output()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_Output "
                      "Constructor()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
    std::string prt = "\t\t\t";
#endif

    m_num_params_plot_single_level = 0;
    m_write_after_init = 0;

    m_num_params_plot_single_level = ReadData();

    if (_raw_fields_to_plot_flag) _extra_dirs.emplace_back("raw_fields");

    m_p_mf.resize(m_map_param_all.size());

#ifdef PRINT_LOW
    amrex::Print() << prt
                   << "number of all parameters: " << m_map_param_all.size()
                   << "\n";
    amrex::Print() << prt
                   << "_raw_fields_to_plot_flag: " << _raw_fields_to_plot_flag
                   << "\n\n";
    amrex::Print()
        << "\n"
        << prt << "number of parameters to plot in a single level plot file: "
        << m_num_params_plot_single_level << "\n";
    amrex::Print() << prt << "m_write_after_init: " << m_write_after_init
                   << "\n\n";
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_Output "
                      "Constructor()************************\n";
#endif
}

c_Output::~c_Output()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_Output "
                      "Destructor()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    m_map_param_all.clear();
    m_map_field_to_plot_after_init.clear();

    m_num_params_plot_single_level = 0;
    m_write_after_init = 0;
    m_p_mf.clear();
#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_Output "
                      "Destructor()************************\n";
#endif
}

int c_Output::ReadData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t{************************c_Output::ReadData("
                      ")************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t";
#endif

    amrex::Vector<std::string> fields_to_plot_withGhost_str;

    amrex::ParmParse pp_plot("plot");

    _foldername_str = "output";
    pp_plot.query("folder_name", _foldername_str);

    _filename_prefix_str = _foldername_str + "/plt";

    pp_plot.query("write_after_init", m_write_after_init);

    _write_interval = 1;           // default
    _rawfield_write_interval = 1;  // default
    queryWithParser(pp_plot, "write_interval", _write_interval);
    queryWithParser(pp_plot, "rawfield_write_interval",
                    _rawfield_write_interval);

    [[maybe_unused]] bool varnames_specified =
        pp_plot.queryarr("fields_to_plot", fields_to_plot_withGhost_str);

    int num_params_without_option_2 =
        SpecifyOutputOption(fields_to_plot_withGhost_str, m_map_param_all);

    fields_to_plot_withGhost_str.clear();

    amrex::Print() << "\n##### OUTPUT #####\n\n";
    amrex::Print() << "##### file name: " << _filename_prefix_str << "\n";
    amrex::Print() << "##### fields_to_plot:\n";
    amrex::Print() << "##### " << std::setw(20) << "name" << std::setw(10)
                   << "number" << std::setw(14) << "output_option\n";
    for (auto it : m_map_param_all)
    {
        amrex::Print() << "##### " << std::setw(20) << it.first << std::setw(10)
                       << it.second << std::setw(10)
                       << _output_option[it.second] << "\n";
    }
    amrex::Print() << "\n##### write_after_init?: " << m_write_after_init
                   << "\n";

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t}************************c_Output::ReadData()***"
                      "*********************\n";
#endif

    return num_params_without_option_2;
}

void c_Output::InitData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_Output::InitData()***"
                      "*********************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
    std::string prt = "\t\t";
#endif

    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    auto &geom = rGprop.geom;
    auto &ba = rGprop.ba;
    auto &dm = rGprop.dm;

    Init_Plot_Field_Essentials(geom, rGprop.is_eb_enabled());

    int Nghost0 = 0;

#ifdef AMREX_USE_EB
    if (rGprop.is_eb_enabled())
    {
        m_p_mf_all =
            std::make_unique<amrex::MultiFab>(ba, dm,
                                              m_num_params_plot_single_level,
                                              Nghost0, MFInfo(),
                                              *rGprop.pEB->p_factory_union);
    }
#endif
    if (!rGprop.is_eb_enabled())
    {
        m_p_mf_all =
            std::make_unique<amrex::MultiFab>(ba, dm,
                                              m_num_params_plot_single_level,
                                              Nghost0);
    }

    if (m_write_after_init)
    {
        int counter = 0;
        for (auto it : m_map_param_all)
        {
            std::string field_name = it.first;
            int field_number = it.second;

            if (Evaluate_TypeOf_MacroStr(field_name) ==
                0) /*mf is in MacroscopicProperties*/
            {
                m_map_field_to_plot_after_init[field_name] = field_number;

                if (_output_option[field_number] == 0 or
                    _output_option[field_number] == 1)
                {
                    ++counter;
                }
            }
        }
#ifdef AMREX_USE_EB
        if (rGprop.is_eb_enabled())
        {
            m_p_mf_all_init =
                std::make_unique<amrex::MultiFab>(ba, dm, counter, Nghost0,
                                                  MFInfo(),
                                                  *rGprop.pEB->p_factory_union);
        }
#endif
        if (!rGprop.is_eb_enabled())
        {
            m_p_mf_all_init =
                std::make_unique<amrex::MultiFab>(ba, dm, counter, Nghost0);
        }

#ifdef PRINT_HIGH
        amrex::Print() << "\n" << prt << "m_write_after_init is true! \n";
        for (auto it : m_map_field_to_plot_after_init)
        {
            std::string field_name = it.first;
            int field_number = it.second;
            amrex::Print() << prt << "field to plot after init, name "
                           << field_name << ", field_number " << field_number
                           << ", output option " << _output_option[field_number]
                           << "\n";
        }
        amrex::Print() << prt << "size of m_p_mf_all_init: " << counter << "\n";
#endif
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_Output::InitData()*******"
                      "*****************\n";
#endif
}

void c_Output::WriteOutput(int step, amrex::Real time)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_Output::WriteOutput()"
                      "************************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
    std::string prt = "\t\t";
#endif

    if ((step + 1) % _write_interval == 0)
    {
        AssimilateDataPointers();
        _plot_file_name =
            amrex::Concatenate(_filename_prefix_str, step, _plt_name_digits);

        WriteSingleLevelPlotFile(step, time, m_p_mf, m_p_mf_all,
                                 m_map_param_all);
    }

    if ((_raw_fields_to_plot_flag == 1) and
        ((step + 1) % _rawfield_write_interval == 0))
        WriteRawFields(m_p_mf, m_map_param_all);

#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_Output::WriteOutput()****"
                      "********************\n";
#endif
}

void c_Output::WriteOutput_AfterInit()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_Output::WriteOutput_"
                      "AfterInit()************************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
    std::string prt = "\t\t";
#endif

    AssimilateDataPointers();
    int step = 0;
    amrex::Real time = 0;
    _plot_file_name =
        amrex::Concatenate(_filename_prefix_str + "_init", step, 0);
    WriteSingleLevelPlotFile(step, time, m_p_mf, m_p_mf_all_init,
                             m_map_field_to_plot_after_init);

    if (_raw_fields_to_plot_flag)
        WriteRawFields(m_p_mf, m_map_field_to_plot_after_init);

#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_Output::WriteOutput_"
                      "AfterInit()************************\n";
#endif
}

void c_Output::AssimilateDataPointers()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_Output::"
                      "AssimilateDataPointers()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
    std::string prt = "\t\t\t";
#endif

    auto &rCode = c_Code::GetInstance();
    auto &rPost = rCode.get_PostProcessor();
    auto &rMprop = rCode.get_MacroscopicProperties();

    for (auto it : m_map_param_all)
    {
        std::string str = it.first;
        int field_counter = it.second;

        std::string second_last_str = str.substr(str.length() - 2, 1);
        std::string last_str = str.substr(str.length() - 1);
#ifdef PRINT_HIGH
        amrex::Print() << "\n"
                       << prt << "macro_str " << str << " second_last_str "
                       << second_last_str << "\n";
#endif

        if (second_last_str != "_" or
            (second_last_str == "_" and last_str != "x" and last_str != "y" and
             last_str != "z"))
        {
#ifdef PRINT_HIGH
            amrex::Print() << prt << str << " doesn't have a subscript \n";
#endif
            switch (Evaluate_TypeOf_MacroStr(str))
            {
                case 0:
                {
#ifdef PRINT_HIGH
                    amrex::Print() << prt << "field counter: " << field_counter
                                   << " fetching pointer for : " << str << "\n";
#endif
                    m_p_mf[field_counter] = rMprop.get_p_mf(str);
                    break;
                }
                case 1:
                {
#ifdef PRINT_HIGH
                    amrex::Print() << prt << "field counter: " << field_counter
                                   << " fetching pointer for : " << str << "\n";
#endif
                    m_p_mf[field_counter] = rPost.get_p_mf(str);
                    break;
                }
                default:
                {
#ifdef PRINT_HIGH
                    amrex::Print() << prt << "field counter: " << field_counter
                                   << " fetching pointer for: " << str << "\n";
#endif
                    amrex::Print() << "macro " << str << " is not defined. \n";
                    break;
                }
            }
        }
        else
        {
            std::string subscript = str.substr(str.length() - 2);
            switch (_map_subscript_name[subscript])
            {
                case s_Subscript::x:
                {
                    std::string str_without_subscript =
                        str.substr(0, str.length() - 2);
                    std::string vector_str = "vec" + str_without_subscript;
#ifdef PRINT_HIGH
                    amrex::Print()
                        << prt << "field counter: " << field_counter << "\n";
                    amrex::Print() << prt << "Inside c_Subscript::x \n";
                    amrex::Print() << prt << "Getting x component of vector: "
                                   << vector_str << "\n";
#endif

                    m_p_mf[field_counter] =
                        rPost.get_p_array_mf_component(vector_str, 0);
                    break;
                }
                case s_Subscript::y:
                {
                    std::string str_without_subscript =
                        str.substr(0, str.length() - 2);
                    std::string vector_str = "vec" + str_without_subscript;
#ifdef PRINT_HIGH
                    amrex::Print()
                        << prt << "field counter: " << field_counter << "\n";
                    amrex::Print() << prt << "Inside c_Subscript::y \n";
                    amrex::Print() << prt << "Getting y component of vector: "
                                   << vector_str << "\n";
#endif

                    m_p_mf[field_counter] =
                        rPost.get_p_array_mf_component(vector_str, 1);
                    break;
                }
                case s_Subscript::z:
                {
                    std::string str_without_subscript =
                        str.substr(0, str.length() - 2);
                    std::string vector_str = "vec" + str_without_subscript;
#ifdef PRINT_HIGH
                    amrex::Print()
                        << prt << "field counter: " << field_counter << "\n";
                    amrex::Print() << prt << "Inside c_Subscript::z \n";
                    amrex::Print() << prt << "Getting z component of vector: "
                                   << vector_str << "\n";
#endif

                    m_p_mf[field_counter] =
                        rPost.get_p_array_mf_component(vector_str, 2);
                    break;
                }
            }
        }
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_Output::"
                      "AssimilateDataPointers()************************\n";
#endif
}
