#include "PostProcessor.H"

#include <AMReX_ParmParse.H>

#include "Code.H"
#include "Solver/Electrostatics/MLMG.H"

using namespace amrex;

c_PostProcessor::c_PostProcessor()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_PostProcessor "
                      "Constructor()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    DefineParameterNameMap();
    num_all_params = ReadData();
    num_arraymf_params = SortParameterType();
    num_mf_params = num_all_params - num_arraymf_params;

    // amrex::Print() << "num_all_params: " << num_all_params << "\n";
    // amrex::Print() << "num_arraymf_params: " << num_arraymf_params << "\n";
    // amrex::Print() << "num_mf_params: " << num_mf_params << "\n";

    m_p_mf.resize(num_mf_params);
    m_p_array_mf.resize(num_arraymf_params);

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_PostProcessor "
                      "Constructor()************************\n";
#endif
}

c_PostProcessor::~c_PostProcessor()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_PostProcessor "
                      "Destructor()************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_PostProcessor "
                      "Destructor()************************\n";
#endif
}

int c_PostProcessor::ReadData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t{************************c_PostProcessor:"
                      "ReadData()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    amrex::Vector<std::string> m_fields_to_process;

    amrex::ParmParse pp_postprocess("post_process");

    [[maybe_unused]] bool varnames_specified =
        pp_postprocess.queryarr("fields_to_process", m_fields_to_process);

    std::map<std::string, int>::iterator it_map_param_all;

    int c = 0;
    for (auto it : m_fields_to_process)
    {
        //    amrex::Print() << "reading field to process: " << it  << "\n";

        std::string second_last_char = it.substr(it.length() - 2, 1);

        if (second_last_char == "_")
        {
            std::string str_without_subscript = it.substr(0, it.length() - 2);
            std::string vector_str = "vec" + str_without_subscript;

            it_map_param_all = map_param_all.find(vector_str);

            if (it_map_param_all == map_param_all.end())
            {
                map_param_all[vector_str] = c;
                ++c;
            }
        }
        else
        {
            it_map_param_all = map_param_all.find(it);

            if (it_map_param_all == map_param_all.end())
            {
                map_param_all[it] = c;
                ++c;
            }
        }
    }
    m_fields_to_process.clear();

    amrex::Print() << "\n##### POST PROCESSOR #####\n\n";
    amrex::Print() << "##### fields_to_process:\n";
    for (auto it : map_param_all)
    {
        amrex::Print() << "##### " << it.second << "   " << it.first << "\n";
    }
    // amrex::Print() << "##### Total fields to process: " <<
    // map_param_all.size() << "\n\n";

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t}************************c_PostProcessor:"
                      "ReadData()************************\n";
#endif

    return map_param_all.size();
}

int c_PostProcessor::SortParameterType()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t{************************c_PostProcessor:"
                      "SortParameterMap()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    int i = 0;
    int j = 0;
    for (auto it : map_param_all)
    {
        std::string str = it.first;
        std::string first_three_char_str = str.substr(0, 3);

        if (first_three_char_str == "vec")
        {
            map_param_arraymf[it.first] = i;
            // amrex::Print() << "key: " << it.first << "  value: " << it.second
            // << "  dimension: " << AMREX_SPACEDIM << "  counter, i: "<< i<<
            // "\n";
            ++i;
        }
        else
        {
            map_param_mf[it.first] = j;
            // amrex::Print() << "key: " << it.first << "  value: " << it.second
            // << "  dimension: " << AMREX_SPACEDIM << "  counter, j: "<< j<<
            // "\n";
            ++j;
        }
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t}************************c_PostProcessor:"
                      "SortParameterMap()************************\n";
#endif

    return i;
}

void c_PostProcessor::InitData()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_PostProcessor::"
                      "InitData()************************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    for (auto it : map_param_mf)
    {
        m_p_mf[it.second] =
            std::make_unique<amrex::MultiFab>();  // cell-centered multifab
    }
    for (auto it : map_param_arraymf)
    {
        m_p_array_mf[it.second] =
            std::make_unique<std::array<amrex::MultiFab, AMREX_SPACEDIM>>();
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_PostProcessor::InitData()"
                      "************************\n";
#endif
}

void c_PostProcessor::Compute()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_PostProcessor::"
                      "Compute()************************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    auto &rCode = c_Code::GetInstance();
    std::map<std::string, s_PostProcessMacroName::macro_name>::iterator it_Post;

    for (auto it : map_param_all)
    {
        auto macro_str = it.first;

        it_Post = map_macro_name.find(macro_str);

        if (it_Post == map_macro_name.end())
        {
            amrex::Print() << "Computation of " << macro_str
                           << " is not implemented at present.\n";
        }
        else
        {
            switch (map_macro_name[macro_str])
            {
                case s_PostProcessMacroName::vecField:
                {
                    auto val = map_param_arraymf[macro_str];
                    auto &rMLMGsolver = rCode.get_MLMGSolver();
                    rMLMGsolver.Compute_vecField(*m_p_array_mf[val]);
                    break;
                }
                case s_PostProcessMacroName::vecFlux:
                {
                    auto val = map_param_arraymf[macro_str];
                    auto &rMLMGsolver = rCode.get_MLMGSolver();
                    rMLMGsolver.Compute_vecFlux(*m_p_array_mf[val]);
                    break;
                }
            }
        }
    }
#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_PostProcessor::Compute()*"
                      "***********************\n";
#endif
}
