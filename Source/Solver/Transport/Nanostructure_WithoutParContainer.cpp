#include <AMReX.H>

#include "Nanostructure.H"
//#include <AMReX_GpuContainers.H>

#include "../../Code.H"
#include "../../Code_Definitions.H"
//#include "../../Utils/CodeUtils/CodeUtil.H"
//#include "../../Utils/SelectWarpXUtils/TextMsg.H"
//#include "../../Utils/SelectWarpXUtils/WarpXConst.H"
//#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"

#include <iostream>

template class c_Nanostructure<c_AtomicChain,
                               RequiresParticleContainer<c_AtomicChain>::value>;

template <typename NSType>
c_Nanostructure<NSType, false>::c_Nanostructure(
    const std::string NS_name_str, const int NS_id_counter,
    const amrex::Real NS_initial_deposit_value, const int use_negf,
    const std::string negf_foldername_str)
{
    _use_negf = use_negf;

    if (_use_negf)
    {
        NSType::Initialize_NEGF_Params(NS_name_str, NS_id_counter,
                                       NS_initial_deposit_value,
                                       negf_foldername_str);

        pos_vec.resize(NSType::num_atoms);

        Read_AtomLocations();

        bool use_electrostatics = false;
        NSType::Initialize_NEGF(negf_foldername_str + "/transport_common",
                                use_electrostatics);
        pos_vec.clear();
    }
}

template <typename NSType>
void c_Nanostructure<NSType, false>::Read_AtomLocations()
{
    if (ParallelDescriptor::IOProcessor())
    {
        std::string read_filename = NSType::get_read_atom_filename();

        if (read_filename.empty())
        {
            NSType::Generate_AtomLocations(pos_vec);
        }
        else
        {
            std::ifstream infile;
            infile.open(read_filename.c_str());

            if (infile.fail())
            {
                amrex::Abort("Failed to read file " + read_filename);
            }
            else
            {
                int filesize = 0;
                std::string line;
                while (infile.peek() != EOF)
                {
                    std::getline(infile, line);
                    filesize++;
                }
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    filesize == NSType::num_atoms,
                    "Number of atoms, " + std::to_string(NSType::num_atoms) +
                        ", are not equal to the filesize, " +
                        std::to_string(filesize) + " !");

                infile.seekg(0, std::ios_base::beg);

                std::string id[2];

                for (int i = 0; i < NSType::num_atoms; ++i)
                {
                    infile >> id[0] >> id[1];

                    for (int j = 0; j < AMREX_SPACEDIM; ++j)
                    {
                        infile >> pos_vec[i].dir[j];
                        pos_vec[i].dir[j] += NSType::offset[j];
                    }
                }
                infile.close();
            }
        }
    }
}
