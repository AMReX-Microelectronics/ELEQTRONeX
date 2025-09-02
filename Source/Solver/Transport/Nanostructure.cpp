#include "Nanostructure.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>

#include "../../Code.H"
#include "../../Code_Definitions.H"
#include "../../Input/GeometryProperties/GeometryProperties.H"
#include "../../Input/MacroscopicProperties/MacroscopicProperties.H"
#include "../../Utils/CodeUtils/CloudInCell.H"
#include "../../Utils/CodeUtils/CodeUtil.H"
#include "../../Utils/SelectWarpXUtils/TextMsg.H"
#include "../../Utils/SelectWarpXUtils/WarpXConst.H"
#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
//
#include <iostream>

template class c_Nanostructure<c_CNT>;
template class c_Nanostructure<c_Graphene>;

template <typename NSType>
c_Nanostructure<NSType>::c_Nanostructure(
    const amrex::Geometry &geom, const amrex::DistributionMapping &dm,
    const amrex::BoxArray &ba, const std::string NS_name_str,
    const int NS_id_counter, const std::string NS_gather_str,
    const std::string NS_deposit_str,
    const amrex::Real NS_initial_deposit_value, const int use_negf,
    const std::string negf_foldername_str)
    : amrex::ParticleContainer<realPD::NUM, intPD::NUM, realPA::NUM,
                               intPA::NUM>(geom, dm, ba),
      _geom(&geom)
{
    auto &rCode = c_Code::GetInstance();
    _use_electrostatic = rCode.use_electrostatic;
    _use_negf = use_negf;

    if (_use_negf)
    {
        NSType::Initialize_NEGF_Params(NS_name_str, NS_id_counter,
                                       NS_initial_deposit_value,
                                       negf_foldername_str);

        pos_vec.resize(NSType::num_atoms);

        Read_AtomLocations();
    }

    if (_use_electrostatic)
    {
        auto &rMprop = rCode.get_MacroscopicProperties();
        p_mf_gather = rMprop.get_p_mf(NS_gather_str);
        p_mf_deposit = rMprop.get_p_mf(NS_deposit_str);

        Compute_CellVolume();

        Fill_AtomLocations();

        Evaluate_LocalFieldSites();

        NSType::Define_MPISendCountAndDisp();

        Mark_CellsWithAtoms();

        NSType::Initialize_ChargeAtFieldSites();

        Deposit_ChargeDensityToMesh();
    }

    if (_use_negf)
    {
        NSType::Initialize_NEGF(negf_foldername_str + "/transport_common",
                                _use_electrostatic);
        pos_vec.clear();
    }
}

template <typename NSType>
amrex::Real c_Nanostructure<NSType>::Compute_CellVolume()
{
    auto dx = _geom->CellSizeArray();
    amrex::Real cell_volume = 1.;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) cell_volume *= dx[i];

    amrex::Print() << "cell_volume: " << cell_volume << "\n";

    return cell_volume;
}

template <typename NSType>
void c_Nanostructure<NSType>::Fill_AtomLocations()
{
    if (ParallelDescriptor::IOProcessor())
    {
        for (int i = 0; i < NSType::num_atoms; ++i)
        {
            ParticleType p;
            p.id() = ParticleType::NextID();

            if (i == 0) particle_id_offset = p.id() - 1;

            p.cpu() = ParallelDescriptor::MyProc();

            for (int j = 0; j < AMREX_SPACEDIM; ++j)
            {
                p.pos(j) = pos_vec[i].dir[j];
            }

            int par_id_local_to_NS = p.id() - particle_id_offset;
            int site_id = NSType::get_1D_site_id(par_id_local_to_NS);

            std::array<int, intPA::NUM> int_attribs;
            int_attribs[intPA::site_id] = site_id;

            std::array<ParticleReal, realPA::NUM> real_attribs;
            real_attribs[realPA::gather] = 0.0;
            real_attribs[realPA::deposit] = 0.0;

            std::pair<int, int> key{0, 0};  //{grid_index, tile index}
            int lev = 0;
            auto &particle_tile = GetParticles(lev)[key];

            particle_tile.push_back(p);
            particle_tile.push_back_int(int_attribs);
            particle_tile.push_back_real(real_attribs);
        }
    }
    ParallelDescriptor::Bcast(&particle_id_offset, 1,
                              ParallelDescriptor::IOProcessorNumber());

    Redistribute();
}

template <typename NSType>
void c_Nanostructure<NSType>::Evaluate_LocalFieldSites()
{
    int lev = 0;
    std::unordered_map<int, int> map;
    int counter = 0;
    int min_site_id = std::numeric_limits<int>::max();

    int hasValidBox = false;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        hasValidBox = true;
        auto np = pti.numParticles();

        const auto &particles = pti.GetArrayOfStructs();
        [[maybe_unused]] const auto p_par = particles().data();

        auto &par_site_id = pti.get_site_id();
        auto *p_site_id = par_site_id.data();

        for (int p = 0; p < np; ++p)
        {
            int site_id = p_site_id[p];
            if (map.find(site_id) == map.end())
            {
                map[site_id] = counter++;
                min_site_id = std::min(min_site_id, site_id);
            }
        }
    }

    if (hasValidBox)
    {
        NSType::min_local_site_id = min_site_id;
    }
    else
    {
        NSType::min_local_site_id = 0;
    }
    NSType::num_local_field_sites = counter;

    // std::cout << " process/min_local_site_id/num_local_field_sites: "
    //           << NSType::my_rank << " "
    //           << NSType::min_local_site_id << " "
    //           << NSType::num_local_field_sites << "\n";

    /*define local charge density 1D table, h_n_curr_in_loc_data */
    NSType::h_n_curr_in_loc_data.resize({0}, {NSType::num_local_field_sites},
                                        The_Pinned_Arena());
#if AMREX_USE_GPU
    NSType::d_n_curr_in_loc_data.resize({0}, {NSType::num_local_field_sites},
                                        The_Arena());
#endif
}

template <typename NSType>
void c_Nanostructure<NSType>::Read_AtomLocations()
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

template <typename NSType>
void c_Nanostructure<NSType>::Mark_CellsWithAtoms()
{
    auto &rCode = c_Code::GetInstance();
    [[maybe_unused]] auto &rPost = rCode.get_PostProcessor();
    auto &rMprop = rCode.get_MacroscopicProperties();

    const auto &plo = _geom->ProbLoArray();
    const auto dx = _geom->CellSizeArray();

    auto &mf = rMprop.get_mf(
        "atom_locations");  // define in macroscopic.fields_to_define

    // amrex::Print() << "Marking atom locations\n";
    int lev = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto np = pti.numParticles();

        const auto &particles = pti.GetArrayOfStructs();
        [[maybe_unused]] const auto p_par = particles().data();
        auto mf_arr = mf.array(pti);
        amrex::ParallelFor(
            np,
            [=] AMREX_GPU_DEVICE(int p) noexcept
            {
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> l = {
                    AMREX_D_DECL((p_par[p].pos(0) - plo[0]) / dx[0],
                                 (p_par[p].pos(1) - plo[1]) / dx[1],
                                 (p_par[p].pos(2) - plo[2]) / dx[2])};

                amrex::GpuArray<int, AMREX_SPACEDIM> index = {
                    AMREX_D_DECL(static_cast<int>(amrex::Math::floor(l[0])),
                                 static_cast<int>(amrex::Math::floor(l[1])),
                                 static_cast<int>(amrex::Math::floor(l[2])))};

                mf_arr(index[0], index[1], index[2]) = 1;
            });
    }
    mf.FillBoundary(_geom->periodicity());
}

template <typename NSType>
void c_Nanostructure<NSType>::Gather_PotentialAtAtoms()
{
    const auto &plo = _geom->ProbLoArray();
    const auto dx = _geom->CellSizeArray();

    int lev = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto np = pti.numParticles();
        // amrex::Print() << "np in Gather: " << np << "\n";
        const auto &particles = pti.GetArrayOfStructs();
        [[maybe_unused]] const auto p_par = particles().data();

        auto &par_gather = pti.get_realPA_comp(realPA::gather);
        auto p_par_gather = par_gather.data();

        auto phi = p_mf_gather->array(pti);

        // amrex::Print() << "p_par[0].pos: " << p_par[0].pos(0) << " "
        //                                    << p_par[0].pos(1) << " "
        //                                    << p_par[0].pos(2) << "\n";
        amrex::ParallelFor(np,
                           [=] AMREX_GPU_DEVICE(int p) noexcept
                           {
                               p_par_gather[p] =
                                   CloudInCell::Gather_Trilinear(p_par[p].pos(),
                                                                 plo, dx, phi);
                           });
    }

    Obtain_PotentialAtSites();
}

template <typename NSType>
void c_Nanostructure<NSType>::Deposit_ZeroToMesh()
{
    const auto &plo = _geom->ProbLoArray();
    const auto dx = _geom->CellSizeArray();
    int lev = 0;

    for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto np = pti.numParticles();

        const auto &particles = pti.GetArrayOfStructs();
        [[maybe_unused]] const auto p_par = particles().data();

        auto rho = p_mf_deposit->array(pti);

        amrex::ParallelFor(np,
                           [=] AMREX_GPU_DEVICE(int p) noexcept {
                               CloudInCell::Deposit_Trilinear(p_par[p].pos(),
                                                              plo, dx, rho);
                           });
    }
}

template <typename NSType>
void c_Nanostructure<NSType>::Deposit_ChargeDensityToMesh()
{
    const auto &plo = _geom->ProbLoArray();
    const auto dx = _geom->CellSizeArray();
    int lev = 0;

    // Deposit_ZeroToMesh();

#ifdef AMREX_USE_GPU
    NSType::d_n_curr_in_loc_data.copy(NSType::h_n_curr_in_loc_data);
    auto const &n_curr_in_loc = NSType::d_n_curr_in_loc_data.table();

    amrex::Gpu::streamSynchronize();
#else
    auto const &n_curr_in_loc = NSType::h_n_curr_in_loc_data.table();
#endif

    for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto np = pti.numParticles();

        const auto &particles = pti.GetArrayOfStructs();
        [[maybe_unused]] const auto p_par = particles().data();

        const auto &par_deposit = pti.get_realPA_comp(realPA::deposit);
        [[maybe_unused]] const auto p_par_deposit = par_deposit.data();

        auto &par_site_id = pti.get_site_id();
        auto *p_site_id = par_site_id.data();

        auto rho = p_mf_deposit->array(pti);

        amrex::Real unit_charge = PhysConst::q_e;
        int atoms_per_field_site = NSType::num_atoms_per_field_site;
        int SIO = NSType::min_local_site_id;

        amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

        [[maybe_unused]] auto const &h_n_curr_in_loc = NSType::h_n_curr_in_loc_data.table();

        amrex::ParallelFor(np,
                           [=] AMREX_GPU_DEVICE(int p) noexcept
                           {
                               int site_id = p_site_id[p];
                               amrex::Real qp = n_curr_in_loc(site_id - SIO) *
                                                unit_charge / vol /
                                                atoms_per_field_site;
                               CloudInCell::Deposit_Trilinear(p_par[p].pos(),
                                                              plo, dx, rho, qp);
                           });
    }
}

template <typename NSType>
void c_Nanostructure<NSType>::Obtain_PotentialAtSites()
{
    const int num_field_sites = NSType::num_field_sites;
    const int num_atoms_per_field_site = NSType::num_atoms_per_field_site;
    const int num_atoms_to_avg_over = NSType::num_atoms_to_avg_over;
    const int average_field_flag = NSType::average_field_flag;
    const int blkCol_size_loc = NSType::blkCol_size_loc;

    amrex::Gpu::DeviceVector<amrex::Real> d_vec_V(num_field_sites);
    amrex::Gpu::HostVector<amrex::Real> h_vec_V(num_field_sites);
    std::fill(d_vec_V.begin(), d_vec_V.end(), 0);
    std::fill(h_vec_V.begin(), h_vec_V.end(), 0);

    amrex::Real *p_dV = d_vec_V.dataPtr();
    amrex::Real *p_hV = h_vec_V.dataPtr();

    amrex::Print() << "\n";

    int lev = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto np = pti.numParticles();

        const auto &particles = pti.GetArrayOfStructs();
        [[maybe_unused]] const auto p_par = particles().data();

        auto &par_gather = pti.get_realPA_comp(realPA::gather);
        auto p_par_gather = par_gather.data();

        auto &par_site_id = pti.get_site_id();
        auto *p_site_id = par_site_id.data();

        if (average_field_flag)
        {
            if (NSType::avg_type == s_AVG_Type::ALL)
            {
                amrex::ParallelFor(np,
                                   [=] AMREX_GPU_DEVICE(int p) noexcept
                                   {
                                       int site_id = p_site_id[p];

                                       amrex::HostDevice::Atomic::Add(
                                           &(p_dV[site_id]), p_par_gather[p]);
                                   });
            }
            else if (NSType::avg_type == s_AVG_Type::SPECIFIC)
            {
                auto get_atom_id_at_site = NSType::get_atom_id_at_site();
#ifdef AMREX_USE_GPU
                auto avg_indices_ptr = NSType::gpuvec_avg_indices.dataPtr();
#else
                auto avg_indices_ptr = NSType::vec_avg_indices.dataPtr();
#endif
                auto par_id_offset = particle_id_offset;

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(int p) noexcept
                    {
                        int par_id_local_to_NS = p_par[p].id() - par_id_offset;

                        int site_id = p_site_id[p];

                        int atom_id_at_site =
                            get_atom_id_at_site(par_id_local_to_NS);

                        int remainder =
                            atom_id_at_site % num_atoms_per_field_site;

                        for (int m = 0; m < num_atoms_to_avg_over; ++m)
                        {
                            if (remainder == avg_indices_ptr[m])
                            {
                                amrex::HostDevice::Atomic::Add(&(p_dV[site_id]),
                                                               p_par_gather[p]);
                            }
                        }
                    });
            }
        }
        else
        {
            amrex::ParallelFor(np,
                               [=] AMREX_GPU_DEVICE(int p) noexcept
                               {
                                   int site_id = p_site_id[p];
                                   p_dV[site_id] = p_par_gather[p];
                               });
        }
    }
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_vec_V.begin(), d_vec_V.end(),
                     h_vec_V.begin());
    amrex::Gpu::streamSynchronize();
    MPI_Allreduce(MPI_IN_PLACE, &(p_hV[0]), num_field_sites, MPI_DOUBLE,
                  MPI_SUM, ParallelDescriptor::Communicator());

    auto const &h_U_loc = NSType::h_U_loc_data.table();
    for (int l = 0; l < blkCol_size_loc; ++l)
    {
        int gid = NSType::vec_blkCol_gids[l];
        h_U_loc(l) = -p_hV[gid] / num_atoms_to_avg_over;
    }
    for (int c = 0; c < NUM_CONTACTS; ++c)
    {
        NSType::U_contact[c] =
            -p_hV[NSType::global_contact_index[c]] / num_atoms_to_avg_over;
    }
    d_vec_V.clear();
    h_vec_V.clear();
    // for (int l=0; l < blkCol_size_loc; ++l)
    //{
    //     int gid = vec_blkCol_gids(l);
    //     if(h_U_loc(i) - (-1*p_V[gid]/num_atoms_to_avg_over) > 1e-8 )
    //     {
    //         amrex::Abort("h_U_loc != p_V: "
    //                 + std::to_string(i) + " " + std::to_string(h_U_loc(i)) +
    //                 " "
    //                 + std::to_string(p_V[gid]));
    //     }
    // }
}
