#include "PointChargeContainer.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>

#include <filesystem>
#include <iomanip>
#include <limits>

#include "../../Output/Output.H"
#include "../../Solver/Transport/Transport.H"
#include "../../Utils/CodeUtils/CodeUtil.H"
#include "../../Utils/CodeUtils/MathFunctions.H"
#include "../../Utils/SelectWarpXUtils/TextMsg.H"
#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Code.H"
#include "GeometryProperties.H"

// amrex::Real c_PointChargeContainer::V0 = 0.;
// amrex::Real c_PointChargeContainer::Et = 0.;
// amrex::Real c_PointChargeContainer::charge_density =
// std::numeric_limits<double>::quiet_NaN();; std::mutex
// c_PointChargeContainer::mtx;

c_PointChargeContainer::c_PointChargeContainer(
    const amrex::Geometry &geom, const amrex::DistributionMapping &dm,
    const amrex::BoxArray &ba)
    : amrex::ParticleContainer<PCA::realPD::NUM, PCA::intPD::NUM,
                               PCA::realPA::NUM, PCA::intPA::NUM>(geom, dm, ba)
{
    Read_PointCharges();
    Define_OutputFile();
}

void c_PointChargeContainer::Define_OutputFile()
{
    if (ParallelDescriptor::IOProcessor())
    {
        auto &rCode = c_Code::GetInstance();
        auto &rOutput = rCode.get_Output();
        int step = rCode.get_step();
        std::string main_output_foldername = rOutput.get_folder_name();
        std::string pc_foldername = main_output_foldername + "/point_charge";

        // if (std::filesystem::exists(pc_foldername)) {
        //     std::cout << "Directory exists: " << pc_foldername << "\n";
        // } else {
        //     try {
        //         if (CreateDirectory(pc_foldername)) {
        //         } else {
        //             std::cout << "Failed to create directory, unknown
        //             reason.\n";
        //         }
        //     } catch (const std::filesystem::filesystem_error& e) {
        //         std::cerr << "Filesystem error: " << e.what() << "\n";
        //     }
        // }a
        UtilCreateCleanDirectory(pc_foldername, false);
        pc_stepwise_filename = pc_foldername + "/total_charge_vs_step.dat";
        outfile_pc_step.open(pc_stepwise_filename.c_str());
        outfile_pc_step << "'step', ";
#ifdef USE_TRANSPORT
        outfile_pc_step << "'Vds', 'Vgs', 'Broyden_Step', ";
#endif
        outfile_pc_step
            << "'total_charge / (q.nm^-3)', 'fractional active charge'\n";
        outfile_pc_step.close();
    }
}

void c_PointChargeContainer::Write_OutputFile()
{
    if (ParallelDescriptor::IOProcessor())
    {
        auto &rCode = c_Code::GetInstance();
        int step = rCode.get_step();
        outfile_pc_step.open(pc_stepwise_filename.c_str(), std::ios_base::app);
        outfile_pc_step << std::setw(10) << step;
#ifdef USE_TRANSPORT
        auto &rTransport = rCode.get_TransportSolver();
        const auto &biases = rTransport.Get_Vec_Biases();
        if (!biases.empty())
        {
            auto [Vds, Vgs] =
                biases[0];  // here biases on all nanostructures are equal
            outfile_pc_step << std::setw(12) << Vds << std::setw(12) << Vgs;
        }
        outfile_pc_step << std::setw(10) << rTransport.get_Broyden_Step() - 1;
#endif
        outfile_pc_step << std::setw(15) << Get_total_charge() << std::setw(15)
                        << Get_total_charge() / static_cast<amrex::Real>(
                                                    Get_total_charge_units())
                        << "\n";
        outfile_pc_step.close();
    }
}

void c_PointChargeContainer::Check_PositionBounds(
    const amrex::Vector<amrex::Vector<amrex::Real>> &particles,
    const amrex::Vector<amrex::Real> &vec_offset,
    const amrex::Vector<amrex::Real> &vec_max_bound)
{
    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();
    const auto &plo = rGprop.geom.ProbLoArray();
    const auto &dx = rGprop.geom.CellSizeArray();

    amrex::Vector<int> minID(AMREX_SPACEDIM, -1);
    amrex::Vector<int> maxID(AMREX_SPACEDIM, -1);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        minID[d] = static_cast<int>(
            amrex::Math::floor((vec_offset[d] - plo[d] - dx[d] * 0.5) / dx[d]));
        maxID[d] = static_cast<int>(amrex::Math::floor(
            (vec_max_bound[d] - plo[d] - dx[d] * 0.5) / dx[d]));
    }
    amrex::Print() << "Min index of bounding box: ( ";
    for (int v : minID) amrex::Print() << v << " ";
    amrex::Print() << ")\n";
    amrex::Print() << "Max index of bounding box: ( ";
    for (int v : maxID) amrex::Print() << v << " ";
    amrex::Print() << ")\n";

    for (int s = 0; s < particles.size(); ++s)
    {
        // check bounds
        amrex::Vector<int> ID(AMREX_SPACEDIM, -1);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            ID[d] = static_cast<int>(amrex::Math::floor(
                (particles[s][d] - plo[d] - dx[d] * 0.5) / dx[d]));
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            GeomUtils::Is_ID_Within_Bounds(ID, minID, maxID) == true,
            "Point charge " + std::to_string(s) + " is out of bounds. ID: [" +
                std::to_string(ID[0]) + ", " + std::to_string(ID[1]) + ", " +
                std::to_string(ID[2]) + "] ");
    }
}

void c_PointChargeContainer::Read_PointCharges()
{
    amrex::ParmParse pp_main("pc");

    amrex::Vector<amrex::Real> vec_offset(AMREX_SPACEDIM, 0.0);
    queryArrWithParser(pp_main, "offset", vec_offset, 0, AMREX_SPACEDIM);

    amrex::Print() << "pc.offset / (nm): ";
    for (const auto &v : vec_offset) amrex::Print() << v * 1e9 << " ";
    amrex::Print() << "\n";

    amrex::Vector<amrex::Real> vec_scaling(AMREX_SPACEDIM, 1.0);
    queryArrWithParser(pp_main, "scaling", vec_scaling, 0, AMREX_SPACEDIM);

    amrex::Print() << "pc.scaling / (nm): ";
    for (const auto &v : vec_scaling) amrex::Print() << v * 1e9 << " ";
    amrex::Print() << "\n";

    amrex::Vector<amrex::Real> vec_max_bound(AMREX_SPACEDIM, 1.0);
    amrex::Print() << "max bound / (nm): ";
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        vec_max_bound[d] = vec_offset[d] + vec_scaling[d];
        amrex::Print() << vec_max_bound[d] * 1e9 << " ";
    }
    amrex::Print() << "\n";

    pp_main.query("V0", V0);
    pp_main.query("Et", Et);
    pp_main.query("flag_vary_occupation", flag_vary_occupation);
    pp_main.query("mixing_factor", mixing_factor);

    amrex::Real default_charge_unit = 1.;
    pp_main.query("default_charge_unit", default_charge_unit);

    amrex::Real default_occupation = 1.;
    pp_main.query("default_occupation", default_occupation);

    pp_main.query("flag_random_positions", flag_random_positions);

    amrex::Print() << "pc.flag_vary_occupation: " << flag_vary_occupation
                   << "\n";
    amrex::Print() << "pc.V0: " << V0 << "\n";
    amrex::Print() << "pc.Et: " << Et << "\n";
    amrex::Print() << "pc.mixing_factor: " << mixing_factor << "\n";
    amrex::Print() << "pc.default_charge_unit: " << default_charge_unit << "\n";
    amrex::Print() << "pc.default_occupation: " << default_occupation << "\n";
    amrex::Print() << "pc.flag_random_positions: " << flag_random_positions
                   << "\n";

    int num = 0;
    int seed = std::random_device{}();
    if (flag_random_positions)
    {
        // amrex::Real density;
        // pp_main.get("charge_density", density);
        // Set_StaticVar(charge_density,density);
        pp_main.get("charge_density", charge_density);

        amrex::Print() << "pc.charge_density: " << charge_density << "\n";
        amrex::Real vol = 1.;
        for (int k = 0; k < AMREX_SPACEDIM; ++k)
        {
            if(vec_scaling[k] != 0) vol *= vec_scaling[k];
        }
        num = static_cast<int>(charge_density * vol);
        amrex::Print() << "pc.num: " << num << "\n";

        // Seed and random engine can be specified in the input file
        pp_main.query("random_seed", seed);
        amrex::Print() << "pc.random_seed: " << seed << "\n";
    }
    else
    {
        pp_main.get("num", num);
        amrex::Print() << "pc.num: " << num << "\n";
    }

    amrex::Vector<amrex::Vector<amrex::Real>> particles(
        num, amrex::Vector<amrex::Real>(AMREX_SPACEDIM, 0.0));
    amrex::Vector<amrex::Real> charge_units(num, default_charge_unit);
    amrex::Vector<amrex::Real> occupations(num, default_occupation);

    if (flag_random_positions)
    {
        MathFunctionsHost::GenerateRandomParticles(particles, num, seed,
                                                   vec_scaling, vec_offset);
    }
    else
    {
        for (int s = 0; s < num; ++s)
        {
            amrex::ParmParse pp_pc("pc_" + std::to_string(s + 1));

            amrex::Vector<amrex::Real> pos(AMREX_SPACEDIM);
            getArrWithParser(pp_pc, "location", pos, 0, AMREX_SPACEDIM);
            for (int k = 0; k < AMREX_SPACEDIM; ++k)
            {
                particles[s][k] = pos[k] * vec_scaling[k] + vec_offset[k];
            }
        }
    }

    Check_PositionBounds(particles, vec_offset, vec_max_bound);

    for (int s = 0; s < num; ++s)
    {
        amrex::ParmParse pp_pc("pc_" + std::to_string(s + 1));
        pp_pc.query("charge_unit", charge_units[s]);
        pp_pc.query("occupation", occupations[s]);
    }

    Define(particles, charge_units, occupations);
}

void c_PointChargeContainer::Define(
    const amrex::Vector<amrex::Vector<amrex::Real>> &particles,
    const amrex::Vector<amrex::Real> &charge_units,
    const amrex::Vector<amrex::Real> &occupations)
{
    if (ParallelDescriptor::IOProcessor())
    {
        for (int p = 0; p < particles.size(); ++p)
        {
            ParticleType new_particle;
            new_particle.id() = ParticleType::NextID();
            new_particle.cpu() = ParallelDescriptor::MyProc();

            for (int j = 0; j < AMREX_SPACEDIM; ++j)
            {
                new_particle.pos(j) = particles[p][j];
            }

            std::array<int, PCA::intPA::NUM> int_attribs = {};
            int_attribs[PCA::intPA::id] = new_particle.id();

            std::array<ParticleReal, PCA::realPA::NUM> real_attribs = {};
            real_attribs[PCA::realPA::charge_unit] = charge_units[p];
            real_attribs[PCA::realPA::occupation] = occupations[p];
            real_attribs[PCA::realPA::potential] = 0.0;
            real_attribs[PCA::realPA::rel_diff] =
                std::numeric_limits<double>::max();

            std::pair<int, int> key{0, 0};  // {grid_index, tile_index}
            int lev = 0;
            auto &particle_tile = GetParticles(lev)[key];

            particle_tile.push_back(new_particle);
            particle_tile.push_back_int(int_attribs);
            particle_tile.push_back_real(real_attribs);

            // Print the details of the newly added particle
            amrex::Print() << std::left << "ID: " << std::setw(8)
                           << new_particle.id() << ", Pos/(nm): (";
            for (int j = 0; j < AMREX_SPACEDIM; ++j)
            {
                amrex::Print() << std::setw(12) << std::setprecision(8)
                               << new_particle.pos(j) * 1e9;
                if (j < AMREX_SPACEDIM - 1)
                {
                    amrex::Print() << ", ";
                }
            }
            amrex::Print() << ")"
                           << ", q: " << std::setw(10) << charge_units[p]
                           << ", occup.: " << std::setw(10) << occupations[p]
                           << "\n";
        }
    }

    Redistribute();
    amrex::Print() << "Total number of particles after Redistribute: "
                   << TotalNumberOfParticles() << "\n";
    AMREX_ALWAYS_ASSERT(particles.size() == TotalNumberOfParticles());
}

void c_PointChargeContainer::Print_Container(bool print_positions)
{
    int lev = 0;
    Vector<int> particle_ids;
    Vector<amrex::Real> charge_units, occupations, potentials, rel_diff, pos_x,
        pos_y, pos_z;

    for (c_PointChargeContainer::ParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto np = pti.numParticles();
        const auto &particles = pti.GetArrayOfStructs();
        const auto *p_par = particles().data();
        const auto &soa_real = pti.get_realPA();

        for (int p = 0; p < np; ++p)
        {
            particle_ids.push_back(p_par[p].id());
            charge_units.push_back(soa_real[PCA::realPA::charge_unit][p]);
            occupations.push_back(soa_real[PCA::realPA::occupation][p]);
            potentials.push_back(soa_real[PCA::realPA::potential][p]);
            rel_diff.push_back(soa_real[PCA::realPA::rel_diff][p]);
            if (print_positions)
            {
                pos_x.push_back(p_par[p].pos(0));
                pos_y.push_back(p_par[p].pos(1));
#if AMREX_SPACEDIM == 3
                pos_z.push_back(p_par[p].pos(2));
#endif
            }
        }
    }

    // MPI Gatherv setup
    int local_num_particles = particle_ids.size();
    int total_particles = TotalNumberOfParticles();

    // Get recvcounts and calculate displacements for Gatherv
    std::vector<int> recvcounts(ParallelDescriptor::NProcs());
    ParallelDescriptor::Gather(&local_num_particles, 1, recvcounts.data(), 1,
                               ParallelDescriptor::IOProcessorNumber());

    std::vector<int> displs(ParallelDescriptor::NProcs());
    if (ParallelDescriptor::IOProcessor())
    {
        displs[0] = 0;
        for (int i = 1; i < ParallelDescriptor::NProcs(); ++i)
        {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    // allocate global vectors
    Vector<int> all_particle_ids;
    Vector<amrex::Real> all_charge_units, all_occupations, all_potentials,
        all_rel_diff, all_pos_x, all_pos_y, all_pos_z;

    if (ParallelDescriptor::IOProcessor())
    {
        all_particle_ids.resize(total_particles);
        all_charge_units.resize(total_particles);
        all_occupations.resize(total_particles);
        all_potentials.resize(total_particles);
        all_rel_diff.resize(total_particles);

        if (print_positions)
        {
            all_pos_x.resize(total_particles);
            all_pos_y.resize(total_particles);
#if AMREX_SPACEDIM == 3
            all_pos_z.resize(total_particles);
#endif
        }
    }

    // Gather data using Gatherv
    ParallelDescriptor::Gatherv(particle_ids.data(), local_num_particles,
                                all_particle_ids.data(), recvcounts, displs,
                                ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Gatherv(charge_units.data(), local_num_particles,
                                all_charge_units.data(), recvcounts, displs,
                                ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Gatherv(occupations.data(), local_num_particles,
                                all_occupations.data(), recvcounts, displs,
                                ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Gatherv(potentials.data(), local_num_particles,
                                all_potentials.data(), recvcounts, displs,
                                ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Gatherv(rel_diff.data(), local_num_particles,
                                all_rel_diff.data(), recvcounts, displs,
                                ParallelDescriptor::IOProcessorNumber());

    if (print_positions)
    {
        ParallelDescriptor::Gatherv(pos_x.data(), local_num_particles,
                                    all_pos_x.data(), recvcounts, displs,
                                    ParallelDescriptor::IOProcessorNumber());

        ParallelDescriptor::Gatherv(pos_y.data(), local_num_particles,
                                    all_pos_y.data(), recvcounts, displs,
                                    ParallelDescriptor::IOProcessorNumber());
#if AMREX_SPACEDIM == 3
        ParallelDescriptor::Gatherv(pos_z.data(), local_num_particles,
                                    all_pos_z.data(), recvcounts, displs,
                                    ParallelDescriptor::IOProcessorNumber());
#endif
    }

    // Printing at root process and computing total charge
    total_charge = 0.;
    total_charge_units = 0;
    amrex::Real THRESHOLD_REL_DIFF_TO_PRINT = 1.e-5;
    if (ParallelDescriptor::IOProcessor())
    {
        amrex::Print() << "Point Charges: \n";
        int np = all_charge_units.size();
        amrex::Print() << "number of charges: " << np << "\n";
        for (int p = 0; p < np; ++p)
        {
            if (all_rel_diff[p] > THRESHOLD_REL_DIFF_TO_PRINT)
            {
                amrex::Print()
                    << std::left << "ID: " << std::setw(6)
                    << all_particle_ids[p] << ", charge/(e): " << std::setw(6)
                    << all_charge_units[p]
                    << ", occupation: " << std::setprecision(8) << std::setw(12)
                    << all_occupations[p]
                    << ", phi/(V): " << std::setprecision(8) << std::setw(12)
                    << all_potentials[p] << ", phi_rel_diff: " << std::setw(15)
                    << std::scientific << all_rel_diff[p];

                if (print_positions)
                {
                    amrex::Print()
                        << ", Position: (" << std::setw(10) << all_pos_x[p]
                        << ", " << std::setw(10) << all_pos_y[p];
#if AMREX_SPACEDIM == 3
                    amrex::Print() << ", " << std::setw(10) << all_pos_z[p];
#endif
                    amrex::Print() << ")";
                }
                amrex::Print() << "\n";
            }
            total_charge += all_charge_units[p] * all_occupations[p];
            total_charge_units += all_charge_units[p];
        }
        amrex::Print() << "total charge: " << total_charge << "\n";
        Write_OutputFile();
    }
}

void c_PointChargeContainer::Compute_Occupation()
{
    auto &rCode = c_Code::GetInstance();
    auto &rGprop = rCode.get_GeometryProperties();

    const auto &plo = rGprop.geom.ProbLoArray();
    const auto &dx = rGprop.geom.CellSizeArray();

    int lev = 0;
    for (c_PointChargeContainer::ParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto np = pti.numParticles();
        const auto &particles = pti.GetArrayOfStructs();
        const auto *p_par = particles().data();

        const auto &soa_real = pti.get_realPA();

        auto get_V0 = Get_V0();
        auto get_Et = Get_Et();

        auto &par_potential = pti.get_potential();
        auto *p_potential = par_potential.data();

        auto &par_occupation = pti.get_occupation();
        auto *p_occupation = par_occupation.data();

        auto &par_charge_unit = pti.get_charge_unit();
        auto *p_charge_unit = par_charge_unit.data();

        auto &par_rel_diff = pti.get_relative_difference();
        auto *p_par_rel_diff = par_rel_diff.data();

        amrex::Real V0 = Get_V0();
        amrex::Real Et = Get_Et();
        // amrex::Real MF = 0.5;
        // int THRESHOLD_REL_DIFF_OCCUPATION_COMPUT = 1.e-2;
        amrex::ParallelFor(np,
                           [=] AMREX_GPU_DEVICE(int p)
                           {
                               // if(p_par_rel_diff[p] >
                               // THRESHOLD_REL_DIFF_OCCUPATION_COMPUT) {
                               amrex::Real argument =
                                   -(p_potential[p] - V0) / Et;
                               p_occupation[p] =
                                   MathFunctions::Sigmoid(argument);
                               //}
                           });
    }
}
