#include "Graphene.H"

#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "../../Utils/CodeUtils/CodeUtil.H"

void c_Graphene::Read_GrapheneParameters(amrex::ParmParse &pp)
{
    queryWithParser(pp, "acc", acc);
    queryWithParser(pp, "gamma", gamma);
    queryWithParser(pp, "num_zigzag_chains", num_zigzag_chains);
}


void c_Graphene::Read_MaterialSpecificNanostructureProperties()
{
    amrex::ParmParse pp_default("Graphene_default");
    amrex::ParmParse pp_ns(name);

    Read_GrapheneParameters(pp_default);
    Read_GrapheneParameters(pp_ns);
}


void c_Graphene::Set_MaterialSpecificParameters()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_zigzag_chains > 0,
        "Graphene_default.num_zigzag_chains must be a positive integer.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        2 * num_zigzag_chains == BLOCK_SIZE,
        "A graphene zigzag ribbon has 2*num_zigzag_chains atoms per transport "
        "slice. Set ELEQTRONeX_BLOCK_SIZE to that value when configuring the "
        "build.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        acc > 0. && gamma > 0.,
        "Graphene bond length acc and hopping gamma must be positive.");
}


void c_Graphene::Print_MaterialSpecificReadData()
{
    amrex::Print() << "##### Graphene zigzag nanoribbon parameters:\n";
    amrex::Print() << "##### acc / (m): " << acc << "\n";
    amrex::Print() << "##### gamma / (eV): " << gamma << "\n";
    amrex::Print() << "##### num_zigzag_chains: " << num_zigzag_chains << "\n";
    amrex::Print() << "##### atoms per transport slice: " << BLOCK_SIZE << "\n";
}


void c_Graphene::Set_BlockDegeneracyVector(amrex::Vector<int> &vec)
{
    vec.resize(BLOCK_SIZE);
    std::fill(vec.begin(), vec.end(), 1);
}


void c_Graphene::Define_MPI_BlkType()
{
    MPI_Type_contiguous(BLOCK_SIZE * BLOCK_SIZE, MPI_DOUBLE_COMPLEX,
                        &MPI_BlkType);
    MPI_Type_commit(&MPI_BlkType);
}


void c_Graphene::Construct_Hamiltonian()
{
    const ComplexType zero(0., 0.);
    h_minusH0 = zero;

    const int N = num_zigzag_chains;
    for (int n = 0; n < N; ++n)
    {
        const int atom_A = n;

        // Within-slice A_n--B_n bond. B_n is present for n < N-1.
        if (n + 1 < N)
        {
            const int atom_B_n = N + n + 1;
            h_minusH0.block[atom_A][atom_B_n] = gamma;
            h_minusH0.block[atom_B_n][atom_A] = gamma;
        }

        // Within-slice A_n--B_(n-1) bond.
        const int atom_B_previous = N + n;
        h_minusH0.block[atom_A][atom_B_previous] = gamma;
        h_minusH0.block[atom_B_previous][atom_A] = gamma;
    }

    auto const &h_minusHa = h_minusHa_loc_data.table();
    auto const &h_Hb = h_Hb_loc_data.table();
    auto const &h_Hc = h_Hc_loc_data.table();
    for (std::size_t i = 0; i < blkCol_size_loc; ++i)
    {
        h_minusHa(i) = h_minusH0;
    }

    // The off-diagonal blocks store -(H_hopping). A_n in this slice is
    // coupled to B_(n-1) in the next slice.
    h_Hb(0) = zero;
    h_Hc(0) = zero;
    for (int n = 0; n < N; ++n)
    {
        const int atom_A = n;
        const int atom_B_previous = N + n;
        h_Hb(0).block[atom_A][atom_B_previous] = gamma;
        h_Hc(0).block[atom_B_previous][atom_A] = gamma;
    }
}


void c_Graphene::Define_ContactInfo()
{
    global_contact_index[0] = 0;
    global_contact_index[1] = Hsize_glo - 1;
    contact_transmission_index[0] = Hsize_glo - 1;
    contact_transmission_index[1] = 0;

    auto const &h_tau = h_tau_glo_data.table();
    auto const &h_Hb = h_Hb_loc_data.table();
    auto const &h_Hc = h_Hc_loc_data.table();
    h_tau(0) = h_Hc(0);
    h_tau(1) = h_Hb(0);
}


void c_Graphene::Compute_SurfaceGreensFunctionImpl(
    MatrixBlock<BlkType> &gr, const ComplexType E, const ComplexType U,
    const bool right_lead)
{
    MatrixBlock<BlkType> epsilon = h_minusH0;
    epsilon.SetDiag(E - U);
    MatrixBlock<BlkType> epsilon_surface = epsilon;

    MatrixBlock<BlkType> alpha = h_Hb_loc_data.table()(0);
    MatrixBlock<BlkType> beta = h_Hc_loc_data.table()(0);
    if (!right_lead) std::swap(alpha, beta);

    const amrex::Real tolerance = 1.e-12 * std::max(1., gamma);
    for (int iter = 0; iter < 100; ++iter)
    {
        const MatrixBlock<BlkType> g = epsilon.Inverse();
        const MatrixBlock<BlkType> alpha_g_beta = alpha * g * beta;
        const MatrixBlock<BlkType> beta_g_alpha = beta * g * alpha;

        epsilon_surface = epsilon_surface - alpha_g_beta;
        epsilon = epsilon - alpha_g_beta - beta_g_alpha;

        MatrixBlock<BlkType> alpha_next = -1. * alpha * g * alpha;
        MatrixBlock<BlkType> beta_next = -1. * beta * g * beta;
        const amrex::Real error =
            std::sqrt(amrex::norm(alpha_next.FrobeniusNorm())) +
            std::sqrt(amrex::norm(beta_next.FrobeniusNorm()));

        alpha = alpha_next;
        beta = beta_next;
        if (error < tolerance) break;
    }

    gr = epsilon_surface.Inverse();
}


void c_Graphene::Compute_SurfaceGreensFunction(MatrixBlock<BlkType> &gr,
                                               const ComplexType E,
                                               const ComplexType U)
{
    Compute_SurfaceGreensFunctionImpl(gr, E, U, true);
}


void c_Graphene::Compute_SurfaceGreensFunctionForContact(
    MatrixBlock<BlkType> &gr, const ComplexType E, const ComplexType U,
    const int contact)
{
    Compute_SurfaceGreensFunctionImpl(gr, E, U, contact == 1);
}


void c_Graphene::Generate_AtomLocations(amrex::Vector<s_Position3D> &pos)
{
    const amrex::Real sqrt3 = std::sqrt(3.);
    const int N = num_zigzag_chains;

    const amrex::Real x_min = -0.5 * acc;
    const amrex::Real x_max = 1.5 * acc * (N - 1);
    const amrex::Real x_center = 0.5 * (x_min + x_max);
    const amrex::Real y_min = -0.5 * sqrt3 * acc;
    const amrex::Real y_max = sqrt3 * acc *
                              (num_unitcells - 1. + 0.5 * (N - 1));
    const amrex::Real y_center = 0.5 * (y_min + y_max);

    for (int cell = 0; cell < num_unitcells; ++cell)
    {
        for (int n = 0; n < N; ++n)
        {
            const int atom_A = cell * BLOCK_SIZE + n;
            const int atom_B = cell * BLOCK_SIZE + N + n;

            // A_n at (sqrt(3)a(m+n/2), 3an/2).
            pos[atom_A].dir[0] = 1.5 * acc * n - x_center;
            pos[atom_A].dir[1] = sqrt3 * acc * (cell + 0.5 * n) - y_center;

            // B_(n-1), the shifted basis row that gives zigzag terminations.
            pos[atom_B].dir[0] = 1.5 * acc * n - 0.5 * acc - x_center;
            pos[atom_B].dir[1] =
                sqrt3 * acc * (cell + 0.5 * (n - 1)) - y_center;
#if AMREX_SPACEDIM == 3
            pos[atom_A].dir[2] = 0.;
            pos[atom_B].dir[2] = 0.;
#endif
        }
    }

    c_NEGF_Common<BlkType>::Generate_AtomLocations(pos);
}
