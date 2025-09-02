#include <limits>

#include "Transport.H"

using namespace amrex;

enum class s_Algorithm_Type : int
{
    broyden_second,
    simple_mixing
};

#ifdef BROYDEN_PARALLEL
void c_TransportSolver::Define_MPI_Vector_Type_and_MPI_Vector_Sum()
{
    // MPI_Type_vector(1, Broyden_Threshold_MaxStep, 2, MPI_DOUBLE_COMPLEX,
    // &MPI_Vector_Type);
    MPI_Type_contiguous(Broyden_Threshold_MaxStep, MPI_DOUBLE,
                        &MPI_Vector_Type);
    MPI_Type_commit(&MPI_Vector_Type);

    /* Note:  https://www.mpich.org/static/docs/v3.2/www3/MPI_Op_create.html
     *
     * int MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op)
     * user_fn: user defined function
     * commute: true if commutative
     * op: operation (handle)
     *
     * Also, note:
     * https://stackoverflow.com/questions/29285883/mpi-allreduce-sum-over-a-derived-datatype-vector
     */

    MPI_Op_create((MPI_User_function *)Vector_Add_Func, true, &Vector_Add);
}

void c_TransportSolver::Free_MPIDerivedDataTypes()
{
    MPI_Type_free(&MPI_Vector_Type);
    MPI_Op_free(&Vector_Add);
}

void c_TransportSolver::Define_Broyden_Partition()
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

void c_TransportSolver::Set_Broyden_Parallel()
{
    amrex::Print() << "\nSetting Broyden PARALLEL\n";

    Define_Broyden_Partition();

    Broyden_Step = 1;
    Broyden_Norm = 1.;
    Broyden_Scalar = 1.;
    Broyden_NormSum_Curr = 1.e10;
    Broyden_NormSum_Prev = 1.e10;
    Broyden_fraction = Broyden_Original_Fraction;

    h_n_curr_in_data.resize({0}, {site_size_loc_all_NS}, The_Pinned_Arena());
    SetVal_RealTable1D(h_n_curr_in_data, 0.);

    [[maybe_unused]] auto const &h_n_curr_in = h_n_curr_in_data.table();

    /*Need generalization for multiple CNTs*/
    for (int c = 0; c < vp_NS.size(); ++c)
    {
        int NS_offset = site_size_loc_cumulative[c];
        vp_NS[c]->Copy_ForBroydenInput_LocalChargeFromNanostructure(
            h_n_curr_in_data, NS_offset);

        // amrex::Print() << "Fetching h_n_curr_in for NS_id: " <<
        // vp_NS[c]->NS_Id << "\n"; for(int i=NS_offset; i< NS_offset +
        // vp_NS[c]->MPI_recv_count[my_rank]; ++i) {
        //     amrex::Print() << i << " " << h_n_curr_in(i) << "\n";
        // }
    }

#ifdef BROYDEN_SKIP_GPU_OPTIMIZATION
    h_n_curr_out_data.resize({0}, {site_size_loc_all_NS}, The_Pinned_Arena());
    h_n_prev_in_data.resize({0}, {site_size_loc_all_NS}, The_Pinned_Arena());
    h_F_curr_data.resize({0}, {site_size_loc_all_NS}, The_Pinned_Arena());
    h_delta_F_curr_data.resize({0}, {site_size_loc_all_NS}, The_Pinned_Arena());
    h_Norm_data.resize({0}, {site_size_loc_all_NS}, The_Pinned_Arena());

    SetVal_RealTable1D(h_n_curr_out_data, 0.);
    SetVal_RealTable1D(h_n_prev_in_data, 0.);
    SetVal_RealTable1D(h_F_curr_data, 0.);
    SetVal_RealTable1D(h_delta_F_curr_data, 0.);
    SetVal_RealTable1D(h_Norm_data, 0.);
#else
    d_n_curr_in_data.resize({0}, {site_size_loc_all_NS}, The_Arena());
    d_n_curr_in_data.copy(h_n_curr_in_data);

    d_n_curr_out_data.resize({0}, {site_size_loc_all_NS}, The_Arena());
    d_n_prev_in_data.resize({0}, {site_size_loc_all_NS}, The_Arena());
    d_F_curr_data.resize({0}, {site_size_loc_all_NS}, The_Arena());
    d_delta_F_curr_data.resize({0}, {site_size_loc_all_NS}, The_Arena());
    d_Norm_data.resize({0}, {site_size_loc_all_NS}, The_Arena());

    auto const &n_curr_out = d_n_curr_out_data.table();
    auto const &n_prev_in = d_n_prev_in_data.table();
    auto const &F_curr = d_F_curr_data.table();
    auto const &delta_F_curr = d_delta_F_curr_data.table();
    auto const &Norm = d_Norm_data.table();

    amrex::ParallelFor(site_size_loc_all_NS,
                       [=] AMREX_GPU_DEVICE(int site) noexcept
                       {
                           n_curr_out(site) = 0.;
                           n_prev_in(site) = 0.;
                           F_curr(site) = 0.;
                           delta_F_curr(site) = 0.;
                           Norm(site) = 0.;
                       });
    amrex::Gpu::streamSynchronize();
#endif

    switch (c_TransportSolver::map_AlgorithmType.at(Algorithm_Type))
    {
        case s_Algorithm_Type::broyden_second:
        {
            h_intermed_vector_data.resize({0}, {Broyden_Threshold_MaxStep},
                                          The_Pinned_Arena());
            SetVal_RealTable1D(h_intermed_vector_data, 0.);

            Define_MPI_Vector_Type_and_MPI_Vector_Sum();

/*Matrices*/
// We would like to compute: VmatTran (i.e. V^T) x delta_F_curr
//
// delta_F_curr is a vector of size num_field_sites_all_NS.
//
// VmatTran is a matrix of size (rows x columns): (number of iterations x
// num_field_sites_all_NS) We set a limit on number of iterations using input
// parameter, 'Broyden_Threshold_MaxStep'. Note that locally, each process
// stores only 'site_size_loc_all_NS' portion out of num_field_sites_all_NS.
//
// (VmatTran x delta_F_curr) is a vector of size Broyden_Threshold_MaxStep.
//
// For RealTable2D inner index is the fast moving index.
// So the multiplication VmatTran x delta_F_curr is  going to be fast
// if we do: VmatTran(*,iteration)*delta_F_curr(*)
// Similar considerations went into storing W, which is used in the
// multiplication, W(*,site)*VmatTran_DeltaF(*), where VmatTran_DeltaF is a
// vector of size Broyden_Threshold_MaxStep.
#ifdef BROYDEN_SKIP_GPU_OPTIMIZATION
            h_sum_vector_data.resize({0}, {site_size_loc_all_NS},
                                     The_Pinned_Arena());
            h_VmatTran_data.resize({0, 0},
                                   {site_size_loc_all_NS,
                                    Broyden_Threshold_MaxStep},
                                   The_Pinned_Arena());
            h_Wmat_data.resize({0, 0},
                               {Broyden_Threshold_MaxStep,
                                site_size_loc_all_NS},
                               The_Pinned_Arena());

            SetVal_RealTable1D(h_sum_vector_data, 0.);
            SetVal_RealTable2D(h_VmatTran_data, 0.);
            SetVal_RealTable2D(h_Wmat_data, 0.);
#else
            d_sum_vector_data.resize({0}, {site_size_loc_all_NS}, The_Arena());
            d_intermed_vector_data.resize({0}, {Broyden_Threshold_MaxStep},
                                          The_Arena());

            d_VmatTran_data.resize({0, 0},
                                   {site_size_loc_all_NS,
                                    Broyden_Threshold_MaxStep},
                                   The_Arena());
            d_Wmat_data.resize({0, 0},
                               {Broyden_Threshold_MaxStep,
                                site_size_loc_all_NS},
                               The_Arena());

            auto const &sum_vector = d_sum_vector_data.table();
            auto const &intermed_vector = d_intermed_vector_data.table();
            auto const &VmatTran = d_VmatTran_data.table();
            auto const &Wmat = d_Wmat_data.table();

            const int BTM = Broyden_Threshold_MaxStep;
            const int SSL = site_size_loc_all_NS;
            amrex::ParallelFor(site_size_loc_all_NS,
                               [=] AMREX_GPU_DEVICE(int site) noexcept
                               {
                                   for (int iter = 0; iter < BTM; ++iter)
                                   {
                                       VmatTran(site, iter) = 0.;
                                       Wmat(iter, site) = 0.;
                                   }
                                   sum_vector(site) = 0.;
                                   for (int iter = site; iter < BTM;
                                        iter += SSL)
                                   {
                                       intermed_vector(iter) = 0.;
                                   }
                               });
            amrex::Gpu::streamSynchronize();
#endif

            break;
        }
        case s_Algorithm_Type::simple_mixing:
        {
            amrex::Abort(
                "Algorithm, simple_Mixing is not parallelized. Compile "
                "with preprocessor directive, BROYDEN_PARALLEL=False, or "
                "use broyden_second algorithm.");
            break;
        }
        default:
        {
            amrex::Abort("In Set_Broyden: selfconsistency_algorithm, " +
                         Algorithm_Type + ", is not yet defined.");
        }
    }

    amrex::Print() << "\nBroyden parameters are set to the following: \n";
    amrex::Print() << " Broyden_Step: " << Broyden_Step << "\n";
    amrex::Print() << " Broyden_Scalar: " << Broyden_Scalar << "\n";
    amrex::Print() << " Broyden_fraction: " << Broyden_fraction << "\n";
    amrex::Print() << " Broyden_Max_Norm: " << Broyden_Norm << "\n";
    amrex::Print() << " Broyden_NormSum_Curr: " << Broyden_NormSum_Curr << "\n";
    amrex::Print() << " Broyden_NormSum_Prev: " << Broyden_NormSum_Prev << "\n";
    amrex::Print() << " Broyden_Threshold_MaxStep: "
                   << Broyden_Threshold_MaxStep << "\n";
}

void c_TransportSolver::Reset_Broyden_Parallel()
{
    amrex::Print() << "\n\n\n\n**********************************Resetting "
                      "Broyden**********************************\n";

    if (flag_reset_with_previous_charge_distribution == 0)
    {
        SetVal_RealTable1D(h_n_curr_in_data, NS_initial_deposit_value);
#ifndef BROYDEN_SKIP_GPU_OPTIMIZATION
        d_n_curr_in_data.copy(h_n_curr_in_data);
#endif
        amrex::Print() << "Input charge distribution is reset to: "
                       << NS_initial_deposit_value << "\n";
    }
    /*else n_curr_in from previous iteration is used*/

#ifdef BROYDEN_SKIP_GPU_OPTIMIZATION
    SetVal_RealTable1D(h_n_curr_out_data, 0.);
    SetVal_RealTable1D(h_n_prev_in_data, 0.);
    SetVal_RealTable1D(h_F_curr_data, 0.);
    SetVal_RealTable1D(h_delta_F_curr_data, 0.);
    SetVal_RealTable1D(h_Norm_data, 0.);
#else
    auto const &n_curr_out = d_n_curr_out_data.table();
    auto const &n_prev_in = d_n_prev_in_data.table();
    auto const &F_curr = d_F_curr_data.table();
    auto const &delta_F_curr = d_delta_F_curr_data.table();
    auto const &Norm = d_Norm_data.table();

    amrex::ParallelFor(site_size_loc_all_NS,
                       [=] AMREX_GPU_DEVICE(int site) noexcept
                       {
                           n_curr_out(site) = 0.;
                           n_prev_in(site) = 0.;
                           F_curr(site) = 0.;
                           delta_F_curr(site) = 0.;
                           Norm(site) = 0.;
                       });
    amrex::Gpu::streamSynchronize();
#endif

    switch (c_TransportSolver::map_AlgorithmType.at(Algorithm_Type))
    {
        case s_Algorithm_Type::broyden_second:
        {
#ifdef BROYDEN_SKIP_GPU_OPTIMIZATION
            SetVal_RealTable1D(h_sum_vector_data, 0.);
            SetVal_RealTable2D(h_VmatTran_data, 0.);
            SetVal_RealTable2D(h_Wmat_data, 0.);
#else
            auto const &sum_vector = d_sum_vector_data.table();
            auto const &intermed_vector = d_intermed_vector_data.table();

            auto const &VmatTran = d_VmatTran_data.table();
            auto const &Wmat = d_Wmat_data.table();

            const int BTM = Broyden_Threshold_MaxStep;
            const int SSL = site_size_loc_all_NS;
            amrex::ParallelFor(site_size_loc_all_NS,
                               [=] AMREX_GPU_DEVICE(int site) noexcept
                               {
                                   for (int iter = 0; iter < BTM; ++iter)
                                   {
                                       VmatTran(site, iter) = 0.;
                                       Wmat(iter, site) = 0.;
                                   }
                                   sum_vector(site) = 0.;
                                   for (int iter = site; iter < BTM;
                                        iter += SSL)
                                   {
                                       intermed_vector(iter) = 0.;
                                   }
                               });
            amrex::Gpu::streamSynchronize();
#endif

            SetVal_RealTable1D(h_intermed_vector_data, 0.);

            break;
        }
        case s_Algorithm_Type::simple_mixing:
        {
            break;
        }
        default:
        {
            amrex::Abort("In Reset_Broyden: selfconsistency_algorithm, " +
                         Algorithm_Type + ", is not yet defined.");
        }
    }

    Broyden_Step = 1;
    Broyden_Norm = 1;
    Broyden_Scalar = 1.;
    Broyden_NormSum_Curr = 1.e10;
    Broyden_NormSum_Prev = 1.e10;
    Broyden_fraction = Broyden_Original_Fraction;

    amrex::Print() << "\nBroyden parameters are reset to the following: \n";
    amrex::Print() << " Broyden_Step: " << Broyden_Step << "\n";
    amrex::Print() << " Broyden_Scalar: " << Broyden_Scalar << "\n";
    amrex::Print() << " Broyden_fraction: " << Broyden_fraction << "\n";
    amrex::Print() << " Broyden_Max_Norm: " << Broyden_Norm << "\n";
    amrex::Print() << " Broyden_NormSum_Curr: " << Broyden_NormSum_Curr << "\n";
    amrex::Print() << " Broyden_NormSum_Prev: " << Broyden_NormSum_Prev << "\n";
}
#endif
