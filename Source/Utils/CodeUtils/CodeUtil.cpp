#include <CodeUtil.H>
#include <sys/stat.h>

#include "../../Code.H"
#include "../../Input/GeometryProperties/GeometryProperties.H"
#include "../../Input/MacroscopicProperties/MacroscopicProperties.H"
#include "../../PostProcessor/PostProcessor.H"

using namespace amrex;

void Multifab_Manipulation::InitializeMacroMultiFabUsingParser_3vars(
    amrex::MultiFab *macro_mf, amrex::ParserExecutor<3> const &macro_parser,
    amrex::Geometry &geom)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::"
           "InitializeMacroMultiFabUsingParser************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    auto dx = geom.CellSizeArray();
    // amrex::Print() << "dx: " << dx[0] << " " << dx[1] << " " << dx[2] <<
    // "\n";

    auto &real_box = geom.ProbDomain();
    // amrex::Print() << "real_box_lo: " << real_box.lo(0) << " " <<
    // real_box.lo(1) << " " << real_box.lo(2) << "\n";

    auto iv = macro_mf->ixType().toIntVect();
    // amrex::Print() << "iv: " << iv[0] << " " << iv[1] << " " << iv[2] <<
    // "\n";

    for (amrex::MFIter mfi(*macro_mf, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const auto &tb = mfi.tilebox(
            iv, macro_mf->nGrowVect()); /** initialize ghost cells in addition
                                           to valid cells. auto = amrex::Box
                                         */
        auto const &mf_array =
            macro_mf->array(mfi);  // auto = amrex::Array4<amrex::Real>

        amrex::ParallelFor(
            tb,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;

                amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;

                amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;

                mf_array(i, j, k) = macro_parser(x, y, z);
            });
    }
#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t\t}************************Multifab_Manipulation::"
           "InitializeMacroMultiFabUsingParser************************\n";
#endif
}

void Multifab_Manipulation::InitializeMacroMultiFabUsingParser_4vars(
    amrex::MultiFab *macro_mf, amrex::ParserExecutor<4> const &macro_parser,
    amrex::Geometry &geom, const amrex::Real t)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::"
           "InitializeMacroMultiFabUsingParser************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    auto dx = geom.CellSizeArray();

    auto &real_box = geom.ProbDomain();

    auto iv = macro_mf->ixType().toIntVect();

    for (amrex::MFIter mfi(*macro_mf, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const auto &tb = mfi.tilebox(
            iv, macro_mf->nGrowVect()); /** initialize ghost cells in addition
                                           to valid cells. auto = amrex::Box
                                         */
        auto const &mf_array =
            macro_mf->array(mfi);  // auto = amrex::Array4<amrex::Real>

        amrex::ParallelFor(
            tb,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;

                amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;

                amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;

                mf_array(i, j, k) = macro_parser(x, y, z, t);
            });
    }
#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t\t}************************Multifab_Manipulation::"
           "InitializeMacroMultiFabUsingParser************************\n";
#endif
}

void Multifab_Manipulation::AverageCellCenteredMultiFabToCellFaces(
    const amrex::MultiFab &cc_arr,
    std::array<amrex::MultiFab, AMREX_SPACEDIM> &face_arr)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::"
           "AverageCellCenteredMultiFabToCellFaces()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif
    for (MFIter mfi(cc_arr, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Array4<Real const> &cc = cc_arr.array(mfi);
        AMREX_D_TERM(const Array4<Real> &facex = face_arr[0].array(mfi);
                     , const Array4<Real> &facey = face_arr[1].array(mfi);
                     , const Array4<Real> &facez = face_arr[2].array(mfi););

        AMREX_D_TERM(const Box &nodal_x = mfi.nodaltilebox(0);
                     , const Box &nodal_y = mfi.nodaltilebox(1);
                     , const Box &nodal_z = mfi.nodaltilebox(2););

        amrex::ParallelFor(
            nodal_x, nodal_y, nodal_z,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { facex(i, j, k) = 0.5 * (cc(i, j, k) + cc(i - 1, j, k)); },
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { facey(i, j, k) = 0.5 * (cc(i, j, k) + cc(i, j - 1, k)); },
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { facez(i, j, k) = 0.5 * (cc(i, j, k) + cc(i, j, k - 1)); });
    }
#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t\t}************************Multifab_Manipulation::"
           "AverageCellCenteredMultiFabToCellFaces()************************\n";
#endif
}

#ifdef AMREX_USE_EB
void Multifab_Manipulation::SpecifyValueOnlyOnCutcells(amrex::MultiFab &mf,
                                                       amrex::Real const value)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::"
           "SpecifyValueOnlyOnCutcells************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    auto factory =
        dynamic_cast<amrex::EBFArrayBoxFactory const *>(&(mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    auto iv = mf.ixType().toIntVect();

    for (amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const auto &box = mfi.tilebox(iv, mf.nGrowVect());

        auto const &mf_array = mf.array(mfi);

        amrex::FabType fab_type = flags[mfi].getType(box);

        if (fab_type == amrex::FabType::regular)
        {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { mf_array(i, j, k) = amrex::Real(0.); });
        }
        else if (fab_type == amrex::FabType::covered)
        {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { mf_array(i, j, k) = amrex::Real(0.); });
        }
        else  // box contains some cutcells
        {
            auto const &vfrac_array = vfrac.const_array(mfi);

            amrex::ParallelFor(box,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   if (vfrac_array(i, j, k) > 0 and
                                       vfrac_array(i, j, k) < 1)
                                   {
                                       mf_array(i, j, k) = value;
                                   }
                               });
        }
    }

#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t\t}************************Multifab_Manipulation::"
           "SpecifyValueOnlyOnCutcells************************\n";
#endif
}

void Multifab_Manipulation::SpecifyValueOnlyOnCutcells_UsingParser_4vars(
    amrex::MultiFab &mf, amrex::ParserExecutor<4> const &macro_parser,
    const amrex::Geometry &geom, const amrex::Real t)
{
    auto factory =
        dynamic_cast<amrex::EBFArrayBoxFactory const *>(&(mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    auto iv = mf.ixType().toIntVect();
    auto dx = geom.CellSizeArray();
    auto &real_box = geom.ProbDomain();

    for (amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const auto &box = mfi.tilebox(iv, mf.nGrowVect());

        auto const &mf_array = mf.array(mfi);

        amrex::FabType fab_type = flags[mfi].getType(box);

        if (fab_type == amrex::FabType::regular)
        {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { mf_array(i, j, k) = amrex::Real(0.); });
        }
        else if (fab_type == amrex::FabType::covered)
        {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { mf_array(i, j, k) = amrex::Real(0.); });
        }
        else  // box contains some cutcells
        {
            auto const &vfrac_array = vfrac.const_array(mfi);

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    if (vfrac_array(i, j, k) > 0 and vfrac_array(i, j, k) < 1)
                    {
                        amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                        amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;

                        amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                        amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;

                        amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                        amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;

                        mf_array(i, j, k) = macro_parser(x, y, z, t);
                    }
                });
        }
    }
}

void Multifab_Manipulation::SpecifyValueOnlyOnCutcells_UsingParser_3vars(
    amrex::MultiFab &mf, amrex::ParserExecutor<3> const &macro_parser,
    const amrex::Geometry &geom)
{
    auto factory =
        dynamic_cast<amrex::EBFArrayBoxFactory const *>(&(mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    auto iv = mf.ixType().toIntVect();
    auto dx = geom.CellSizeArray();
    auto &real_box = geom.ProbDomain();

    for (amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const auto &box = mfi.tilebox(iv, mf.nGrowVect());

        auto const &mf_array = mf.array(mfi);

        amrex::FabType fab_type = flags[mfi].getType(box);

        if (fab_type == amrex::FabType::regular)
        {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { mf_array(i, j, k) = amrex::Real(0.); });
        }
        else if (fab_type == amrex::FabType::covered)
        {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { mf_array(i, j, k) = amrex::Real(0.); });
        }
        else  // box contains some cutcells
        {
            auto const &vfrac_array = vfrac.const_array(mfi);

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    if (vfrac_array(i, j, k) > 0 and vfrac_array(i, j, k) < 1)
                    {
                        amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                        amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;

                        amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                        amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;

                        amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                        amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;

                        mf_array(i, j, k) = macro_parser(x, y, z);
                    }
                });
        }
    }
}

void Multifab_Manipulation::CopyValuesIntoAMultiFabOnCutcells(
    amrex::MultiFab &target_mf, amrex::MultiFab &source_mf)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::"
           "CopyValuesIntoAMultiFabOnCutcells************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif
    /*target_mf is initialized to 0 and contains cutcell information through its
     * factory*/
    /*source_mf is a regular mf with some field information that we would like
     * to copy into target_mf*/

    auto factory =
        dynamic_cast<amrex::EBFArrayBoxFactory const *>(&(target_mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    auto iv = target_mf.ixType().toIntVect();

    for (amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const auto &box = mfi.tilebox(iv);

        auto const &target_mf_array = target_mf.array(mfi);
        auto const &source_mf_array = source_mf.array(mfi);

        amrex::FabType fab_type = flags[mfi].getType(box);
        if ((fab_type != amrex::FabType::regular) &&
            (fab_type != amrex::FabType::covered))
        {
            /*box has some cutcells*/
            auto const &vfrac_array = vfrac.const_array(mfi);

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    if (vfrac_array(i, j, k) > 0. + VFRAC_THREASHOLD and
                        vfrac_array(i, j, k) < 1. - VFRAC_THREASHOLD)
                    {
                        target_mf_array(i, j, k) = source_mf_array(i, j, k);
                    }
                });
        }
    }

#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t\t}************************Multifab_Manipulation::"
           "CopyValuesIntoAMultiFabOnCutcells************************\n";
#endif
}

amrex::Real Multifab_Manipulation::GetTotalNumberOfCutcells(amrex::MultiFab &mf)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::"
           "GetTotalNumberOfCutcells************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif
    /*mf must be created with a factory that holds cutcell info*/

    auto factory =
        dynamic_cast<amrex::EBFArrayBoxFactory const *>(&(mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const auto &box = mfi.tilebox();

        auto const &mf_array = mf.array(mfi);

        amrex::FabType fab_type = flags[mfi].getType(box);
        if ((fab_type != amrex::FabType::regular) &&
            (fab_type != amrex::FabType::covered))
        {
            /*box has some cutcells*/
            auto const &vfrac_array = vfrac.const_array(mfi);

            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple
                {
                    amrex::Real weight =
                        (vfrac_array(i, j, k) >= 0. + VFRAC_THREASHOLD and
                         vfrac_array(i, j, k) <= 1.0 - VFRAC_THREASHOLD)
                            ? 1.0
                            : 0;
                    return weight;
                });
        }
    }
    amrex::Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t\t}************************Multifab_Manipulation::"
           "GetTotalNumberOfCutcells************************\n";
#endif
}

void Multifab_Manipulation::Copy_3DCartesian_To_2DAzimuthalLongitudinal(
    const amrex::Geometry &geom, amrex::MultiFab &mf,
    amrex::Array2D<amrex::Real, 1, Max_Ncell_Azim, 1, Max_Ncell_Long,
                   amrex::Order::C> *cyl_surf_grid,
    int cyl_ncell_ring, int ncell_ncell_axis, amrex::RealArray &center,
    int dir_axial, int dir_ref2)
{
    auto factory =
        dynamic_cast<amrex::EBFArrayBoxFactory const *>(&(mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    auto dx = geom.CellSizeArray();
    auto &real_box = geom.ProbDomain();
    auto iv = mf.ixType().toIntVect();

    int dir_ref1 = Get_Third_Reference_Direction(dir_axial, dir_ref2);

    for (amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const auto &box = mfi.tilebox();

        auto const &mf_array = mf.array(mfi);

        amrex::FabType fab_type = flags[mfi].getType(box);

        if ((fab_type != amrex::FabType::regular) &&
            (fab_type != amrex::FabType::covered))
        {
            auto const &vfrac_array = vfrac.const_array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    if (vfrac_array(i, j, k) > 0. + VFRAC_THREASHOLD and
                        vfrac_array(i, j, k) < 1. - VFRAC_THREASHOLD)
                    {
                        amrex::RealArray cart;
                        amrex::Array<int, 3> index = {i, j, k};
                        amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                        cart[0] =
                            i * dx[0] + real_box.lo(0) + fac_x - center[0];

                        amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                        cart[1] =
                            j * dx[1] + real_box.lo(1) + fac_y - center[1];

                        amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                        cart[2] =
                            k * dx[2] + real_box.lo(2) + fac_z - center[2];

                        amrex::Real theta =
                            atan2(cart[dir_ref2], cart[dir_ref1]);
                    }
                });
        }
    }
}

#endif  // for AMREX_USE_EB

void PrintRunDiagnostics(amrex::Real initial_time)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************PrintRunDiagnostics()****"
                      "********************\n";
    amrex::Print() << "\tin file: " << __FILE__ << " at line: " << __LINE__
                   << "\n";
#endif

    // MultiFab memory usage
    const int IOProc = ParallelDescriptor::IOProcessorNumber();

    amrex::Long min_fab_megabytes =
        amrex::TotalBytesAllocatedInFabsHWM() / 1048576;
    amrex::Long max_fab_megabytes = min_fab_megabytes;

    ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
    ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

    amrex::Print() << "\n\tHigh-water FAB megabyte spread across MPI nodes: ["
                   << min_fab_megabytes << " ... " << max_fab_megabytes
                   << "]\n";

    min_fab_megabytes = amrex::TotalBytesAllocatedInFabs() / 1048576;
    max_fab_megabytes = min_fab_megabytes;

    ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
    ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

    amrex::Print() << "\tCurent     FAB megabyte spread across MPI nodes: ["
                   << min_fab_megabytes << " ... " << max_fab_megabytes
                   << "]\n";

    amrex::Real total_run_time = ParallelDescriptor::second() - initial_time;
    ParallelDescriptor::ReduceRealMax(total_run_time);

    amrex::Print() << "\tTotal run time " << total_run_time << " seconds\n\n";

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************PrintRunDiagnostics()********"
                      "****************\n";
#endif
}

amrex::RealArray vecToArr(amrex::Vector<amrex::Real> &vec)
{
    amrex::RealArray array;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) array[i] = vec[i];
    return array;
}

amrex::IntArray vecToArr(amrex::Vector<int> &vec)
{
    amrex::IntArray array;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) array[i] = vec[i];
    return array;
}

int Evaluate_TypeOf_MacroStr(std::string macro_str)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t{************************c_Output::Evaluate_"
                      "TypeOf_MacroStr(*)************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t";
#endif
    /** if macro_str is defined in c_MacroscopicProperties then return is 0.
     *  if macro_str is defined in c_PostProcessor then return is 1.
     **/
    int return_type = -1;
    auto &rCode = c_Code::GetInstance();
    auto &rPost = rCode.get_PostProcessor();
    auto &rMprop = rCode.get_MacroscopicProperties();

    std::map<std::string, int>::iterator it_Mprop;
    std::map<std::string, s_PostProcessMacroName::macro_name>::iterator it_Post;

    it_Mprop = rMprop.map_param_all.find(macro_str);
    it_Post = rPost.map_macro_name.find(macro_str);

    if (it_Mprop != rMprop.map_param_all.end())
    {
#ifdef PRINT_HIGH
        amrex::Print() << prt << "macro_string " << macro_str
                       << " is a part of c_MacroscopicProperties. \n";
#endif
        return_type = 0;
    }
    else if (it_Post != rPost.map_macro_name.end())
    {
#ifdef PRINT_HIGH
        amrex::Print() << prt << "macro_string " << macro_str
                       << " is a part of c_PostProcessor. \n";
#endif
        return_type = 1;
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t}************************c_Output::Evaluate_"
                      "TypeOf_MacroStr(*)************************\n";
#endif
    return return_type;
}

int Get_Third_Reference_Direction(int dir1, int dir2)
{
    int dir3;
    if ((dir1 == 0 && dir2 == 1) || (dir1 == 1 && dir2 == 0))
    {
        dir3 = 2;
    }
    else if ((dir1 == 1 && dir2 == 2) || (dir1 == 2 && dir2 == 1))
    {
        dir3 = 0;
    }
    else if ((dir1 == 0 && dir2 == 2) || (dir1 == 2 && dir2 == 0))
    {
        dir3 = 1;
    }
    return dir3;
}

void Quadrature::Gauss_Legendre(amrex::Vector<amrex::Real> &x,
                                amrex::Vector<amrex::Real> &w, const int n)
{
    /*given degree n, function returns abscissa x, and weight w, of size n,
      over interval -1 to 1*/

    amrex::Real eps = 3.0e-11;
    int m = (n + 1) / 2;
    /*roots are symmetric in the interval, so we only need to find half of
     * them*/
    amrex::Real z = 0.;

    for (int i = 1; i <= m; ++i)
    {
        z = cos(MathConst::pi * (i - 0.25) / (n + 0.5));
        amrex::Real z1 = z;
        amrex::Real pp = 0.;
        /*Starting with the above approximation to the ith root, enter the main
         * loop of refinement by Newton's method*/
        do
        {
            amrex::Real p1 = 1.0;
            amrex::Real p2 = 0.0;
            for (int j = 1; j <= n; ++j)
            {
                amrex::Real p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            /*p1 is the desired Legendre polynomial. Next, compute pp, it's
              derivative, by a standard relation involving p2, the polynomial of
              one lower order*/

            pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;

        } while (std::fabs(z - z1) > eps);

        x[i - 1] = -z;
        x[n - i] = z;
        w[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - i] = w[i - 1];
    }
}

void CreateDirectory(std::string foldername)
{
    if (ParallelDescriptor::IOProcessor())
    {
        const int dir_err =
            mkdir(foldername.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if (dir_err == -1)
        {
            amrex::Print() << "Directory exists with name: " << foldername
                           << "\n";
        }
        else
        {
            amrex::Print() << "Directory created with name: " << foldername
                           << "\n";
        }
    }
}

bool GeomUtils::Is_ID_Within_Bounds(const amrex::Vector<int> &ID,
                                    const amrex::Vector<int> &minID,
                                    const amrex::Vector<int> &maxID)
{
    bool is_inside = true;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        is_inside = is_inside && (ID[d] >= minID[d] && ID[d] <= maxID[d]);
    }
    return is_inside;
}
