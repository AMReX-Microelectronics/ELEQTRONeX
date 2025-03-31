#ifndef TRANSPORT_H_
#define TRANSPORT_H_

#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_TableData.H>

#include <string>

#include "Nanostructure.H"
#include "Transport_fwd.H"

enum class s_NS_Type : int;
enum class s_Algorithm_Type : int;

class c_TransportSolver
{
   private:
    using RealTable1D = TableData<amrex::Real, 1>;
    using RealTable2D = TableData<amrex::Real, 2>;

    bool flag_isDefined_InitialDepositValue = false;
    bool flag_isDefined_NS_type_default = false;

    int NS_num = 0;
    int m_step = 0;
    int m_iter = 1;
    int total_iter = 0;
    int use_selfconsistent_potential = 0;
    int use_negf = 0;
    const int negf_plt_name_digits = 4;
    int flag_reset_with_previous_charge_distribution = 0;
    int total_proc;
    int my_rank;
    int num_procs_with_sites;
    int site_size_loc_all_NS = 0;

    std::vector<std::pair<amrex::Real, amrex::Real>> vec_biases;

    /*Broyden*/
    int num_field_sites_all_NS = 0;
    int Broyden_Step = 1;
    int Broyden_Threshold_MaxStep = 200;

    amrex::Real NS_initial_deposit_value = 0.;
    amrex::Real total_max_time_across_all_steps[7] = {0., 0., 0., 0.,
                                                      0., 0., 0.};
    amrex::Real Broyden_fraction = 0.1;
    amrex::Real Broyden_max_norm = 1.e-5;
    amrex::Real Broyden_Original_Fraction = 0.1;
    amrex::Real Broyden_Scalar = 1.0;
    amrex::Real Broyden_NormSum_Prev = 1.e100;
    amrex::Real Broyden_Norm = 0.;
    amrex::Real Broyden_NormSum_Curr = 0.;

    std::string NS_type_default = "";
    std::string NS_gather_field_str = "phi";
    std::string NS_deposit_field_str = "charge_density";
    std::string negf_foldername_str = "output/negf";
    std::string common_foldername_str = "output/negf/transport_common";
    std::string common_step_folder_str;
    std::string Algorithm_Type = "broyden_second";
    std::string Broyden_Norm_Type = "relative";

    static const std::map<std::string, s_NS_Type> map_NSType_enum;
    static const std::map<std::string, s_Algorithm_Type> map_AlgorithmType;
    std::map<std::string, s_Norm_Type> map_NormType = {
        {"absolute", s_Norm_Type::Absolute},
        {"relative", s_Norm_Type::Relative},
        {"Absolute", s_Norm_Type::Absolute},
        {"Relative", s_Norm_Type::Relative}};
    std::map<std::string, std::string> map_NSNameToTypeStr;

    amrex::Geometry const *_geom = nullptr;
    amrex::BoxArray const *_ba = nullptr;
    amrex::DistributionMapping const *_dm = nullptr;

    amrex::Vector<std::string> vec_NS_names;
    amrex::Vector<int> site_size_loc_cumulative;

    amrex::Vector<std::unique_ptr<c_Nanostructure_Base>> vp_NS;

    /*Tables for Broyden*/
    RealTable1D h_n_curr_in_data;
    RealTable1D h_n_curr_out_data;
    RealTable1D h_n_prev_in_data;
    RealTable1D h_F_curr_data;
    RealTable1D h_delta_F_curr_data;
    RealTable1D h_Norm_data;
#ifdef BROYDEN_PARALLEL
    RealTable1D n_curr_in_glo_data;
    RealTable1D h_intermed_vector_data;
    amrex::Gpu::HostVector<amrex::Real> h_Intermed_values_vec = {0., 0., 0.};
    /*stores Broyden_Norm, Broyden_NormSum_Curr,  Broyden_Denom*/
#ifdef BROYDEN_SKIP_GPU_OPTIMIZATION
    RealTable1D h_sum_vector_data;
    RealTable2D h_Wmat_data;
    RealTable2D h_VmatTran_data;
#else
    RealTable1D d_n_curr_in_data;
    RealTable1D d_n_curr_out_data;
    RealTable1D d_n_prev_in_data;
    RealTable1D d_F_curr_data;
    RealTable1D d_delta_F_curr_data;
    RealTable1D d_Norm_data;
    RealTable1D d_sum_vector_data;
    RealTable1D d_intermed_vector_data;

    RealTable2D d_Wmat_data;
    RealTable2D d_VmatTran_data;
    amrex::Gpu::DeviceVector<amrex::Real> d_Intermed_values_vec = {0., 0., 0.};
/*stores Broyden_Norm, Broyden_NormSum_Curr,  Broyden_Denom*/
#endif
#endif

    /***Methods***/
    void ReadData();
    void Read_NSNames(amrex::ParmParse &pp);
    void Read_NSTypes(amrex::ParmParse &pp);
    void Read_ControlFlags(amrex::ParmParse &pp);
    void Read_GatherAndDepositFields(amrex::ParmParse &pp);
    void Read_SelfConsistencyInput(amrex::ParmParse &pp);
    void Set_NEGFFolderDirectories();

    void Create_NEGFFolderDirectories();
    void Assert_GatherAndDepositFields();
    void Define_InitialDepositValue();
    template <typename NSType>
    void Create_Nanostructure(const std::string &name, const int NS_id_counter);
    int Instantiate_Materials();
    std::string Get_NS_type_str(const std::string &name);
    void Group_ChargeDepositedByAllNS();
    void Set_CommonStepFolder(const int step);
    void Perform_SelfConsistencyAlgorithm();
    void Copy_BroydenPredictedChargeToHost(int NS_id);
    void Copy_DataToBeWrittenToHost(int NS_id);
    void Reset_ForNextBiasStep();
    void Obtain_maximum_time(amrex::Real const *total_time_counter_diff);

#ifdef BROYDEN_PARALLEL
    void Deallocate_Broyden_Parallel();
    void Define_Broyden_Partition();
    void Define_MPI_Vector_Type_and_MPI_Vector_Sum();
    void Free_MPIDerivedDataTypes();

    MPI_Datatype MPI_Vector_Type;
    MPI_Op Vector_Add;

    static void Vector_Add_Func(double *A, double *B, int *veclen,
                                MPI_Datatype *dtype)
    {
        /* Note: https://www.mpich.org/static/docs/v3.2/www3/MPI_Op_create.html
         * The calling list for the user function type is
         * typedef void (MPI_User_function) ( void * A, void * B, int * veclen,
         * MPI_Datatype * ); where the operation is B[i] = A[i] op B[i], for
         * i=0,...,veclen-1. A pointer to the datatype given to the MPI
         * collective computation routine (i.e., MPI_Reduce, MPI_Allreduce,
         * MPI_Scan, or MPI_Reduce_scatter) is also passed to the user-specified
         * routine.
         */
        // double* Avec = A;
        // double* Bvec = B;
        for (int i = 0; i < *veclen; ++i)
        {
            B[i] += A[i];
        }
    }
#endif

    void SetVal_RealTable1D(RealTable1D &Tab1D_data, amrex::Real val);
    void SetVal_RealTable2D(RealTable2D &Tab2D_data, amrex::Real val);

    template <typename TableType>
    void Read_Table1D(int assert_size, TableType &Tab1D_data,
                      std::string filename);

    template <typename TableType>
    void Read_Table2D(int assert_size, TableType &Tab2D_data,
                      std::string filename);

    template <typename VectorType, typename TableType>
    void Write_Table1D(const amrex::Vector<VectorType> &Vec,
                       const TableType &Arr_data, std::string filename,
                       std::string header);

    template <typename TableType>
    void Write_Table2D(const TableData<TableType, 2> &Arr_data,
                       std::string filename, std::string header);

   public:
    c_TransportSolver();
    void InitData();
    const std::vector<std::pair<amrex::Real, amrex::Real>> &Get_Vec_Biases()
        const
    {
        return vec_biases;
    }
    void Solve(const int step, const amrex::Real time);
    void Cleanup();

/* The following methods are public because
 * __device__ lambda cannot have private or protected access within its class.
 * Need refactoring. */
#ifdef BROYDEN_PARALLEL
    void Set_Broyden_Parallel();
    void Reset_Broyden_Parallel();
    void Execute_Broyden_Modified_Second_Algorithm_Parallel();
    void Execute_Broyden_Modified_Second_Algorithm_Parallel_SkipGPU();
#endif

    amrex::Real get_Broyden_Step() { return Broyden_Step; }
};

#endif
