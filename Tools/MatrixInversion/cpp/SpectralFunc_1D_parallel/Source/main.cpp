
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Array.H>
#include <AMReX_MFParallelForC.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GpuComplex.H>
#include<AMReX_Print.H>
#include<AMReX_TableData.H>
#include <AMReX_TinyProfiler.H>
#include <AMReX_GpuUtility.H>
#include <AMReX_Config.H>
#include <AMReX_Loop.H>

#include <cmath>
#include <iomanip>
#include <math.h> 
#include<stdlib.h>
using namespace amrex;
using MatrixDType = amrex::GpuComplex<amrex::Real>;
using Matrix1D = TableData<MatrixDType, 1>;
using Matrix2D = TableData<MatrixDType, 2>;

template<typename T>
class TD;

namespace MathConst
{
    static constexpr amrex::Real pi = static_cast<amrex::Real>(3.14159265358979323846);
}

namespace PhysConst
{
    static constexpr auto q_e   = static_cast<amrex::Real>( 1.602176634e-19 );
    static constexpr auto ep0   = static_cast<amrex::Real>( 8.8541878128e-12 );
//    static constexpr auto hbar  = static_cast<amrex::Real>( 1.054571817e-34 );
}


MatrixDType get_beta(amrex::Real gamma, int M, int J) 
{
   amrex::GpuComplex arg(0., -MathConst::pi*J/M);
   return 2 * gamma * cos(-1*arg.imag()) * exp(arg); 
}


AMREX_GPU_HOST_DEVICE 
MatrixDType conjugate(MatrixDType a) 
{
   amrex::GpuComplex a_conj(a.real(), -1.*a.imag());
   return a_conj;
}


void PrintTable_loc(Matrix2D& G_blk)
{

    auto tlo = G_blk.lo();
    auto thi = G_blk.hi();
	
    auto const& G = G_blk.table();

	std::cout << "tlo: " << tlo[0] << " " << tlo[1] 
	          << "thi: " << thi[0] << " " << thi[1] << "\n";

    for (int i = tlo[0]; i < thi[0]; ++i) 
    { 
        for (int j = tlo[1]; j < thi[1]; ++j) //slow moving index. printing slow
	{ 
            std::cout << std::setw(12) << std::setprecision(2) << G(i,j);
        }
        std::cout << "\n";
    }
}


void PrintTable(Matrix2D& h_G_loc_data, 
		int N_total, 
		const amrex::Vector<int>& cumu_blk_size,
                const amrex::Vector<int>& vec_col_gids,
                const int num_proc_with_blk)
{

    /*Gatherv G_loc into G_glo*/
    amrex::Real G_locglo_gatherv_time_beg = amrex::second();

    int num_proc = ParallelDescriptor::NProcs(); 
    int my_rank = ParallelDescriptor::MyProc();
    int num_cols_loc = vec_col_gids.size();

    Matrix2D h_G_glo_data;
    Table2D<MatrixDType> h_G_glo;
    int* recv_count;
    int* disp;

    if(my_rank == ParallelDescriptor::IOProcessorNumber()) 
    {
        h_G_glo_data.resize({0,0},{N_total, N_total}, The_Pinned_Arena());
        h_G_glo = h_G_glo_data.table();
        recv_count = new int[num_proc];
        disp = new int [num_proc];
        for(int p=0; p < num_proc; ++p) {

           if(p < num_proc_with_blk) {
              recv_count[p] = cumu_blk_size[p+1] - cumu_blk_size[p];
              disp[p] = cumu_blk_size[p];
           }
           else {
              recv_count[p] = 0;
              disp[p] = 0;
           }
           //amrex::Print() << "p/recv_count/disp: " << p << " " << recv_count[p] << " " << disp[p] << "\n";
        }
    }

    MPI_Datatype column_type;

    MPI_Type_vector(1, N_total+1, N_total+1, MPI_DOUBLE_COMPLEX, &column_type);

    MPI_Type_commit(&column_type);

    auto h_G_loc = h_G_loc_data.table();

    MPI_Gatherv(&h_G_loc(0,0),
                num_cols_loc,
                column_type,
                &h_G_glo(0,0),
                recv_count,
                disp,
                column_type,
                ParallelDescriptor::IOProcessorNumber(),
                ParallelDescriptor::Communicator());
    ParallelDescriptor::Barrier();  

    if(my_rank == ParallelDescriptor::IOProcessorNumber()) 
    { 
        delete [] recv_count;
        delete [] disp;
    } 
    MPI_Type_free(&column_type);

    amrex::Real G_locglo_gatherv_time = amrex::second() - G_locglo_gatherv_time_beg;
    amrex::Print() << "\nG_locglo_gatherv_time: " << std::setw(15) << G_locglo_gatherv_time<< "\n";



    if(my_rank == ParallelDescriptor::IOProcessorNumber()) 
    {
        amrex::Print() << "G_glo:\n";
        auto tlo = h_G_glo_data.lo();
        auto thi = h_G_glo_data.hi();
            
        auto const& G = h_G_glo_data.table();

            amrex::Print() << "tlo: " << tlo[0] << " " << tlo[1] << "\n"
                           << "thi: " << thi[0] << " " << thi[1] << "\n";

        for (int i = tlo[0]; i < thi[0]; ++i) 
        { 
            for (int j = tlo[1]; j < thi[1]; ++j) //slow moving index. printing slow
           	{ 
                amrex::Print() << std::setw(12) << std::setprecision(2) << G(i,j);
            }
            amrex::Print() << "\n";
        }
    }

}


//template<std::size_t N_total>
void MatInv_BlockTriDiagonal(int N_total,
		             const Matrix1D& h_Alpha_loc_data, 
                             const Matrix1D& h_B_data, 
                             const Matrix1D& h_C_data,
                             Matrix2D& hd_G_loc_data, 
                             Matrix2D& hd_A_loc_data, 
                             const amrex::Vector<int>& cumu_blk_size,
                             const amrex::Vector<int>& vec_col_gids,
                             const int num_proc_with_blk)
{
/* Six step procedure:
 * 1) MPI_Allgatherv local Hamiltonian diagonal elements in Alpha_loc_data into Alpha_glo_data, both allocated on the PinnedArena.
 * 2) Compute on all processes, 1DTable arrays Xtil, Ytil, X, Y recursively, on the PinnedArena and then delete Alpha_glo_data. 
 * These arrays have a size equal to the column (or row) size of the full (or global) matrix to be inverted.
 * 3) All processes copy the respective portion of X and Y needed later into arrays allocated on managed memory. 
 * These portions have a size equal to the number of columns in the local matrix block. 
 * 4) Copy needed arrays on to the device. These include Xtil, Ytil, X_local, Y_local, Alpha_loc. 
 * 5) Launce parallelFor, assign each gpu-thread to compute a column of the inverted matrix. 
 * Each GPU is responsible for computing it's own block.
 * 6) Do a stream synchronise and copy the inverted matrix block on to the CPU.
 */ 

    int num_contacts = 2;
    global_contact_index[num_contacts] = {0, N_total-1};

    BL_PROFILE_VAR("steps_1_2_3", steps_1_2_3);

    /*Step 1*/	
    int num_proc = ParallelDescriptor::NProcs(); 
    int my_rank = ParallelDescriptor::MyProc();
    int num_cols_loc = vec_col_gids.size();
    int ioproc = ParallelDescriptor::IOProcessorNumber();  
    auto const& h_Alpha_loc = h_Alpha_loc_data.table();
    auto const& h_B = h_B_data.const_table();
    auto const& h_C = h_C_data.const_table();


    Matrix1D h_Alpha_glo_data({0},{N_total}, The_Pinned_Arena());
    Table1D<MatrixDType> h_Alpha_glo = h_Alpha_glo_data.table();
    int recv_count[num_proc];
    int disp[num_proc];

    for(int p=0; p < num_proc; ++p) {

       if(p < num_proc_with_blk) {
          recv_count[p] = cumu_blk_size[p+1] - cumu_blk_size[p];
          disp[p] = cumu_blk_size[p];
       }
       else {
          recv_count[p] = 0;
          disp[p] = 0;
       }
    }

    MPI_Allgatherv(&h_Alpha_loc(0),
                    num_cols_loc,
                    MPI_DOUBLE_COMPLEX,
                   &h_Alpha_glo(0),
                    recv_count,
                    disp,
                    MPI_DOUBLE_COMPLEX,
                    ParallelDescriptor::Communicator());


    /*Step 2*/	
    Matrix1D h_Xtil_glo_data({0},{N_total},The_Pinned_Arena());
    Matrix1D h_Ytil_glo_data({0},{N_total},The_Pinned_Arena());
    Matrix1D h_X_glo_data({0},{N_total},The_Pinned_Arena());
    Matrix1D h_Y_glo_data({0},{N_total},The_Pinned_Arena());
    auto const& h_Ytil_glo = h_Ytil_glo_data.table();
    auto const& h_Xtil_glo = h_Xtil_glo_data.table();
    auto const& h_Y_glo = h_Y_glo_data.table();
    auto const& h_X_glo = h_X_glo_data.table();

    h_Y_glo(0) = 0;
    for (int n = 1; n < N_total; ++n)
    {
    int p = (n-1)%2;
        h_Ytil_glo(n) = h_C(p) / ( h_Alpha_glo(n-1) - h_Y_glo(n-1) );
        h_Y_glo(n) = h_B(p) * h_Ytil_glo(n);
    }

    h_X_glo(N_total-1) = 0;
    for (int n = N_total-2; n > -1; n--)
    {
    int p = n%2;
        h_Xtil_glo(n) = h_B(p)/(h_Alpha_glo(n+1) - h_X_glo(n+1));
        h_X_glo(n) = h_C(p)*h_Xtil_glo(n);
    }

    Matrix1D h_Alpha_contact_data({0},{num_contacts},The_Pinned_Arena()); 
    for (int c = 0; c < num_contacts; ++c) /*loop over elements of a block*/
    {
        int n = global_contact_index[c];
        h_Alpha_contact(c) = h_Alpha_glo(n);
    } 

    /**deallocate A_global on the host*/
    h_Alpha_glo_data.clear();


    /*Step 3*/	
    Matrix1D h_X_loc_data({0},{num_cols_loc},The_Pinned_Arena()); 
    Matrix1D h_Y_loc_data({0},{num_cols_loc},The_Pinned_Arena()); 
    auto const& h_Y_loc = h_Y_loc_data.table();
    auto const& h_X_loc = h_X_loc_data.table();

    for (int e = 0; e < num_cols_loc; ++e) /*loop over elements of a block*/
    {
        int n = e + cumu_blk_size[my_rank]; 
        h_Y_loc(e) = h_Y_glo(n);
        h_X_loc(e) = h_X_glo(n);
    } 

    Matrix1D h_X_contact_data({0},{num_contacts},The_Pinned_Arena()); 
    Matrix1D h_Y_contact_data({0},{num_contacts},The_Pinned_Arena()); 
    for (int c = 0; c < num_contacts; ++c) /*loop over elements of a block*/
    {
        int n = global_contact_index[c];
        h_Y_contact(c) = h_Y_glo(n);
        h_X_contact(c) = h_X_glo(n);
    } 

    /**deallocate X and Y global on the host*/
    h_X_glo_data.clear();
    h_Y_glo_data.clear();

    BL_PROFILE_VAR_STOP(steps_1_2_3);


    /*Step 4*/	

    #if (defined AMREX_USE_GPU && GPU_MATRIX_OP)
    BL_PROFILE_VAR("step4_CpuToGpuCopy_time", step4_CpuToGpuCopy_time);

    Matrix1D d_Alpha_loc_data({0},{num_cols_loc}, The_Arena()); 
    d_Alpha_loc_data.copy(h_Alpha_loc_data); 

    Matrix1D  d_Xtil_glo_data({0},{N_total}, The_Arena()); 
    Matrix1D  d_Ytil_glo_data({0},{N_total}, The_Arena()); 

    d_Xtil_glo_data.copy(h_Xtil_glo_data); 
    d_Ytil_glo_data.copy(h_Ytil_glo_data); 

    Matrix1D d_X_loc_data({0},{num_cols_loc}, The_Arena()); 
    Matrix1D d_Y_loc_data({0},{num_cols_loc}, The_Arena()); 
    d_X_loc_data.copy(h_X_loc_data); 
    d_Y_loc_data.copy(h_Y_loc_data); 


    Matrix1D d_X_contact_data({0},{num_contacts}, The_Arena()); 
    Matrix1D d_X_contact_data({0},{num_contacts}, The_Arena()); 
    Matrix1D d_Y_contact_data({0},{num_contacts}, The_Arena()); 
    d_Alpha_contact_data.copy(h_Alpha_contact_data); 
    d_X_contact_data.copy(h_X_contact_data); 
    d_Y_contact_data.copy(h_Y_contact_data); 
    
    amrex::Gpu::streamSynchronize();

    BL_PROFILE_VAR_STOP(step4_CpuToGpuCopy_time);
    #endif

    Matrix2D d_G_contact_data({0,0},{N_total,num_contacts}, The_Arena()); 
 
    /*Step 5*/	
    BL_PROFILE_VAR("step5_pFor_time", step5_pFor_time);

    #if(defined AMREX_USE_GPU && GPU_MATRIX_OP)
    auto const& Alpha    =    d_Alpha_loc_data.const_table();
    auto const& Xtil_glo =    d_Xtil_glo_data.const_table();
    auto const& Ytil_glo =    d_Ytil_glo_data.const_table();
    auto const& X        =    d_X_loc_data.const_table();
    auto const& Y        =    d_Y_loc_data.const_table();
    auto const& Alpha_contact = d_Alpha_contact_data.const_table();
    auto const& X_contact     = d_X_contact_data.const_table();
    auto const& Y_contact     = d_Y_contact_data.const_table();
    #else 
    auto const& Alpha    =    h_Alpha_loc_data.const_table();
    auto const& Xtil_glo =    h_Xtil_glo_data.const_table();
    auto const& Ytil_glo =    h_Ytil_glo_data.const_table();
    auto const& X        =    h_X_loc_data.const_table();
    auto const& Y        =    h_Y_loc_data.const_table();
    auto const& Alpha_contact = h_Alpha_contact_data.const_table();
    auto const& X_contact     = h_X_contact_data.const_table();
    auto const& Y_contact     = h_Y_contact_data.const_table();
    #endif
    auto const& G_loc    =    hd_G_loc_data.table(); //Green's function
    auto const& A_loc    =    hd_A_loc_data.table(); //Spectral function
    auto const& G_contact =   d_G_contact_data.table();

    int cumulative_columns = cumu_blk_size[my_rank]; 

    #if(defined AMREX_USE_GPU && GPU_MATRIX_OP)
    amrex::ParallelFor(num_cols_loc, [=] AMREX_GPU_DEVICE (int n) noexcept
    {
        int n_glo = n + cumulative_columns; /*global column number*/

        G_loc(n_glo,n) =  1./(Alpha(n) - X(n) - Y(n)); 

        for (int m = n_glo; m > 0; m--)
        {   
            G_loc(m-1,n) =  -Ytil_glo(m)*G_loc(m,n);
        }
        for (int m = n_glo; m < N_total-1; ++m)
        {   
            G_loc(m+1,n) = -Xtil_glo(m)*G_loc(m,n);
        }

        for (int k=0; k < num_contacts; ++k) 
        {
            int k_glo = global_contact_index[k];
            G_contact(k_glo,k) =  1./(Alpha_contact(k) - X_contact(k) - Y_contact(k)); 

            for (int m = k_glo; m >= n_glo; m--)
            {   
                G_contact(m-1,k) = -Ytil_glo(m)*G_contact(m,k);
            }
            for (int m = k_glo; m <= n_glo; ++m)
            {   
                G_contact(m+1,k) = -Xtil_glo(m)*G_contact(m,k);
            }
        }  

        A_loc(n_glo,n) = 0.; 
        for (int k=0; k < num_contacts; ++k) 
        {
            A_loc(n_glo,n) += G_contact(n_glo,k) * Gamma(k) * conjugate( G_contact(n_glo,k) );
        }
  
        for (int m = n_glo; m > 0; m--)
        {   
            A_loc(m-1,n) =  -Ytil_glo(m)*A_loc(m,n);
        }

        for (int m = n_glo; m < N_total-1; ++m)
        {   
            A_loc(m+1,n) = -Xtil_glo(m)*A_loc(m,n);
        }

    });
    #else 
    for (int n=0; n < num_cols_loc; ++n) {
        int n_glo = n + cumulative_columns; /*global column number*/

        G_loc(n_glo,n) =  1./(Alpha(n) - X(n) - Y(n)); 

        for (int m = n_glo; m > 0; m--)
        {   
            G_loc(m-1,n) =  -Ytil_glo(m)*G_loc(m,n);
        }
        for (int m = n_glo; m < N_total-1; ++m)
        {   
            G_loc(m+1,n) = -Xtil_glo(m)*G_loc(m,n);
        }
    }
    for (int n=0; n < num_cols_loc; ++n) {

        for (int m = n_glo; m > 0; m--)
        {   
            A_loc(m-1,n) =  -Ytil_glo(m)*A_loc(m,n);
        }

        for (int m = n_glo; m < N_total-1; ++m)
        {   
            A_loc(m+1,n) = -Xtil_glo(m)*A_loc(m,n);
        }
    }
    #endif 

    BL_PROFILE_VAR_STOP(step5_pFor_time);
   
    /*deallocate memory*/
    #if(defined AMREX_USE_GPU && GPU_MATRIX_OP)
    d_Alpha_loc_data.clear();
    d_X_loc_data.clear();
    d_Y_loc_data.clear();
    d_Xtil_glo_data.clear();
    d_Ytil_glo_data.clear();
    d_Alpha_contact_data.clear();
    d_X_contact_data.clear();
    d_Y_contact_data.clear();
    #endif 
    h_X_loc_data.clear();
    h_Y_loc_data.clear();
    h_Alpha_contact_data.clear();
    h_X_contact_data.clear();
    h_Y_contact_data.clear();

}

int main (int argc, char* argv[])
{

    amrex::Initialize(argc,argv);
    int num_proc = ParallelDescriptor::NProcs(); 
    int my_rank = ParallelDescriptor::MyProc();
    amrex::Print() << "total number of procs: " << num_proc << "\n";

    //Construct Tridiagonal Dummy Hamiltonian
    std::array<amrex::Real,3> point_charge_loc {0., 0., 1e-9};

    int N_total = 4; /*matrix size*/
    amrex::ParmParse pp;
    pp.query("N_total", N_total);

    //ParallelDescriptor::Bcast(&N_total, 1, ParallelDescriptor::IOProcessorNumber());
    if(my_rank == 1) {
	    std::cout << "Rank 1 reporting, total matrix size: " << N_total << "\n";
    }

    amrex::Print() << "total matrix size: " << N_total << "\n";
    const int MAX_THRESHOLD_BLK_SIZE = 40000; /*matrix size*/

    bool flag_fixed_blk_size = false;

    int max_blk_size = 2;   /*max block size, may change depending on flag_fixed_blk_size*/

    if(flag_fixed_blk_size) amrex::Print() << "max_blk_size is fixed by user\n";
    else {
         amrex::Print() << "max_blk_size is computed at run-time\n";
         max_blk_size = ceil(static_cast<amrex::Real>(N_total)/num_proc);
         if(max_blk_size > MAX_THRESHOLD_BLK_SIZE) {
            max_blk_size = MAX_THRESHOLD_BLK_SIZE;
            /*assert that use larger number of procs*/
         }
    }
    amrex::Print() << "maximum block size (in number of columns): " << max_blk_size << "\n";

    const int num_blk_glo = ceil(static_cast<amrex::Real>(N_total)/max_blk_size);
    amrex::Print() << "global number of blocks: " << num_blk_glo << "\n";
     
    const int num_proc_with_blk = num_blk_glo;
    /*if num_proc_with_blk >= num_proc, assert.*/

    amrex::Vector<int> cumu_blk_size(num_proc_with_blk + 1);

    cumu_blk_size[0] = 0;
    for(int p=1; p < num_proc_with_blk; ++p)
    {
        cumu_blk_size[p] = cumu_blk_size[p-1] + max_blk_size;
        /*note that all proc except the last proc in "all procs with blocks" have N_blk number of columns. The last proc in "all procs with blocks" may have number of columns smaller than N_blk.
        */
    }
    cumu_blk_size[num_proc_with_blk] = cumu_blk_size[num_proc_with_blk-1] 
                                     + N_total - (num_proc_with_blk-1)*max_blk_size;

    //amrex::Print() << "cumulative blk size: \n";
    //for(int p=0; p < num_proc_with_blk+1; ++p)
    //{
    //    amrex::Print() << cumu_blk_size[p] << "\n";
    //}


    amrex::Vector<int>  vec_col_gids;
   
    if(my_rank < num_proc_with_blk) 
    {

        int blk_gid = my_rank;
        int blk_size = cumu_blk_size[blk_gid+1] - cumu_blk_size[blk_gid];

        for (int c=0; c < blk_size; ++c) 
        {
            int col_gid = cumu_blk_size[blk_gid] + c;
            vec_col_gids.push_back(col_gid);
        } 

        //if(my_rank == 1) {
        //    std::cout      << "(proc) My rank: " << my_rank << "\n";
        //    std::cout << " blk_size: " << vec_col_gids.size() << "\n";
        //    std::cout << " vec_col_gids: ";
        //    for(auto& col_gid: vec_col_gids) {
        //       std::cout << std::setw(5) << col_gid << " ";
        //    } 
        //    std::cout <<  "\n ";
        //}
    }


    /*setting A, the diagonal 1D vector, which is partitioned in the form of some blocks*/
    /*in general A can be 2D*/
    int num_cols_loc = vec_col_gids.size();
    Matrix1D h_Alpha_loc_data({0},{num_cols_loc}, The_Pinned_Arena());

    /*R this is the size of repeated B and C vectors in Hamiltonian in the mode-space approximation for CNT*/
    const int R = 2; 

    /*In general, B and C are 2D blocks of different sizes 2D blocks*/
    Matrix1D h_B_data({0},{R},The_Pinned_Arena());
    Matrix1D h_C_data({0},{R},The_Pinned_Arena());
    auto const& h_B = h_B_data.table();
    auto const& h_C = h_C_data.table();

    /*specifying A block vector for CNT with point charge potential*/

    auto blk_size = vec_col_gids.size();
    int c=0;
    auto const& h_Alpha_loc = h_Alpha_loc_data.table();

    for(auto& col_gid: vec_col_gids) 
    {
        amrex::Real layer_loc_y = -10e-9 
		+ (static_cast<amrex::Real>(col_gid)/(N_total-1))*20e-9;        
        
        amrex::Real r = pow( (  pow((0.          - point_charge_loc[0]),2)
                              + pow((layer_loc_y - point_charge_loc[1]),2)
                              + pow((0.          - point_charge_loc[2]),2) ), 0.5);

        h_Alpha_loc(c) = -(PhysConst::q_e)/(4. * MathConst::pi * PhysConst::ep0 * r);

        //amrex::Print() << "(proc 0) column, A: " << col_gid << " "<< A_loc(c) << "\n";
        ++c;  
    }

    /*below are some parameters for (17,0) carbon nanotube*/
    amrex::Real gamma = 2.5; //eV
    int M=17;
    MatrixDType beta = get_beta(gamma,M,6);

    /*specifying B and C vectors for the special case of mode-space Hamiltonian of CNT*/
    for (std::size_t i = 0; i < R; ++i)
    {
       if(i%2 == 0) {
          h_B(i) = conjugate(beta);
          h_C(i) = beta;
       }
       else {
          h_B(i) = gamma;
          h_C(i) = gamma;
       } 
    }
  
    /*allocate local G*/
    #if(defined AMREX_USE_GPU && GPU_MATRIX_OP)
    Matrix2D d_G_loc_data({0,0},{N_total, num_cols_loc}, The_Arena());
    Matrix2D d_A_loc_data({0,0},{N_total, num_cols_loc}, The_Arena());
    
    MatInv_BlockTriDiagonal(N_total, h_Alpha_loc_data, h_B_data, h_C_data, d_G_loc_data, 
                            d_A_loc_data,
                            cumu_blk_size, vec_col_gids, num_proc_with_blk);   
    #else 
    Matrix2D h_G_loc_data({0,0},{N_total, num_cols_loc}, The_Pinned_Arena());
    Matrix2D h_A_loc_data({0,0},{N_total, num_cols_loc}, The_Pinned_Arena());

    MatInv_BlockTriDiagonal(N_total, h_Alpha_loc_data, h_B_data, h_C_data, h_G_loc_data, 
                            h_A_loc_data,
                            cumu_blk_size, vec_col_gids, num_proc_with_blk);   
    #endif



    int print_matrix_flag = false;
    pp.query("print_matrix", print_matrix_flag);

    if(print_matrix_flag) {
        /**copy G_loc from device to host*/
        
        #if (defined AMREX_USE_GPU && GPU_MATRIX_OP)
        BL_PROFILE_VAR("step6_CopyGpuToCpu", step6_copyGpuToCpu);
        
	amrex::Gpu::streamSynchronize();
        Matrix2D h_G_loc_data({0,0},{N_total, num_cols_loc}, The_Pinned_Arena());
        h_G_loc_data.copy(d_G_loc_data); //copy from gpu to cpu

        BL_PROFILE_VAR_STOP(step6_copyGpuToCpu);
        #endif
        
        PrintTable(h_G_loc_data,N_total,cumu_blk_size, vec_col_gids, num_proc_with_blk);
        //if(my_rank == 1) 
        //{
        //    amrex::Print() << "(rank 1) G_loc:\n";
        //    PrintTable_loc(h_G_blk_loc);
        //}
        #if (defined AMREX_USE_GPU && GPU_MATRIX_OP)
        h_G_loc_data.clear();
        #endif
    }
    

    /*deallocate memory*/
    h_Alpha_loc_data.clear();
    h_B_data.clear();
    h_C_data.clear();
    #if (defined AMREX_USE_GPU && GPU_MATRIX_OP)
    d_G_loc_data.clear();
    #else 
    h_G_loc_data.clear();
    #endif


    amrex::Finalize();

}
