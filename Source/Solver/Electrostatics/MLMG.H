#ifndef MLMG_H_
#define MLMG_H_

#include "MLMG_fwd.H"

#include <AMReX_REAL.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_IntVect.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>

//#include <AMReX_BaseFwd.H>



#include <string>


/**
 * Multi-Level Multi-Grid solver is used which solves for phi in (A*alpha_cc - B * div (beta_fc grad)) phi = rho
 * For Poisson solver, A and alpha_cc are set to zero.
 * see AMReX_MLLinOp.H for the defaults, accessors, and mutators
 */

template<typename T>
class TD;

class 
c_MLMGSolver
{
public: 
    amrex::Real ascalar, bscalar;
    int set_verbose;
    int max_order;
    amrex::Real  relative_tolerance, absolute_tolerance;

    amrex::LPInfo info;
    amrex::MLABecLaplacian mlabec;
    std::unique_ptr<amrex::MLMG> pMLMG;
    std::unique_ptr<amrex::MultiFab> alpha_cc; 
    std::array< amrex::MultiFab, AMREX_SPACEDIM > beta_fc; 
    c_MLMGSolver();

    void ReadData();

    void Setup_MLABecLaplacian_ForPoissonEqn();

    void Solve_PoissonEqn();

    void Compute_vecE(std::array<amrex::MultiFab, AMREX_SPACEDIM>& E);

    void Compute_vecFlux(std::array<amrex::MultiFab, AMREX_SPACEDIM>& flux);

    void AverageCellCenteredMultiFabToCellFaces(const amrex::MultiFab& cc_arr,
                                                std::array< amrex::MultiFab, AMREX_SPACEDIM >& face_arr);


//    void InitData();
    
//    amrex::MultiFab& get_mf  (std::string macro_str) {auto macro_num = param_map[macro_str]; return (*m_p_mf[macro_num]);}
//    amrex::MultiFab* get_p_mf (std::string macro_str) {auto macro_num = param_map[macro_str]; return m_p_mf[macro_num].get();}

private:
//    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_p_mf;
//    void ParseBasicDomainInput();
//    void InitializeBoxArrayAndDistributionMap ();

};

#endif