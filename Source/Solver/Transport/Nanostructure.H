#ifndef NANOSTRUCTURE_H_
#define NANOSTRUCTURE_H_

#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <string>

#include "../../Utils/CodeUtils/ParticleStructure.H"
#include "NEGF/AtomicChain.H"
#include "NEGF/CNT.H"
#include "NEGF/Graphene.H"
#include "Nanostructure_fwd.H"

class c_Nanostructure_Base
{
   public:
    using RealTable1D = TableData<amrex::Real, 1>;

    virtual ~c_Nanostructure_Base() = default;

    virtual void Set_StepFilenames(int step) = 0;
    virtual void Set_IterationFilenames(int iter) = 0;
    virtual int Get_NanostructureID() = 0;
    virtual int Get_NumFieldSites() = 0;
    virtual int Get_NumFieldSites_Local() = 0;
    virtual void Set_NumFieldSites_Local_NSOffset(int val) = 0;
    virtual int Get_Flag_WriteAtIter() = 0;

    virtual int Solve_NEGF(RealTable1D &n_curr_out_data, int iter,
                           bool flag_update_terminal_bias) = 0;

    virtual void Copy_ForBroydenInput_LocalChargeFromNanostructure(
        RealTable1D &container_data, const int NS_offset) = 0;

    virtual void Copy_BroydenPredictedChargeToNanostructure(
        const RealTable1D &) = 0;

    virtual void PostProcess(bool isIter, int iter_or_step) = 0;
    virtual void Write_Output(const bool isIter, const int iter_or_step,
                              const int total_iterations_in_step,
                              const amrex::Real avg_intg_pts,
                              RealTable1D const &h_n_curr_out_data,
                              RealTable1D const &h_Norm_data) = 0;

    virtual std::pair<int, int> Get_Biases() { return {0, 0}; }  // Default:
    virtual void Gather_PotentialAtAtoms() {}
    virtual void Deposit_ChargeDensityToMesh() {}
};

// Default trait: assume all materials need particle container.
template <typename NSType>
struct RequiresParticleContainer
{
    static constexpr bool value = NSType::uses_particle_container;
};

// Specialize for materials that do not require particle container
template <typename NSType,
          bool UseParticles = RequiresParticleContainer<NSType>::value>
class c_Nanostructure;

// Partial specialization for nanostructures that need particle container
template <typename NSType>
class c_Nanostructure<NSType, true>
    : public c_Nanostructure_Base,
      private amrex::ParticleContainer<realPD::NUM, intPD::NUM, realPA::NUM,
                                       intPA::NUM>,
      public NSType
{
    int _use_electrostatic = 0;
    int _use_negf = 0;
    const amrex::Geometry *const _geom;
    amrex::MultiFab *p_mf_gather = nullptr;
    amrex::MultiFab *p_mf_deposit = nullptr;
    const amrex::GpuArray<int, AMREX_SPACEDIM> *_n_cell;
    amrex::Vector<s_Position3D> pos_vec;
    int particle_id_offset = 0;

    void Fill_AtomLocations();
    void Read_AtomLocations();
    void Set_GatherAndDepositMultiFabs();
    amrex::Real Compute_CellVolume();

   public:
    c_Nanostructure(const amrex::Geometry &geom,
                    const amrex::DistributionMapping &dm,
                    const amrex::BoxArray &ba, const std::string NS_name,
                    const int NS_id, const std::string NS_gather_str,
                    const std::string NS_deposit_str,
                    const amrex::Real NS_initial_deposit_value,
                    const int use_negf, const std::string negf_foldername_str);

    void Gather_PotentialAtAtoms() override;

    void Deposit_ChargeDensityToMesh() override;

    void Set_StepFilenames(int step) override
    {
        NSType::Set_StepFilenameString(step);
    }

    void Set_IterationFilenames(int iter) override
    {
        NSType::Set_IterationFilenameString(iter);
    }

    int Get_NanostructureID() override { return NSType::get_NS_Id(); }

    int Get_NumFieldSites() override { return NSType::get_num_field_sites(); }

    int Get_NumFieldSites_Local() override
    {
        return NSType::get_num_field_sites_loc();
    }

    void Set_NumFieldSites_Local_NSOffset(int val) override
    {
        return NSType::set_num_field_sites_loc_NS_offset(val);
    }

    int Get_Flag_WriteAtIter() override
    {
        return NSType::get_flag_write_at_iter();
    }

    std::pair<int, int> Get_Biases() override
    {
        return std::make_pair(NSType::get_Vds(), NSType::get_Vgs());
    }

    virtual int Solve_NEGF(RealTable1D &n_curr_out_data, const int iter,
                           bool flag_update_terminal_bias) override
    {
        NSType::Solve_NEGF(n_curr_out_data, iter, flag_update_terminal_bias);
        return NSType::get_Total_Integration_Pts();
    }

    virtual void Copy_ForBroydenInput_LocalChargeFromNanostructure(
        RealTable1D &container_data, const int NS_offset) override
    {
        NSType::Copy_LocalChargeFromNanostructure(container_data, NS_offset);
    }

    virtual void Copy_BroydenPredictedChargeToNanostructure(
        const RealTable1D &h_n_curr_in_data) override
    {
        NSType::Copy_BroydenPredictedCharge(h_n_curr_in_data);
    }

    virtual void PostProcess(bool isIter, int iter_or_step) override
    {
        NSType::Compute_DensityOfStates_General(isIter, iter_or_step);
        if (!isIter) NSType::Compute_Current();
    }

    void Write_Output(const bool isIter, const int iter_or_step,
                      const int total_iterations_in_step,
                      const amrex::Real avg_intg_pts,
                      RealTable1D const &h_n_curr_out_data,
                      RealTable1D const &h_Norm_data) override
    {
        NSType::Write_Data_General(isIter, iter_or_step,
                                   total_iterations_in_step, avg_intg_pts,
                                   h_n_curr_out_data, h_Norm_data);
    }

    /* These are public because functions with lambda need public access.
     * Code needs refactoring
     */
    void Evaluate_LocalFieldSites();
    void Deposit_ZeroToMesh();
    void Obtain_PotentialAtSites();
    void Mark_CellsWithAtoms();
};

// Partial specialization for nanostructures that do not need particle container
template <typename NSType>
class c_Nanostructure<NSType, false> : public c_Nanostructure_Base,
                                       public NSType
{
    int _use_electrostatic = 0;
    int _use_negf = 0;
    amrex::Vector<s_Position3D> pos_vec;
    void Read_AtomLocations();

    // int particle_id_offset = 0;
    // const amrex::GpuArray<int, AMREX_SPACEDIM> *_n_cell;

   public:
    c_Nanostructure(const std::string NS_name, const int NS_id,
                    const amrex::Real NS_initial_deposit_value,
                    const int use_negf, const std::string negf_foldername_str);

    void Set_StepFilenames(int step) override
    {
        NSType::Set_StepFilenameString(step);
    }

    void Set_IterationFilenames(int iter) override
    {
        NSType::Set_IterationFilenameString(iter);
    }

    int Get_NanostructureID() override { return NSType::get_NS_Id(); }

    int Get_NumFieldSites() override { return NSType::get_num_field_sites(); }

    int Get_NumFieldSites_Local() override
    {
        return NSType::get_num_field_sites_loc();
    }

    void Set_NumFieldSites_Local_NSOffset(int val) override
    {
        return NSType::set_num_field_sites_loc_NS_offset(val);
    }

    int Get_Flag_WriteAtIter() override
    {
        return NSType::get_flag_write_at_iter();
    }

    virtual int Solve_NEGF(RealTable1D &n_curr_out_data, const int iter,
                           bool flag_update_terminal_bias) override
    {
        NSType::Solve_NEGF(n_curr_out_data, iter, flag_update_terminal_bias);
        return NSType::get_Total_Integration_Pts();
    }

    virtual void Copy_ForBroydenInput_LocalChargeFromNanostructure(
        RealTable1D &container_data, const int NS_offset) override
    {
        NSType::Copy_LocalChargeFromNanostructure(container_data, NS_offset);
    }

    virtual void Copy_BroydenPredictedChargeToNanostructure(
        const RealTable1D &h_n_curr_in_data) override
    {
        NSType::Copy_BroydenPredictedCharge(h_n_curr_in_data);
    }

    virtual void PostProcess(bool isIter, int iter_or_step) override
    {
        NSType::Compute_DensityOfStates_General(isIter, iter_or_step);
        if (!isIter) NSType::Compute_Current();
    }

    void Write_Output(const bool isIter, const int iter_or_step,
                      const int total_iterations_in_step,
                      const amrex::Real avg_intg_pts,
                      RealTable1D const &h_n_curr_out_data,
                      RealTable1D const &h_Norm_data) override
    {
        NSType::Write_Data_General(isIter, iter_or_step,
                                   total_iterations_in_step, avg_intg_pts,
                                   h_n_curr_out_data, h_Norm_data);
    }
};
#endif
