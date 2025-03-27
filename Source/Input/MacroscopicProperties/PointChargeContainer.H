#ifndef POINT_CHARGE_CONTAINER_H_
#define POINT_CHARGE_CONTAINER_H_

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>

#include <limits>
#include <mutex>

struct PointChargeAttributes
{
    struct realPA
    {
        enum
        {
            charge_unit,
            occupation,
            potential,
            rel_diff,
            NUM
        };
    };
    struct intPA
    {
        enum
        {
            id,
            NUM
        };
    };
    struct realPD
    {
        enum
        {
            NUM
        };
    };
    struct intPD
    {
        enum
        {
            NUM
        };
    };
};
using PCA = PointChargeAttributes;

class c_PointChargeContainer
    : public amrex::ParticleContainer<PCA::realPD::NUM, PCA::intPD::NUM,
                                      PCA::realPA::NUM, PCA::intPA::NUM>
{
   public:
    using ParticleContainerType =
        amrex::ParticleContainer<PCA::realPD::NUM, PCA::intPD::NUM,
                                 PCA::realPA::NUM, PCA::intPA::NUM>;

    // using ParticleType = typename ParticleContainerType::ParticleType;

    c_PointChargeContainer(const amrex::Geometry &geom,
                           const amrex::DistributionMapping &dm,
                           const amrex::BoxArray &ba);

    void Define(const amrex::Vector<amrex::Vector<amrex::Real>> &particles,
                const amrex::Vector<amrex::Real> &charge_units = {},
                const amrex::Vector<amrex::Real> &occupations = {});

    amrex::Real Get_V0() { return V0; }
    amrex::Real Get_Et() { return Et; }
    amrex::Real Get_mixing_factor() { return mixing_factor; }
    amrex::Real Get_Charge_Density() { return charge_density; }
    amrex::Real Get_total_charge() { return total_charge; }
    amrex::Real Get_total_charge_units() { return total_charge_units; }

    void Print_Container(bool print_positions = false);

    void Compute_Occupation();

    bool Is_Occupation_Variable() { return flag_vary_occupation; }

    void Define_OutputFile();
    void Write_OutputFile();

    // Define the iterator class to iterate over the particles
    class ParIter : public amrex::ParIter<PCA::realPD::NUM, PCA::intPD::NUM,
                                          PCA::realPA::NUM, PCA::intPA::NUM>
    {
       public:
        using amrex::ParIter<PCA::realPD::NUM, PCA::intPD::NUM,
                             PCA::realPA::NUM, PCA::intPA::NUM>::ParIter;

        using RealVector =
            amrex::ParIter<PCA::realPD::NUM, PCA::intPD::NUM, PCA::realPA::NUM,
                           PCA::intPA::NUM>::ContainerType::RealVector;

        // Get realPA
        const std::array<RealVector, PCA::realPA::NUM> &get_realPA() const
        {
            return this->GetStructOfArrays().GetRealData();
        }

        std::array<RealVector, PCA::realPA::NUM> &get_realPA()
        {
            return this->GetStructOfArrays().GetRealData();
        }

        const RealVector &get_charge_unit() const
        {
            return this->GetStructOfArrays().GetRealData(
                PCA::realPA::charge_unit);
        }

        const RealVector &get_potential() const
        {
            return this->GetStructOfArrays().GetRealData(
                PCA::realPA::potential);
        }

        const RealVector &get_occupation() const
        {
            return this->GetStructOfArrays().GetRealData(
                PCA::realPA::occupation);
        }

        RealVector &get_charge_unit()
        {
            return this->GetStructOfArrays().GetRealData(
                PCA::realPA::charge_unit);
        }

        RealVector &get_potential()
        {
            return this->GetStructOfArrays().GetRealData(
                PCA::realPA::potential);
        }

        RealVector &get_occupation()
        {
            return this->GetStructOfArrays().GetRealData(
                PCA::realPA::occupation);
        }

        RealVector &get_relative_difference()
        {
            return this->GetStructOfArrays().GetRealData(PCA::realPA::rel_diff);
        }
    };

   private:
    // static std::mutex mtx;

    bool flag_vary_occupation = false;
    bool flag_random_positions = false;
    amrex::Real charge_density = 0.;
    amrex::Real V0 = 0.;
    amrex::Real Et = 0.;
    amrex::Real mixing_factor = 0.;

    std::ofstream outfile_pc_step;
    std::string pc_stepwise_filename;
    amrex::Real total_charge = 0.;
    int total_charge_units = 0;

    void Read_PointCharges();
    void Check_PositionBounds(
        const amrex::Vector<amrex::Vector<amrex::Real>> &particles,
        const amrex::Vector<amrex::Real> &vec_offset,
        const amrex::Vector<amrex::Real> &vec_max_bound);

    std::string Get_PointChargeDump_Filename();
};

#endif
