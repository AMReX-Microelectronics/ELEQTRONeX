#ifndef CHARGE_DENSITY_SOURCE_H_
#define CHARGE_DENSITY_SOURCE_H_

#include "PointChargeContainer.H"
//#include "HasDeposit.H"
#include <type_traits>

template <typename ParticleContainerType = c_PointChargeContainer>
class c_ChargeDensitySource
{
   private:
    ParticleContainerType m_container;

   public:
    c_ChargeDensitySource(const amrex::Geometry &geom,
                          const amrex::DistributionMapping &dm,
                          const amrex::BoxArray &ba)
        : m_container(geom, dm, ba)
    {
    }

    void Define(const amrex::Vector<amrex::Vector<amrex::Real>> &particles)
    {
        m_container.Define(particles);
    }

    void Print_Container() { m_container.Print_Container(); }

    void Compute_Occupation() { m_container.Compute_Occupation(); }

    bool Is_Occupation_Variable()
    {
        return m_container.Is_Occupation_Variable();
    }
    amrex::Real Get_mixing_factor() { return m_container.Get_mixing_factor(); }

    void Deposit(amrex::MultiFab *const p_rho_mf);

    void Gather(amrex::MultiFab *const p_potential_mf);
};

using PointChargeSource = c_ChargeDensitySource<c_PointChargeContainer>;
#endif
