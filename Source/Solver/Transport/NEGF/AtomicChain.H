#ifndef ATOMIC_CHAIN_H_
#define ATOMIC_CHAIN_H_

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>

#include "NEGF_Common.H"

class c_AtomicChain : public c_NEGF_Common<ComplexType[BLOCK_SIZE]>
{
    using BlkType = ComplexType[BLOCK_SIZE];

    amrex::Real f_d;
    amrex::Real f_c;
    amrex::Real f_cd;
    amrex::Real m_d;
    amrex::Real m_c;
    amrex::Real spacing;
    amrex::Real omega_max;
    amrex::Real omega_scaling_factor = 1;

    void Read_AtomicChainParameters(amrex::ParmParse &pp_ns);
    // void Set_AtomicChainParameters();
    void Print_AtomicChainParameters();

    s_Position3D get_AtomPosition(int field_id);

    ComplexType Compute_zPlus(const amrex::Real omega_sq)
    {
        ComplexType zPlus(0., omega_scaling_factor *
                                  (1. - sqrt(omega_sq) / omega_max) * omega_sq);
        // ComplexType zPlus(0.,1e28);
        return zPlus;
    };

   protected:
    virtual void Read_MaterialSpecificNanostructureProperties() final;
    virtual void Set_MaterialSpecificParameters() final;
    virtual void Print_MaterialSpecificReadData() final;

    virtual void Construct_Hamiltonian() final;
    virtual void Construct_ContactHamiltonian() final;
    virtual void Define_ContactInfo() final;
    virtual void Define_MPI_BlkType() final;

    AMREX_GPU_HOST_DEVICE
    virtual ComplexType FermiFunction(ComplexType E_minus_Mu,
                                      const amrex::Real kT) final;

    virtual int Compute_NumAtoms() final { return num_unitcells; }

    virtual int Compute_AtomsPerUnitcell() final { return 1; }

    virtual int Compute_NumFieldSites() final { return num_unitcells; }

    virtual int Compute_NumAtomsPerFieldSite() final { return 1; }
    virtual int Set_PrimaryTransportDir() final { return 1; /*Y*/ }
    virtual int Set_AverageFieldFlag() final { return 0; }
    virtual int Compute_NumAtomsToAvgOver() final { return 1; }

    virtual void Set_BlockDegeneracyVector(amrex::Vector<int> &vec) final;

    virtual int get_Hsize() final { return num_unitcells; }
    virtual int get_offDiag_repeatBlkSize() final { return 1; }

    virtual void Compute_SurfaceGreensFunction(MatrixBlock<BlkType> &gr,
                                               const ComplexType EmU) final;

    virtual ComplexType Get_DOS_const_multiplier(const ComplexType E) final
    {
        // ComplexType val(sqrt(E.real())/ (MathConst::pi * spacing), 0.);
        // return val;
        return sqrt(E) / (MathConst::pi * spacing);
    };

    virtual void Define_EnergyLimits() final;
    virtual void Define_IntegrationPaths() final;

   public:
    static constexpr bool uses_particle_container = false;

    static int get_1D_site_id(int par_id_local_to_NS)
    {
        return par_id_local_to_NS;
    }

    struct GetAtomIDAtSite
    {
        AMREX_GPU_HOST_DEVICE int operator()(int par_id_local_to_NS) const
        {
            return par_id_local_to_NS;
        }
    };

    static GetAtomIDAtSite get_atom_id_at_site() { return GetAtomIDAtSite{}; }

    virtual void Generate_AtomLocations(amrex::Vector<s_Position3D> &pos) final;
};
#endif
