#ifndef CNT_H_
#define CNT_H_

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>

#include "NEGF_Common.H"

class c_CNT : public c_NEGF_Common<ComplexType[BLOCK_SIZE]>
{
    using BlkType = ComplexType[BLOCK_SIZE];

    amrex::Real acc = 1.42e-10;
    amrex::Real gamma = 2.5;
    amrex::Real R_cnt;
    int rings_per_unitcell;
    int atoms_per_ring;

    amrex::Vector<int> mode_vec;
    amrex::Vector<int> mode_degen_vec;
    amrex::Vector<amrex::Real> bandgap_vec;
    MatrixBlock<BlkType> beta;

    void Read_NanotubeParameters(amrex::ParmParse &pp_ns);
    void Set_NanotubeParameters();
    void Define_SortedModeVector();
    void Print_NanotubeParameters();
    amrex::Real Get_Bandgap_Of_Mode(int p);
    ComplexType get_beta(int J);
    s_Position3D get_AtomPosition_ZigZag_CNT(int ring_id, int atom_id);

    static amrex::Array<int, 2> type_id;
    // type_id[0]: m, type_id[1]: n for (m,n) tube

   protected:
    virtual void Read_MaterialSpecificNanostructureProperties() final;
    virtual void Set_MaterialSpecificParameters() final;
    virtual void Print_MaterialSpecificReadData() final;
    virtual void Construct_Hamiltonian() final;
    virtual void Construct_ContactHamiltonian() final;
    virtual void Define_ContactInfo() final;
    virtual void Define_MPI_BlkType() final;

    virtual int Compute_NumAtoms() final
    {
        return num_unitcells * rings_per_unitcell * atoms_per_ring;
    }

    virtual int Compute_AtomsPerUnitcell() final
    {
        return atoms_per_ring * rings_per_unitcell;
    }

    virtual int Compute_NumFieldSites() final
    {
        return num_unitcells * rings_per_unitcell;
    }

    virtual int Compute_NumAtomsPerFieldSite() final { return atoms_per_ring; }
    virtual int Set_PrimaryTransportDir() final { return 1; /*Y*/ }
    virtual int Set_AverageFieldFlag() final { return 1; }

    virtual int Compute_NumAtomsToAvgOver() final
    {
        return (avg_type == s_AVG_Type::ALL)        ? num_atoms_per_field_site
               : (avg_type == s_AVG_Type::SPECIFIC) ? vec_avg_indices.size()
                                                    : 1;
    }
    virtual void Set_BlockDegeneracyVector(amrex::Vector<int> &vec) final;

    virtual int get_Hsize() final { return rings_per_unitcell * num_unitcells; }
    virtual int get_offDiag_repeatBlkSize() final { return 2; }

    virtual void Compute_SurfaceGreensFunction(MatrixBlock<BlkType> &gr,
                                               const ComplexType EmU) final;

   public:
    static constexpr bool uses_particle_container = true;

    static int get_1D_site_id(int par_id_local_to_NS)
    {
        return static_cast<int>(
            amrex::Math::floor((par_id_local_to_NS - 1) / type_id[0]));
    }

    struct GetAtomIDAtSite
    {
        AMREX_GPU_HOST_DEVICE int operator()(int par_id_local_to_NS) const
        {
            return static_cast<int>((par_id_local_to_NS - 1) % type_id[0]);
        }
        amrex::Array<int, 2> type_id;
    };

    static GetAtomIDAtSite get_atom_id_at_site()
    {
        return GetAtomIDAtSite{type_id};
    }

    virtual void Generate_AtomLocations(amrex::Vector<s_Position3D> &pos) final;
};
#endif
