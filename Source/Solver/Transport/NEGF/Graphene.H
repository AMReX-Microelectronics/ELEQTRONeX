#ifndef GRAPHENE_H_
#define GRAPHENE_H_

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>

#include "NEGF_Common.H"

class c_Graphene : public c_NEGF_Common<ComplexType[BLOCK_SIZE][BLOCK_SIZE]>
{
    using BlkType = ComplexType[BLOCK_SIZE][BLOCK_SIZE];

    amrex::Real acc = 1.42e-10;
    amrex::Real gamma = 2.7;
    int num_zigzag_chains = 0;
    MatrixBlock<BlkType> h_minusH0;

    void Read_GrapheneParameters(amrex::ParmParse &pp);
    void Compute_SurfaceGreensFunctionImpl(MatrixBlock<BlkType> &gr,
                                           const ComplexType E,
                                           const ComplexType U,
                                           bool right_lead);

   protected:
    virtual void Read_MaterialSpecificNanostructureProperties() final;
    virtual void Set_MaterialSpecificParameters() final;
    virtual void Print_MaterialSpecificReadData() final;

    virtual void Construct_Hamiltonian() final;
    virtual void Define_ContactInfo() final;
    virtual void Define_MPI_BlkType() final;

    virtual int Compute_NumAtoms() final
    {
        return num_unitcells * BLOCK_SIZE;
    }
    virtual int Compute_AtomsPerUnitcell() final { return BLOCK_SIZE; }
    virtual int Compute_NumFieldSites() final { return num_unitcells; }
    virtual int Compute_NumAtomsPerFieldSite() final { return BLOCK_SIZE; }
    virtual int Set_PrimaryTransportDir() final { return 1; /*Y*/ }
    virtual int Set_AverageFieldFlag() final { return 1; }
    virtual int Compute_NumAtomsToAvgOver() final
    {
        return (avg_type == s_AVG_Type::ALL) ? num_atoms_per_field_site
                                             : vec_avg_indices.size();
    }
    virtual void Set_BlockDegeneracyVector(amrex::Vector<int> &vec) final;
    virtual int get_Hsize() final { return num_unitcells; }
    virtual int get_offDiag_repeatBlkSize() final { return 1; }

    virtual void Compute_SurfaceGreensFunction(MatrixBlock<BlkType> &gr,
                                               const ComplexType E,
                                               const ComplexType U) final;
    virtual void Compute_SurfaceGreensFunctionForContact(
        MatrixBlock<BlkType> &gr, const ComplexType E, const ComplexType U,
        const int contact) final;

   public:
    static int get_1D_site_id(int par_id_local_to_NS)
    {
        return (par_id_local_to_NS - 1) / BLOCK_SIZE;
    }

    struct GetAtomIDAtSite
    {
        AMREX_GPU_HOST_DEVICE int operator()(int global_par_id) const
        {
            return (global_par_id - 1) % BLOCK_SIZE;
        }
    };

    static GetAtomIDAtSite get_atom_id_at_site()
    {
        return {};
    }

    virtual void Generate_AtomLocations(
        amrex::Vector<s_Position3D> &pos) final;
};
#endif
