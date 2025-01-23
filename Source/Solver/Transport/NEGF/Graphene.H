#ifndef GRAPHENE_H_
#define GRAPHENE_H_

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>

#include "NEGF_Common.H"

class c_Graphene : public c_NEGF_Common<ComplexType[BLOCK_SIZE][BLOCK_SIZE]>
{
    using BlkType = ComplexType[BLOCK_SIZE][BLOCK_SIZE];

    static amrex::Array<int, 2> type_id;

   protected:
    virtual void Read_MaterialSpecificNanostructureProperties() final {}
    virtual void Set_MaterialSpecificParameters() final {}
    virtual void Print_MaterialSpecificReadData() final {}

    virtual void Construct_Hamiltonian() final {}
    virtual void Construct_ContactHamiltonian() final {}
    virtual void Define_ContactInfo() final {}
    virtual void Define_MPI_BlkType() final{};

    virtual int Compute_NumAtoms() final { return 0; }
    virtual int Compute_AtomsPerUnitcell() final { return 0; }
    virtual int Compute_NumFieldSites() final { return 0; }
    virtual int Compute_NumAtomsPerFieldSite() final { return 0; }
    virtual int Set_PrimaryTransportDir() final { return 1; /*Y*/ }
    virtual int Set_AverageFieldFlag() final { return 1; }
    virtual int Compute_NumAtomsToAvgOver() final { return 1; }
    virtual void Set_BlockDegeneracyVector(amrex::Vector<int> & vec) final {}
    virtual int get_Hsize() final { return 0; }
    virtual int get_offDiag_repeatBlkSize() final { return 1; }

    // virtual void Compute_SurfaceGreensFunction(
    //     MatrixBlock<BlkType> & gr, const ComplexType EmU) final
    //{
    // }

   public:
    static int get_1D_site_id(int par_id_local_to_NS)
    {
        return static_cast<int>(
            amrex::Math::floor((par_id_local_to_NS - 1) / type_id[0]));
    }

    struct GetAtomIDAtSite
    {
        AMREX_GPU_HOST_DEVICE int operator()(int global_par_id) const
        {
            return static_cast<int>((global_par_id - 1) % type_id[0]);
        }
        amrex::Array<int, 2> type_id;
    };

    static GetAtomIDAtSite get_atom_id_at_site()
    {
        return GetAtomIDAtSite{type_id};
    }

    virtual void Generate_AtomLocations(amrex::Vector<s_Position3D> & pos) final
    {
    }
};
#endif
