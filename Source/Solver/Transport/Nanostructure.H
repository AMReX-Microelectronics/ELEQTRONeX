#ifndef NANOSTRUCTURE_H_
#define NANOSTRUCTURE_H_

#include "Nanostructure_fwd.H"
#include "NEGF/CNT.H"
#include "NEGF/Graphene.H"

#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_Particles.H>

#include <string>

//Internal variables: x,y,z, id, cpu. I think id is the local id (not sure).

//Extra Particle data in Array-of-Structs form 
//real: 0
//int: 0

//Extra Particle attributes in Struct-of-Arrays form
//real: phi, charge
//int: 2 //cell_id, atom_id (from atom_id we can obtain (ring_number, azimuthal_number)

struct realPA //real Particle Attribute
{
    enum { 
          gather,
          deposit,
          NUM 
    };
};
struct intPA //integer Particle Attribute
{
    enum { 
          cid, //cell id
          NUM 
    };
};

struct realPD //real Particle Data
{
    enum {
          NUM 
    };
};

struct intPD //integer Particle Data
{
    enum {
          NUM 
    };
};

class MyParIter
    : public amrex::ParIter<realPD::NUM, intPD::NUM, realPA::NUM, intPA::NUM>
{
public:
    using amrex::ParIter<realPD::NUM, intPD::NUM, realPA::NUM, intPA::NUM>::ParIter;
    using RealVector = ParIter::ContainerType::RealVector;
    using IntVector = ParIter::ContainerType::IntVector;

    //get realPA
    const std::array<RealVector, realPA::NUM>& get_realPA () const {
        return GetStructOfArrays().GetRealData();
    }
    
    std::array<RealVector, realPA::NUM>& get_realPA () {
        return GetStructOfArrays().GetRealData();
    }
    //get realPA component
    const RealVector& get_realPA_comp (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }
    
    RealVector& get_realPA_comp (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }

    //get intPA
    const std::array<IntVector, intPA::NUM>& get_intPA () const {
        return GetStructOfArrays().GetIntData();
    }
    
    std::array<IntVector, intPA::NUM>& get_intPA () {
        return GetStructOfArrays().GetIntData();
    }
    //get intPA component
    const IntVector& get_intPA_comp (int comp) const {
        return GetStructOfArrays().GetIntData(comp);
    }
    
    IntVector& get_intPA_comp (int comp) {
        return GetStructOfArrays().GetIntData(comp);
    }

    //get realPA
    //std::array<> (int comp) {
    //    return GetArrayOfStructs();
    //}

};


template<typename NSType>
class 
c_Nanostructure
               : public amrex::ParticleContainer<realPD::NUM, 
                                                 intPD::NUM, 
                                                 realPA::NUM, 
                                                 intPA::NUM>, public NSType
{

public:

   ~c_Nanostructure();
    c_Nanostructure (const amrex::Geometry            & geom,
                     const amrex::DistributionMapping & dm,
                     const amrex::BoxArray            & ba,
                     const std::string ns_name,
                     const std::string NS_gather_str,
                     const std::string NS_deposit_str,
                     const amrex::Real NS_initial_deposit_value,
                     const amrex::Real NS_Broyden_frac,
                     const int use_selfconsistent_potential,
                     const int use_negf    );

    virtual void ReadNanostructureProperties () override;

    void MarkCellsWithAtoms(); 

    void Gather_MeshAttributeAtAtoms();
    void Deposit_AtomAttributeToMesh();
    void Obtain_PotentialAtSites();
    void Write_PotentialAtSites();
    void Solve_NEGF();
    void Reset();

//    const amrex::Vector<amrex::Real>* get_avg_gatherField() { &avg_gatherField };

private:
    const amrex::GpuArray<int, AMREX_SPACEDIM>* _n_cell; 
    const amrex::Geometry * _geom;

    void Fill_AtomLocations ();
    void Read_AtomLocations ();
    std::string read_filename;

    amrex::MultiFab* p_mf_gather=nullptr;
    amrex::MultiFab* p_mf_deposit=nullptr;

    //amrex::Vector<amrex::Real> avg_gatherField;

    int _use_electrostatic=0;
    int _use_selfconsistent_potential=0;
    int _use_negf=0;
    amrex::Vector<s_Position3D> pos_vec;

    void InitializeNEGF();
};

#endif