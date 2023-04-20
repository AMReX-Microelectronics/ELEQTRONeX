#include "Nanostructure.H"

#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
#include "../../Utils/SelectWarpXUtils/WarpXConst.H"
#include "../../Utils/SelectWarpXUtils/TextMsg.H"
#include "../../Utils/CodeUtils/CodeUtil.H"
#include "../../Code.H"
#include "../../Input/GeometryProperties/GeometryProperties.H"
#include "../../Input/MacroscopicProperties/MacroscopicProperties.H"

#include "../../Code_Definitions.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
//
#include <iostream>

template class c_Nanostructure<c_CNT>; 
template class c_Nanostructure<c_Graphene>; 
//template class c_Nanostructure<c_Silicon>; 

template<typename NSType>
c_Nanostructure<NSType>::c_Nanostructure (const amrex::Geometry            & geom,
                                  const amrex::DistributionMapping & dm,
                                  const amrex::BoxArray            & ba,
                                  const std::string NS_name_str,
                                  const std::string NS_gather_str,
                                  const std::string NS_deposit_str,
                                  const amrex::Real NS_initial_deposit_value,
                                  const int use_selfconsistent_potential)
                 : amrex::ParticleContainer<realPD::NUM, intPD::NUM, 
                                            realPA::NUM, intPA::NUM> (geom, dm, ba)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_Nanostructure() Constructor************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    NSType::name = NS_name_str;
    amrex::Print() << "Nanostructure: " << NS_name_str << "\n";
    ReadNanostructureProperties();
     
    auto& rCode = c_Code::GetInstance();
    _use_electrostatic = rCode.use_electrostatic;
    if(use_selfconsistent_potential) 
    {
        auto& rGprop = rCode.get_GeometryProperties();

        _n_cell = &rGprop.n_cell;
        _geom = &geom;

        auto& rMprop = rCode.get_MacroscopicProperties();

        p_mf_gather = rMprop.get_p_mf(NS_gather_str);  
        p_mf_deposit = rMprop.get_p_mf(NS_deposit_str);  

        gpuvec_avg_indices.resize(NSType::vec_avg_indices.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, NSType::vec_avg_indices.begin(), NSType::vec_avg_indices.end(), gpuvec_avg_indices.begin());

        amrex::Print() << "Number of layers: " << NSType::get_num_layers() << "\n";
        NSType::avg_gatherField.resize(NSType::get_num_layers());

        ReadAtomLocations();
        Redistribute(); //This function is in amrex::ParticleContainer

        MarkCellsWithAtoms();
        InitializeAttributeToDeposit(NS_initial_deposit_value);
        DepositToMesh();
    }

    _num_proc = amrex::ParallelDescriptor::NProcs();
    _my_rank = amrex::ParallelDescriptor::MyProc();

    //DefineMatrixPartition(_num_proc);
    //AllocateArrays();
    //ConstructHamiltonian();

    //NSType::DefineIntegrationPaths();
    //NSType::DefineSelfEnergy();

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_Nanostructure() Constructor************************\n";
#endif
}

template<typename NSType>
c_Nanostructure<NSType>::~c_Nanostructure ()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_Nanostructure() Destructor************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    NSType::avg_gatherField.clear();

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t}************************c_Nanostructure() Destructor************************\n";
#endif
}

template<typename NSType>
void
c_Nanostructure<NSType>:: ReadNanostructureProperties ()
{
    amrex::ParmParse pp_ns(NSType::name);
    read_filename = "__NONE__";
    pp_ns.query("read_filename", read_filename);
    amrex::Print() << "##### read_filename: " << read_filename << "\n";

    NSType::ReadNanostructureProperties();
}


template<typename NSType>
void 
c_Nanostructure<NSType>::ReadAtomLocations() 
{

    if (ParallelDescriptor::IOProcessor()) 
    {
        std::ifstream infile;
        infile.open(read_filename.c_str());
         
        if(infile.fail())
        {
            amrex::Abort("Failed to read file " + read_filename);
        }
        else
        {
            int filesize=0;
            std::string line; 
            while(infile.peek()!=EOF)
            {
                std::getline(infile, line);
                filesize++;
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(filesize == NSType::num_atoms,
            "Number of atoms, " + std::to_string(NSType::num_atoms) 
             + ", are not equal to the filesize, " + std::to_string(filesize) + " !");

            infile.seekg(0, std::ios_base::beg);
           
            std::string id[2];

            for(int i=0; i < NSType::num_atoms; ++i) 
            {
                ParticleType p;
                p.id() = ParticleType::NextID();
                p.cpu() = ParallelDescriptor::MyProc();
                
                infile >> id[0] >> id[1];

		auto get_1Dlayer_id = NSType::get_1Dlayer_id();
                int layer = get_1Dlayer_id(p.id());

		auto get_atom_in_1Dlayer_id = NSType::get_atom_in_1Dlayer_id();
                int atom_id_in_layer = get_atom_in_1Dlayer_id(p.id());

                for(int j=0; j < AMREX_SPACEDIM; ++j) {
                    infile >> p.pos(j);
                    p.pos(j) += NSType::offset[j];
                }
                
                std::array<int,intPA::NUM> int_attribs;
                int_attribs[intPA::cid]  = 0;

                std::array<ParticleReal,realPA::NUM> real_attribs;
                real_attribs[realPA::gather]  = 0.0;
                real_attribs[realPA::deposit]  = 0.0;

                std::pair<int,int> key {0,0}; //{grid_index, tile index}
                int lev=0;
                auto& particle_tile = GetParticles(lev)[key];
              
                particle_tile.push_back(p);
                particle_tile.push_back_int(int_attribs);
                particle_tile.push_back_real(real_attribs);
            }
            infile.close();
        } 
    }

}


template<typename NSType>
void 
c_Nanostructure<NSType>::MarkCellsWithAtoms() 
{
    auto& rCode = c_Code::GetInstance();
    auto& rPost = rCode.get_PostProcessor();
    auto& rMprop = rCode.get_MacroscopicProperties();
    
    const auto& plo = _geom->ProbLoArray();
    const auto dx =_geom->CellSizeArray();

    auto& mf = rMprop.get_mf("atom_locations"); //define in macroscopic.fields_to_define
    
    int lev = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) 
    { 
        auto np = pti.numParticles();

        const auto& particles = pti.GetArrayOfStructs();
        const auto p_par = particles().data();

        auto mf_arr = mf.array(pti);
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int p) noexcept 
        {
            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> 
                   l = { AMREX_D_DECL((p_par[p].pos(0) - plo[0])/dx[0], 
                                      (p_par[p].pos(1) - plo[1])/dx[1], 
                                      (p_par[p].pos(2) - plo[2])/dx[2]) };

            amrex::GpuArray<int,AMREX_SPACEDIM> 
               index = { AMREX_D_DECL(static_cast<int>(amrex::Math::floor(l[0])), 
                                      static_cast<int>(amrex::Math::floor(l[1])), 
                                      static_cast<int>(amrex::Math::floor(l[2]))) };

            mf_arr(index[0],index[1],index[2]) = -1;                            
        });
    }
    mf.FillBoundary(_geom->periodicity());
   
}


template<typename NSType>
void 
c_Nanostructure<NSType>::GatherFromMesh() 
{
    
    const auto& plo = _geom->ProbLoArray();
    const auto dx =_geom->CellSizeArray();

    int lev = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) 
    { 
        auto np = pti.numParticles();

        const auto& particles = pti.GetArrayOfStructs();
        const auto p_par = particles().data();

        auto& par_gather  = pti.get_realPA_comp(realPA::gather);
        auto p_par_gather = par_gather.data();

        auto mesh_mf_arr = p_mf_gather->array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int p) noexcept 
        {
            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> 
                   l = { AMREX_D_DECL((p_par[p].pos(0) - plo[0])/dx[0], 
                                      (p_par[p].pos(1) - plo[1])/dx[1], 
                                      (p_par[p].pos(2) - plo[2])/dx[2]) };

            amrex::GpuArray<int,AMREX_SPACEDIM> 
               index = { AMREX_D_DECL(static_cast<int>(amrex::Math::floor(l[0])), 
                                      static_cast<int>(amrex::Math::floor(l[1])), 
                                      static_cast<int>(amrex::Math::floor(l[2]))) };

            p_par_gather[p] = mesh_mf_arr(index[0],index[1],index[2]);
        });
    }
}


template<typename NSType>
void 
c_Nanostructure<NSType>::DepositToMesh() 
{
    const auto& plo = _geom->ProbLoArray();
    const auto dx =_geom->CellSizeArray();

    int lev = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) 
    { 
        auto np = pti.numParticles();

        const auto& particles = pti.GetArrayOfStructs();
        const auto p_par = particles().data();

        const auto& par_deposit  = pti.get_realPA_comp(realPA::deposit);
        const auto p_par_deposit = par_deposit.data();

        auto mesh_mf_arr = p_mf_deposit->array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int p) noexcept 
        {
            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> 
                   l = { AMREX_D_DECL((p_par[p].pos(0) - plo[0])/dx[0], 
                                      (p_par[p].pos(1) - plo[1])/dx[1], 
                                      (p_par[p].pos(2) - plo[2])/dx[2]) };

            amrex::GpuArray<int,AMREX_SPACEDIM> 
               index = { AMREX_D_DECL(static_cast<int>(amrex::Math::floor(l[0])), 
                                      static_cast<int>(amrex::Math::floor(l[1])), 
                                      static_cast<int>(amrex::Math::floor(l[2]))) };

            mesh_mf_arr(index[0],index[1],index[2]) = p_par_deposit[p];
        });
    }
}


template<typename NSType>
void 
c_Nanostructure<NSType>::InitializeAttributeToDeposit(const amrex::Real value) 
{
    const auto& plo = _geom->ProbLoArray();
    const auto dx =_geom->CellSizeArray();

    int lev = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) 
    { 
        auto np = pti.numParticles();

        auto& par_deposit  = pti.get_realPA_comp(realPA::deposit);
        auto p_par_deposit = par_deposit.data();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int p) noexcept 
        {
            p_par_deposit[p] = value;
        });
    }
}


template<typename NSType>
void 
c_Nanostructure<NSType>::AverageFieldGatheredFromMesh() 
{

    const int num_layers = NSType::get_num_layers();
    const int atoms_per_layer = NSType::get_num_atoms_per_layer();
    int atoms_to_avg_over = atoms_per_layer;

    if(NSType::avg_type == s_AVG_TYPE::SPECIFIC) 
    {
        atoms_to_avg_over = NSType::vec_avg_indices.size();
    }

    amrex::Gpu::DeviceVector<amrex::Real> vec_sum_gatherField(num_layers);
    //amrex::Gpu::DeviceVector<amrex::Real> vec_axial_loc(num_layers);
    for (int l=0; l<num_layers; ++l) 
    {
        vec_sum_gatherField[l] = 0.;
        //vec_axial_loc[l] = 0.;
    }
    amrex::Real* p_sum = vec_sum_gatherField.dataPtr();  
    //amrex::Real* p_axial_loc = vec_sum_axial_loc.dataPtr();  
    
    int lev = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto np = pti.numParticles();

        const auto& particles = pti.GetArrayOfStructs();
        const auto p_par = particles().data();

        auto& par_gather  = pti.get_realPA_comp(realPA::gather);
        auto p_par_gather = par_gather.data();
        auto get_1Dlayer_id = NSType::get_1Dlayer_id();

        if(NSType::avg_type == s_AVG_TYPE::ALL) 
        {
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int p) noexcept 
            {
                int global_id = p_par[p].id();
                int layer_id = get_1Dlayer_id(global_id); 
                amrex::HostDevice::Atomic::Add(&(p_sum[layer_id]), p_par_gather[p]);
                //amrex::HostDevice::Atomic::Add(&(p_axial_loc[layer_id]), p_par[p].pos(1));

            });
        }
        else if(NSType::avg_type == s_AVG_TYPE::SPECIFIC) 
        {
            auto get_atom_in_1Dlayer_id = NSType::get_atom_in_1Dlayer_id();
	    auto avg_indices_ptr = gpuvec_avg_indices.dataPtr();
            int size = NSType::vec_avg_indices.size(); //size of index to average

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int p) noexcept 
            {
                int global_id = p_par[p].id();
                int layer_id = get_1Dlayer_id(global_id); 
                int atom_id_in_layer = get_atom_in_1Dlayer_id(global_id); 
                
                int remainder = atom_id_in_layer%atoms_per_layer;

                //for(auto index: gpuvec_avg_indices) 
		for(int m=0; m < size; ++m)
		{
                    if(remainder == avg_indices_ptr[m]) 
		    {
                        amrex::HostDevice::Atomic::Add(&(p_sum[layer_id]), p_par_gather[p]);
                        //amrex::HostDevice::Atomic::Add(&(p_axial_loc[layer]), p_par[p].pos(1));
                    }
                } 
            });
        }
    }

    for (int l=0; l<num_layers; ++l) 
    {
        ParallelDescriptor::ReduceRealSum(p_sum[l]);
        //ParallelDescriptor::ReduceRealSum(p_axial_loc[l]);
    }

    for (int l=0; l<num_layers; ++l) 
    {
        NSType::avg_gatherField[l] = p_sum[l]/atoms_to_avg_over;
        //avg_axial_loc[l] = p_axial_loc[l]/atoms_to_avg_over;
    }

}


template<typename NSType>
void 
c_Nanostructure<NSType>::Write_AveragedGatherField() 
{
    if (ParallelDescriptor::IOProcessor()) 
    {
        std::ofstream outfile;
        outfile.open("avg_gatherField.dat");
        
        for (int l=0; l<NSType::avg_gatherField.size(); ++l)
        {
            outfile << l << std::setw(15) << NSType::avg_gatherField[l] << "\n";
        }  
        outfile.close();
    }
}


//template<typename NSType>
//void 
//c_Nanostructure<NSType>::ComputeChargeDensity() 
//{
//    NSType::ComputeChargeDensity();
//}
//
//
template<typename NSType>
void
c_Nanostructure<NSType>::DefineMatrixPartition (int _num_proc)
{

    NSType::DefineMatrixPartition(_num_proc); /*material specific*/
    amrex::Print() << "size of the Hamiltonian matrix: " << NSType::Hsize_glo << "\n";
    amrex::Print() << "max block columns per proc: " << NSType::max_blkCol_perProc << "\n";


    NSType::num_proc_with_blkCol = ceil(static_cast<amrex::Real>(NSType::Hsize_glo)/NSType::max_blkCol_perProc);
    amrex::Print() << "number of procs with block columns: " << NSType::num_proc_with_blkCol<< "\n";
    /*if num_proc_with_blk >= num_proc, assert.*/


    NSType::vec_cumu_blkCol_size.resize(NSType::num_proc_with_blkCol + 1);
    NSType::vec_cumu_blkCol_size[0] = 0;
    for(int p=1; p < NSType::num_proc_with_blkCol; ++p)
    {
        NSType::vec_cumu_blkCol_size[p] = NSType::vec_cumu_blkCol_size[p-1] + NSType::max_blkCol_perProc;
        /*All proc except the last one contains max_blkCol_perProc number of column blks.*/
    }
    NSType::vec_cumu_blkCol_size[NSType::num_proc_with_blkCol] = NSType::Hsize_glo;


    NSType::blkCol_size_loc = 0;
    if(_my_rank < NSType::num_proc_with_blkCol)
    {
        int blk_gid = _my_rank;
        NSType::blkCol_size_loc = NSType::vec_cumu_blkCol_size[blk_gid+1] - NSType::vec_cumu_blkCol_size[blk_gid];

        /*check later:setting vec_blkCol_gids may not be necessary*/
        for (int c=0; c < NSType::blkCol_size_loc; ++c)
        {
            int col_gid = NSType::vec_cumu_blkCol_size[blk_gid] + c;
            NSType::vec_blkCol_gids.push_back(col_gid);
        }
    }

}


template<typename NSType>
void
c_Nanostructure<NSType>::AllocateArrays () 
{
    NSType::AllocateArrays();
    h_E_RealPath_data.resize({0},{NUM_ENERGY_PTS_REAL},The_Pinned_Arena());


}


template<typename NSType>
void
c_Nanostructure<NSType>::ConstructHamiltonian () 
{

    NSType::ConstructHamiltonian();
    if(_use_electrostatic) NSType::AddPotentialToHamiltonian();

}
