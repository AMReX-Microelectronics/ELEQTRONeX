#include "CNT.H"
#include "../../Utils/CodeUtils/CodeUtil.H"

#include <AMReX_Particles.H>

#include <cmath>
#include <math.h>
#include<stdlib.h>

amrex::Array<int,2> c_CNT::type_id; //very important

/*Next, member class functions are defined*/

void
c_CNT:: ReadNanostructureProperties ()
{
    amrex::Print() << "\n##### NANOSTRUCTURE PROPERTIES #####\n\n";

    c_Common_Properties<BlkType>::ReadNanostructureProperties();

    amrex::ParmParse pp_ns(name);

    amrex::Print() << "##### Properties Specific to CNT: \n";

    getWithParser(pp_ns,"acc", acc);
    amrex::Print() << "##### acc: " << acc << "\n";

    amrex::Vector<int> vec_type_id;
    getArrWithParser(pp_ns, "type_id", vec_type_id, 0, 2);
    type_id = vecToArr_Templated<int, 2>(vec_type_id);
    amrex::Print() << "##### type_id: ";
    for (int i=0; i<2; ++i) amrex::Print() << type_id[i] << "  ";
    amrex::Print() << "\n";

    //getWithParser(pp_ns,"num_unitcells", num_unitcells);
    //amrex::Print() << "##### num_unitcells: " << num_unitcells << "\n";

    getWithParser(pp_ns,"gamma", gamma); 
    amrex::Print() << "##### gamma: " << gamma << "\n";

    //amrex::Vector<amrex::Real> vec_offset;
    //getArrWithParser(pp_ns, "offset", vec_offset, 0, AMREX_SPACEDIM);
    //offset = vecToArr(vec_offset);

    //amrex::Print() << "##### offset: ";
    //for (int i=0; i<AMREX_SPACEDIM; ++i) amrex::Print() << offset[i] << "  ";
    //amrex::Print() << "\n";

    if(type_id[1] == 0) {
        rings_per_unitcell = 4;
        atoms_per_ring = type_id[0];
    } else {
        //need to verify this 
        rings_per_unitcell = 2;
        atoms_per_ring = type_id[0]-1;
    }
    num_atoms = num_unitcells*rings_per_unitcell*atoms_per_ring;
    amrex::Print() << "#####* rings_per_unitcell: " << rings_per_unitcell << "\n";
    amrex::Print() << "#####* atoms_per_ring: " << atoms_per_ring << "\n";
    amrex::Print() << "#####* num_atoms: " << num_atoms << "\n";

}


AMREX_GPU_HOST_DEVICE ComplexType 
c_CNT::get_beta(int J)
{
   ComplexType arg(0., -MathConst::pi*J/type_id[0]);
   return 2. * gamma * cos(-1*arg.imag());// * exp(arg); 
}


//AMREX_GPU_HOST_DEVICE 
//void
//c_CNT::get_Sigma(MatrixBlock<BlkType>& Sigma, const ComplexType EmU)
//{
//
//   MatrixBlock<BlkType> gr;
//   Compute_SurfaceGreensFunction(gr, EmU);
//
//   amrex::Print() << "Printing gr: \n";
//   amrex::Print() << gr << "\n";
//
//   Sigma = gr*pow(gamma,2.);
//}
//
//
void 
c_CNT::AllocateArrays () 
{
    c_Common_Properties<BlkType>:: AllocateArrays (); 

}
//
//
//void 
//c_CNT::DeallocateArrays () 
//{
//
//}
//
//


void 
c_CNT::ConstructHamiltonian() 
{

    auto const& h_Ha = h_Ha_loc_data.table();
    auto const& h_Hb = h_Hb_loc_data.table();
    auto const& h_Hc = h_Hc_loc_data.table();

    for (std::size_t i = 0; i < blkCol_size_loc; ++i)
    {
        h_Ha(i) = 0.;
    }

    for (int j=0; j<NUM_MODES; ++j) 
    {
        int J = mode_arr[j];
        beta.block[j] = get_beta(J);
    }
    amrex::Print() << "\n Printing beta: "<< "\n";
    amrex::Print() << beta << "\n";

    for (std::size_t i = 0; i < offDiag_repeatBlkSize; ++i)
    {
       if(i%offDiag_repeatBlkSize == 0) {
          h_Hb(i) = -1*beta; /*negative sign because (E[I] - [H]) will have negative B and C*/
          h_Hc(i) = -1*beta;
       }
       else {
          h_Hb(i) = -gamma;
          h_Hc(i) = -gamma;  
       }
    }

}

void
c_CNT:: Define_tau ()
{
    auto const& h_tau = h_tau_glo_data.table();
    for (std::size_t c = 0; c < NUM_CONTACTS; ++c)
    {
        h_tau(c) = gamma;
    }
}

AMREX_GPU_HOST_DEVICE
void
c_CNT:: Compute_SurfaceGreensFunction(MatrixBlock<BlkType> gr, const ComplexType EmU)
{

   auto EmU_sq = pow(EmU,2.);
   auto gamma_sq = pow(gamma,2.);

   for(int i=0; i < NUM_MODES; ++i) 
   {
       auto Factor = EmU_sq + gamma_sq - pow(beta.block[i],2);

       auto Sqrt = sqrt(pow(Factor,2) - 4. * EmU_sq * gamma_sq);

       auto Denom = 2. * gamma_sq * EmU;

       auto val1 = (Factor + Sqrt) / Denom;
       auto val2 = (Factor - Sqrt) / Denom;

       if(val1.imag() < 0.) gr.block[i] = val1;
       else if(val2.imag() < 0.) gr.block[i] = val2; 
       //amrex::Print() << "EmU: " << EmU << "\n";
       //amrex::Print() << "Factor: " << Factor << "\n";
       //amrex::Print() << "Sqrt: "  << Sqrt << "\n";
       //amrex::Print() << "Denom: " << Denom << "\n";
       //amrex::Print() << "Numerator: " << Factor+Sqrt << "\n";
       //amrex::Print() << "Value: " << (Factor+Sqrt)/Denom << "\n";
   }
}




void 
c_CNT::Define_SelfEnergy ()
{

    c_Common_Properties<BlkType>:: Define_SelfEnergy (); 
    auto const& h_Sigma = h_Sigma_glo_data.table();
    ////MatrixBlock<BlkType> test_sigma;
    ////ComplexType EmU(1, 0.);
    ////get_Sigma(test_sigma, EmU);
    amrex::Print() << "Printing Sigma: \n";
    std::cout<< h_Sigma(0,0) << "\n";

}

//void 
//c_CNT::DefineIntegrationPaths ()
//{
//
//
//}
//
//
//
//
//void 
//c_CNT::ComputeChargeDensity () 
//{
//
//}
