#include "AtomicChain.H"

#include <AMReX_Particles.H>
#include <math.h>
#include <stdlib.h>

#include <cmath>

#include "../../Utils/CodeUtils/CodeUtil.H"

void c_AtomicChain::Read_AtomicChainParameters(amrex::ParmParse &pp_ns)
{
    queryWithParser(pp_ns, "spring_constant_device_device", f_d);
    queryWithParser(pp_ns, "spring_constant_contact_contact", f_c);
    queryWithParser(pp_ns, "spring_constant_contact_device", f_cd);
    queryWithParser(pp_ns, "atom_spacing", spacing);
    queryWithParser(pp_ns, "mass_contact_atoms", m_c);
    queryWithParser(pp_ns, "mass_device_atoms", m_d);
}

// void c_AtomicChain::Set_AtomicChainParameters() {}

void c_AtomicChain::Read_MaterialSpecificNanostructureProperties()
{
    amrex::ParmParse pp_ns_default("atomic_chain_default");
    amrex::ParmParse pp_ns(name);
    amrex::ParmParse *pp = &pp_ns_default;

    amrex::Print() << "##### Reading ParmParse atomic_chain_default\n";
    for (int i = 0; i < 2; ++i)
    {
        if (i == 1)
        {
            pp = &pp_ns;
            amrex::Print() << "##### Reading ParmParse: " << name << "\n";
        }
        Read_AtomicChainParameters(*pp);
    }
}

void c_AtomicChain::Set_MaterialSpecificParameters()
{
    // Set_AtomicChainParameters();
}

void c_AtomicChain::Print_AtomicChainParameters()
{
    amrex::Print() << "##### Properties Specific to 1D Atomic Chain: \n";
    amrex::Print() << "##### spring_constant_device_device: " << f_d << "\n";
    amrex::Print() << "##### spring_constant_contact_contact: " << f_c << "\n";
    amrex::Print() << "##### spring_constant_contact_device: " << f_cd << "\n";
    amrex::Print() << "##### atom_spacing: " << spacing << "\n";
    amrex::Print() << "##### mass_contact_atoms: " << m_c << "\n";
    amrex::Print() << "##### mass_device_atoms: " << m_d << "\n";
}

void c_AtomicChain::Print_MaterialSpecificReadData()
{
    Print_AtomicChainParameters();
}

void c_AtomicChain::Set_BlockDegeneracyVector(amrex::Vector<int> &vec)
{
    vec.resize(BLOCK_SIZE);
    for (int m = 0; m < BLOCK_SIZE; ++m)
    {
        vec[m] = 1;
    }
}

s_Position3D c_AtomicChain::get_AtomPosition(int field_id)
{
    s_Position3D pos;

    double Y_center_offset = -(num_unitcells / 2) * spacing;

    if (num_unitcells % 2 == 0) Y_center_offset += spacing / 2.;

    pos.dir[0] = 0.;
    pos.dir[1] = field_id * spacing + Y_center_offset;
    pos.dir[2] = 0.;

    return pos;
}

void c_AtomicChain::Generate_AtomLocations(amrex::Vector<s_Position3D> &pos)
{
    // num_atoms_per_field_site is 1
    for (int i = 0; i < num_field_sites; ++i)
    {
        pos[i] = get_AtomPosition(i);
        amrex::Print() << i << " " << pos[i].dir[0] / 1.e-9 << " "
                       << pos[i].dir[1] / 1.e-9 << " " << pos[i].dir[2] / 1.e-9
                       << "\n";
    }
    c_NEGF_Common<BlkType>::Generate_AtomLocations(pos);
}

void c_AtomicChain::Define_MPI_BlkType()
{
    MPI_Type_vector(1, BLOCK_SIZE, BLOCK_SIZE, MPI_DOUBLE_COMPLEX,
                    &MPI_BlkType);
    MPI_Type_commit(&MPI_BlkType);
}

void c_AtomicChain::Construct_Hamiltonian()
{
    /*Here we define -H0 where H0 is device Hamiltonian*/

    auto const &h_minusHa = h_minusHa_loc_data.table();
    auto const &h_Hb = h_Hb_loc_data.table();
    auto const &h_Hc = h_Hc_loc_data.table();
    auto const &h_HcontactBeta = h_HcontactBeta_loc_data.table();

    for (std::size_t i = 0; i < blkCol_size_loc; ++i)
    {
        h_minusHa(i) = -2. * f_d / m_d;
    }

    if ((vec_cumu_blkCol_size[my_rank] <= 0 &&
         0 < vec_cumu_blkCol_size[my_rank + 1]))
    {
        h_minusHa(0) = -(f_d + f_cd) / m_d;
    }

    if ((vec_cumu_blkCol_size[my_rank] <= (Hsize_glo - 1) &&
         (Hsize_glo - 1) < vec_cumu_blkCol_size[my_rank + 1]))
    {
        int n = Hsize_glo - 1 - vec_cumu_blkCol_size[my_rank];
        h_minusHa(n) = -(f_d + f_cd) / m_d;
    }

    for (std::size_t i = 0; i < offDiag_repeatBlkSize; ++i)
    {
        h_Hb(i) = f_d / m_d;
        h_Hc(i) = f_d / m_d;
    }
}

void c_AtomicChain::Construct_ContactHamiltonian()
{
    /* Here we define the right contact Hamiltonian with
     *
     * H_right = | Alpha0  Beta0                                  |
     *           | Beta0^D Alpha1  Beta1                          |
     *           |         Beta1^D Alpha2  Beta2                  |
     *           |                 Beta2^D Alpha3  Beta3          |
     *           |                         Beta3^D Alpha4  Beta4  |
     *           |                                 Beta4^D Alpha0 |
     *
     * Alpha, Beta size: decimation layers
     * Alpha, Beta have size decimation layers.
     * In the above example there are 5 decimation layers.
     */
    auto const &h_HcontactAlpha = h_HcontactAlpha_loc_data.table();
    auto const &h_HcontactBeta = h_HcontactBeta_loc_data.table();

    for (std::size_t i = 0; i < decimation_layers; ++i)
    {
        h_HcontactBeta(i) = -f_c / m_c;
    }

    h_HcontactAlpha(0) = (f_c + f_cd) / m_c;
    for (std::size_t i = 1; i < decimation_layers; ++i)
    {
        h_HcontactAlpha(i) = 2 * f_c / m_c;
    }
}

void c_AtomicChain::Define_ContactInfo()
{
    /*define arrays depending on Hsize_glo*/
    global_contact_index[0] = 0;
    global_contact_index[1] = Hsize_glo - 1;
    contact_transmission_index[0] = Hsize_glo - 1;
    contact_transmission_index[1] = 0;

    /*define tau*/
    auto const &h_tau = h_tau_glo_data.table();
    for (std::size_t c = 0; c < NUM_CONTACTS; ++c)
    {
        h_tau(c) = -f_cd / sqrt(m_c * m_d);
    }
}

void c_AtomicChain::Compute_SurfaceGreensFunction(MatrixBlock<BlkType> &gr,
                                                  const ComplexType EmU)
{
    if (use_decimation)
    {
        c_NEGF_Common<BlkType>::DecimationTechnique(gr, EmU);
    }
    else
    {
        //  auto EmU_sq = pow(EmU, 2.);
        //  auto gamma_sq = pow(gamma, 2.);

        //  for (int i = 0; i < BLOCK_SIZE; ++i)
        //  {
        //      auto Factor = EmU_sq + gamma_sq - pow(beta.block[i], 2);

        //      auto Sqrt = sqrt(pow(Factor, 2) - 4. * EmU_sq * gamma_sq);

        //      auto Denom = 2. * gamma_sq * EmU;

        //      auto val1 = (Factor + Sqrt) / Denom;
        //      auto val2 = (Factor - Sqrt) / Denom;

        //      if (val1.imag() < 0.)
        //          gr.block[i] = val1;
        //      else if (val2.imag() < 0.)
        //          gr.block[i] = val2;

        //      // amrex::Print() << "EmU: " << EmU << "\n";
        //      // amrex::Print() << "Factor: " << Factor << "\n";
        //      // amrex::Print() << "Sqrt: "  << Sqrt << "\n";
        //      // amrex::Print() << "Denom: " << Denom << "\n";
        //      // amrex::Print() << "Numerator: " << Factor+Sqrt << "\n";
        //      // amrex::Print() << "Value: " << (Factor+Sqrt)/Denom << "\n";
        //  }
        //  // amrex::Print() << "Using quadratic, gr: " << gr << "\n";
    }
}

ComplexType c_AtomicChain::FermiFunction(ComplexType E_minus_Mu,
                                         const amrex::Real kT)
{
    ComplexType one(1., 0.);
    return one / (exp((sqrt(E_minus_Mu)) / kT) - one);
}
