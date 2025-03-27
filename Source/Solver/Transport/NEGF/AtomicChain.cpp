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
    queryWithParser(pp_ns, "omega_scaling_factor", omega_scaling_factor);
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

ComplexType c_AtomicChain::FermiFunction(ComplexType omega_sq,
                                         const amrex::Real kT)
{
    ComplexType one(1., 0.);
    static constexpr auto hbar_eVperHz = static_cast<amrex::Real>(6.582119569e-16);
    ComplexType hbarOmega = hbar_eVperHz * sqrt(omega_sq);

    return one / (exp(hbarOmega / kT) - one);
}

void c_AtomicChain::Define_EnergyLimits()
{
    for (int c = 0; c < NUM_CONTACTS; ++c)
    {
        kT_contact[c] = PhysConst::kb_eVperK *
                        Contact_Temperature[c]; /*set Temp in the input*/
    }
    mu_min = mu_contact[0];
    mu_max = mu_contact[0];
    kT_min = kT_contact[0];
    kT_max = kT_contact[0];

    flag_noneq_exists = false;

    for (int c = 1; c < NUM_CONTACTS; ++c)
    {
        if (mu_min > mu_contact[c])
        {
            mu_min = mu_contact[c];
        }
        if (mu_max < mu_contact[c])
        {
            mu_max = mu_contact[c];
        }
        if (kT_min > kT_contact[c])
        {
            kT_min = kT_contact[c];
        }
        if (kT_max < kT_contact[c])
        {
            kT_max = kT_contact[c];
        }
    }
    if (fabs(mu_min - mu_max) > 1e-8) flag_noneq_exists = true;
    if (fabs(kT_min - kT_max) > 0.01) flag_noneq_exists = true;

    /* Only the real part is set here. The imaginary part is set in
     * Define_IntegrationPaths */
    E_contour_left = pow((E_valence_min / PhysConst::hbar_eVperHz), 2);

    omega_max =
        (mu_max + Fermi_tail_factor_upper * kT_max) / PhysConst::hbar_eVperHz;
    E_contour_right = pow(omega_max, 2) + E_zPlus;

    E_rightmost = E_contour_right;

    // if (flag_noneq_exists)
    //{
    //     amrex::Print() << "\n Nonequilibrium exists!\n";
    // }
    // else
    //{
    //     amrex::Print() << "\n Nonequilibrium doesn't exist!\n";
    // }
    amrex::Print() << " U_contact: ";
    for (int c = 0; c < NUM_CONTACTS; ++c)
    {
        amrex::Print() << U_contact[c] << " ";
    }
    amrex::Print() << "\n";
    amrex::Print() << " E_f: " << E_f << "\n";
    amrex::Print() << " mu_min/max: " << mu_min << " " << mu_max << "\n";
    amrex::Print() << " kT_min/max: " << kT_min << " " << kT_max << "\n";
    amrex::Print() << " E_zPlus: " << E_zPlus << "\n";
    amrex::Print() << " E_contour_left/E_contour_right: " << E_contour_left
                   << "  " << E_contour_right << "\n";
}

void c_AtomicChain::Define_IntegrationPaths()
{
    /* Define_ContourPath_Rho0 */
    ContourPath_Rho0.resize(1);
    ContourPath_Rho0[0].Define_GaussLegendrePoints(E_contour_left,
                                                   E_contour_right,
                                                   eq_integration_pts[0], 0);

    /* Here we add the imaginary part to each energy point */
    amrex::Print() << "Printing E for equilibrium Rho0 path: \n";
    for (ComplexType E : ContourPath_Rho0[0].E_vec)
    {
        E += Compute_zPlus(E.real());

        amrex::Print() << E << "\n";
    }

    /* Define_ContourPath_DOS */
    if (flag_compute_flatband_dos)
    {
        ContourPath_DOS.resize(1);
        ComplexType min(flatband_dos_integration_limits[0], 0.);
        ComplexType max(flatband_dos_integration_limits[1], 0.);
        ContourPath_DOS[0].Define_GaussLegendrePoints(
            min, max, flatband_dos_integration_pts, 0);

        amrex::Print() << "Printing E for DOS path: \n";
        for (ComplexType E : ContourPath_DOS[0].E_vec)
        {
            E += Compute_zPlus(E.real());
            amrex::Print() << E << "\n";
        }
    }
}
