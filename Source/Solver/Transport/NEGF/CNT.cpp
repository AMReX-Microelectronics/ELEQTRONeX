#include "CNT.H"

#include <AMReX_Particles.H>
#include <math.h>
#include <stdlib.h>

#include <cmath>

#include "../../Utils/CodeUtils/CodeUtil.H"

amrex::Array<int, 2> c_CNT::type_id = {17, 0};

void c_CNT::Read_NanotubeParameters(amrex::ParmParse &pp_ns)
{
    queryWithParser(pp_ns, "acc", acc);
    queryWithParser(pp_ns, "gamma", gamma);
    amrex::Vector<int> vec_type_id;
    auto typeID_isDefined =
        queryArrWithParser(pp_ns, "type_id", vec_type_id, 0, 2);
    if (typeID_isDefined) type_id = vecToArr_Templated<int, 2>(vec_type_id);
}

void c_CNT::Set_NanotubeParameters()
{
    R_cnt = acc *
            sqrt(3. * (pow(type_id[0], 2) + pow(type_id[1], 2) +
                       type_id[0] * type_id[1])) /
            (2. * MathConst::pi);

    if (type_id[1] == 0)
    {
        rings_per_unitcell = 4;
        atoms_per_ring = type_id[0];
    }
    else
    {
        // need to verify this
        rings_per_unitcell = 2;
        atoms_per_ring = type_id[0] - 1;
    }
}

void c_CNT::Read_MaterialSpecificNanostructureProperties()
{
    amrex::ParmParse pp_ns_default("CNT_default");
    amrex::ParmParse pp_ns(name);
    amrex::ParmParse *pp = &pp_ns_default;

    amrex::Print() << "##### Reading ParmParse CNT_default\n";
    for (int i = 0; i < 2; ++i)
    {
        if (i == 1)
        {
            pp = &pp_ns;
            amrex::Print() << "##### Reading ParmParse: " << name << "\n";
        }
        Read_NanotubeParameters(*pp);
    }
}

void c_CNT::Set_MaterialSpecificParameters()
{
    Set_NanotubeParameters();
    Define_SortedModeVector();
}

void c_CNT::Set_BlockDegeneracyVector(amrex::Vector<int> &vec)
{
    vec.resize(BLOCK_SIZE);
    for (int m = 0; m < BLOCK_SIZE; ++m)
    {
        vec[m] = mode_degen_vec[m];
    }
}

void c_CNT::Print_NanotubeParameters()
{
    amrex::Print() << "##### Properties Specific to CNT: \n";
    amrex::Print() << "##### acc: " << acc << "\n";
    amrex::Print() << "##### type_id: ";
    for (int i = 0; i < 2; ++i) amrex::Print() << type_id[i] << "  ";
    amrex::Print() << "\n";
    amrex::Print() << "#####* R_cnt / (nm): " << R_cnt / 1.e-9 << "\n";
    amrex::Print() << "#####* D_cnt / (nm): " << 2 * R_cnt / 1.e-9 << "\n";
    amrex::Print() << "##### gamma: " << gamma << "\n";
    amrex::Print() << "#####* rings_per_unitcell: " << rings_per_unitcell
                   << "\n";
    amrex::Print() << "#####* atoms_per_ring: " << atoms_per_ring << "\n";
}

void c_CNT::Print_MaterialSpecificReadData() { Print_NanotubeParameters(); }

s_Position3D c_CNT::get_AtomPosition_ZigZag_CNT(int ring_id, int atom_id)
{
    amrex::Real theta = 0;
    amrex::Real m = static_cast<amrex::Real>(num_atoms_per_field_site);

    s_Position3D center_offset;

    center_offset.dir[0] = 0.;
    center_offset.dir[1] = -(num_unitcells / 2.) * 3. * acc - acc / 2.;
    center_offset.dir[2] = 0.;

    s_Position3D pos;

    int unit_cell_id = ring_id / rings_per_unitcell;
    amrex::Real unit_cell_offset = 3 * acc * unit_cell_id;

    if (ring_id % rings_per_unitcell == 0) pos.dir[1] = acc + unit_cell_offset;
    if (ring_id % rings_per_unitcell == 1)
        pos.dir[1] = 1.5 * acc + unit_cell_offset;
    if (ring_id % rings_per_unitcell == 2)
        pos.dir[1] = 2.5 * acc + unit_cell_offset;
    if (ring_id % rings_per_unitcell == 3)
        pos.dir[1] = 3 * acc + unit_cell_offset;

    pos.dir[1] += center_offset.dir[1];

    if (ring_id % 4 == 0 or ring_id % 4 == 3)
    {
        theta = (atom_id / m) * 2. * MathConst::pi;
    }
    else
    {
        theta = (atom_id / m) * 2. * MathConst::pi +
                (sqrt(3.) / 2.) * (acc / R_cnt);
    }
    pos.dir[2] = R_cnt * cos(theta) + center_offset.dir[2];
    pos.dir[0] = R_cnt * sin(theta) + center_offset.dir[0];

    return pos;
}

void c_CNT::Generate_AtomLocations(amrex::Vector<s_Position3D> &pos)
{
    int m = num_atoms_per_field_site;
    int counter = 0;
    for (int i = 0; i < num_field_sites; ++i)
    {
        for (int j = 0; j < num_atoms_per_field_site; ++j)
        {
            pos[i * m + j] = get_AtomPosition_ZigZag_CNT(i, j);
            counter += 1;
            // amrex::Print() << counter << " " << pos[i*m+j].dir[0]/1.e-9 << "
            // "
            //                                  << pos[i*m+j].dir[1]/1.e-9 << "
            //                                  "
            //                                  << pos[i*m+j].dir[2]/1.e-9 <<
            //                                  "\n";
        }
    }
    c_NEGF_Common<BlkType>::Generate_AtomLocations(pos);
}

amrex::Real c_CNT::Get_Bandgap_Of_Mode(int p)
{
    int m = type_id[0];
    int n = type_id[1];
    amrex::Array<amrex::Real, 6> Eg_case;
    Eg_case[0] = (acc * gamma / R_cnt) * fabs(3 * p - (2 * m + n));
    Eg_case[1] = (acc * gamma / R_cnt) * fabs(3 * p - (m + 2 * n));
    Eg_case[2] = (acc * gamma / R_cnt) * fabs(3 * p - (m - n));
    Eg_case[3] = (acc * gamma / R_cnt) * fabs(-3 * p + (2 * m + n));
    Eg_case[4] = (acc * gamma / R_cnt) * fabs(-3 * p + (m + 2 * n));
    Eg_case[5] = (acc * gamma / R_cnt) * fabs(3 * p - (m - n));

    amrex::Real Eg_min = Eg_case[0];
    for (int i = 1; i < 6; ++i)
    {
        if (Eg_min > Eg_case[i])
        {
            Eg_min = Eg_case[i];
        }
    }
    return Eg_min;
}

void swap(amrex::Vector<amrex::Real> &vec, amrex::Vector<int> &mode_vec,
          int pos1, int pos2)
{
    amrex::Real temp;
    temp = vec[pos1];
    vec[pos1] = vec[pos2];
    vec[pos2] = temp;

    int temp_id;
    temp_id = mode_vec[pos1];
    mode_vec[pos1] = mode_vec[pos2];
    mode_vec[pos2] = temp_id;
}

int partition(amrex::Vector<amrex::Real> &vec, amrex::Vector<int> &mode_vec,
              int low, int high, amrex::Real pivot)
{
    int i = low;
    int j = low;
    while (i <= high)
    {
        if (vec[i] > pivot)
        {
            i++;
        }
        else
        {
            swap(vec, mode_vec, i, j);
            i++;
            j++;
        }
    }
    return j - 1;
}

void quickSort(amrex::Vector<amrex::Real> &vec, amrex::Vector<int> &mode_vec,
               int low, int high)
{
    // Credit: https://favtutor.com/blogs/quick-sort-cpp
    if (low < high)
    {
        amrex::Real pivot = vec[high];
        int pos = partition(vec, mode_vec, low, high, pivot);

        quickSort(vec, mode_vec, low, pos - 1);
        quickSort(vec, mode_vec, pos + 1, high);
    }
}

void c_CNT::Define_SortedModeVector()
{
    amrex::Vector<amrex::Real> Eg_vec;
    Eg_vec.resize(atoms_per_ring);

    amrex::Vector<int> mode_index_vec;
    mode_index_vec.resize(atoms_per_ring);

    amrex::Print() << "All modes: \n";
    for (int p = 1; p < atoms_per_ring + 1; ++p)
    {
        amrex::Real Eg = Get_Bandgap_Of_Mode(p);

        Eg_vec[p - 1] = Eg;
        mode_index_vec[p - 1] = p;

        amrex::Print() << "mode: " << p << " bandgap (nm): " << Eg << "\n";
    }

    quickSort(Eg_vec, mode_index_vec, 0, atoms_per_ring - 1);

    mode_vec.resize(0);
    bandgap_vec.resize(0);
    mode_degen_vec.resize(0);

    mode_vec.push_back(mode_index_vec[0]);
    bandgap_vec.push_back(Eg_vec[0]);
    mode_degen_vec.push_back(1);

    int counter = 1;
    for (int p = 1; p < atoms_per_ring; ++p)
    {
        if ((Eg_vec[p] - bandgap_vec[counter - 1]) > 1e-6)
        {
            mode_vec.push_back(mode_index_vec[p]);
            bandgap_vec.push_back(Eg_vec[p]);
            mode_degen_vec.push_back(1);
            counter++;
        }
        else
        {
            mode_degen_vec[counter - 1] += 1;
        }
    }

    int total_modes = mode_vec.size();

    amrex::Print() << "\nSorted, distinct modes: " << total_modes << "\n";
    for (int i = 0; i < total_modes; ++i)
    {
        amrex::Print() << "mode: " << std::setw(5) << mode_vec[i]
                       << std::setw(5) << ", bandgap / (nm): " << std::setw(15)
                       << bandgap_vec[i] << std::setw(15)
                       << ", degeneracy: " << std::setw(5) << mode_degen_vec[i]
                       << std::setw(5) << "\n";
    }

    Eg_vec.clear();
    mode_index_vec.clear();
}

ComplexType c_CNT::get_beta(int J)
{
    ComplexType arg(0., -MathConst::pi * J / type_id[0]);
    return 2. * gamma * cos(-1 * arg.imag());  // * exp(arg);
}

void c_CNT::Define_MPI_BlkType()
{
    MPI_Type_vector(1, BLOCK_SIZE, BLOCK_SIZE, MPI_DOUBLE_COMPLEX,
                    &MPI_BlkType);
    MPI_Type_commit(&MPI_BlkType);
}

void c_CNT::Construct_Hamiltonian()
{
    /*Here we define -H0 where H0 is Hamiltonian of flat bands*/

    auto const &h_minusHa = h_minusHa_loc_data.table();
    auto const &h_Hb = h_Hb_loc_data.table();
    auto const &h_Hc = h_Hc_loc_data.table();

    for (std::size_t i = 0; i < blkCol_size_loc; ++i)
    {
        h_minusHa(i) = 0.;
    }

    for (int j = 0; j < BLOCK_SIZE; ++j)
    {
        int J = mode_vec[j];
        beta.block[j] = get_beta(J);
    }
    // amrex::Print() << "\n Printing beta: "<< "\n";
    // amrex::Print() << beta << "\n";

    for (std::size_t i = 0; i < offDiag_repeatBlkSize; ++i)
    {
        if (i % offDiag_repeatBlkSize == 0)
        {
            h_Hb(i) = -1. * beta; /* negative sign because we are storing -H0 */
            h_Hc(i) = -1. * beta;
        }
        else
        {
            h_Hb(i) = -gamma;
            h_Hc(i) = -gamma;
        }
    }
}

void c_CNT::Construct_ContactHamiltonian()
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
    auto const &h_Hb = h_Hb_loc_data.const_table();

    for (std::size_t i = 0; i < decimation_layers; ++i)
    {
        std::size_t j = i % offDiag_repeatBlkSize;
        h_HcontactBeta(i) = -1. * h_Hb(j);
    }

    for (std::size_t i = 0; i < decimation_layers; ++i)
    {
        h_HcontactAlpha(i) = 0;
    }
}

void c_CNT::Define_ContactInfo()
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
        h_tau(c) = gamma;
    }
}

void c_CNT::Compute_SurfaceGreensFunction(MatrixBlock<BlkType> &gr,
                                          const ComplexType EmU)
{
    if (use_decimation)
    {
        c_NEGF_Common<BlkType>::DecimationTechnique(gr, EmU);
    }
    else
    {
        auto EmU_sq = pow(EmU, 2.);
        auto gamma_sq = pow(gamma, 2.);

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            auto Factor = EmU_sq + gamma_sq - pow(beta.block[i], 2);

            auto Sqrt = sqrt(pow(Factor, 2) - 4. * EmU_sq * gamma_sq);

            auto Denom = 2. * gamma_sq * EmU;

            auto val1 = (Factor + Sqrt) / Denom;
            auto val2 = (Factor - Sqrt) / Denom;

            if (val1.imag() < 0.)
                gr.block[i] = val1;
            else if (val2.imag() < 0.)
                gr.block[i] = val2;

            // amrex::Print() << "EmU: " << EmU << "\n";
            // amrex::Print() << "Factor: " << Factor << "\n";
            // amrex::Print() << "Sqrt: "  << Sqrt << "\n";
            // amrex::Print() << "Denom: " << Denom << "\n";
            // amrex::Print() << "Numerator: " << Factor+Sqrt << "\n";
            // amrex::Print() << "Value: " << (Factor+Sqrt)/Denom << "\n";
        }
        // amrex::Print() << "Using quadratic, gr: " << gr << "\n";
    }
}
