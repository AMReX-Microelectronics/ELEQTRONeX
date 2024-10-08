#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

#include <any>
#include <cmath>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "../../Code_Definitions.H"
#include "../SelectWarpXUtils/WarpXConst.H"

using namespace amrex;

using ComplexType = amrex::GpuComplex<amrex::Real>;

namespace Multifab_Manipulation
{
void InitializeMacroMultiFabUsingParser_3vars(
    amrex::MultiFab *macro_mf, amrex::ParserExecutor<3> const &macro_parser,
    amrex::Geometry &geom);

void InitializeMacroMultiFabUsingParser_4vars(
    amrex::MultiFab *macro_mf, amrex::ParserExecutor<4> const &macro_parser,
    amrex::Geometry &geom, const amrex::Real t);

void AverageCellCenteredMultiFabToCellFaces(
    const amrex::MultiFab &cc_arr,
    std::array<amrex::MultiFab, AMREX_SPACEDIM> &face_arr);

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void ConvertParserIntoMultiFab_3vars(
    const int i, const int j, const int k,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
    const amrex::RealBox &real_box, const amrex::IntVect &iv,
    amrex::ParserExecutor<3> const &macro_parser,
    amrex::Array4<amrex::Real> const &mf_array)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::"
           "ConvertParserIntoMultiFab_3vars()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
    amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;

    amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
    amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;

    amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
    amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;

    mf_array(i, j, k) = macro_parser(x, y, z);

#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t\t}************************Multifab_Manipulation::"
           "ConvertParserIntoMultiFab_3vars()************************\n";
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void ConvertParserIntoMultiFab_4vars(
    const int i, const int j, const int k, const amrex::Real t,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
    const amrex::RealBox &real_box, const amrex::IntVect &iv,
    amrex::ParserExecutor<4> const &macro_parser,
    amrex::Array4<amrex::Real> const &mf_array)
{
#ifdef PRINT_NAME
    amrex::Print()
        << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::"
           "ConvertParserIntoMultiFab_4vars()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__
                   << " at line: " << __LINE__ << "\n";
#endif

    amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
    amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;

    amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
    amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;

    amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
    amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;

    mf_array(i, j, k) = macro_parser(x, y, z, t);

#ifdef PRINT_NAME
    amrex::Print()
        << "\t\t\t\t\t}************************Multifab_Manipulation::"
           "ConvertParserIntoMultiFab_4vars()************************\n";
#endif
}

#ifdef AMREX_USE_EB
void SpecifyValueOnlyOnCutcells(amrex::MultiFab &mf, amrex::Real const value);

void SpecifyValueOnlyOnCutcells_UsingParser_4vars(
    amrex::MultiFab &mf, amrex::ParserExecutor<4> const &macro_parser,
    const amrex::Geometry &geom, const amrex::Real t);

void SpecifyValueOnlyOnCutcells_UsingParser_3vars(
    amrex::MultiFab &mf, amrex::ParserExecutor<3> const &macro_parser,
    const amrex::Geometry &geom);

void CopyValuesIntoAMultiFabOnCutcells(amrex::MultiFab &target_mf,
                                       amrex::MultiFab &source_mf);

amrex::Real GetTotalNumberOfCutcells(amrex::MultiFab &mf);

void Copy_3DCartesian_To_2DAzimuthalLongitudinal(
    const amrex::Geometry &geom, amrex::MultiFab &mf,
    amrex::Array2D<amrex::Real, 1, Max_Ncell_Azim, 1, Max_Ncell_Long,
                   amrex::Order::C> *cyl_surf_grid,
    int cyl_ncell_ring, int ncell_ncell_axis, amrex::RealArray &center,
    int dir_axial, int dir_ref2);
#endif

}  // namespace Multifab_Manipulation

void PrintRunDiagnostics(amrex::Real initial_time);

template <class T, class F>
inline std::pair<const std::type_index, std::function<void(std::any const &)>>
to_any_visitor(F const &f)
{
    return {std::type_index(typeid(T)), [g = f](std::any const &a)
            {
                if constexpr (std::is_void_v<T>)
                    g();
                else
                    g(std::any_cast<T const &>(a));
            }};
}

static std::unordered_map<std::type_index,
                          std::function<void(std::any const &)>>
    any_visitor{
        to_any_visitor<float>([](float x)
                              { amrex::Print() << "contains a value: " << x; }),
        to_any_visitor<double>(
            [](double x) { amrex::Print() << "contains a value: " << x; }),
        to_any_visitor<std::string>(
            [](std::string s)
            { amrex::Print() << "contains a function parser name: " << s; })
        // ... add more handlers for your types ...
    };

inline void process_std_any(const std::any &a)
{
    if (const auto it = any_visitor.find(std::type_index(a.type()));
        it != any_visitor.cend())
    {
        it->second(a);
    }
    else
    {
        //        std::cout << "unregistered type! "<<
        //        std::quoted(a.type().name());
        amrex::Print() << "contents do not matter! ";
    }
}

template <typename K, typename V>
bool findByValue(std::vector<K> &vec, std::map<K, V> mapOfElemen, V value)
{
    bool bResult = false;
    auto it = mapOfElemen.begin();

    while (it != mapOfElemen.end())
    {
        if (it->second == value)
        {
            bResult = true;
            vec.push_back(it->first);
        }
        it++;
    }
    return bResult;
}

amrex::RealArray vecToArr(amrex::Vector<amrex::Real> &vec);
amrex::IntArray vecToArr(amrex::Vector<int> &vec);

int Evaluate_TypeOf_MacroStr(std::string macro_str);

int Get_Third_Reference_Direction(int dir1, int dir2);

template <typename ARR_TYPE, std::size_t ARR_SIZE>
amrex::Array<ARR_TYPE, ARR_SIZE> vecToArr_Templated(
    amrex::Vector<ARR_TYPE> &vec)
{
    amrex::Array<ARR_TYPE, ARR_SIZE> array;
    for (int i = 0; i < vec.size(); ++i) array[i] = vec[i];
    return array;
}

// template<typename T>
// T Compute_Cell_ID(const amrex::GpuArray<int,AMREX_SPACEDIM>& index, const
// amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell) {
//
//     T cell_id;
//
//     switch(AMREX_SPACEDIM-1)
//     {
//         case 0:
//         {
//             cell_id = index[0];
//             break;
//         }
//         case 1:
//         {
//             cell_id =   index[0]
//                       + index[1] * n_cell[0];
//             break;
//         }
//         case 2:
//         {
//             cell_id =   index[0]
//                       + index[1] * n_cell[0]
//                       + index[2] * n_cell[0] * n_cell[1];
//             break;
//         }
//     }
//     return cell_id;
// }

namespace Quadrature
{
void Gauss_Legendre(amrex::Vector<amrex::Real> &x,
                    amrex::Vector<amrex::Real> &w, const int n);
}

void CreateDirectory(std::string foldername);

namespace GeomUtils
{
bool Is_ID_Within_Bounds(const amrex::Vector<int> &pos,
                         const amrex::Vector<int> &minID,
                         const amrex::Vector<int> &maxID);
}
