/* Copyright 2019-2020 Andrew Myers, Luca Fedeli, Maxence Thevenet
 * Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_H_
#define WARPX_UTILS_H_

#include <AMReX_BaseFwd.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_LayoutData.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

void ParseGeometryInput();

void ReadBoostedFrameParameters(amrex::Real &gamma_boost,
                                amrex::Real &beta_boost,
                                amrex::Vector<int> &boost_direction);

void ConvertLabParamsToBoost();

/**
 * Reads the user-defined field and particle boundary condition parameters
 */
void ReadBCParams();

/** Check the warpx.dims matches the binary name
 */
void CheckDims();

/** Check the warpx.dims matches the binary name & set up RZ gridding
 *
 * Ensures that the blocks are setup correctly for the RZ spectral solver
 * When using the RZ spectral solver, the Hankel transform cannot be
 * divided among multiple blocks. Each block must extend over the
 * entire radial extent.
 * The grid can be divided up along z, but the number of blocks
 * must be >= the number of processors.
 */
void CheckGriddingForRZSpectral();

void NullifyMF(amrex::MultiFab &mf, int lev, amrex::Real zmin,
               amrex::Real zmax);

/**
 * \brief Parse a string (typically a mathematical expression) from the
 * input file and store it into a variable.
 *
 * \param pp used to read the query_string `pp.<function>=string`
 * \param query_string ParmParse.query will look for this string
 * \param stored_string variable in which the string to parse is stored
 */
void Store_parserString(const amrex::ParmParse &pp, std::string query_string,
                        std::string &stored_string);

namespace WarpXUtilIO
{
/**
 * A helper function to write binary data on disk.
 * @param[in] filename where to write
 * @param[in] data Vector containing binary data to write on disk
 * return true if it succeeds, false otherwise
 */
bool WriteBinaryDataOnFile(std::string filename,
                           const amrex::Vector<char> &data);

}  // namespace WarpXUtilIO

namespace WarpXUtilAlgo
{

/** \brief Returns a pointer to the first element in the range [first, last)
 * that is greater than val
 *
 * A re-implementation of the upper_bound algorithm suitable for GPU kernels.
 *
 * @param first: pointer to left limit of the range to consider
 * @param last: pointer to right limit of the range to consider
 * @param val: value to compare the elements of [first, last) to
 */
template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE const T *upper_bound(const T *first,
                                                         const T *last,
                                                         const T &val)
{
    const T *it;
    size_t count, step;
    count = last - first;
    while (count > 0)
    {
        it = first;
        step = count / 2;
        it += step;
        if (!(val < *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

/** \brief Performs a linear interpolation
 *
 * Performs a linear interpolation at x given the 2 points
 * (x0, f0) and (x1, f1)
 */
template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE T linear_interp(T x0, T x1, T f0, T f1, T x)
{
    return ((x1 - x) * f0 + (x - x0) * f1) / (x1 - x0);
}

/** \brief Performs a bilinear interpolation
 *
 * Performs a bilinear interpolation at (x,y) given the 4 points
 * (x0, y0, f00), (x0, y1, f01), (x1, y0, f10), (x1, y1, f11).
 */
template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE T bilinear_interp(T x0, T x1, T y0, T y1,
                                                      T f00, T f01, T f10,
                                                      T f11, T x, T y)
{
    const T fx0 = linear_interp(x0, x1, f00, f10, x);
    const T fx1 = linear_interp(x0, x1, f01, f11, x);
    return linear_interp(y0, y1, fx0, fx1, y);
}

/** \brief Performs a trilinear interpolation
 *
 * Performs a trilinear interpolation at (x,y,z) given the 8 points
 * (x0, y0, z0, f000), (x0, y0, z1, f001), (x0, y1, z0, f010), (x0, y1, z1,
 * f011), (x1, y0, z0, f100), (x1, y0, z1, f101), (x1, y1, z0, f110), (x1, y1,
 * z1, f111)
 */
template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE T trilinear_interp(T x0, T x1, T y0, T y1,
                                                       T z0, T z1, T f000,
                                                       T f001, T f010, T f011,
                                                       T f100, T f101, T f110,
                                                       T f111, T x, T y, T z)
{
    const T fxy0 =
        bilinear_interp(x0, x1, y0, y1, f000, f010, f100, f110, x, y);
    const T fxy1 =
        bilinear_interp(x0, x1, y0, y1, f001, f011, f101, f111, x, y);
    return linear_interp(z0, z1, fxy0, fxy1, z);
}

/** \brief Compute physical coordinates (x,y,z) that correspond to a given
 * (i,j,k) and the corresponding staggering, mf_type.
 *
 * \param[in] i    index along x
 * \param[in] j    index along y
 * \param[in] k    index along z
 * \param[in] mf_type GpuArray containing the staggering type to convert (i,j,k)
 * to (x,y,z) \param[in] domain_lo Physical coordinates of the lowest corner of
 * the simulation domain \param[in] dx   Cell size of the simulation domain
 *
 * \param[out] x   physical coordinate along x
 * \param[out] y   physical coordinate along y
 * \param[out] z   physical coordinate along z
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void getCellCoordinates(
    int i, int j, int k, amrex::GpuArray<int, 3> const mf_type,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_lo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx, amrex::Real &x,
    amrex::Real &y, amrex::Real &z)
{
    using namespace amrex::literals;
    x = domain_lo[0] + i * dx[0] + (1._rt - mf_type[0]) * dx[0] * 0.5_rt;
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::ignore_unused(j);
    y = 0._rt;
    z = domain_lo[1] + k * dx[1] + (1._rt - mf_type[1]) * dx[1] * 0.5_rt;
#else
    y = domain_lo[1] + j * dx[1] + (1._rt - mf_type[1]) * dx[1] * 0.5_rt;
    z = domain_lo[2] + k * dx[2] + (1._rt - mf_type[2]) * dx[2] * 0.5_rt;
#endif
}

}  // namespace WarpXUtilAlgo

/**
 * \brief Do a safe cast of a real to an int
 * This ensures that the float value is within the range of ints and if not,
 * raises an exception.
 *
 * \param x Real value to cast
 * \param real_name String, the name of the variable being casted to use in the
 * error message
 */
int safeCastToInt(amrex::Real x, const std::string &real_name);

/**
 * \brief Initialize an amrex::Parser object from a string containing a math
 * expression
 *
 * \param parse_function String to read to initialize the parser.
 * \param varnames A list of predefined independent variables
 */
amrex::Parser makeParser(std::string const &parse_function,
                         amrex::Vector<std::string> const &varnames);

/** Parse a string and return as a real
 *
 * In case the string cannot be interpreted as Real,
 * this function ... <throws an exception? aborts with error message?>
 *
 * \param str The string to be parsed
 * \return representation as real
 */
double parseStringtoReal(std::string str);

/** Parse a string and return an int
 *
 * In case the string cannot be interpreted as Real,
 * this function ... <throws an exception? aborts with error message?>
 *
 * \param str The string to be parsed
 * \param name For integers, the name, to be used in error messages
 * \return rounded closest integer
 */
int parseStringtoInt(std::string str, std::string name);

template <int N>
amrex::ParserExecutor<N> compileParser(amrex::Parser const *parser)
{
    if (parser)
    {
        return parser->compile<N>();
    }
    else
    {
        return amrex::ParserExecutor<N>{};
    }
}

/** Similar to amrex::ParmParse::query, but also supports math expressions for
 * the value.
 *
 * amrex::ParmParse::query reads a name and a value from the input file. This
 * function does the same, and applies the amrex::Parser to the value, so the
 * user has the choice to specify a value or a math expression (including
 * user-defined constants). Works for amrex::Real numbers and integers.
 *
 * \param[in] a_pp amrex::ParmParse object
 * \param[in] str name of the parameter to read
 * \param[out] val where the value queried and parsed is stored, either a scalar
 * or vector
 */
int queryWithParser(const amrex::ParmParse &a_pp, char const *const str,
                    float &val);
int queryWithParser(const amrex::ParmParse &a_pp, char const *const str,
                    double &val);

/** Similar to amrex::ParmParse::query, but also supports math expressions for
 * the value.
 *
 * amrex::ParmParse::query reads a name and a value from the input file. This
 * function does the same, and applies the amrex::Parser to the value, so the
 * user has the choice to specify a value or a math expression (including
 * user-defined constants). Works for amrex::Real numbers and integers.
 *
 * \param[in] a_pp amrex::ParmParse object
 * \param[in] str name of the parameter to read
 * \param[out] val where the value queried and parsed is stored, either a scalar
 * or vector \param[in] start_ix start index in the list of inputs values
 * (optional with arrays, default is amrex::ParmParse::FIRST for starting with
 * the first input value) \param[in] num_val number of input values to use
 * (optional with arrays, default is amrex::ParmParse::LAST for reading until
 * the last input value)
 */
int queryArrWithParser(const amrex::ParmParse &a_pp, char const *const str,
                       std::vector<amrex::Real> &val,
                       const int start_ix = amrex::ParmParse::FIRST,
                       const int num_val = amrex::ParmParse::LAST);
int queryWithParser(const amrex::ParmParse &a_pp, char const *const str,
                    int &val);
int queryArrWithParser(const amrex::ParmParse &a_pp, char const *const str,
                       std::vector<int> &val,
                       const int start_ix = amrex::ParmParse::FIRST,
                       const int num_val = amrex::ParmParse::LAST);

/** Similar to amrex::ParmParse::get, but also supports math expressions for the
 * value.
 *
 * amrex::ParmParse::get reads a name and a value from the input file. This
 * function does the same, and applies the Parser to the value, so the user has
 * the choice to specify a value or a math expression (including user-defined
 * constants). Works for amrex::Real numbers and integers.
 *
 * \param[in] a_pp amrex::ParmParse object
 * \param[in] str name of the parameter to read
 * \param[out] val where the value queried and parsed is stored
 */
void getWithParser(const amrex::ParmParse &a_pp, char const *const str,
                   float &val);
void getWithParser(const amrex::ParmParse &a_pp, char const *const str,
                   double &val);

/** Similar to amrex::ParmParse::get, but also supports math expressions for the
 * value.
 *
 * amrex::ParmParse::get reads a name and a value from the input file. This
 * function does the same, and applies the Parser to the value, so the user has
 * the choice to specify a value or a math expression (including user-defined
 * constants). Works for amrex::Real numbers and integers.
 *
 * \param[in] a_pp amrex::ParmParse object
 * \param[in] str name of the parameter to read
 * \param[out] val where the value queried and parsed is stored
 * \param[in] start_ix start index in the list of inputs values (optional with
 * arrays, default is amrex::ParmParse::FIRST for starting with the first input
 * value) \param[in] num_val number of input values to use (optional with
 * arrays, default is amrex::ParmParse::LAST for reading until the last input
 * value)
 */
void getArrWithParser(const amrex::ParmParse &a_pp, char const *const str,
                      std::vector<amrex::Real> &val,
                      const int start_ix = amrex::ParmParse::FIRST,
                      const int num_val = amrex::ParmParse::LAST);
void getWithParser(const amrex::ParmParse &a_pp, char const *const str,
                   int &val);
void getArrWithParser(const amrex::ParmParse &a_pp, char const *const str,
                      std::vector<int> &val,
                      const int start_ix = amrex::ParmParse::FIRST,
                      const int num_val = amrex::ParmParse::LAST);

namespace WarpXUtilStr
{
/** Return true if elem is in vect, false otherwise
 * @param[in] vect vector of strings, typically names
 * @param[in] elem single string
 * @return true if elem is in vect, false otherwise
 */
bool is_in(const std::vector<std::string> &vect, const std::string &elem);

/** Return true if any element in elems is in vect, false otherwise
 * @param[in] vect vector of strings, typically names
 * @param[in] elems vector of string
 * @return true if any element in elems is in vect, false otherwise
 */
bool is_in(const std::vector<std::string> &vect,
           const std::vector<std::string> &elems);

/** \brief Splits a string using a string separator. This is somewhat similar to
 *  amrex::Tokenize. The main difference is that, if the separator ":" is used,
 *  amrex::Tokenize will split ":3::2" into ["3","2"] while this function will
 *  split ":3::2" into ["","3","","2"]. This function can also perform a
 * trimming to remove whitespaces (or any other arbitrary string) from the split
 * string.
 *
 * @tparam Container the type of the split string.
 *
 * @param[in] instr the input string
 * @param[in] separator the separator string
 * @param[in] trim true to trim the split string, false otherwise.
 * @param[in] trim_space the string to trim if trim is true.
 * @return cont the split string
 */
template <typename Container>
auto split(std::string const &instr, std::string const &separator,
           bool const trim = false, std::string const &trim_space = " \t")
{
    Container cont;
    std::size_t current = instr.find(separator);
    std::size_t previous = 0;
    while (current != std::string::npos)
    {
        if (trim)
        {
            cont.push_back(
                amrex::trim(instr.substr(previous, current - previous),
                            trim_space));
        }
        else
        {
            cont.push_back(instr.substr(previous, current - previous));
        }
        previous = current + separator.size();
        current = instr.find(separator, previous);
    }
    if (trim)
    {
        cont.push_back(amrex::trim(instr.substr(previous, current - previous),
                                   trim_space));
    }
    else
    {
        cont.push_back(instr.substr(previous, current - previous));
    }
    return cont;
}

/** \brief This function performs automatic text wrapping on a string,
 *  returning an array of strings each not exceeding the maximum line length
 *  (unless the text contains a word exceeding the maximum line length).
 *
 * @param[in] text the string containing the text to be wrapped
 * @param[in] max_line_length the maximum line length
 * @return an std::vector containing the lines of the wrapped text
 */
std::vector<std::string> automatic_text_wrap(const std::string &text,
                                             const int max_line_length);

}  // namespace WarpXUtilStr

namespace WarpXUtilLoadBalance
{
/** \brief We only want to update the cost data if the grids we are working on
 *  are the main grids, i.e. not the PML grids. This function returns whether
 *   this is the case or not.
 * @param[in] cost pointer to the cost data
 * @param[in] ba the grids to check
 * @param[in] dm the dmap to check
 * @return consistent whether the grids are consistent or not.
 */
bool doCosts(const amrex::LayoutData<amrex::Real> *cost,
             const amrex::BoxArray ba, const amrex::DistributionMapping &dm);
}  // namespace WarpXUtilLoadBalance

#endif  // WARPX_UTILS_H_
