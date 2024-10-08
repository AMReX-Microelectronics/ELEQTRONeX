#ifndef EMBEDDED_BOUNDARIES_H_
#define EMBEDDED_BOUNDARIES_H_

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBSupport.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <any>
#include <string>

#include "EmbeddedBoundaries_fwd.H"

struct s_STL
{
    std::string file;
    amrex::Real scale;
    std::vector<amrex::Real> center;
    int reverse_normal;

    s_STL()
    {
        using namespace amrex;
        scale = 1._rt;
        center = {0.0_rt, 0.0_rt, 0.0_rt};
        reverse_normal = 0;
    };
};

struct s_PARSER
{
    std::string parser_function_str;
};

struct s_ObjectType
{
    enum object
    {
        sphere,
        box,
        cylinder,
        plane,
        torus,
        parser,
        stl,
        cntfet_contact,
        cntfet_contact_cyl,
        cntfet_contact_rect
    };
};

class c_EmbeddedBoundaries
{
   public:
    c_EmbeddedBoundaries(const amrex::Array<amrex::Real, AMREX_SPACEDIM> *dx);

    int required_coarsening_level;
    int max_coarsening_level;
    int specify_input_using_eb2;
    // int specify_separate_surf_beta;
    int specify_separate_surf_soln;
    // amrex::Real surf_beta;
    amrex::Real surf_soln;
    int specify_inhomogeneous_dirichlet;
    int num_objects;
    amrex::EBSupport support;

    std::unique_ptr<amrex::EBFArrayBoxFactory> p_factory_union;
    // std::unique_ptr<amrex::MultiFab> p_surf_beta_union;
    std::unique_ptr<amrex::MultiFab> p_surf_soln_union;

    const amrex::Geometry *geom;
    const amrex::BoxArray *ba;
    const amrex::DistributionMapping *dm;

    amrex::MultiFab &get_soln_mf(int i) { return (*m_p_soln_mf[i]); }
    // amrex::MultiFab& get_beta_mf  (int i) {return (*m_p_beta_mf[i]);}
    void clear_soln_mf()
    {
        for (int i = 0; i < m_p_soln_mf.size(); ++i)
        {
            m_p_soln_mf[i].reset();
        }
        m_p_soln_mf.clear();
    };
    // void clear_beta_mf() { m_p_beta_mf.clear(); };

    void BuildGeometry(const amrex::Geometry *geom, const amrex::BoxArray *ba,
                       const amrex::DistributionMapping *dm);

    void Update_SurfaceSolution(const amrex::Real time);

    amrex::Real Read_SurfSoln(const std::string name);

   private:
    amrex::Vector<std::string> vec_object_names;
    std::map<std::string, std::string> map_basic_objects_type;
    std::map<std::string, std::any> map_basic_objects_info;
    std::map<std::string, amrex::Real> map_basic_objects_soln;
    std::map<std::string, int> map_basic_objects_soln_parser_flag;
    std::map<std::string, std::unique_ptr<amrex::Parser>>
        map_basic_objects_soln_parser;
    std::map<std::string, amrex::Real> map_basic_objects_beta;
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> *_dx;

    std::map<std::string, s_ObjectType::object> map_object_type_enum = {
        {"sphere", s_ObjectType::object::sphere},
        {"box", s_ObjectType::object::box},
        {"cylinder", s_ObjectType::object::cylinder},
        {"plane", s_ObjectType::object::plane},
        {"torus", s_ObjectType::object::torus},
        {"parser", s_ObjectType::object::parser},
        {"stl", s_ObjectType::object::stl},
        {"cntfet_contact", s_ObjectType::object::cntfet_contact},
        {"cntfet_contact_cyl", s_ObjectType::object::cntfet_contact_cyl},
        {"cntfet_contact_rect", s_ObjectType::object::cntfet_contact_rect},

        {"Sphere", s_ObjectType::object::sphere},
        {"Box", s_ObjectType::object::box},
        {"Cylinder", s_ObjectType::object::cylinder},
        {"Plane", s_ObjectType::object::plane},
        {"Torus", s_ObjectType::object::torus},
        {"Parser", s_ObjectType::object::parser},
        {"Stl", s_ObjectType::object::stl},
        {"Cntfet_contact", s_ObjectType::object::cntfet_contact},
        {"Cntfet_contact_cyl", s_ObjectType::object::cntfet_contact_cyl},
        {"Cntfet_contact_rect", s_ObjectType::object::cntfet_contact_rect},

        {"STL", s_ObjectType::object::stl},
        {"CNTFET_contact", s_ObjectType::object::cntfet_contact},
        {"CNTFET_contact_cyl", s_ObjectType::object::cntfet_contact_cyl},
        {"CNTFET_contact_rect", s_ObjectType::object::cntfet_contact_rect},
    };
    using cntfet_contact_type =
        amrex::EB2::IntersectionIF<amrex::EB2::CylinderIF, amrex::EB2::BoxIF>;
    using cntfet_contact_cyl_type =
        amrex::EB2::IntersectionIF<amrex::EB2::CylinderIF,
                                   amrex::EB2::CylinderIF>;
    using cntfet_contact_rect_type =
        amrex::EB2::IntersectionIF<amrex::EB2::BoxIF, amrex::EB2::BoxIF>;

    int num_final_objects;
    std::map<int, int> map_basic_objects;

    std::map<std::string, amrex::EBSupport> map_eb_support = {
        {"full", amrex::EBSupport::full},
        {"basic", amrex::EBSupport::basic},
        {"volume", amrex::EBSupport::volume},
        {"Full", amrex::EBSupport::full},
        {"Basic", amrex::EBSupport::basic},
        {"Volume", amrex::EBSupport::volume},
    };

    std::map<std::string, amrex::Real> map_SurfSoln_Name_Value;

    void ReadGeometry();

    void ReadObjectInfo(std::string object_name, std::string object_type,
                        amrex::ParmParse pp_object);

    template <typename ObjectType>
    void BuildSingleObject(std::string name);

    template <typename ObjectType>
    void BuildSingleParserObject(std::string name);

    template <typename ObjectType1, typename ObjectType2>
    void BuildUnionObject(std::string name1, std::string name2);

    template <typename ObjectType1, typename ObjectType2, typename ObjectType3>
    void BuildUnionObject(std::string name1, std::string name2,
                          std::string name3);

    template <typename ObjectType1, typename ObjectType2>
    void BuildUnionParserObject(std::string name1, std::string name2);

    template <typename ObjectType>
    void BuildSingleSTLObject(std::string name);

    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_p_soln_mf;
    // amrex::Vector< std::unique_ptr<amrex::MultiFab> > m_p_beta_mf;
    amrex::Vector<std::unique_ptr<amrex::EBFArrayBoxFactory>> m_p_factory;

#ifdef TIME_DEPENDENT
    amrex::Vector<std::string> m_parser_varname_vector = {"x", "y", "z", "t"};
#else
    amrex::Vector<std::string> m_parser_varname_vector = {"x", "y", "z"};
#endif

    void Specify_SurfSolnOnCutcells(amrex::MultiFab &mf, const std::string name,
                                    const amrex::Real time);
};
#endif
