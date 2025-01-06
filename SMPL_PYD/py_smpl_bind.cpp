#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "smpl_model.h"

namespace py = pybind11;

/**
 * SMPL_Calc을 직접 호출하면 (void) 반환이며, verts / joints는 레퍼런스로 받음.
 * Python 사용 편의를 위해, 아래 래퍼 함수를 만들어 (verts, joints)를 리턴값으로 돌려주도록 한다.
 */
std::pair<Eigen::MatrixXd, Eigen::VectorXd>
SMPL_Calc_wrapper(const Eigen::VectorXd& thetas,
                  const Eigen::VectorXd& betas,
                  const SMPLModel& model)
{
    Eigen::MatrixXd verts;
    Eigen::VectorXd joints;
    SMPL_Calc(thetas, betas, model, verts, joints);
    return {verts, joints};
}

PYBIND11_MODULE(PySMPL, m) {
    m.doc() = "Python binding for C++ SMPLModel example";

    // SMPLModel 클래스를 파이썬에 노출
    py::class_<SMPLModel>(m, "SMPLModel")
        .def(py::init<>())
        .def_readwrite("N", &SMPLModel::N)
        .def_readwrite("K", &SMPLModel::K)
        // 필요하다면 faces, kintree_table 등도 .def_readwrite(...)로 노출 가능
        ;

    // buildSMPLModel 함수 바인딩
    m.def("buildSMPLModel", &buildSMPLModel,
          "Always read model.json and build SMPLModel.");

    // SMPL_Calc 래퍼 바인딩
    m.def("SMPL_Calc", &SMPL_Calc_wrapper,
          "SMPL calculation => returns (verts, joints).");
}
