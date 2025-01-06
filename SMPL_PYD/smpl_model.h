#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>

/**
 * SMPLModel: SMPL에 필요한 데이터(버텍스/관절 수, 템플릿, shapedirs, posedirs, ...)를 모은 구조체
 */
struct SMPLModel
{
    int N;                          // 버텍스(vertex) 개수
    int K;                          // 관절(joint) 개수
    Eigen::MatrixXi faces;          // 삼각형 폴리곤 면 정보
    Eigen::MatrixXi kintree_table;  // 관절 트리 (부모-자식)
    Eigen::MatrixXd weights;        // 각 버텍스별 관절 K개에 대한 스키닝 가중치 (N x K)
    Eigen::MatrixXd shapedirs_vec;  // (4*N x shapedim)
    Eigen::MatrixXd posedirs_vec;   // (4*N x posedim)
    Eigen::VectorXd v_tem_vec;      // 템플릿 버텍스 (4*N)
    Eigen::SparseMatrix<double> J_reg_vec; // 관절 위치 추출용 Regressor (4K x 4N)
};

/**
 * buildSMPLModel
 *  - JSON 데이터로부터 SMPLModel 구조체를 구성하는 함수
 */
SMPLModel buildSMPLModel();

/**
 * SMPL_Calc
 *  - thetas, betas를 이용해 최종 버텍스(verts)와 관절(joints)을 계산
 */
void SMPL_Calc(
    const Eigen::VectorXd& thetas,  // (3*K) or (3*(K-1)) 등
    const Eigen::VectorXd& betas,   // (shapedim) 보통 300차원
    const SMPLModel& model,
    Eigen::MatrixXd& verts,         // 출력: 최종 버텍스 (N x 4 또는 N x 3 형태)
    Eigen::VectorXd& joints         // 출력: 관절 위치 (K*3)
);

