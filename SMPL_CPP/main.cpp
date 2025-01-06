#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>

struct SMPLModel
{
    int N;
    int K;
    Eigen::MatrixXi faces;
    Eigen::MatrixXi kintree_table;
    Eigen::MatrixXd weights;
    Eigen::MatrixXd shapedirs_vec;
    Eigen::MatrixXd posedirs_vec;
    Eigen::VectorXd v_tem_vec;
    Eigen::SparseMatrix<double> J_reg_vec;
};

SMPLModel buildSMPLModel(const nlohmann::json& data);
void loadJSON(const std::string& filename, nlohmann::json& j);
void SMPL_Calc(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& betas,
    const SMPLModel& model,
    Eigen::MatrixXd& verts,
    Eigen::VectorXd& joints
);
void shapeblend(
    const Eigen::VectorXd& betas,
    const SMPLModel& model,
    Eigen::VectorXd& j_shaped,
    Eigen::VectorXd& v_shaped
);
void transforms(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& j_shaped,
    const SMPLModel& model,
    Eigen::MatrixXd& global_transform,
    Eigen::MatrixXd& global_transform_remove
);
Eigen::Matrix3d so3exp(const Eigen::Vector3d& omega);
void poseblend(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& betas,
    const Eigen::VectorXd& v_shaped,
    const SMPLModel& model,
    Eigen::VectorXd& v_shaped2
);
void poserot(
    const Eigen::MatrixXd& global_transform_remove,
    const Eigen::VectorXd& v_shaped,
    const SMPLModel& model,
    Eigen::MatrixXd& v_rot
);

int main()
{
    nlohmann::json modelJson;
    loadJSON("model.json", modelJson);
    SMPLModel smplModel = buildSMPLModel(modelJson);

    nlohmann::json motionJson;
    loadJSON("walking_20240125.json", motionJson);

    if (!motionJson.is_array() || motionJson.size() == 0)
    {
        std::cerr << "Motion JSON is empty or not an array!" << std::endl;
        return -1;
    }

    size_t numFrame = motionJson.size();
    size_t motionDim = motionJson[0].size();
    Eigen::MatrixXd motionData(
        static_cast<Eigen::Index>(numFrame),
        static_cast<Eigen::Index>(motionDim)
    );

    for (size_t i = 0; i < numFrame; i++)
    {
        for (size_t j = 0; j < motionDim; j++)
        {
            motionData(
                static_cast<Eigen::Index>(i),
                static_cast<Eigen::Index>(j)
            ) = motionJson[i][j].get<double>();
        }
    }

    Eigen::VectorXd betas = Eigen::VectorXd::Zero(300);

    for (size_t i = 0; i < numFrame; i++)
    {
        Eigen::Vector3d trans;
        trans(0) = motionData(static_cast<Eigen::Index>(i), 0);
        trans(1) = motionData(static_cast<Eigen::Index>(i), 1);
        trans(2) = motionData(static_cast<Eigen::Index>(i), 2);

        Eigen::VectorXd thetas(motionDim - 3);
        for (size_t k = 3; k < motionDim; k++)
        {
            thetas(static_cast<Eigen::Index>(k - 3)) =
                motionData(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k))
                + 1e-5;
        }

        Eigen::MatrixXd verts;
        Eigen::VectorXd joints;
        SMPL_Calc(thetas, betas, smplModel, verts, joints);
    }

    return 0;
}

void loadJSON(const std::string& filename, nlohmann::json& j)
{
    std::ifstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    f >> j;
}

SMPLModel buildSMPLModel(const nlohmann::json& data)
{
    SMPLModel model;
    std::vector<std::vector<double>> v_template_json = data["v_template"];
    int N = static_cast<int>(v_template_json.size());
    Eigen::MatrixXd v_template(N, 3);
    for (int i = 0; i < N; i++) {
        v_template(i, 0) = v_template_json[i][0];
        v_template(i, 1) = v_template_json[i][1];
        v_template(i, 2) = v_template_json[i][2];
    }

    auto shapedirs_json = data["shapedirs"];
    int shapedim = 300;
    Eigen::MatrixXd shapedirs(N * 3, shapedim);
    for (int dd = 0; dd < shapedim; dd++) {
        for (int i = 0; i < N; i++) {
            shapedirs(i * 3 + 0, dd) = shapedirs_json[i][0][dd].get<double>();
            shapedirs(i * 3 + 1, dd) = shapedirs_json[i][1][dd].get<double>();
            shapedirs(i * 3 + 2, dd) = shapedirs_json[i][2][dd].get<double>();
        }
    }

    auto posedirs_json = data["posedirs"];
    int posedim = 93;
    Eigen::MatrixXd posedirs(N * 3, posedim);
    for (int dd = 0; dd < posedim; dd++) {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < 3; c++) {
                posedirs(r + c * N, dd) = posedirs_json[r][c][dd].get<double>();
            }
        }
    }

    std::vector<std::vector<int>> faces_json = data["f"];
    int F = static_cast<int>(faces_json.size());
    Eigen::MatrixXi faces(F, 3);
    for (int i = 0; i < F; i++) {
        faces(i, 0) = faces_json[i][0] + 1;
        faces(i, 1) = faces_json[i][1] + 1;
        faces(i, 2) = faces_json[i][2] + 1;
    }

    std::vector<std::vector<int>> kt_json = data["kintree_table"];
    int K = static_cast<int>(kt_json[0].size());
    Eigen::MatrixXi kintree_table(2, K);
    for (int col = 0; col < K; col++) {
        kintree_table(0, col) = kt_json[0][col] + 1;
        kintree_table(1, col) = kt_json[1][col] + 1;
    }

    std::vector<std::vector<double>> weights_json = data["weights"];
    Eigen::MatrixXd weights(N, K);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            weights(i, k) = weights_json[i][k];
        }
    }

    std::vector<std::vector<double>> Jreg_json = data["J_regressor"];
    int actualK = (int)Jreg_json.size();
    int actualN = (int)Jreg_json[0].size();
    Eigen::SparseMatrix<double> J_reg(actualK, actualN);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int k = 0; k < actualK; k++) {
        for (int n = 0; n < actualN; n++) {
            double val = Jreg_json[k][n];
            if (std::abs(val) > 1e-12) {
                triplets.push_back(Eigen::Triplet<double>(k, n, val));
            }
        }
    }
    J_reg.setFromTriplets(triplets.begin(), triplets.end());

    model.N = N;
    model.K = K;
    model.faces = faces;
    model.kintree_table = kintree_table;
    model.weights = weights;

    Eigen::MatrixXd shapedirs_vec(4 * N, shapedim);
    shapedirs_vec << shapedirs,
        Eigen::MatrixXd::Zero(N, shapedim);
    model.shapedirs_vec = shapedirs_vec;

    Eigen::MatrixXd posedirs_vec(4 * N, posedim);
    posedirs_vec << posedirs,
        Eigen::MatrixXd::Zero(N, posedim);
    model.posedirs_vec = posedirs_vec;

    Eigen::VectorXd v_tem_vec(4 * N);
    int idx = 0;
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < N; row++) {
            v_tem_vec(idx++) = v_template(row, col);
        }
    }
    for (int i = 0; i < N; i++) {
        v_tem_vec(3 * N + i) = 1.0;
    }
    model.v_tem_vec = v_tem_vec;

    Eigen::SparseMatrix<double> J_reg_vec(4 * K, 4 * N);
    {
        std::vector<Eigen::Triplet<double>> triplets_big;
        for (int c = 0; c < J_reg.cols(); c++) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(J_reg, c); it; ++it) {
                int r = static_cast<int>(it.row());
                double val = it.value();
                triplets_big.push_back(Eigen::Triplet<double>(r, c, val));
                triplets_big.push_back(Eigen::Triplet<double>(K + r, N + c, val));
                triplets_big.push_back(Eigen::Triplet<double>(2 * K + r, 2 * N + c, val));
            }
        }
        J_reg_vec.setFromTriplets(triplets_big.begin(), triplets_big.end());
    }
    model.J_reg_vec = J_reg_vec;

    return model;
}

void SMPL_Calc(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& betas,
    const SMPLModel& model,
    Eigen::MatrixXd& verts,
    Eigen::VectorXd& joints
)
{
    Eigen::VectorXd j_shaped, v_shaped;
    shapeblend(betas, model, j_shaped, v_shaped);
    Eigen::MatrixXd global_transform, global_transform_remove;
    transforms(thetas, j_shaped, model, global_transform, global_transform_remove);
    Eigen::VectorXd v_shaped2;
    poseblend(thetas, betas, v_shaped, model, v_shaped2);
    Eigen::MatrixXd v_rot;
    poserot(global_transform_remove, v_shaped2, model, v_rot);
    verts = v_rot;
    joints.resize(model.K * 3);
    auto idx = [&](int row, int col) {
        return row * 4 + col;
        };
    for (int i = 0; i < model.K; i++) {
        joints(i) = global_transform(i, idx(0, 3));
        joints(i + model.K) = global_transform(i, idx(1, 3));
        joints(i + 2 * model.K) = global_transform(i, idx(2, 3));
    }
}

void shapeblend(
    const Eigen::VectorXd& betas,
    const SMPLModel& model,
    Eigen::VectorXd& j_shaped,
    Eigen::VectorXd& v_shaped
)
{
    Eigen::VectorXd v_shapeblend = model.shapedirs_vec * betas;
    v_shaped = model.v_tem_vec + v_shapeblend;
    Eigen::VectorXd j_temp = model.J_reg_vec * v_shaped;
    j_shaped = j_temp;
}

void transforms(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& j_shaped,
    const SMPLModel& model,
    Eigen::MatrixXd& global_transform,
    Eigen::MatrixXd& global_transform_remove
)
{
    global_transform.resize(model.K, 16);
    global_transform.setZero();
    global_transform_remove.resize(model.K, 16);
    global_transform_remove.setZero();
    auto colMajorIdx = [&](int row, int col) {
        return col * 4 + row;
        };
    for (int i = 0; i < model.K; i++)
    {
        Eigen::Vector3d omega;
        omega << thetas(3 * i), thetas(3 * i + 1), thetas(3 * i + 2);
        Eigen::Matrix3d rotmat = so3exp(omega);
        if (i == 0)
        {
            double tx = j_shaped(i);
            double ty = j_shaped(i + model.K);
            double tz = j_shaped(i + 2 * model.K);
            global_transform(i, colMajorIdx(0, 0)) = rotmat(0, 0);
            global_transform(i, colMajorIdx(1, 0)) = rotmat(1, 0);
            global_transform(i, colMajorIdx(2, 0)) = rotmat(2, 0);
            global_transform(i, colMajorIdx(0, 1)) = rotmat(0, 1);
            global_transform(i, colMajorIdx(1, 1)) = rotmat(1, 1);
            global_transform(i, colMajorIdx(2, 1)) = rotmat(2, 1);
            global_transform(i, colMajorIdx(0, 2)) = rotmat(0, 2);
            global_transform(i, colMajorIdx(1, 2)) = rotmat(1, 2);
            global_transform(i, colMajorIdx(2, 2)) = rotmat(2, 2);
            global_transform(i, colMajorIdx(0, 3)) = tx;
            global_transform(i, colMajorIdx(1, 3)) = ty;
            global_transform(i, colMajorIdx(2, 3)) = tz;
            global_transform(i, colMajorIdx(3, 3)) = 1.0;
        }
        else
        {
            int parentIdx = model.kintree_table(0, i) - 1;
            double tx = j_shaped(i) - j_shaped(parentIdx);
            double ty = j_shaped(i + model.K) - j_shaped(parentIdx + model.K);
            double tz = j_shaped(i + 2 * model.K) - j_shaped(parentIdx + 2 * model.K);
            Eigen::Matrix4d localTrans = Eigen::Matrix4d::Identity();
            localTrans.block<3, 3>(0, 0) = rotmat;
            localTrans(0, 3) = tx;
            localTrans(1, 3) = ty;
            localTrans(2, 3) = tz;
            Eigen::Matrix4d parentMat = Eigen::Matrix4d::Zero();
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    parentMat(r, c) = global_transform(parentIdx, colMajorIdx(r, c));
                }
            }
            Eigen::Matrix4d result = parentMat * localTrans;
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    global_transform(i, colMajorIdx(r, c)) = result(r, c);
                }
            }
        }
        Eigen::Vector4d jHere;
        jHere << j_shaped(i),
            j_shaped(i + model.K),
            j_shaped(i + 2 * model.K),
            0.0;
        Eigen::Matrix4d G;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                G(r, c) = global_transform(i, colMajorIdx(r, c));
            }
        }
        Eigen::Vector4d fx = G * jHere;
        Eigen::Matrix4d pack = Eigen::Matrix4d::Zero();
        pack(0, 3) = fx(0);
        pack(1, 3) = fx(1);
        pack(2, 3) = fx(2);
        Eigen::Matrix4d Gremove = G - pack;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                global_transform_remove(i, colMajorIdx(r, c)) = Gremove(r, c);
            }
        }
    }
}

Eigen::Matrix3d so3exp(const Eigen::Vector3d& omega)
{
    double theta = omega.norm();
    if (theta < 1e-6)
    {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d u = omega / theta;
    double w1 = u(0), w2 = u(1), w3 = u(2);
    Eigen::Matrix3d A;
    A << 0.0, -w3, w2,
        w3, 0.0, -w1,
        -w2, w1, 0.0;
    double alpha = std::cos(theta);
    double beta = std::sin(theta);
    double gamma = 1.0 - alpha;
    Eigen::Matrix3d R = alpha * Eigen::Matrix3d::Identity()
        + beta * A
        + gamma * (u * u.transpose());
    return R;
}

Eigen::Vector4d axis2quat_single(const Eigen::Vector3d& axisAngle)
{
    double angle = axisAngle.norm();
    if (angle < 1e-12) {
        return Eigen::Vector4d(0.0, 0.0, 0.0, -1.0);
    }
    Eigen::Vector3d u = axisAngle / angle;
    double half = angle * 0.5;
    double cos_val = std::cos(half);
    double sin_val = std::sin(half);
    double qx = u(0) * sin_val;
    double qy = u(1) * sin_val;
    double qz = u(2) * sin_val;
    double qw = cos_val - 1.0;
    return Eigen::Vector4d(qx, qy, qz, qw);
}

void poseblend(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& betas,
    const Eigen::VectorXd& v_shaped,
    const SMPLModel& model,
    Eigen::VectorXd& v_shaped2
)
{
    Eigen::VectorXd pose = thetas.segment(3, thetas.size() - 3);
    int nJoints = static_cast<int>(pose.size() / 3);
    Eigen::VectorXd quaternionAngle(4 * nJoints);
    for (int j = 0; j < nJoints; j++) {
        Eigen::Vector3d axisAngle(
            pose(3 * j + 0),
            pose(3 * j + 1),
            pose(3 * j + 2)
        );
        Eigen::Vector4d q = axis2quat_single(axisAngle);
        quaternionAngle(4 * j + 0) = q(0);
        quaternionAngle(4 * j + 1) = q(1);
        quaternionAngle(4 * j + 2) = q(2);
        quaternionAngle(4 * j + 3) = q(3);
    }
    double shape_feat = betas(0);
    Eigen::VectorXd feat(4 * nJoints + 1);
    feat.head(4 * nJoints) = quaternionAngle;
    feat(4 * nJoints) = shape_feat;
    v_shaped2 = v_shaped + model.posedirs_vec * feat;
}

void poserot(
    const Eigen::MatrixXd& global_transform_remove,
    const Eigen::VectorXd& v_shaped,
    const SMPLModel& model,
    Eigen::MatrixXd& v_rot
)
{
    using namespace Eigen;
    const int N = model.N;
    const int K = model.K;
    const MatrixXd& weights = model.weights;
    MatrixXd coefficients = weights * global_transform_remove;
    Map<const Matrix<double, Dynamic, 4, ColMajor>> v_shaped_nor(v_shaped.data(), N, 4);
    v_rot.resize(N, 4);
    for (int n = 0; n < N; ++n) {
        Matrix4d M;
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                M(row, col) = coefficients(n, col * 4 + row);
            }
        }
        Vector4d v = v_shaped_nor.row(n).transpose();
        Vector4d result = M * v;
        v_rot.row(n) = result.transpose();
    }
}
