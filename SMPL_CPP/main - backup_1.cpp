#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>

//-----------------------------------------------------------------------------
// SMPL 모델을 보관할 구조체 정의
//-----------------------------------------------------------------------------
struct SMPLModel
{
    int N;  // number of vertices
    int K;  // number of joints (kintree_table의 열 수)

    Eigen::MatrixXi faces;         // faces: (F x 3)
    Eigen::MatrixXi kintree_table; // (2 x K)
    Eigen::MatrixXd weights;       // (N x K)
    
    // reshape된 shapedirs, posedirs: (4N x numCoeffs)
    // 4N = (N*3 + N) = 동차좌표 포함
    Eigen::MatrixXd shapedirs_vec; // (4N x 300)
    Eigen::MatrixXd posedirs_vec;  // (4N x 93)

    // template vertex (4N x 1)
    Eigen::VectorXd v_tem_vec;

    // joint regressor (4K x 4N)
    Eigen::SparseMatrix<double> J_reg_vec;
};

// 함수 선언
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

//-----------------------------------------------------------------------------
// 메인 함수
//-----------------------------------------------------------------------------
int main()
{
    // 1) Load model data (model.json)
    nlohmann::json modelJson;
    loadJSON("model.json", modelJson);

    // SMPL_Model build
    SMPLModel smplModel = buildSMPLModel(modelJson);

    // 2) Load motion data (walking_20240125.json)
    nlohmann::json motionJson;
    loadJSON("walking_20240125.json", motionJson);

    // motionData가 Nx(3 + numAngles) 구조라고 가정
    // 예: motionJson은 [[transX, transY, transZ, theta1, theta2, ...], [...], ...]
    if(!motionJson.is_array() || motionJson.size() == 0)
    {
        std::cerr << "Motion JSON is empty or not an array!" << std::endl;
        return -1;
    }

    // numFrame, motionDim을 size_t로
    size_t numFrame = motionJson.size();
    size_t motionDim = motionJson[0].size(); 

    // Eigen::MatrixXd 생성 시, 인자는 Eigen::Index로 변환
    Eigen::MatrixXd motionData(
        static_cast<Eigen::Index>(numFrame),
        static_cast<Eigen::Index>(motionDim)
    );

    // for문도 size_t로
    for(size_t i = 0; i < numFrame; i++)
    {
        for(size_t j = 0; j < motionDim; j++)
        {
            motionData(
                static_cast<Eigen::Index>(i),
                static_cast<Eigen::Index>(j)
            ) = motionJson[i][j].get<double>();
        }
    }

    // betas 예시 (300-dim)
    Eigen::VectorXd betas = Eigen::VectorXd::Zero(300);

    // 프레임 단위로 SMPL 계산 수행
    for(size_t i = 0; i < numFrame; i++)
    {
        // trans (이동)
        Eigen::Vector3d trans;
        trans(0) = motionData(static_cast<Eigen::Index>(i), 0);
        trans(1) = motionData(static_cast<Eigen::Index>(i), 1);
        trans(2) = motionData(static_cast<Eigen::Index>(i), 2);

        // thetas (축-각) (+ 작은 offset)
        Eigen::VectorXd thetas(motionDim - 3);
        for(size_t k = 3; k < motionDim; k++)
        {
            thetas(static_cast<Eigen::Index>(k - 3)) =
                motionData(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k))
                + 1e-5;
        }

        // SMPL 계산
        Eigen::MatrixXd verts;
        Eigen::VectorXd joints;
        SMPL_Calc(thetas, betas, smplModel, verts, joints);

        // 시각화 또는 후처리
        if(verts.rows() > 0)
        {
            // (N x 4)라면 x,y,z 만 추출
            Eigen::MatrixXd xyz = verts.leftCols(3);

            // translation 적용
            for(Eigen::Index vi = 0; vi < xyz.rows(); vi++)
            {
                xyz(vi, 0) += trans(0);
                xyz(vi, 1) += trans(1);
                xyz(vi, 2) += trans(2);
            }

            // 첫 번째 정점 좌표 출력
            std::cout << "[Frame " << i << "] First vertex: ("
                      << xyz(0,0) << ", "
                      << xyz(0,1) << ", "
                      << xyz(0,2) << ")" << std::endl;
        }
    }

    return 0;
}

//-----------------------------------------------------------------------------
// JSON 로더
//-----------------------------------------------------------------------------
void loadJSON(const std::string& filename, nlohmann::json& j)
{
    std::ifstream f(filename);
    if(!f.is_open())
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    f >> j;
}

//-----------------------------------------------------------------------------
// SMPL 모델 빌드 (MATLAB의 buildSMPLModel 대응)
//-----------------------------------------------------------------------------
SMPLModel buildSMPLModel(const nlohmann::json& data)
{
    SMPLModel model;

    // 예시로 v_template, shapedirs, posedirs 등 파싱
    std::vector<std::vector<double>> v_template_json = data["v_template"];
    int N = static_cast<int>(v_template_json.size());

    Eigen::MatrixXd v_template(N, 3);
    for(int i=0; i<N; i++){
        v_template(i, 0) = v_template_json[i][0];
        v_template(i, 1) = v_template_json[i][1];
        v_template(i, 2) = v_template_json[i][2];
    }

    // shapedirs (N x 3 x 300)
    auto shapedirs_json = data["shapedirs"];
    int shapedim = 300;
    Eigen::MatrixXd shapedirs(N*3, shapedim);
    for(int dd=0; dd<shapedim; dd++){
        for(int i=0; i<N; i++){
            shapedirs(i*3 + 0, dd) = shapedirs_json[i][0][dd].get<double>();
            shapedirs(i*3 + 1, dd) = shapedirs_json[i][1][dd].get<double>();
            shapedirs(i*3 + 2, dd) = shapedirs_json[i][2][dd].get<double>();
        }
    }

    // posedirs (N x 3 x 93)
    auto posedirs_json = data["posedirs"];
    int posedim = 93;
    Eigen::MatrixXd posedirs(N*3, posedim);
    for(int dd=0; dd<posedim; dd++){
        for(int i=0; i<N; i++){
            posedirs(i*3 + 0, dd) = posedirs_json[i][0][dd].get<double>();
            posedirs(i*3 + 1, dd) = posedirs_json[i][1][dd].get<double>();
            posedirs(i*3 + 2, dd) = posedirs_json[i][2][dd].get<double>();
        }
    }

    // faces (F x 3)
    std::vector<std::vector<int>> faces_json = data["f"];
    int F = static_cast<int>(faces_json.size());
    Eigen::MatrixXi faces(F, 3);
    for(int i=0; i<F; i++){
        faces(i,0) = faces_json[i][0] + 1;
        faces(i,1) = faces_json[i][1] + 1;
        faces(i,2) = faces_json[i][2] + 1;
    }

    // kintree_table (2 x K)
    std::vector<std::vector<int>> kt_json = data["kintree_table"];
    int K = static_cast<int>(kt_json[0].size());
    Eigen::MatrixXi kintree_table(2, K);
    for(int col=0; col<K; col++){
        kintree_table(0, col) = kt_json[0][col] + 1;
        kintree_table(1, col) = kt_json[1][col] + 1;
    }

    // weights (N x K)
    std::vector<std::vector<double>> weights_json = data["weights"];
    Eigen::MatrixXd weights(N, K);
    for(int i=0; i<N; i++){
        for(int k=0; k<K; k++){
            weights(i,k) = weights_json[i][k];
        }
    }

    // J_regressor (K x N) -> Sparse
    std::vector<std::vector<double>> Jreg_json = data["J_regressor"];
    Eigen::SparseMatrix<double> J_reg(K, N);
    std::vector<Eigen::Triplet<double>> triplets;
    for(int r=0; r<K; r++){
        for(int c=0; c<N; c++){
            double val = Jreg_json[r][c];
            if(std::abs(val) > 1e-12){
                triplets.push_back(Eigen::Triplet<double>(r, c, val));
            }
        }
    }
    J_reg.setFromTriplets(triplets.begin(), triplets.end());

    // 구조체에 저장
    model.N = N;
    model.K = K;
    model.faces = faces;
    model.kintree_table = kintree_table;
    model.weights = weights;

    // shapedirs_vec: (4N x 300)
    Eigen::MatrixXd shapedirs_vec(4*N, shapedim);
    shapedirs_vec << shapedirs,
                     Eigen::MatrixXd::Zero(N, shapedim);
    model.shapedirs_vec = shapedirs_vec;

    // posedirs_vec: (4N x 93)
    Eigen::MatrixXd posedirs_vec(4*N, posedim);
    posedirs_vec << posedirs,
                    Eigen::MatrixXd::Zero(N, posedim);
    model.posedirs_vec = posedirs_vec;

    // v_template -> (N*3 x 1) -> (4N x 1)
    Eigen::VectorXd v_tem_vec(4*N);
    for(int i=0; i<N; i++){
        v_tem_vec(i*3 + 0) = v_template(i,0);
        v_tem_vec(i*3 + 1) = v_template(i,1);
        v_tem_vec(i*3 + 2) = v_template(i,2);
    }
    for(int i=0; i<N; i++){
        v_tem_vec(3*N + i) = 1.0;
    }
    model.v_tem_vec = v_tem_vec;

    // J_reg_vec: (4K x 4N)
    Eigen::SparseMatrix<double> J_reg_vec(4*K, 4*N);
    std::vector<Eigen::Triplet<double>> triplets_big;
    // 블록 확장
    for(int kRow=0; kRow<K; kRow++){
        for(Eigen::SparseMatrix<double>::InnerIterator it(J_reg, kRow); it; ++it){
            int colN = it.col();
            double val = it.value();

            // x block
            triplets_big.push_back(Eigen::Triplet<double>(kRow, colN, val));
            // y block
            triplets_big.push_back(Eigen::Triplet<double>(K + kRow, N + colN, val));
            // z block
            triplets_big.push_back(Eigen::Triplet<double>(2*K + kRow, 2*N + colN, val));
        }
    }
    // 나머지(4번째 block)는 0

    J_reg_vec.setFromTriplets(triplets_big.begin(), triplets_big.end());
    model.J_reg_vec = J_reg_vec;

    return model;
}

//-----------------------------------------------------------------------------
// [verts, joints] = SMPL_Calc(thetas, betas, SMPL_Model)
//-----------------------------------------------------------------------------
void SMPL_Calc(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& betas,
    const SMPLModel& model,
    Eigen::MatrixXd& verts,
    Eigen::VectorXd& joints
)
{
    // 1) shape blend
    Eigen::VectorXd j_shaped, v_shaped;
    shapeblend(betas, model, j_shaped, v_shaped);

    // 2) compute joint transforms
    Eigen::MatrixXd global_transform, global_transform_remove;
    transforms(thetas, j_shaped, model, global_transform, global_transform_remove);

    // 3) pose blend
    Eigen::VectorXd v_shaped2;
    poseblend(thetas, betas, v_shaped, model, v_shaped2);

    // 4) pose rotation
    Eigen::MatrixXd v_rot;
    poserot(global_transform_remove, v_shaped2, model, v_rot);

    // v_rot: (N x 4)
    verts = v_rot;

    // joints: global_transform의 (x,y,z) 추출
    // global_transform은 (K x 16) 형태로 (K개 4x4) 행렬을 flatten
    joints.resize(model.K * 3);

    // 4x4 인덱스를 1D로 변환하는 람다
    auto idx = [&](int row, int col){
        return row*4 + col; 
    };

    for(int i = 0; i < model.K; i++){
        // 4번째 열(= idx(0,3), idx(1,3), idx(2,3))이 translation
        joints(i)             = global_transform(i, idx(0,3));
        joints(i + model.K)   = global_transform(i, idx(1,3));
        joints(i + 2*model.K) = global_transform(i, idx(2,3));
    }
}

//-----------------------------------------------------------------------------
// shapeblend
//-----------------------------------------------------------------------------
void shapeblend(
    const Eigen::VectorXd& betas,
    const SMPLModel& model,
    Eigen::VectorXd& j_shaped,
    Eigen::VectorXd& v_shaped
)
{
    // v_shaped = v_tem_vec + shapedirs_vec * betas
    Eigen::VectorXd v_shapeblend = model.shapedirs_vec * betas;
    v_shaped = model.v_tem_vec + v_shapeblend;

    // j_shaped = J_reg_vec * v_shaped  (4K x 1)
    Eigen::VectorXd j_temp = model.J_reg_vec * v_shaped;
    j_shaped = j_temp;
}

//-----------------------------------------------------------------------------
// transforms
//-----------------------------------------------------------------------------
void transforms(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& j_shaped,
    const SMPLModel& model,
    Eigen::MatrixXd& global_transform,
    Eigen::MatrixXd& global_transform_remove
)
{
    // global_transform, global_transform_remove: (K x 16)
    global_transform.resize(model.K, 16);
    global_transform.setZero();
    global_transform_remove.resize(model.K, 16);
    global_transform_remove.setZero();

    // 보조 람다
    auto idx = [&](int row, int col) {
        return row*4 + col;
    };

    for(int i=0; i<model.K; i++)
    {
        // thetas -> 3D 축각
        Eigen::Vector3d omega;
        omega << thetas(3*i), thetas(3*i+1), thetas(3*i+2);
        Eigen::Matrix3d rotmat = so3exp(omega);

        // root joint
        if(i == 0)
        {
            double tx = j_shaped(i);
            double ty = j_shaped(i + model.K);
            double tz = j_shaped(i + 2*model.K);

            global_transform(i, idx(0,0)) = rotmat(0,0);
            global_transform(i, idx(0,1)) = rotmat(0,1);
            global_transform(i, idx(0,2)) = rotmat(0,2);
            global_transform(i, idx(1,0)) = rotmat(1,0);
            global_transform(i, idx(1,1)) = rotmat(1,1);
            global_transform(i, idx(1,2)) = rotmat(1,2);
            global_transform(i, idx(2,0)) = rotmat(2,0);
            global_transform(i, idx(2,1)) = rotmat(2,1);
            global_transform(i, idx(2,2)) = rotmat(2,2);

            global_transform(i, idx(0,3)) = tx;
            global_transform(i, idx(1,3)) = ty;
            global_transform(i, idx(2,3)) = tz;
            global_transform(i, idx(3,3)) = 1.0;
        }
        else
        {
            int parentIdx = model.kintree_table(0, i) - 1;
            double tx = j_shaped(i)          - j_shaped(parentIdx);
            double ty = j_shaped(i+model.K)  - j_shaped(parentIdx+model.K);
            double tz = j_shaped(i+2*model.K)- j_shaped(parentIdx+2*model.K);

            Eigen::Matrix4d localTrans = Eigen::Matrix4d::Identity();
            localTrans.block<3,3>(0,0) = rotmat;
            localTrans(0,3) = tx;
            localTrans(1,3) = ty;
            localTrans(2,3) = tz;

            Eigen::Matrix4d parentMat = Eigen::Matrix4d::Zero();
            for(int r=0; r<4; r++){
                for(int c=0; c<4; c++){
                    parentMat(r,c) = global_transform(parentIdx, idx(r,c));
                }
            }
            Eigen::Matrix4d result = parentMat * localTrans;

            // global_transform(i,:)
            for(int r=0; r<4; r++){
                for(int c=0; c<4; c++){
                    global_transform(i, idx(r,c)) = result(r,c);
                }
            }
        }

        // global_transform_remove(i)
        Eigen::Vector4d jHere;
        jHere << j_shaped(i), j_shaped(i+model.K), j_shaped(i+2*model.K), 0.0;

        Eigen::Matrix4d G;
        for(int r=0; r<4; r++){
            for(int c=0; c<4; c++){
                G(r,c) = global_transform(i, idx(r,c));
            }
        }
        Eigen::Vector4d fx = G * jHere;

        Eigen::Matrix4d pack = Eigen::Matrix4d::Zero();
        pack(0,3) = fx(0);
        pack(1,3) = fx(1);
        pack(2,3) = fx(2);

        Eigen::Matrix4d Gremove = G - pack;
        for(int r=0; r<4; r++){
            for(int c=0; c<4; c++){
                global_transform_remove(i, idx(r,c)) = Gremove(r,c);
            }
        }
    }
}

//-----------------------------------------------------------------------------
// so3exp
//-----------------------------------------------------------------------------
Eigen::Matrix3d so3exp(const Eigen::Vector3d& omega)
{
    double theta = omega.norm();
    if(theta < 1e-6)
    {
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Vector3d u = omega / theta;
    double w1 = u(0), w2 = u(1), w3 = u(2);

    Eigen::Matrix3d A;
    A <<  0.0, -w3,  w2,
          w3,  0.0, -w1,
         -w2,  w1,  0.0;

    double alpha = std::cos(theta);
    double beta  = std::sin(theta);
    double gamma = 1.0 - alpha;

    Eigen::Matrix3d R = alpha * Eigen::Matrix3d::Identity()
                      + beta  * A
                      + gamma * (u * u.transpose());
    return R;
}

//-----------------------------------------------------------------------------
// poseblend
//-----------------------------------------------------------------------------
void poseblend(
    const Eigen::VectorXd& thetas,
    const Eigen::VectorXd& betas,
    const Eigen::VectorXd& v_shaped,
    const SMPLModel& model,
    Eigen::VectorXd& v_shaped2
)
{
    int totalPoseDim = static_cast<int>(model.posedirs_vec.cols()); // 93
    Eigen::VectorXd posePart = thetas.segment(3, thetas.size() - 3);

    // 예시로 posePart를 그대로 feat에 복사 후, 맨 끝에 betas(0)을 넣음
    Eigen::VectorXd feat(totalPoseDim);
    int minSize = std::min<int>(posePart.size(), totalPoseDim - 1);
    for(int i = 0; i < minSize; i++){
        feat(i) = posePart(i);
    }
    feat(totalPoseDim - 1) = betas(0);

    for(int i = minSize; i < totalPoseDim - 1; i++){
        feat(i) = 0.0;
    }

    // v_shaped2 = v_shaped + posedirs_vec * feat
    v_shaped2 = v_shaped + model.posedirs_vec * feat;
}

//-----------------------------------------------------------------------------
// poserot
//-----------------------------------------------------------------------------
void poserot(
    const Eigen::MatrixXd& global_transform_remove,
    const Eigen::VectorXd& v_shaped,
    const SMPLModel& model,
    Eigen::MatrixXd& v_rot
)
{
    // global_transform_remove: (K x 16)
    // weights: (N x K)
    Eigen::MatrixXd coefficients = model.weights * global_transform_remove; // (N x 16)

    // v_shaped: (4N x 1) → (N x 4)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>>
        v_shaped_mat(v_shaped.data(), model.N, 4);

    // 결과 v_rot: (N x 4)
    v_rot.resize(model.N, 4);

    for(int n=0; n<model.N; n++)
    {
        double m00 = coefficients(n, 0);
        double m01 = coefficients(n, 1);
        double m02 = coefficients(n, 2);
        double m03 = coefficients(n, 3);
        double m10 = coefficients(n, 4);
        double m11 = coefficients(n, 5);
        double m12 = coefficients(n, 6);
        double m13 = coefficients(n, 7);
        double m20 = coefficients(n, 8);
        double m21 = coefficients(n, 9);
        double m22 = coefficients(n,10);
        double m23 = coefficients(n,11);
        double m30 = coefficients(n,12);
        double m31 = coefficients(n,13);
        double m32 = coefficients(n,14);
        double m33 = coefficients(n,15);

        double x = v_shaped_mat(n,0);
        double y = v_shaped_mat(n,1);
        double z = v_shaped_mat(n,2);
        double w = v_shaped_mat(n,3);

        double rx = m00*x + m01*y + m02*z + m03*w;
        double ry = m10*x + m11*y + m12*z + m13*w;
        double rz = m20*x + m21*y + m22*z + m23*w;
        double rw = m30*x + m31*y + m32*z + m33*w;

        v_rot(n,0) = rx;
        v_rot(n,1) = ry;
        v_rot(n,2) = rz;
        v_rot(n,3) = rw;
    }
}
