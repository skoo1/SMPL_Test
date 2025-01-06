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
    // for(size_t i = 0; i < numFrame; i++)
    for(size_t i = 0; i < 1; i++)
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

        std::cout << "verts (first 10 rows):" << std::endl;
        for(int i = 0; i < 10; i++)
        {
            for(int j = 0; j < verts.cols(); j++)
            {
                std::cout << std::fixed << std::setw(10) << std::setprecision(4) << verts(i, j);
            }
            std::cout << std::endl;
        }

        std::ofstream file("frame1verts.csv");
        if(!file.is_open())
        {
            std::cerr << "Error: Unable to open file " << "frame1verts.csv" << std::endl;
        }

        // 여기서 원하는 출력 포맷 설정 (소수점 4자리, 고정소수점)
        file << std::fixed << std::setprecision(4);

        // verts: (rows() x cols())
        for(int i = 0; i < verts.rows(); ++i)
        {
            for(int j = 0; j < verts.cols(); ++j)
            {
                file << verts(i,j);
                // 열 사이에는 쉼표를, 마지막 열에는 쉼표 대신 줄바꿈
                if(j < verts.cols() - 1)
                    file << ",";
            }
            file << "\n";
        }

        file.close();
        std::cout << "Saved matrix to " << "frame1verts.csv" << std::endl;







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

    //--------------------------------------------------------------------------
    // 1) v_template JSON 파싱
    //--------------------------------------------------------------------------
    std::vector<std::vector<double>> v_template_json = data["v_template"];
    int N = static_cast<int>(v_template_json.size());

    // v_template: (N x 3) 행렬 (일단 row-major 형태로 저장되지만,
    // 내부 값은 [ i행 ][ 0..2 ] = (x, y, z)
    Eigen::MatrixXd v_template(N, 3);
    for(int i = 0; i < N; i++){
        v_template(i, 0) = v_template_json[i][0];
        v_template(i, 1) = v_template_json[i][1];
        v_template(i, 2) = v_template_json[i][2];
    }

    //--------------------------------------------------------------------------
    // 2) shapedirs (N x 3 x 300)
    //--------------------------------------------------------------------------
    auto shapedirs_json = data["shapedirs"];
    int shapedim = 300; // 실제 JSON 구조 확인 필요
    Eigen::MatrixXd shapedirs(N * 3, shapedim);
    for(int dd = 0; dd < shapedim; dd++){
        for(int i = 0; i < N; i++){
            shapedirs(i*3 + 0, dd) = shapedirs_json[i][0][dd].get<double>();
            shapedirs(i*3 + 1, dd) = shapedirs_json[i][1][dd].get<double>();
            shapedirs(i*3 + 2, dd) = shapedirs_json[i][2][dd].get<double>();
        }
    }

    //--------------------------------------------------------------------------
    // 3) posedirs (N x 3 x 93)
    //--------------------------------------------------------------------------
    auto posedirs_json = data["posedirs"];
    int posedim = 93;
    Eigen::MatrixXd posedirs(N * 3, posedim);
    for(int dd = 0; dd < posedim; dd++){
        for(int i = 0; i < N; i++){
            posedirs(i*3 + 0, dd) = posedirs_json[i][0][dd].get<double>();
            posedirs(i*3 + 1, dd) = posedirs_json[i][1][dd].get<double>();
            posedirs(i*3 + 2, dd) = posedirs_json[i][2][dd].get<double>();
        }
    }

    //--------------------------------------------------------------------------
    // 4) faces (F x 3)
    //--------------------------------------------------------------------------
    std::vector<std::vector<int>> faces_json = data["f"];
    int F = static_cast<int>(faces_json.size());
    Eigen::MatrixXi faces(F, 3);
    for(int i=0; i<F; i++){
        faces(i,0) = faces_json[i][0] + 1;  // MATLAB 인덱스 보정
        faces(i,1) = faces_json[i][1] + 1;
        faces(i,2) = faces_json[i][2] + 1;
    }

    //--------------------------------------------------------------------------
    // 5) kintree_table (2 x K)
    //--------------------------------------------------------------------------
    std::vector<std::vector<int>> kt_json = data["kintree_table"];
    int K = static_cast<int>(kt_json[0].size());
    Eigen::MatrixXi kintree_table(2, K);
    for(int col=0; col<K; col++){
        kintree_table(0, col) = kt_json[0][col] + 1;
        kintree_table(1, col) = kt_json[1][col] + 1;
    }

    //--------------------------------------------------------------------------
    // 6) weights (N x K)
    //--------------------------------------------------------------------------
    std::vector<std::vector<double>> weights_json = data["weights"];
    Eigen::MatrixXd weights(N, K);
    for(int i=0; i<N; i++){
        for(int k=0; k<K; k++){
            weights(i,k) = weights_json[i][k];
        }
    }

    //--------------------------------------------------------------------------
    // 7) J_regressor (K x N) -> SparseMatrix
    //--------------------------------------------------------------------------
    std::vector<std::vector<double>> Jreg_json = data["J_regressor"];

    int rowCount = (int)Jreg_json.size();
    int colCount = (int)Jreg_json[0].size();
    std::cout << "J_regressor JSON: rowCount=" << rowCount 
              << ", colCount=" << colCount << std::endl;

    // Check dimension
    int actualK = (int)Jreg_json.size();    // 24
    int actualN = (int)Jreg_json[0].size(); // 6890

    Eigen::SparseMatrix<double> J_reg(actualK, actualN);
    // => rows=24, cols=6890

    std::vector<Eigen::Triplet<double>> triplets;
    for(int k = 0; k < actualK; k++){
        for(int n = 0; n < actualN; n++){
            double val = Jreg_json[k][n]; // row=k, col=n
            if(std::abs(val) > 1e-12){
                triplets.push_back(Eigen::Triplet<double>(k, n, val));
            }
        }
    }
    J_reg.setFromTriplets(triplets.begin(), triplets.end());


    // J_reg 실제 non-zero 개수 찍기
    std::cout << "J_reg.nonZeros() = " << J_reg.nonZeros() << std::endl;

    double checkVal = J_reg.coeff(13, 701); 
    std::cout << "J_reg(13,701) = " << checkVal << std::endl;


    //--------------------------------------------------------------------------
    // 구조체에 저장할 기본 정보
    //--------------------------------------------------------------------------
    model.N = N;
    model.K = K;
    model.faces = faces;
    model.kintree_table = kintree_table;
    model.weights = weights;

    //--------------------------------------------------------------------------
    // 8) shapedirs_vec: (4N x 300)
    //--------------------------------------------------------------------------
    Eigen::MatrixXd shapedirs_vec(4*N, shapedim);
    shapedirs_vec << shapedirs,
                     Eigen::MatrixXd::Zero(N, shapedim);
    model.shapedirs_vec = shapedirs_vec;

    //--------------------------------------------------------------------------
    // 9) posedirs_vec: (4N x 93)
    //--------------------------------------------------------------------------
    Eigen::MatrixXd posedirs_vec(4*N, posedim);
    posedirs_vec << posedirs,
                    Eigen::MatrixXd::Zero(N, posedim);
    model.posedirs_vec = posedirs_vec;

    //--------------------------------------------------------------------------
    // 10) v_template -> (N*3 x 1) -> (4N x 1)
    //    MATLAB: reshape(v_template, N*3, 1) ==> 열(column)우선
    //--------------------------------------------------------------------------
    Eigen::VectorXd v_tem_vec(4*N);

    // -- (A) column-major 방식으로 flatten --
    // MATLAB의 reshape(v_template, [N*3, 1])와 동일한 순서
    // 즉, v_template(:,1) -> v_template(:,2) -> v_template(:,3)
    int idx = 0;
    for(int col = 0; col < 3; col++){
        for(int row = 0; row < N; row++){
            v_tem_vec(idx++) = v_template(row, col);
        }
    }

    // 동차 좌표(homogeneous) 부분
    for(int i = 0; i < N; i++){
        v_tem_vec(3*N + i) = 1.0;
    }
    model.v_tem_vec = v_tem_vec;

    //--------------------------------------------------------------------------
    // 11) J_reg_vec: (4K x 4N)
    //    블록 확장
    //--------------------------------------------------------------------------
    Eigen::SparseMatrix<double> J_reg_vec(4*K, 4*N);
    {
        std::vector<Eigen::Triplet<double>> triplets_big;
        for(int c = 0; c < J_reg.cols(); c++){
            for(Eigen::SparseMatrix<double>::InnerIterator it(J_reg, c); it; ++it){
                int r = it.row();
                double val = it.value();
                // x block
                triplets_big.push_back(Eigen::Triplet<double>(r, c, val));
                // y block
                triplets_big.push_back(Eigen::Triplet<double>(K + r, N + c, val));
                // z block
                triplets_big.push_back(Eigen::Triplet<double>(2*K + r, 2*N + c, val));
            }
        }
        // 4번째 block(동차좌표)은 0
        J_reg_vec.setFromTriplets(triplets_big.begin(), triplets_big.end());
    }
    model.J_reg_vec = J_reg_vec;

    // ---------------------------------------
    // 디버깅: 로딩된 값의 크기와 일부 샘플 출력
    // ---------------------------------------
    std::cout << "[buildSMPLModel DEBUG]\n";
    std::cout << "  N = " << model.N << ", K = " << model.K << std::endl;

    // shapedirs_vec, posedirs_vec, v_tem_vec, J_reg_vec 크기 확인
    std::cout << "  shapedirs_vec: " << model.shapedirs_vec.rows() 
              << " x " << model.shapedirs_vec.cols() << std::endl;
    std::cout << "  posedirs_vec : " << model.posedirs_vec.rows()
              << " x " << model.posedirs_vec.cols() << std::endl;
    std::cout << "  v_tem_vec size = " << model.v_tem_vec.size() << std::endl;

    // J_reg_vec는 희소행렬이므로 nonZeros()로 확인
    std::cout << "  J_reg_vec: " << model.J_reg_vec.rows() 
              << " x " << model.J_reg_vec.cols() 
              << ", nonZeros = " << model.J_reg_vec.nonZeros() << std::endl;

    // v_tem_vec 일부(앞 10개) 출력
    std::cout << "  v_tem_vec (first 10) = ";
    for(int i = 0; i < std::min<int>(10, (int)model.v_tem_vec.size()); i++){
        std::cout << model.v_tem_vec(i) << " ";
    }
    std::cout << std::endl;

    // shapedirs_vec 일부
    // (예: 첫 2행 x 3열 정도)
    std::cout << "  shapedirs_vec(0..1, 0..2):\n";
    for(int r = 0; r < 2; r++){
        std::cout << "    Row " << r << ": ";
        for(int c = 0; c < 3; c++){
            std::cout << model.shapedirs_vec(r, c) << " ";
        }
        std::cout << std::endl;
    }

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
    // ------------------------------------------------------------------------
    // (1) 각종 차원(Dimension) 출력
    // ------------------------------------------------------------------------
    std::cout << "[DEBUG] shapedirs_vec size: " 
              << model.shapedirs_vec.rows() << " x "
              << model.shapedirs_vec.cols() << std::endl;
    std::cout << "[DEBUG] v_tem_vec size: " 
              << model.v_tem_vec.size() << std::endl;
    std::cout << "[DEBUG] betas size: "
              << betas.size() << std::endl;
    std::cout << "[DEBUG] J_reg_vec size: "
              << model.J_reg_vec.rows() << " x "
              << model.J_reg_vec.cols() << std::endl;

    // ------------------------------------------------------------------------
    // (2) v_shapeblend = shapedirs_vec * betas
    // ------------------------------------------------------------------------
    Eigen::VectorXd v_shapeblend = model.shapedirs_vec * betas;

    // v_shapeblend 앞부분 출력 (최대 10개)
    std::cout << "[DEBUG] v_shapeblend (first 10): ";
    for(int i = 0; i < std::min<int>(10, (int)v_shapeblend.size()); i++){
        std::cout << v_shapeblend(i) << " ";
    }
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // (3) v_shaped = v_tem_vec + v_shapeblend
    // ------------------------------------------------------------------------
    v_shaped = model.v_tem_vec + v_shapeblend;

    // v_shaped 앞부분 출력 (최대 10개)
    std::cout << "[DEBUG] v_shaped (first 10): ";
    for(int i = 0; i < std::min<int>(10, (int)v_shaped.size()); i++){
        std::cout << v_shaped(i) << " ";
    }
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // (4) j_shaped = J_reg_vec * v_shaped
    // ------------------------------------------------------------------------
    Eigen::VectorXd j_temp = model.J_reg_vec * v_shaped;
    j_shaped = j_temp;

    // j_shaped 앞부분 출력 (최대 10개)
    std::cout << "[DEBUG] j_shaped (first 10): ";
    for(int i = 0; i < std::min<int>(10, (int)j_shaped.size()); i++){
        std::cout << j_shaped(i) << " ";
    }
    std::cout << std::endl;
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
    //  => 각 row i마다, 4x4 행렬을 col-major로 펼친 16개가 들어감
    global_transform.resize(model.K, 16);
    global_transform.setZero();
    global_transform_remove.resize(model.K, 16);
    global_transform_remove.setZero();

    // --- col-major 플래튼 인덱스 ---
    // col-major: index = col*4 + row
    auto colMajorIdx = [&](int row, int col) {
        return col*4 + row;
    };

    for(int i = 0; i < model.K; i++)
    {
        // thetas -> 3D 축각 (axis-angle)
        Eigen::Vector3d omega;
        omega << thetas(3*i), thetas(3*i+1), thetas(3*i+2);
        Eigen::Matrix3d rotmat = so3exp(omega);  // 3x3 회전행렬

        // -------------------------
        // 1) global_transform(i,:)
        // -------------------------
        if(i == 0)
        {
            // root joint
            double tx = j_shaped(i);
            double ty = j_shaped(i + model.K);
            double tz = j_shaped(i + 2*model.K);

            // 4x4 행렬 (col-major) 로 저장
            // rotmat 부분
            global_transform(i, colMajorIdx(0,0)) = rotmat(0,0);
            global_transform(i, colMajorIdx(1,0)) = rotmat(1,0);
            global_transform(i, colMajorIdx(2,0)) = rotmat(2,0);

            global_transform(i, colMajorIdx(0,1)) = rotmat(0,1);
            global_transform(i, colMajorIdx(1,1)) = rotmat(1,1);
            global_transform(i, colMajorIdx(2,1)) = rotmat(2,1);

            global_transform(i, colMajorIdx(0,2)) = rotmat(0,2);
            global_transform(i, colMajorIdx(1,2)) = rotmat(1,2);
            global_transform(i, colMajorIdx(2,2)) = rotmat(2,2);

            // translation 부분
            global_transform(i, colMajorIdx(0,3)) = tx;
            global_transform(i, colMajorIdx(1,3)) = ty;
            global_transform(i, colMajorIdx(2,3)) = tz;
            global_transform(i, colMajorIdx(3,3)) = 1.0;
        }
        else
        {
            // child joint
            int parentIdx = model.kintree_table(0, i) - 1;

            double tx = j_shaped(i)           - j_shaped(parentIdx);
            double ty = j_shaped(i+model.K)   - j_shaped(parentIdx+model.K);
            double tz = j_shaped(i+2*model.K) - j_shaped(parentIdx+2*model.K);

            // local transform
            Eigen::Matrix4d localTrans = Eigen::Matrix4d::Identity();
            localTrans.block<3,3>(0,0) = rotmat;
            localTrans(0,3) = tx;
            localTrans(1,3) = ty;
            localTrans(2,3) = tz;

            // parentMat (col-major로 읽기)
            Eigen::Matrix4d parentMat = Eigen::Matrix4d::Zero();
            for(int r=0; r<4; r++){
                for(int c=0; c<4; c++){
                    parentMat(r,c) = global_transform(parentIdx, colMajorIdx(r,c));
                }
            }

            // 최종 = parent * local
            Eigen::Matrix4d result = parentMat * localTrans;

            // 다시 col-major로 저장
            for(int r=0; r<4; r++){
                for(int c=0; c<4; c++){
                    global_transform(i, colMajorIdx(r,c)) = result(r,c);
                }
            }
        }

        // --------------------------------
        // 2) global_transform_remove(i,:)
        // --------------------------------
        Eigen::Vector4d jHere;
        jHere << j_shaped(i), 
                 j_shaped(i + model.K), 
                 j_shaped(i + 2*model.K), 
                 0.0;

        // G: 현재 joint의 global_transform (col-major)
        Eigen::Matrix4d G;
        for(int r=0; r<4; r++){
            for(int c=0; c<4; c++){
                G(r,c) = global_transform(i, colMajorIdx(r,c));
            }
        }

        // fx = G * jHere
        Eigen::Vector4d fx = G * jHere;

        // pack
        Eigen::Matrix4d pack = Eigen::Matrix4d::Zero();
        pack(0,3) = fx(0);
        pack(1,3) = fx(1);
        pack(2,3) = fx(2);

        Eigen::Matrix4d Gremove = G - pack;

        // col-major로 저장
        for(int r=0; r<4; r++){
            for(int c=0; c<4; c++){
                global_transform_remove(i, colMajorIdx(r,c)) = Gremove(r,c);
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
// axis2quat_single
//-----------------------------------------------------------------------------
Eigen::Vector4d axis2quat_single(const Eigen::Vector3d& axisAngle)
{
    double angle = axisAngle.norm();
    if(angle < 1e-12) {
        // near zero rotation
        // MATLAB: qw = cos(0/2)-1 = 0? or 1-1=0? 
        // depends on original code, but let's do consistent with MATLAB
        return Eigen::Vector4d(0.0, 0.0, 0.0, -1.0); // for example
    }
    Eigen::Vector3d u = axisAngle / angle;
    double half = angle * 0.5;
    double cos_val = std::cos(half);
    double sin_val = std::sin(half);

    double qx = u(0)*sin_val;
    double qy = u(1)*sin_val;
    double qz = u(2)*sin_val;
    double qw = cos_val - 1.0; // or cos_val (depending on the actual MATLAB code)

    return Eigen::Vector4d(qx, qy, qz, qw);
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
    Eigen::VectorXd pose = thetas.segment(3, thetas.size()-3); // check carefully

    int nJoints = static_cast<int>(pose.size()/3);
    Eigen::VectorXd quaternionAngle(4 * nJoints);

    for(int j=0; j<nJoints; j++){
        Eigen::Vector3d axisAngle(
            pose(3*j+0),
            pose(3*j+1),
            pose(3*j+2)
        );
        Eigen::Vector4d q = axis2quat_single(axisAngle);
        quaternionAngle(4*j+0) = q(0);
        quaternionAngle(4*j+1) = q(1);
        quaternionAngle(4*j+2) = q(2);
        quaternionAngle(4*j+3) = q(3);
    }

    double shape_feat = betas(0); // or betas(1)
    // 4*nJoints + 1 = 93 => => nJoints=23 => 4*23=92 => +1=93
    Eigen::VectorXd feat(4*nJoints + 1);
    feat.head(4*nJoints) = quaternionAngle;
    feat(4*nJoints) = shape_feat;

    // --- Debug print: 
    std::cout << "posedirs_vec first row: ";
    
    for(int c = 0; c < model.posedirs_vec.cols(); c++)
    {
        double val = model.posedirs_vec(0, c);
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "feat (entire vector): ";
    for(int i = 0; i < feat.size(); i++)
    {
        std::cout << feat(i) << " ";
    }
    std::cout << std::endl;
    // --- End of Debug print

    // v_shaped2 = v_shaped + posedirs_vec * feat
    v_shaped2 = v_shaped + model.posedirs_vec * feat;
}

//-----------------------------------------------------------------------------
// poserot
//-----------------------------------------------------------------------------
void poserot(
    const Eigen::MatrixXd& global_transform_remove, // (K x 16)
    const Eigen::VectorXd& v_shaped,                // (4N x 1)
    const SMPLModel& model,
    Eigen::MatrixXd& v_rot
)
{
    using namespace Eigen;

    // 1) N, K, weights
    const int N = model.N;
    const int K = model.K;
    const MatrixXd& weights = model.weights; // [N x K]


    // global_transform_remove(0, 0..7)
    std::cout << "global_transform_remove(0, 0..7)" << std::endl;
    for(int j=0; j<8; j++){
      std::cout << global_transform_remove(0,j) << " ";
    }

    // 간단한 사이즈 체크
    if(global_transform_remove.rows() != K || global_transform_remove.cols() != 16) {
        std::cerr << "[poserot] global_transform_remove must be (K x 16)\n";
        return;
    }
    if((int)v_shaped.size() != 4*N) {
        std::cerr << "[poserot] v_shaped must be (4N x 1)\n";
        return;
    }
    if(weights.rows() != N || weights.cols() != K) {
        std::cerr << "[poserot] weights must be (N x K)\n";
        return;
    }

    // 2) coefficients = weights * global_transform_remove => (N x 16)
    MatrixXd coefficients = weights * global_transform_remove;  // (N x 16)

    // 3) v_shaped => (N x 4), "col-major" map
    //    (Matlab reshape와 동일한 순서 보장)
    Map<const Matrix<double, Dynamic, 4, ColMajor>> v_shaped_nor(v_shaped.data(), N, 4);

    // 4) coefficients의 각 row(n,:) => 4x4 행렬 (col-major)
    //    => M(row,col) = coefficients(n, col*4 + row)
    //    곱해주고 v_rot(n,:)에 저장
    v_rot.resize(N, 4);

    for(int n = 0; n < N; ++n) {
        // (4x4) 만들기
        Matrix4d M;
        for(int col = 0; col < 4; ++col) {
            for(int row = 0; row < 4; ++row) {
                // col-major 기준 => index = col*4 + row
                M(row, col) = coefficients(n, col*4 + row);
            }
        }

        // v_shaped_nor(n,:) => [1x4], -> transpose() => [4x1]
        Vector4d v = v_shaped_nor.row(n).transpose();
        Vector4d result = M * v;     // [4x1]

        // 결과 저장
        v_rot.row(n) = result.transpose();  // [1x4]
    }

}