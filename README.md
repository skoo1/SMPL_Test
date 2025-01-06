# SMPL: Matlab → C++ → Python(.pyd) 변환 과정

이 저장소(Repository)는 **Matlab**에서 구현된 SMPL 모델을 **C++**로 변환하고, 최종적으로 **pybind11**을 통해 **.pyd**(Python 확장 모듈) 형태로 제작하는 과정을 정리합니다. 주의할 것은, 주요 변환 단계마다 주요 함수의 **input/output** 데이터를 `.mat` 또는 `.json` 형식으로 저장하고, 상호 비교해야 한다.

- SMPL_Matlab 을 테스트해보기 위해서는 model.mat 필요
- SMPL_Matlab 을 이용해서 model.json 생성
- SMPL_CPP 와 SMPL_PYD 에 이 model.json 을 넣고, 테스트
- PySMPL.pyd 생성을 위해서는 SMPL_PYD 안에 있는 README*.txt 참고할 것

# SMPL_PYD 구조
SMPL_PYD/ \
├─ smpl_model.cpp # SMPLModel 관련 C++ 구현 \
├─ smpl_model.h # SMPLModel 구조체 및 함수 선언 \
├─ py_smpl_bind.cpp # pybind11 바인딩 코드 (C++ → Python) \
├─ pybind11/ # pybind11 서브모듈 디렉터리 \
│ └─ ... (pybind11 관련 파일들) \
├─ CMakeLists.txt # CMake 빌드 설정 파일 \
├─ build.bat # CMake 명령 호출 스크립트 (Windows) \
├─ README_Build및실행방법.txt # 빌드 및 실행 방법 정리 \
├─ README_Successful_Build_Log.txt# 성공적으로 빌드된 로그 예시 \
│ \
├─ model.json # SMPL 모델 데이터 (JSON) \
├─ walking_20240125.json # 모션 데이터 (JSON) \
│ \
├─ PySMPL.cp310-win_amd64.pyd # 빌드된 Python 확장 모듈 (.pyd) \
│ \
├─ main.py # Python 메인 스크립트 \
├─ main_with_timecheck.py # Python 메인 (시간 측정용) \
└─ frame1verts.csv # 예시 CSV 데이터 \
