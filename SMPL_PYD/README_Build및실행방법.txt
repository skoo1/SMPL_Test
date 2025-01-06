Build 방법

SMPL_PYD/
├─ smpl_model.cpp                 # SMPLModel 관련 C++ 구현
├─ smpl_model.h                   # SMPLModel 구조체 및 함수 선언
├─ py_smpl_bind.cpp               # pybind11 바인딩 코드 (C++ → Python)
├─ pybind11/                      # pybind11 서브모듈 디렉터리
│  └─ ... (pybind11 관련 파일들)
├─ CMakeLists.txt                 # CMake 빌드 설정 파일
├─ build.bat                      # CMake 명령 호출 스크립트 (Windows)
├─ README_Build및실행방법.txt     # 빌드 및 실행 방법 정리
├─ README_Successful_Build_Log.txt# 성공적으로 빌드된 로그 예시
│
├─ model.json                     # SMPL 모델 데이터 (JSON)
├─ walking_20240125.json          # 모션 데이터 (JSON)
│
├─ PySMPL.cp310-win_amd64.pyd     # 빌드된 Python 확장 모듈 (.pyd)
│
├─ main.py                        # Python 메인 스크립트
├─ main_with_timecheck.py         # Python 메인 (시간 측정용)
└─ frame1verts.csv                # 예시 CSV 데이터


target python 버전에 맞는 conda env 로 console 을 연다. 

build.bat 을 실행한다. 

(miniconda_python310) D:\Works\SMPL_PYD>build.bat
(miniconda_python310) D:\Works\SMPL_PYD>copy .\build\Release\PySMPL.cp310-win_amd64.pyd .
(miniconda_python310) D:\Works\SMPL_PYD>python main_with_timecheck.py
[TIME] buildSMPLModel() took 2.279999 sec
[TIME] loadJSON('walking_20240125.json') took 0.001002 sec
[TIME] motionJson -> motionData took 0.000999 sec
[TIME] Total for all frames: 0.372999 sec
model.json 을 읽어와서 smpl_model 을 buld 하는데 약 2.3초 소요
그 이후로 SMPL_Calc() 는 한번 call 할때 약 3.8 ms 소요
