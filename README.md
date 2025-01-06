# SMPL: Matlab → C++ → Python(.pyd) 변환 과정

이 저장소(Repository)는 **Matlab**에서 구현된 SMPL 모델을 **C++**로 변환하고, 최종적으로 **pybind11**을 통해 **.pyd**(Python 확장 모듈) 형태로 제작하는 과정을 정리합니다. 주의할 것은, 주요 변환 단계마다 주요 함수의 **input/output** 데이터를 `.mat` 또는 `.json` 형식으로 저장하고, 상호 비교해야 한다.

- SMPL_Matlab 을 테스트해보기 위해서는 model.mat 필요
- SMPL_Matlab 을 이용해서 model.json 생성
- SMPL_CPP 와 SMPL_PYD 에 이 model.json 을 넣고, 테스트
- PySMPL.pyd 생성을 위해서는 SMPL_PYD 안에 있는 README*.txt 참고할 것
