cmake_minimum_required(VERSION 3.10)
project(PySMPLProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1) pybind11
add_subdirectory(pybind11)

# 2) nlohmann/json
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json NOT FOUND. Fetching it via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.2
    )
    FetchContent_MakeAvailable(json)
endif()

# 3) Eigen3
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 NOT FOUND. Fetching it via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
    )
    FetchContent_MakeAvailable(eigen)

    set(Eigen3_DIR "${eigen_SOURCE_DIR}")
    find_package(Eigen3 REQUIRED NO_MODULE)
endif()

# 4) smpl_model 라이브러리
add_library(smpl_model STATIC smpl_model.cpp)

target_link_libraries(smpl_model
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)

# 5) 파이썬 모듈 (PySMPL)
pybind11_add_module(PySMPL py_smpl_bind.cpp)

# 여기에서 Eigen, nlohmann_json도 링크해야, 
# #include <nlohmann/json.hpp>, #include <pybind11/eigen.h> 등이 인식됨
target_link_libraries(PySMPL
    PRIVATE
        smpl_model
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)

# 6) MSVC Release 빌드 최적화
if(MSVC)
    # 컴파일 단계
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2 /DNDEBUG /GL /openmp /DEIGEN_DONT_PARALLELIZE=0"
    )

    # 링크 단계
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE
        "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG"
    )
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE
        "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG"
    )
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE
        "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /LTCG"
    )
endif()
