cmake_minimum_required(VERSION 3.21)

# 프로젝트 이름 (원하는 이름으로 변경 가능)
project(SMPLExample LANGUAGES CXX)

# C++17 표준 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-------------------------------------------------------------------------------
# nlohmann/json 라이브러리 찾기 or FetchContent로 가져오기
#-------------------------------------------------------------------------------
find_package(nlohmann_json 3.2.0 QUIET)
if (NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json NOT FOUND. Fetching it via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.2  # 원하는 버전 태그
    )
    FetchContent_MakeAvailable(json)
endif()

#-------------------------------------------------------------------------------
# Eigen 라이브러리 찾기 or FetchContent로 가져오기
#-------------------------------------------------------------------------------
find_package(Eigen3 3.4 QUIET NO_MODULE)
if (NOT Eigen3_FOUND)
    message(STATUS "Eigen3 NOT FOUND. Fetching it via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0    # 원하는 버전 태그
    )
    FetchContent_MakeAvailable(eigen)
    set(Eigen3_DIR "${eigen_SOURCE_DIR}")
    find_package(Eigen3 REQUIRED NO_MODULE)
endif()

#-------------------------------------------------------------------------------
# 실행 파일 생성
#-------------------------------------------------------------------------------
add_executable(SMPLExample
    main.cpp
    # 추가 소스가 있다면 여기에 이어서 추가
    # e.g., shapeblend.cpp, transforms.cpp, etc.
)

#-------------------------------------------------------------------------------
# 링크 라이브러리 설정
#-------------------------------------------------------------------------------
target_link_libraries(SMPLExample
    PRIVATE
        nlohmann_json::nlohmann_json
        Eigen3::Eigen
)

#-------------------------------------------------------------------------------
# Visual Studio용 설정(선택 사항)
#-------------------------------------------------------------------------------
# 출력 폴더 설정 등 필요한 경우 추가
# set_target_properties(SMPLExample PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
# )
