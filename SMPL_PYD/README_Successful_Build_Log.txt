(miniconda_python310) D:\Works\SMPL_PYD>build
-- Selecting Windows SDK version 10.0.22000.0 to target Windows 10.0.22631.
-- The C compiler identification is MSVC 19.42.34433.0
-- The CXX compiler identification is MSVC 19.42.34433.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- pybind11 v2.14.0 dev1
-- Found PythonInterp: D:/miniconda3/envs/miniconda_python310/python.exe (found suitable version "3.10.16", minimum required is "3.8")
-- Found PythonLibs: D:/miniconda3/envs/miniconda_python310/libs/python310.lib
-- Performing Test HAS_MSVC_GL_LTCG
-- Performing Test HAS_MSVC_GL_LTCG - Success
-- nlohmann_json NOT FOUND. Fetching it via FetchContent...
CMake Deprecation Warning at build/_deps/json-src/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- Using the multi-header code from D:/Works/SMPL_PYD/build/_deps/json-src/include/
-- Configuring done (42.5s)
-- Generating done (0.1s)
-- Build files have been written to: D:/Works/SMPL_PYD/build
msbuild 버전 17.12.6+db5f6012c(.NET Framework용)

  1>Checking Build System
  Building Custom Rule D:/Works/SMPL_PYD/CMakeLists.txt
  smpl_model.cpp
  smpl_model.vcxproj -> D:\Works\SMPL_PYD\build\Release\smpl_model.lib
  Building Custom Rule D:/Works/SMPL_PYD/CMakeLists.txt
  py_smpl_bind.cpp
     D:/Works/SMPL_PYD/build/Release/PySMPL.lib 라이브러리 및 D:/Works/SMPL_PYD/build/Release/PySMPL.exp 개체를 생성하고 있습니다.
  코드를 생성하고 있습니다.
  코드를 생성했습니다.
  PySMPL.vcxproj -> D:\Works\SMPL_PYD\build\Release\PySMPL.cp310-win_amd64.pyd
  Building Custom Rule D:/Works/SMPL_PYD/CMakeLists.txt

(miniconda_python310) D:\Works\SMPL_PYD>