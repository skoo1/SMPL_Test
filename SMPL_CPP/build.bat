cmake -S . ^
      -B build ^
      -G "Visual Studio 17 2022" ^
      -A x64 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CXX_FLAGS_RELEASE="/O2 /arch:AVX2 /DNDEBUG /GL /LTCG /openmp /DEIGEN_DONT_PARALLELIZE=0"

cmake --build build --config Release