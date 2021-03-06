# ########################################################################
# Copyright (c) 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

if(UNIX)
  # the interface to rocblas for Fortran programs
  add_library(rocblas_module OBJECT
    "${ROCBLAS_INCLUDE_DIR}/rocblas_module.f90"
  )
endif()

# declare sample programs
add_executable(example-c-basic
  example_basic.c
)
add_executable(example-cpp-basic
  example_basic.cpp
)
if(UNIX)
  add_executable(example-fortran-basic
    example_basic.f90
    $<TARGET_OBJECTS:rocblas_module>
  )
endif()
add_executable(example-c-hmm
  example_hmm.c
)
add_executable(example-cpp-logging
  example_logging.cpp
)

# group sample programs by language
set(c_samples
  example-c-basic
  example-c-hmm
)
set(cpp_samples
  example-cpp-basic
  example-cpp-logging
)
if(UNIX)
  set(fortran_samples
    example-fortran-basic
  )
endif()

# set flags for building the sample programs
foreach(exe ${c_samples} ${cpp_samples} ${fortran_samples})
  target_link_libraries(${exe} PRIVATE roc::rocsolver)
endforeach()

foreach(exe ${cpp_samples})
  set_target_properties(${exe} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
endforeach()

foreach(exe ${c_samples})
  set_target_properties(${exe} PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
endforeach()
