cmake_minimum_required(VERSION 3.24.0)
project(ELEQTRONeX VERSION 1.0.0)

# In-source tree builds are not supported
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Change default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Options - mirror GNUmakefile defaults but allow flexibility
option(ELEQTRONeX_APP "Build the ELEQTRONeX executable application" ON)
option(ELEQTRONeX_MPI "Multi-node support (message-passing)" ON)
option(ELEQTRONeX_EB "Embedded boundary support" ON)
option(ELEQTRONeX_TRANSPORT "Transport support" ON)
option(ELEQTRONeX_TIME_DEPENDENT "Time-dependent support" ON)
option(ELEQTRONeX_BROYDEN_PARALLEL "Broyden parallel support" ON)
option(ELEQTRONeX_HYPRE "HYPRE support" OFF)

# Dependency options
option(ELEQTRONeX_amrex_internal "Download & build AMReX" ON)

# Print options
option(ELEQTRONeX_PRINT_HIGH "Enable high level debug printing" OFF)
option(ELEQTRONeX_PRINT_MEDIUM "Enable medium level debug printing" OFF)
option(ELEQTRONeX_PRINT_LOW "Enable low level debug printing" OFF)
option(ELEQTRONeX_PRINT_NAME "Enable function name debug printing" OFF)

# Compute backend selection
set(ELEQTRONeX_COMPUTE_VALUES NOACC OMP CUDA HIP)
set(ELEQTRONeX_COMPUTE NOACC CACHE STRING "On-node, accelerated computing backend (NOACC/OMP/CUDA/HIP)")
set_property(CACHE ELEQTRONeX_COMPUTE PROPERTY STRINGS ${ELEQTRONeX_COMPUTE_VALUES})
if(NOT ELEQTRONeX_COMPUTE IN_LIST ELEQTRONeX_COMPUTE_VALUES)
    message(FATAL_ERROR "ELEQTRONeX_COMPUTE (${ELEQTRONeX_COMPUTE}) must be one of ${ELEQTRONeX_COMPUTE_VALUES}")
endif()

# Set dimensions (3D only for ELEQTRONeX)
set(ELEQTRONeX_DIMS 3 CACHE STRING "Simulation dimensionality (3D only)")

# Include AMReX dependency management
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/AMReX.cmake)

# Create executable with sophisticated naming like other codes
if(ELEQTRONeX_APP)
    # Build binary name
    set(binary_name "main3d")

    # Add compiler info
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(binary_name "${binary_name}.gnu")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(binary_name "${binary_name}.intel")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(binary_name "${binary_name}.clang")
    endif()

    # Add MPI support
    if(ELEQTRONeX_MPI)
        set(binary_name "${binary_name}.MPI")
    endif()

    # Add EB support
    if(ELEQTRONeX_EB)
        set(binary_name "${binary_name}.EB")
    endif()

    # Add time-dependent support
    if(ELEQTRONeX_TIME_DEPENDENT)
        set(binary_name "${binary_name}.TD")
    endif()

    # Add transport support
    if(ELEQTRONeX_TRANSPORT)
        set(binary_name "${binary_name}.TRAN")
    endif()

    # Add Broyden parallel support
    if(ELEQTRONeX_BROYDEN_PARALLEL)
        set(binary_name "${binary_name}.BROYPRLL")
    endif()

    # Add SKIPGPU suffix when not using GPU backend (matching gnumake logic)
    if(NOT ELEQTRONeX_COMPUTE STREQUAL "CUDA" AND NOT ELEQTRONeX_COMPUTE STREQUAL "HIP")
        set(binary_name "${binary_name}.SKIPGPU")
    endif()

    # Add compute backend
    if(ELEQTRONeX_COMPUTE STREQUAL "OMP")
        set(binary_name "${binary_name}.OMP")
    elseif(ELEQTRONeX_COMPUTE STREQUAL "CUDA")
        set(binary_name "${binary_name}.CUDA")
    elseif(ELEQTRONeX_COMPUTE STREQUAL "HIP")
        set(binary_name "${binary_name}.HIP")
    endif()

    # Add HYPRE support
    if(ELEQTRONeX_HYPRE)
        set(binary_name "${binary_name}.HYPRE")
    endif()

    set(binary_name "${binary_name}.ex")

    add_executable(${binary_name})

    # Add main source files
    target_sources(${binary_name} PRIVATE
        Source/main.cpp
        Source/Code.cpp
    )

    # Add all subdirectory sources recursively
    file(GLOB_RECURSE ELEQTRONeX_SOURCES
        "Source/Diagnostics/*.cpp"
        "Source/Input/*.cpp"
        "Source/Output/*.cpp"
        "Source/PostProcessor/*.cpp"
        "Source/Solver/*.cpp"
        "Source/Utils/*.cpp"
    )
    target_sources(${binary_name} PRIVATE ${ELEQTRONeX_SOURCES})

    # Include directories
    target_include_directories(${binary_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Diagnostics
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Input
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Input/BoundaryConditions
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Input/GeometryProperties
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Input/MacroscopicProperties
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Output
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/PostProcessor
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Solver
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Solver/Electrostatics
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Solver/Transport
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Utils
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Utils/CodeUtils
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Utils/SelectWarpXUtils
    )

    # Link AMReX (AMReX will automatically link HYPRE if it was built with HYPRE support)
    target_link_libraries(${binary_name} PRIVATE AMReX::amrex)

    # Set compile definitions based on options
    # Matching gnumake here - this will be a redefinition as a workaround for nonstandard code
    if(ELEQTRONeX_EB)
        target_compile_definitions(${binary_name} PRIVATE AMREX_USE_EB)
    endif()

    if(ELEQTRONeX_TIME_DEPENDENT)
        target_compile_definitions(${binary_name} PRIVATE TIME_DEPENDENT)
    endif()

    if(ELEQTRONeX_TRANSPORT)
        target_compile_definitions(${binary_name} PRIVATE USE_TRANSPORT)
    endif()

    if(ELEQTRONeX_BROYDEN_PARALLEL)
        target_compile_definitions(${binary_name} PRIVATE BROYDEN_PARALLEL)
    endif()

    # Matching gnumake logic: skip GPU optimization when not using GPU backend
    if(NOT ELEQTRONeX_COMPUTE STREQUAL "CUDA" AND NOT ELEQTRONeX_COMPUTE STREQUAL "HIP")
        target_compile_definitions(${binary_name} PRIVATE BROYDEN_SKIP_GPU_OPTIMIZATION)
    endif()

    # Print debug options - matches Make.Code logic
    if(ELEQTRONeX_PRINT_HIGH)
        target_compile_definitions(${binary_name} PRIVATE PRINT_HIGH PRINT_MEDIUM PRINT_LOW PRINT_NAME)
    elseif(ELEQTRONeX_PRINT_MEDIUM)
        target_compile_definitions(${binary_name} PRIVATE PRINT_MEDIUM PRINT_LOW PRINT_NAME)
    elseif(ELEQTRONeX_PRINT_LOW)
        target_compile_definitions(${binary_name} PRIVATE PRINT_LOW PRINT_NAME)
    elseif(ELEQTRONeX_PRINT_NAME)
        target_compile_definitions(${binary_name} PRIVATE PRINT_NAME)
    endif()

    # Handle CUDA compilation like MagneX and FerroX
    if(ELEQTRONeX_COMPUTE STREQUAL "CUDA")
        # AMReX helper function: propagate CUDA specific target & source properties
        setup_target_for_cuda_compilation(${binary_name})
        target_compile_features(${binary_name} PUBLIC cuda_std_17)
        set_target_properties(${binary_name} PROPERTIES
            CUDA_EXTENSIONS OFF
            CUDA_STANDARD_REQUIRED ON
        )
    else()
        target_compile_features(${binary_name} PUBLIC cxx_std_17)
        set_target_properties(${binary_name} PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD_REQUIRED ON
        )
    endif()

    # Set output directory
    set_target_properties(${binary_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Create simple symlink for convenience
    add_custom_command(TARGET ${binary_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${binary_name}
        ${CMAKE_BINARY_DIR}/eleqtronex
        COMMENT "Creating symlink eleqtronex -> ${binary_name}"
    )
endif()