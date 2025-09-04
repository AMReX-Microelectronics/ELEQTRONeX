# AMReX dependency management
# Based on patterns from FerroX, MagneX, and WarpX

# Local source-tree
set(ELEQTRONeX_amrex_src ""
    CACHE PATH
    "Local path to AMReX source directory (preferred if set)")

# Git fetcher
set(ELEQTRONeX_amrex_repo "https://github.com/AMReX-Codes/amrex.git"
    CACHE STRING
    "Repository URI to pull and build AMReX from if(ELEQTRONeX_amrex_internal)")

set(ELEQTRONeX_amrex_branch "development"
    CACHE STRING
    "Repository branch for ELEQTRONeX_amrex_repo if(ELEQTRONeX_amrex_internal)")

# PR testing options (primarily for CI/testing)
set(ELEQTRONeX_amrex_pr "" CACHE STRING "AMReX pull request number for testing")
mark_as_advanced(ELEQTRONeX_amrex_pr)

# Override branch if PR is specified
if(ELEQTRONeX_amrex_pr AND NOT ELEQTRONeX_amrex_pr STREQUAL "")
    set(ELEQTRONeX_amrex_branch "pull/${ELEQTRONeX_amrex_pr}/head" CACHE STRING
        "Using AMReX PR #${ELEQTRONeX_amrex_pr}" FORCE)
    message(STATUS "AMReX: Using pull request #${ELEQTRONeX_amrex_pr}")
endif()

if(ELEQTRONeX_amrex_src)
    message(STATUS "Compiling local AMReX ...")
    message(STATUS "AMReX source path: ${ELEQTRONeX_amrex_src}")
    if(NOT IS_DIRECTORY ${ELEQTRONeX_amrex_src})
        message(FATAL_ERROR "Specified directory ELEQTRONeX_amrex_src='${ELEQTRONeX_amrex_src}' does not exist!")
    endif()
elseif(ELEQTRONeX_amrex_internal)
    message(STATUS "Downloading AMReX ...")
    message(STATUS "AMReX repository: ${ELEQTRONeX_amrex_repo} (${ELEQTRONeX_amrex_branch})")
    include(FetchContent)
endif()

if(ELEQTRONeX_amrex_internal OR ELEQTRONeX_amrex_src)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    # Configure AMReX based on ELEQTRONeX options - only require what's actually needed
    set(AMReX_SPACEDIM         ${ELEQTRONeX_DIMS}         CACHE STRING "" FORCE)
    set(AMReX_MPI              ${ELEQTRONeX_MPI}          CACHE BOOL "" FORCE)
    set(AMReX_EB               ${ELEQTRONeX_EB}           CACHE BOOL "" FORCE)
    set(AMReX_PARTICLES        ${ELEQTRONeX_TRANSPORT}    CACHE BOOL "" FORCE)
    set(AMReX_LINEAR_SOLVERS   ON                         CACHE BOOL "" FORCE)
    
    # Let AMReX handle HYPRE if requested
    if(ELEQTRONeX_HYPRE)
        set(AMReX_HYPRE ON CACHE BOOL "" FORCE)
    endif()

    # Set compute backend
    if(ELEQTRONeX_COMPUTE STREQUAL "CUDA")
        set(AMReX_GPU_BACKEND CUDA CACHE STRING "" FORCE)
    elseif(ELEQTRONeX_COMPUTE STREQUAL "HIP")
        set(AMReX_GPU_BACKEND HIP CACHE STRING "" FORCE)
    elseif(ELEQTRONeX_COMPUTE STREQUAL "OMP")
        set(AMReX_OMP ON CACHE BOOL "" FORCE)
        set(AMReX_GPU_BACKEND NONE CACHE STRING "" FORCE)
    else()
        set(AMReX_GPU_BACKEND NONE CACHE STRING "" FORCE)
    endif()

    # Precision settings
    set(AMReX_PRECISION DOUBLE CACHE STRING "" FORCE)

    # Disable features not needed by ELEQTRONeX
    set(AMReX_AMRLEVEL OFF CACHE INTERNAL "")
    set(AMReX_ENABLE_TESTS OFF CACHE INTERNAL "")
    set(AMReX_FORTRAN OFF CACHE INTERNAL "")
    set(AMReX_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
    set(AMReX_BUILD_TUTORIALS OFF CACHE INTERNAL "")
    set(AMReX_PROBINIT OFF CACHE INTERNAL "")

    if(ELEQTRONeX_amrex_src)
        list(APPEND CMAKE_MODULE_PATH "${ELEQTRONeX_amrex_src}/Tools/CMake")
        if(ELEQTRONeX_COMPUTE STREQUAL CUDA)
            enable_language(CUDA)
        elseif(ELEQTRONeX_COMPUTE STREQUAL HIP)
            if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                message(WARNING "HIP backend works best with Clang-based compilers (clang++, amdclang++, hipcc)")
            endif()
        endif()
        add_subdirectory(${ELEQTRONeX_amrex_src} _deps/localamrex-build/)
    else()
        if(ELEQTRONeX_COMPUTE STREQUAL CUDA)
            enable_language(CUDA)
        elseif(ELEQTRONeX_COMPUTE STREQUAL HIP)
            if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                message(WARNING "HIP backend works best with Clang-based compilers (clang++, amdclang++, hipcc)")
            endif()
        endif()
        FetchContent_Declare(fetchedamrex
            GIT_REPOSITORY ${ELEQTRONeX_amrex_repo}
            GIT_TAG        ${ELEQTRONeX_amrex_branch}
            BUILD_IN_SOURCE 0
        )
        FetchContent_MakeAvailable(fetchedamrex)
        list(APPEND CMAKE_MODULE_PATH "${fetchedamrex_SOURCE_DIR}/Tools/CMake")
    endif()

    # No alias needed - amrex target is available directly

    message(STATUS "AMReX: Using version '${AMREX_PKG_VERSION}' (${AMREX_GIT_VERSION})")
else()
    message(STATUS "Searching for pre-installed AMReX ...")
    # Find external AMReX installation
    find_package(AMReX REQUIRED)
    
    # Verify AMReX was built with required features (only check what's actually needed)
    if(ELEQTRONeX_EB AND NOT AMReX_EB)
        message(FATAL_ERROR "ELEQTRONeX requires AMReX built with EB support when ELEQTRONeX_EB=ON")
    endif()
    
    if(ELEQTRONeX_TRANSPORT AND NOT AMReX_PARTICLES)
        message(FATAL_ERROR "ELEQTRONeX requires AMReX built with PARTICLES support when ELEQTRONeX_TRANSPORT=ON")
    endif()
    
    if(ELEQTRONeX_HYPRE AND NOT AMReX_HYPRE)
        message(FATAL_ERROR "ELEQTRONeX requires AMReX built with HYPRE support when ELEQTRONeX_HYPRE=ON")
    endif()

    # No alias needed - AMReX::amrex target is available directly

    if(ELEQTRONeX_COMPUTE STREQUAL CUDA)
        enable_language(CUDA)
    elseif(ELEQTRONeX_COMPUTE STREQUAL HIP)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(WARNING "HIP backend works best with Clang-based compilers (clang++, amdclang++, hipcc)")
        endif()
    endif()

    message(STATUS "AMReX: Found version '${AMReX_VERSION}'")
endif()