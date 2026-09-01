# 3.21 is the floor for $<TARGET_RUNTIME_DLLS:...>, used to package the vcpkg
# DLLs alongside the Windows executables.
cmake_minimum_required(VERSION 3.21...3.30)

project(neat-dnfs
        DESCRIPTION "Dynamic Neural Field Evolution with NEAT for Robotics"
        LANGUAGES CXX)

# Define library version
# Keep in step with the topmost released section of ../CHANGELOG.md: release
# archives are named after this version, and release.yml refuses to publish a
# tag that disagrees with it. Pre-1.0 on purpose -- the API is still free to
# move between releases.
set(NEAT_DNFS_VERSION_MAJOR 0 CACHE STRING "major version" FORCE)
set(NEAT_DNFS_VERSION_MINOR 3 CACHE STRING "minor version" FORCE)
set(NEAT_DNFS_VERSION_PATCH 0 CACHE STRING "patch version" FORCE)
set(NEAT_DNFS_VERSION ${NEAT_DNFS_VERSION_MAJOR}.${NEAT_DNFS_VERSION_MINOR}.${NEAT_DNFS_VERSION_PATCH} CACHE STRING "version" FORCE)

# Library build type prefix
set(NEAT_DNFS_LIB_PREFIX "release")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(NEAT_DNFS_LIB_PREFIX "debug")
endif()

# Installation directories
set(NEAT_DNFS_INC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(NEAT_DNFS_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")
set(NEAT_DNFS_LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/${NEAT_DNFS_LIB_PREFIX}")
set(NEAT_DNFS_ARCHIVE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/${NEAT_DNFS_LIB_PREFIX}")
set(NEAT_DNFS_FRAMEWORK_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")
set(NEAT_DNFS_CMAKE_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/${CMAKE_PROJECT_NAME}")
set(NEAT_DNFS_ADDITIONAL_FILES_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/${CMAKE_PROJECT_NAME}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC defaults to the system codepage for non-BOM source files, which mangles
# non-ASCII characters (e.g. em dashes in Catch2 test names) at compile time.
if(MSVC)
    add_compile_options(/utf-8)
    # Windows.h reaches nearly every TU transitively via imgui-platform-kit and
    # defines function-like min/max macros that break std::numeric_limits<T>::max()
    # and std::min/std::max at their call sites.
    add_compile_definitions(NOMINMAX)
endif()

# Sanitizers
set(NEAT_DNFS_SANITIZER "" CACHE STRING "Sanitizer: '' | address (ASan+UBSan) | thread (TSan)")
if(NEAT_DNFS_SANITIZER STREQUAL "address")
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
elseif(NEAT_DNFS_SANITIZER STREQUAL "thread")
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=thread)
elseif(NOT NEAT_DNFS_SANITIZER STREQUAL "")
    message(FATAL_ERROR "Unknown NEAT_DNFS_SANITIZER: ${NEAT_DNFS_SANITIZER}")
endif()

# Evolution-loop profiling (GH #80)
option(NEAT_DNFS_PROFILE "Time evolution-loop phases and write profile.csv per run" OFF)
if(NEAT_DNFS_PROFILE)
    add_compile_definitions(NEAT_DNFS_PROFILE)
endif()

# VCPKG integration
if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "ERROR: This project requires VCPKG.\n")
endif()
set(VCPKG_ROOT $ENV{VCPKG_ROOT})
include(${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

# Project directory definitions
set(PROJECT_DIR "${CMAKE_SOURCE_DIR}")
add_compile_definitions(PROJECT_DIR="${PROJECT_DIR}")

set(OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/data")
add_compile_definitions(OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY}")

# Build provenance -- captured at configure time and baked into a generated
# build_info.h so each run's output can be traced to what built it (GH #73).
set(NEAT_DNFS_GIT_SHA "unknown")
set(NEAT_DNFS_GIT_DIRTY "false")
set(NEAT_DNFS_IPK_SHA "unknown")
set(NEAT_DNFS_DNFC_SHA "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NEAT_DNFS_GIT_SHA_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NEAT_DNFS_GIT_SHA_RESULT ERROR_QUIET)
    if(NEAT_DNFS_GIT_SHA_RESULT EQUAL 0)
        set(NEAT_DNFS_GIT_SHA "${NEAT_DNFS_GIT_SHA_OUT}")
    endif()

    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NEAT_DNFS_GIT_STATUS_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NEAT_DNFS_GIT_STATUS_RESULT ERROR_QUIET)
    if(NEAT_DNFS_GIT_STATUS_RESULT EQUAL 0 AND NOT NEAT_DNFS_GIT_STATUS_OUT STREQUAL "")
        set(NEAT_DNFS_GIT_DIRTY "true")
    endif()

    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/deps/imgui-platform-kit
        OUTPUT_VARIABLE NEAT_DNFS_IPK_SHA_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NEAT_DNFS_IPK_SHA_RESULT ERROR_QUIET)
    if(NEAT_DNFS_IPK_SHA_RESULT EQUAL 0)
        set(NEAT_DNFS_IPK_SHA "${NEAT_DNFS_IPK_SHA_OUT}")
    endif()

    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/deps/dynamic-neural-field-composer
        OUTPUT_VARIABLE NEAT_DNFS_DNFC_SHA_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NEAT_DNFS_DNFC_SHA_RESULT ERROR_QUIET)
    if(NEAT_DNFS_DNFC_SHA_RESULT EQUAL 0)
        set(NEAT_DNFS_DNFC_SHA "${NEAT_DNFS_DNFC_SHA_OUT}")
    endif()
endif()

set(NEAT_DNFS_VCPKG_LIST "unknown")
find_program(NEAT_DNFS_VCPKG_EXECUTABLE NAMES vcpkg PATHS "${VCPKG_ROOT}" NO_DEFAULT_PATH)
if(NEAT_DNFS_VCPKG_EXECUTABLE)
    execute_process(COMMAND ${NEAT_DNFS_VCPKG_EXECUTABLE} list
        WORKING_DIRECTORY ${VCPKG_ROOT}
        OUTPUT_VARIABLE NEAT_DNFS_VCPKG_LIST_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NEAT_DNFS_VCPKG_LIST_RESULT ERROR_QUIET)
    if(NEAT_DNFS_VCPKG_LIST_RESULT EQUAL 0 AND NOT NEAT_DNFS_VCPKG_LIST_OUT STREQUAL "")
        set(NEAT_DNFS_VCPKG_LIST "${NEAT_DNFS_VCPKG_LIST_OUT}")
    endif()
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/neat_tools/build_info.h.in"
    "${CMAKE_BINARY_DIR}/generated/include/neat_tools/build_info.h"
    @ONLY)

# Header and source files
set(header
        "include/neat/connection_gene.h"
        "include/neat/field_gene.h"
        "include/neat/genome.h"
        "include/neat/population.h"
        "include/neat/population_file_manager.h"
        "include/neat/solution.h"
        "include/neat/species.h"

        "include/solutions/detection_instability.h"
        "include/solutions/memory_instability.h"
        "include/solutions/selection_instability.h"
        "include/solutions/memory_trace.h"
        "include/solutions/delayed_match_to_sample.h"
        "include/solutions/inhibition_of_return.h"
        "include/solutions/and.h"
        "include/solutions/xor.h"

        "include/neat_tools/logger.h"
        "include/neat_tools/utils.h"
        "include/neat_tools/key_listener.h"
        "include/neat_tools/ablation_presets.h"
        "include/neat_tools/config_loader.h"
        "include/neat_tools/resource_paths.h"
        "include/neat_tools/solution_registry.h"
        "include/neat_tools/machine_info.h"
        "include/neat_tools/profiler.h"

        "include/constants.h"
)
set(src
        "src/neat/connection_gene.cpp"
        "src/neat/field_gene.cpp"
        "src/neat/genome.cpp"
        "src/neat/population.cpp"
        "src/neat/population_file_manager.cpp"
        "src/neat/solution.cpp"
        "src/neat/species.cpp"

        "src/solutions/detection_instability.cpp"
        "src/solutions/memory_instability.cpp"
        "src/solutions/selection_instability.cpp"
        "src/solutions/memory_trace.cpp"
        "src/solutions/delayed_match_to_sample.cpp"
        "src/solutions/inhibition_of_return.cpp"
        "src/solutions/and.cpp"
        "src/solutions/xor.cpp"

        "src/neat_tools/logger.cpp"
        "src/neat_tools/ablation_presets.cpp"
        "src/neat_tools/config_loader.cpp"
        "src/neat_tools/resource_paths.cpp"
        "src/neat_tools/machine_info.cpp"
        "src/neat_tools/solution_registry.cpp"
)

# Library target definition
add_library(${CMAKE_PROJECT_NAME} ${header} ${src})
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${NEAT_DNFS_INC_INSTALL_DIR}>
)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${CMAKE_BINARY_DIR}/generated/include")

# Setup imgui - win32 Directx12
find_package(imgui CONFIG REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE imgui::imgui)

# Setup implot
find_package(implot CONFIG REQUIRED)

# Setup imgui-node-editor
find_package(unofficial-imgui-node-editor CONFIG REQUIRED)

# Setup nlohmann-json
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)

# Setup OpenGL (required on macOS/Linux to satisfy imgui-platform-kit's exported link interface)
if(NOT WIN32)
    find_package(OpenGL REQUIRED)
endif()

# Setup imgui-platform-kit
find_package(imgui-platform-kit REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE imgui-platform-kit)

# Setup FFTW3 (transitive dependency of dynamic-neural-field-composer's spectral convolution path)
find_package(FFTW3 CONFIG REQUIRED)

# Setup dynamic-neural-field-composer
find_package(dynamic-neural-field-composer REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE dynamic-neural-field-composer)


target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC
    NEAT_DNFS=1
    NEAT_DNFS_VERSION_MAJOR=${NEAT_DNFS_VERSION_MAJOR}
    NEAT_DNFS_VERSION_MINOR=${NEAT_DNFS_VERSION_MINOR}
    NEAT_DNFS_VERSION_PATCH=${NEAT_DNFS_VERSION_PATCH}
)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "${CMAKE_PROJECT_NAME}-${NEAT_DNFS_VERSION}"
    POSITION_INDEPENDENT_CODE ON
)

# Install the library (binaries)
# The library, its headers and its package config form the "dev" component;
# the executables and the files they read at startup form "runtime". Release
# archives are built with `cmake --install --component runtime` (see
# scripts/package.sh) so they carry only what is needed to run an experiment;
# a plain `cmake --install` still installs both.
install(TARGETS ${CMAKE_PROJECT_NAME} EXPORT ${CMAKE_PROJECT_NAME}Targets
    RUNTIME       DESTINATION ${NEAT_DNFS_RUNTIME_INSTALL_DIR}
    LIBRARY       DESTINATION ${NEAT_DNFS_LIBRARY_INSTALL_DIR}
    ARCHIVE       DESTINATION ${NEAT_DNFS_ARCHIVE_INSTALL_DIR}
    FRAMEWORK     DESTINATION ${NEAT_DNFS_FRAMEWORK_INSTALL_DIR}
    COMPONENT dev
)

# Install the headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${NEAT_DNFS_INC_INSTALL_DIR}/neat-dnfs
        COMPONENT dev
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Package version configuration
include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/${CMAKE_PROJECT_NAME}-config-version.cmake")
write_basic_package_version_file(${version_file}
    VERSION ${NEAT_DNFS_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# config_loader.h publicly includes <nlohmann/json.hpp>, so the installed
# package config must resolve nlohmann_json for downstream consumers -- a
# plain install(EXPORT ... FILE ...config.cmake) has no room for that
# find_dependency() call, hence the template instead.
set(config_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/${CMAKE_PROJECT_NAME}-config.cmake")
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_PROJECT_NAME}-config.cmake.in"
    ${config_file}
    INSTALL_DESTINATION ${NEAT_DNFS_CMAKE_CONFIG_INSTALL_DIR}
)

install(EXPORT ${CMAKE_PROJECT_NAME}Targets DESTINATION ${NEAT_DNFS_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${CMAKE_PROJECT_NAME}-targets.cmake
    COMPONENT dev
)
install(FILES ${config_file} ${version_file}
    DESTINATION ${NEAT_DNFS_CMAKE_CONFIG_INSTALL_DIR} COMPONENT dev)

# Main executable
set(EXE_PROJECT_EXAMPLE neat-dnfs-evol)
add_executable(${EXE_PROJECT_EXAMPLE} "apps/neat_dnfs_evol.cpp")
target_include_directories(${EXE_PROJECT_EXAMPLE} PRIVATE include)
target_link_libraries(${EXE_PROJECT_EXAMPLE} PRIVATE
        imgui::imgui
        imgui-platform-kit
        dynamic-neural-field-composer
        ${CMAKE_PROJECT_NAME}
)

# Incremental evolution executable
set(EXE_PROJECT_EXAMPLE neat-dnfs-inc-evol)
add_executable(${EXE_PROJECT_EXAMPLE} "apps/neat_dnfs_incremental_evolution.cpp")
target_include_directories(${EXE_PROJECT_EXAMPLE} PRIVATE include)
target_link_libraries(${EXE_PROJECT_EXAMPLE} PRIVATE
        imgui::imgui
        imgui-platform-kit
        dynamic-neural-field-composer
        ${CMAKE_PROJECT_NAME}
)

# Solution evaluation executable
set(EXE_PROJECT_EXAMPLE neat-dnfs-sol-eval)
add_executable(${EXE_PROJECT_EXAMPLE} "apps/neat_dnfs_solution_evaluation.cpp")
target_include_directories(${EXE_PROJECT_EXAMPLE} PRIVATE include)
target_link_libraries(${EXE_PROJECT_EXAMPLE} PRIVATE
        imgui::imgui
        imgui-platform-kit
        dynamic-neural-field-composer
        ${CMAKE_PROJECT_NAME}
)

# Runtime package (the "runtime" component)
#
# The executables resolve config/ and templates/ at startup, relative to their
# own location, via neat_tools/resource_paths.h. Installing them to bin/ and the
# data files to share/neat-dnfs/ is exactly the layout that resolver looks for,
# so an installed or unpacked tree runs without any environment set up.
#
# Relative destinations, unlike the absolute NEAT_DNFS_*_INSTALL_DIR ones above:
# only a relative destination follows `cmake --install --prefix` and the staging
# prefix CPack installs into when it builds an archive.
set(NEAT_DNFS_RUNTIME_DESTINATION "bin")
set(NEAT_DNFS_DATA_DESTINATION "share/${CMAKE_PROJECT_NAME}")

install(TARGETS neat-dnfs-evol neat-dnfs-inc-evol neat-dnfs-sol-eval
    RUNTIME DESTINATION ${NEAT_DNFS_RUNTIME_DESTINATION}
    COMPONENT runtime
)

install(DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/config"
        "${CMAKE_CURRENT_SOURCE_DIR}/templates"
    DESTINATION ${NEAT_DNFS_DATA_DESTINATION}
    COMPONENT runtime
)

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../README.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/../CHANGELOG.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE"
    DESTINATION ${NEAT_DNFS_DATA_DESTINATION}
    COMPONENT runtime
)

# vcpkg's Windows triplet is dynamic, so imgui/fftw3/glfw arrive as DLLs that
# have to sit next to the executables -- the same set CI works around by putting
# vcpkg's bin directory on PATH. All three executables link the identical set,
# so one target's list covers them. Both sibling dependencies
# (imgui-platform-kit, dynamic-neural-field-composer) are static libraries and
# contribute nothing here.
if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:neat-dnfs-evol>
        DESTINATION ${NEAT_DNFS_RUNTIME_DESTINATION}
        COMPONENT runtime
    )
endif()

# Setup Catch2
enable_testing()
find_package(Catch2 CONFIG REQUIRED)
include(CTest)
include(Catch)

# Add test project
set(TEST_PROJECT ${CMAKE_PROJECT_NAME}-test)
add_executable(${TEST_PROJECT}
    "tests/entry.cpp"
    "tests/test_utils.cpp"
    "tests/test_logger.cpp"
    "tests/test_field_gene.cpp"
    "tests/test_connection_gene.cpp"
    "tests/test_genome.cpp"
    "tests/test_ablations.cpp"
    "tests/test_solution.cpp"
    "tests/test_solutions_tasks.cpp"
    "tests/test_species.cpp"
    "tests/test_population.cpp"
    "tests/test_population_file_manager.cpp"
    "tests/test_speciation.cpp"
    "tests/test_key_listener.cpp"
    "tests/test_resource_paths.cpp"
    "tests/test_golden.cpp"
    "tests/test_profiler.cpp"
    "tests/test_evolution_helpers.h"
    "tests/solutions/evol_detection_instability.cpp"
    "tests/solutions/evol_memory_instability.cpp"
    "tests/solutions/evol_selection_instability.cpp"
    "tests/solutions/evol_memory_trace.cpp"
    "tests/solutions/evol_inhibition_of_return.cpp"
    "tests/solutions/evol_and.cpp"
    "tests/solutions/evol_xor.cpp"
)
target_include_directories(${TEST_PROJECT} PRIVATE include tests)
target_link_libraries(${TEST_PROJECT} PRIVATE Catch2::Catch2WithMain imgui-platform-kit  dynamic-neural-field-composer ${CMAKE_PROJECT_NAME})
# Automatically discover and add tests, split into a fast default lane and a
# slow lane (the [Evolution] tier runs real Population::evolve() at
# populationSize=50/numGenerations=10/numRuns=5 per solution -- tens of
# seconds, versus ~2.6s for everything else). `ctest -LE slow` stays fast;
# `ctest -L slow` runs the evolution tier.
catch_discover_tests(${TEST_PROJECT} TEST_SPEC "~[Evolution]" TEST_PREFIX "fast.")
catch_discover_tests(${TEST_PROJECT} TEST_SPEC "[Evolution]" TEST_PREFIX "slow."
    PROPERTIES LABELS "slow" TIMEOUT 600)
catch_discover_tests(${TEST_PROJECT} TEST_SPEC "[Golden]" TEST_PREFIX "golden."
    PROPERTIES LABELS "Golden")
