cmake_minimum_required(VERSION 3.20...3.30)

project(neat-dnfs
        DESCRIPTION "Dynamic Neural Field Evolution with NEAT for Robotics"
        LANGUAGES CXX)

# Define library version
set(NEAT_DNFS_VERSION_MAJOR 1 CACHE STRING "major version" FORCE)
set(NEAT_DNFS_VERSION_MINOR 0 CACHE STRING "minor version" FORCE)
set(NEAT_DNFS_VERSION ${NEAT_DNFS_VERSION_MAJOR}.${NEAT_DNFS_VERSION_MINOR} CACHE STRING "version" FORCE)

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
)

# Library target definition
add_library(${CMAKE_PROJECT_NAME} ${header} ${src})
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${NEAT_DNFS_INC_INSTALL_DIR}> 
)

# Setup imgui - win32 Directx12
find_package(imgui CONFIG REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE imgui::imgui)

# Setup implot
find_package(implot CONFIG REQUIRED)

# Setup imgui-node-editor
find_package(unofficial-imgui-node-editor CONFIG REQUIRED)

# Setup nlohmann-json
find_package(nlohmann_json CONFIG REQUIRED)

# Setup OpenGL (required on macOS/Linux to satisfy imgui-platform-kit's exported link interface)
if(NOT WIN32)
    find_package(OpenGL REQUIRED)
endif()

# Setup imgui-platform-kit
find_package(imgui-platform-kit REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE imgui-platform-kit)

# Setup dynamic-neural-field-composer
find_package(dynamic-neural-field-composer REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE dynamic-neural-field-composer)


target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC
    NEAT_DNFS=1
    NEAT_DNFS_VERSION_MAJOR=${NEAT_DNFS_VERSION_MAJOR}
    NEAT_DNFS_VERSION_MINOR=${NEAT_DNFS_VERSION_MINOR}
)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "${CMAKE_PROJECT_NAME}-${NEAT_DNFS_VERSION}"
    POSITION_INDEPENDENT_CODE ON
)

# Install the library (binaries)
install(TARGETS ${CMAKE_PROJECT_NAME} EXPORT ${CMAKE_PROJECT_NAME}Targets
    RUNTIME       DESTINATION ${NEAT_DNFS_RUNTIME_INSTALL_DIR}
    LIBRARY       DESTINATION ${NEAT_DNFS_LIBRARY_INSTALL_DIR}
    ARCHIVE       DESTINATION ${NEAT_DNFS_ARCHIVE_INSTALL_DIR}
    FRAMEWORK     DESTINATION ${NEAT_DNFS_FRAMEWORK_INSTALL_DIR}
)

# Install the headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${NEAT_DNFS_INC_INSTALL_DIR}/neat-dnfs
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Package version configuration
include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/${CMAKE_PROJECT_NAME}-config-version.cmake")
write_basic_package_version_file(${version_file}
    VERSION ${NEAT_DNFS_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(EXPORT ${CMAKE_PROJECT_NAME}Targets DESTINATION ${NEAT_DNFS_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${CMAKE_PROJECT_NAME}-config.cmake
)

# Main executable
set(EXE_PROJECT_EXAMPLE neat-dnfs-evol)
add_executable(${EXE_PROJECT_EXAMPLE} "examples/tabula_rasa_evolution.cpp")
target_include_directories(${EXE_PROJECT_EXAMPLE} PRIVATE include)
target_link_libraries(${EXE_PROJECT_EXAMPLE} PRIVATE
        imgui::imgui
        imgui-platform-kit
        dynamic-neural-field-composer
        ${CMAKE_PROJECT_NAME}
)

# Incremental evolution executable
set(EXE_PROJECT_EXAMPLE neat-dnfs-inc-evol)
add_executable(${EXE_PROJECT_EXAMPLE} "examples/incremental_evolution.cpp")
target_include_directories(${EXE_PROJECT_EXAMPLE} PRIVATE include)
target_link_libraries(${EXE_PROJECT_EXAMPLE} PRIVATE
        imgui::imgui
        imgui-platform-kit
        dynamic-neural-field-composer
        ${CMAKE_PROJECT_NAME}
)

# Solution evaluation executable
set(EXE_PROJECT_EXAMPLE neat-dnfs-sol-eval)
add_executable(${EXE_PROJECT_EXAMPLE} "examples/solution_evaluation.cpp")
target_include_directories(${EXE_PROJECT_EXAMPLE} PRIVATE include)
target_link_libraries(${EXE_PROJECT_EXAMPLE} PRIVATE
        imgui::imgui
        imgui-platform-kit
        dynamic-neural-field-composer
        ${CMAKE_PROJECT_NAME}
)

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
    "tests/test_solution.cpp"
    "tests/test_species.cpp"
    "tests/test_population.cpp"
    "tests/test_speciation.cpp"
)
target_include_directories(${TEST_PROJECT} PRIVATE include tests)
target_link_libraries(${TEST_PROJECT} PRIVATE Catch2::Catch2 Catch2::Catch2WithMain imgui-platform-kit  dynamic-neural-field-composer ${CMAKE_PROJECT_NAME})
target_compile_definitions(${TEST_PROJECT} PRIVATE NEAT_DNFS_TEST=1)
# Automatically discover and add tests
catch_discover_tests(${TEST_PROJECT})