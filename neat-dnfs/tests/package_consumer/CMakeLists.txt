# A downstream project that uses only an installed neat-dnfs, to check that
# find_package(neat-dnfs) resolves everything the package needs. CI builds and runs it
# after installing the "dev" component; it is not part of the neat-dnfs build.
cmake_minimum_required(VERSION 3.21...3.30)
project(neat-dnfs-package-consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(neat-dnfs CONFIG REQUIRED)

add_executable(neat-dnfs-package-consumer main.cpp)
target_link_libraries(neat-dnfs-package-consumer PRIVATE neat-dnfs)
