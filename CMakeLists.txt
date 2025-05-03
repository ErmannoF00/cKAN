cmake_minimum_required(VERSION 3.16)
project(KANViewer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch nlohmann/json automatically
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(nlohmann_json)

# Find Qt
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Add source directories
add_subdirectory(src)
