cmake_minimum_required(VERSION 3.16)
project(swn_mmcf_public_interface VERSION 0.1.0 LANGUAGES CXX)

add_library(swn_mmcf_api INTERFACE)
target_include_directories(
  swn_mmcf_api
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(swn_mmcf_api INTERFACE cxx_std_17)

option(SWN_MMCF_BUILD_HEADER_CHECK "Compile the public API smoke test" ON)
if(SWN_MMCF_BUILD_HEADER_CHECK)
  add_executable(swn_mmcf_header_check tests/header_smoke_test.cpp)
  target_link_libraries(swn_mmcf_header_check PRIVATE swn_mmcf_api)
endif()

install(DIRECTORY include/ DESTINATION include)

