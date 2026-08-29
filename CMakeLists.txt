cmake_minimum_required(VERSION 3.24)

project(
  onnx_world_model
  VERSION 0.9.0
  DESCRIPTION "Runtime engine for Mobius-exported ONNX world models"
  LANGUAGES CXX
)

option(ONNX_WORLD_MODEL_BUILD_PYTHON "Build the Python extension" OFF)
option(ONNX_WORLD_MODEL_BUILD_TESTS "Build C++ tests" OFF)
set(ONNX_RUNTIME_HEADER_VERSION "1.28.0")
set(
  ONNXRUNTIME_INCLUDE_DIR
  ""
  CACHE PATH
  "Existing directory containing ONNX Runtime C/C++ API headers"
)
set(
  NLOHMANN_JSON_INCLUDE_DIR
  ""
  CACHE PATH
  "Existing include directory containing nlohmann/json.hpp"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchOrtHeaders)
onnx_world_model_resolve_ort_headers(ORT_INCLUDE_DIR)
include(FetchJsonHeader)
onnx_world_model_resolve_json_header(JSON_INCLUDE_DIR)

add_library(
  onnx_world_model
  src/cancellation.cpp
  src/dynamic_library.cpp
  src/model.cpp
  src/ort_backend.cpp
  src/pipeline.cpp
  src/pipeline_manifest_common.cpp
  src/pipeline_manifest_validation.cpp
  src/pipeline_scheduler.cpp
  src/pipeline_session.cpp
  src/pipeline_telemetry.cpp
  src/stage_executor.cpp
  src/stage_executor_autoregressive.cpp
  src/stage_executor_iterative.cpp
  src/stage_registry.cpp
  src/tensor.cpp
  src/world_model.cpp
)
add_library(onnx_world_model::onnx_world_model ALIAS onnx_world_model)

target_compile_features(onnx_world_model PUBLIC cxx_std_20)
set_target_properties(
  onnx_world_model
  PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    SOVERSION 0.9
    VERSION ${PROJECT_VERSION}
)
target_compile_definitions(
  onnx_world_model
  PRIVATE
    ORT_API_MANUAL_INIT
)
target_include_directories(
  onnx_world_model
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
  PRIVATE
    "${ORT_INCLUDE_DIR}"
    "${JSON_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if(MSVC)
  target_compile_options(onnx_world_model PRIVATE /W4 /permissive- /EHsc)
  target_compile_definitions(onnx_world_model PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  target_compile_options(onnx_world_model PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(onnx_world_model PRIVATE ${CMAKE_DL_LIBS})
endif()

if(ONNX_WORLD_MODEL_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native MODULE bindings/python_module.cpp)
  target_link_libraries(_native PRIVATE onnx_world_model)
  target_compile_features(_native PRIVATE cxx_std_20)
  install(
    TARGETS _native
    LIBRARY DESTINATION onnx_world_model COMPONENT python
    RUNTIME DESTINATION onnx_world_model COMPONENT python
  )
endif()

if(ONNX_WORLD_MODEL_BUILD_TESTS)
  enable_testing()
  foreach(
    test_name
    IN ITEMS
      tensor
      model
      cancellation
      pipeline
      pipeline_device
      pipeline_snapshot
      pipeline_stream
      pipeline_cancellation
      pipeline_scheduler
      pipeline_telemetry
      pipeline_trace
      pipeline_precision
  )
    add_executable(${test_name}_test tests/cpp/${test_name}_test.cpp)
    target_link_libraries(${test_name}_test PRIVATE onnx_world_model)
    target_compile_features(${test_name}_test PRIVATE cxx_std_20)
    add_test(NAME ${test_name}_test COMMAND ${test_name}_test)
    # Last-resort hang protection. Every concurrent test already bounds its
    # own waits, so a run that reaches this ceiling is a deadlock, and CTest
    # reporting it beats a build agent waiting forever.
    set_tests_properties(${test_name}_test PROPERTIES TIMEOUT 60)
  endforeach()
endif()

include(GNUInstallDirs)
install(
  TARGETS onnx_world_model
  EXPORT onnx_world_model_targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
)
install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT development
)
install(
  EXPORT onnx_world_model_targets
  NAMESPACE onnx_world_model::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_world_model"
  COMPONENT development
)
