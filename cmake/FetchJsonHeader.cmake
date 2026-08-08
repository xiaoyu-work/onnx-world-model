function(onnx_world_model_resolve_json_header output_variable)
  if(NLOHMANN_JSON_INCLUDE_DIR)
    if(NOT EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
      message(
        FATAL_ERROR
        "NLOHMANN_JSON_INCLUDE_DIR does not contain nlohmann/json.hpp: "
        "${NLOHMANN_JSON_INCLUDE_DIR}"
      )
    endif()
    set(${output_variable} "${NLOHMANN_JSON_INCLUDE_DIR}" PARENT_SCOPE)
    return()
  endif()

  set(version "3.12.0")
  set(destination "${CMAKE_BINARY_DIR}/_deps/nlohmann-json-${version}/include/nlohmann")
  set(target "${destination}/json.hpp")
  if(NOT EXISTS "${target}")
    file(MAKE_DIRECTORY "${destination}")
    message(STATUS "Downloading nlohmann/json ${version}")
    file(
      DOWNLOAD
      "https://github.com/nlohmann/json/releases/download/v${version}/json.hpp"
      "${target}"
      EXPECTED_HASH
        "SHA256=aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63"
      STATUS download_status
      TLS_VERIFY ON
    )
    list(GET download_status 0 status_code)
    list(GET download_status 1 status_message)
    if(NOT status_code EQUAL 0)
      file(REMOVE "${target}")
      message(FATAL_ERROR "Failed to download nlohmann/json: ${status_message}")
    endif()
  endif()
  get_filename_component(include_directory "${destination}" DIRECTORY)
  set(${output_variable} "${include_directory}" PARENT_SCOPE)
endfunction()
