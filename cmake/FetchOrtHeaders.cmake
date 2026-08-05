function(onnx_world_model_resolve_ort_headers output_variable)
  if(ONNXRUNTIME_INCLUDE_DIR)
    if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
      message(
        FATAL_ERROR
        "ONNXRUNTIME_INCLUDE_DIR does not contain onnxruntime_cxx_api.h: "
        "${ONNXRUNTIME_INCLUDE_DIR}"
      )
    endif()
    set(${output_variable} "${ONNXRUNTIME_INCLUDE_DIR}" PARENT_SCOPE)
    return()
  endif()

  set(
    destination
    "${CMAKE_BINARY_DIR}/_deps/onnxruntime-${ONNX_RUNTIME_HEADER_VERSION}/include"
  )
  file(MAKE_DIRECTORY "${destination}")
  set(
    base_url
    "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNX_RUNTIME_HEADER_VERSION}/include/onnxruntime/core/session"
  )

  set(
    headers
    "onnxruntime_c_api.h|b69a133143c0da61782b2b02cc8a620d21ac139231fd211719776d10385135c1"
    "onnxruntime_cxx_api.h|c6bf005e4063b7db911d0a6b030452bcb9d4fc5e87530a39d94c4a5fcd311b61"
    "onnxruntime_cxx_inline.h|97f960b1bc4a372918893ef4c77102dee725ccfce0d76607f4570b03e7e28325"
    "onnxruntime_ep_c_api.h|db86df0f846b8d3bdbf429a5c76c5805293196575ca3c475878612591cd18e51"
    "onnxruntime_error_code.h|5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721"
    "onnxruntime_float16.h|88b242845d25981633a0bbd1c148e273cf8bfb016ea3f57c4af41a06530f72b0"
  )

  foreach(header_and_hash IN LISTS headers)
    string(REPLACE "|" ";" header_parts "${header_and_hash}")
    list(GET header_parts 0 header)
    list(GET header_parts 1 sha256)
    set(target "${destination}/${header}")
    if(NOT EXISTS "${target}")
      message(STATUS "Downloading ONNX Runtime header ${header}")
      file(
        DOWNLOAD
        "${base_url}/${header}"
        "${target}"
        EXPECTED_HASH "SHA256=${sha256}"
        STATUS download_status
        TLS_VERIFY ON
      )
      list(GET download_status 0 status_code)
      list(GET download_status 1 status_message)
      if(NOT status_code EQUAL 0)
        file(REMOVE "${target}")
        message(FATAL_ERROR "Failed to download ${header}: ${status_message}")
      endif()
    endif()
  endforeach()

  set(${output_variable} "${destination}" PARENT_SCOPE)
endfunction()
