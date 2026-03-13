cmake_minimum_required(VERSION 3.22)
project(personaplex_gateway C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# uWebSockets v20 (header-only)
FetchContent_Declare(uWebSockets
    GIT_REPOSITORY https://github.com/uNetworking/uWebSockets.git
    GIT_TAG        v20.66.0
)
FetchContent_Declare(uSockets
    GIT_REPOSITORY https://github.com/uNetworking/uSockets.git
    GIT_TAG        v0.8.8
)
FetchContent_MakeAvailable(uWebSockets uSockets)

# simdjson (fetched as header-only amalgamation)
FetchContent_Declare(simdjson
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git
    GIT_TAG        v3.10.1
)
FetchContent_MakeAvailable(simdjson)

# ---------------------------------------------------------------------------
# uSockets — build the C library that uWebSockets depends on
# ---------------------------------------------------------------------------
file(GLOB USOCKETS_SRCS
    ${usockets_SOURCE_DIR}/src/*.c
    ${usockets_SOURCE_DIR}/src/eventing/*.c
    ${usockets_SOURCE_DIR}/src/crypto/*.c
)
add_library(uSockets STATIC ${USOCKETS_SRCS})
target_include_directories(uSockets PUBLIC ${usockets_SOURCE_DIR}/src)
target_compile_definitions(uSockets PUBLIC LIBUS_NO_SSL)   # TLS handled at container level

# ---------------------------------------------------------------------------
# Triton gRPC C++ client
# Installed by the tritonserver SDK image at /workspace/install/
# ---------------------------------------------------------------------------
set(TRITON_CLIENT_DIR "/workspace/install" CACHE PATH
    "Path to triton client install (set correctly inside Docker)")

find_library(GRPCCLIENT_LIB grpcclient
    HINTS ${TRITON_CLIENT_DIR}/lib
    REQUIRED)

# gRPC/protobuf libs — bundled alongside grpcclient in the SDK image
find_library(GRPC_LIB     grpc      HINTS ${TRITON_CLIENT_DIR}/lib /usr/local/lib)
find_library(GRPCPP_LIB   grpc++    HINTS ${TRITON_CLIENT_DIR}/lib /usr/local/lib)
find_library(GPR_LIB      gpr       HINTS ${TRITON_CLIENT_DIR}/lib /usr/local/lib)
find_library(PROTOBUF_LIB protobuf  HINTS ${TRITON_CLIENT_DIR}/lib /usr/local/lib)

# libcurl — for brain_client HTTP requests
find_package(CURL REQUIRED)

# ---------------------------------------------------------------------------
# Gateway executable
# ---------------------------------------------------------------------------
add_executable(personaplex_gateway
    src/main.cpp
    src/audio_utils.cpp
    src/protocol.cpp
    src/session.cpp
    src/triton_client.cpp
    src/orchestrator.cpp
    src/brain_client.cpp
)

target_include_directories(personaplex_gateway PRIVATE
    src/
    ${uwebsockets_SOURCE_DIR}/src
    ${usockets_SOURCE_DIR}/src
    ${TRITON_CLIENT_DIR}/include
    ${simdjson_SOURCE_DIR}/include
)

# Build the link list — only include libs that were found
set(_LINK_LIBS
    uSockets
    ${GRPCCLIENT_LIB}
    simdjson
    pthread
    z
    CURL::libcurl
)
foreach(_lib GRPCPP_LIB GRPC_LIB GPR_LIB PROTOBUF_LIB)
    if(${_lib})
        list(APPEND _LINK_LIBS ${${_lib}})
    endif()
endforeach()

target_link_libraries(personaplex_gateway PRIVATE ${_LINK_LIBS})

# DGX Spark: enable AVX2 for base64 and SIMD optimisations
target_compile_options(personaplex_gateway PRIVATE
    -O3 -march=native -mtune=native -funroll-loops
    -Wall -Wextra -Wpedantic
)

install(TARGETS personaplex_gateway DESTINATION bin)
