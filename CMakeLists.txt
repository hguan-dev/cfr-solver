cmake_minimum_required(VERSION 3.20)

project(cfr_solver_pipeline LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PROFILING "Enable gprof profiling" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring with Debug symbols")
    add_compile_options(-g)
    add_link_options(-g)
endif()

if(PROFILING)
    message(STATUS "Enabling gprof profiling")
    add_compile_options(-pg)
    add_link_options(-pg)
endif()

# include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
find_package(fmt REQUIRED)

set(CORE_IMPLEMENTATION_SOURCES
    src/cards/Deck.cpp
    src/game/HandEvaluator.cpp
    src/game/HandHashes.cpp
    src/game/GameState.cpp
    src/cfr/Solver.cpp
)

set(COMMON_INCLUDE_DIRS 
    src/cards 
    src/game 
    src/common 
    src/cfr 
)

set(COMMON_LINK_LIBRARIES fmt::fmt pthread)

add_executable(cfr_solver
    src/main.cpp
    ${CORE_IMPLEMENTATION_SOURCES}
)

target_include_directories(cfr_solver PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(cfr_solver PRIVATE ${COMMON_LINK_LIBRARIES})
set_target_properties(cfr_solver PROPERTIES RUNTIME_OUTPUT_DIREdTORY ${CMAKE_BINARY_DIR}/bin)
