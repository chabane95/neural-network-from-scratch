cmake_minimum_required(VERSION 3.16)
project(spiralnet LANGUAGES C)

add_executable(spiralnet src/main.c src/experiment.c src/nn.c)
set_target_properties(spiralnet PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
if(MSVC)
    target_compile_options(spiralnet PRIVATE /W4)
else()
    target_compile_options(spiralnet PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(spiralnet PRIVATE m)
endif()

enable_testing()
add_test(NAME numerical_checks COMMAND spiralnet check)
add_test(NAME experiment_roundtrip
    COMMAND ${CMAKE_COMMAND} -DPROGRAM=$<TARGET_FILE:spiralnet>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/experiment.cmake)
