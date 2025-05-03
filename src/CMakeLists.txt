file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    *.cpp
    *.cu
    *.h
)

add_executable(KANViewer ${SRC_FILES})

# Link Qt and JSON
target_link_libraries(KANViewer PRIVATE Qt6::Widgets nlohmann_json::nlohmann_json)

# Enable CUDA language for device code
set_target_properties(KANViewer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
)

# Optional: Suppress warnings from CUDA headers
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    target_compile_options(KANViewer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif
