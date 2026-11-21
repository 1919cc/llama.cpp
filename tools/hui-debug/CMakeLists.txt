# hui_infer numeric debugging tools (not part of upstream llama.cpp)

foreach(TARGET llama-hui-logit-cmp llama-hui-kvdiff llama-hui-quantize-f32)
    string(REPLACE "llama-hui-" "" SRC ${TARGET})
    add_executable(${TARGET} ${SRC}.cpp)
    target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endforeach()
