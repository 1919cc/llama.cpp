# hui_infer: the Android JNI engine apps embed (not part of upstream llama.cpp).
# libhui_engine.so + the Kotlin binding java/com/hui/infer/HuiEngine.kt; installed next to
# libllama*.so so the Android package (scripts/snapdragon/build.py --target adb) carries it.

set(TARGET hui-engine)

add_library(${TARGET} SHARED hui_engine.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME hui_engine)
target_link_libraries(${TARGET} PRIVATE llama-common llama ggml log ${CMAKE_DL_LIBS})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# JNI entry points are looked up by name, they have no prototypes to declare
target_compile_options(${TARGET} PRIVATE -Wno-missing-prototypes)
# 16 KB page-size devices (Android 15+) need 16 KB aligned segments
target_link_options(${TARGET} PRIVATE -Wl,-z,max-page-size=16384)

install(TARGETS ${TARGET} LIBRARY)
