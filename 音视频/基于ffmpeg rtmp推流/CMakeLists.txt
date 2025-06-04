# CMakeList.txt: CMakeProject3 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CMakeProject3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# 设置 FFmpeg 根路径
set(FFMPEG_DIR "D:/ffmpeg-7.1.1-full_build/ffmpeg-7.1.1-full_build")

# 头文件路径
include_directories(${FFMPEG_DIR}/include)

# 库文件路径
link_directories(${FFMPEG_DIR}/lib)

# SDL2 路径
set(SDL2_DIR "D:/SDL2-2.32.6")
include_directories(${SDL2_DIR}/include)
link_directories(${SDL2_DIR}/lib/x64)

# 将源代码添加到此项目的可执行文件。
add_executable (CMakeProject3 "CMakeProject3.cpp" "CMakeProject3.h")
# 链接的 FFmpeg 库（.lib）
target_link_libraries(CMakeProject3
    avformat avcodec avutil swscale
    SDL2 SDL2main
)

add_custom_command(TARGET CMakeProject3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "D:/SDL2-2.32.6/lib/x64/SDL2.dll"
    $<TARGET_FILE_DIR:CMakeProject3>
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeProject3 PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
