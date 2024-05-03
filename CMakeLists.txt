# CMakeの最小バージョンを指定します。プロジェクトで使われているCMakeのコマンドや機能が、このバージョン以降で提供されていることを保証します。
cmake_minimum_required(VERSION 3.0)

# プロジェクトの名前を指定します。ここではプロジェクト名として'cognitiveGames'を設定しています。
project(cognitiveGames)

# ソースファイルを指定します。ここでは、main.cppをソースファイルとして設定しています。
set(SOURCE_FILE main.cpp)

# OpenCVが必要であることを示します。CMakeは、システム内でOpenCVを見つけ出し、それをビルドにリンクします。
find_package(OpenCV REQUIRED)

# CPP VERSION
set(CMAKE_CXX_STANDARD 17)

# OpenCVのヘッダーファイルへのパスをインクルードパスに追加します。これにより、ソースコード内でOpenCVのヘッダーを直接参照できます。
# include 2 subdirectories of tflite include: tensorflow and flatbuffers
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/include/
)

# add lib libtensorflowlite.dylib
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/libs/macM1/libtensorflowlite.dylib)

# 実行可能なバイナリー'main'を作成します。そのソースコードとして、上で定義したSOURCE_FILE（つまり、main.cpp）を使用します。
add_executable(main
    ${SOURCE_FILE}
)

# バイナリー'main'に対するリンクを設定します。ここでは、OpenCVライブラリをリンクしています。
target_link_libraries(main
    ${OpenCV_LIBRARIES}
    tensorflowlite
)
