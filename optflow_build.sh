# https://github.com/ucisysarch/opencvjs/blob/master/make.py
cd opencv
mkdir build
cd build
OPTS='-O2 --llvm-lto 1 -s SIMD=1'
# configure
echo emcmake cmake
emcmake cmake \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DBUILD_DOCS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_PACKAGE=OFF \
  -DBUILD_WITH_DEBUG_INFO=OFF \
  -DBUILD_opencv_bioinspired=OFF \
  -DBUILD_opencv_calib3d=OFF \
  -DBUILD_opencv_cuda=OFF \
  -DBUILD_opencv_cudaarithm=OFF \
  -DBUILD_opencv_cudabgsegm=OFF \
  -DBUILD_opencv_cudacodec=OFF \
  -DBUILD_opencv_cudafeatures2d=OFF \
  -DBUILD_opencv_cudafilters=OFF \
  -DBUILD_opencv_cudaimgproc=OFF \
  -DBUILD_opencv_cudaoptflow=OFF \
  -DBUILD_opencv_cudastereo=OFF \
  -DBUILD_opencv_cudawarping=OFF \
  -DBUILD_opencv_gpu=OFF \
  -DBUILD_opencv_gpuarithm=OFF \
  -DBUILD_opencv_gpubgsegm=OFF \
  -DBUILD_opencv_gpucodec=OFF \
  -DBUILD_opencv_gpufeatures2d=OFF \
  -DBUILD_opencv_gpufilters=OFF \
  -DBUILD_opencv_gpuimgproc=OFF \
  -DBUILD_opencv_gpuoptflow=OFF \
  -DBUILD_opencv_gpustereo=OFF \
  -DBUILD_opencv_gpuwarping=OFF \
  -BUILD_opencv_hal=OFF \
  -DBUILD_opencv_highgui=OFF \
  -DBUILD_opencv_java=OFF \
  -DBUILD_opencv_legacy=OFF \
  -DBUILD_opencv_ml=ON \
  -DBUILD_opencv_nonfree=OFF \
  -DBUILD_opencv_optim=OFF \
  -DBUILD_opencv_photo=ON \
  -DBUILD_opencv_shape=ON \
  -DBUILD_opencv_objdetect=ON \
  -DBUILD_opencv_softcascade=ON \
  -DBUILD_opencv_stitching=OFF \
  -DBUILD_opencv_superres=OFF \
  -DBUILD_opencv_ts=ON \
  -DBUILD_opencv_videostab=ON \
  -DBUILD_opencv_videoio=OFF \
  -DBUILD_opencv_imgcodecs=OFF \
  -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DWITH_1394=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_CUFFT=OFF \
  -DWITH_EIGEN=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_GIGEAPI=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_GTK=OFF \
  -DWITH_JASPER=OFF \
  -DWITH_JPEG=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_OPENCLAMDBLAS=OFF \
  -DWITH_OPENCLAMDFFT=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_PNG=OFF \
  -DWITH_PVAPI=OFF \
  -DWITH_TIFF=OFF \
  -DWITH_LIBV4L=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_PTHREADS_PF=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DWITH_IPP=OFF \
  -DENABLE_SSE=OFF \
  -DENABLE_SSE2=OFF \
  -DENABLE_SSE3=OFF \
  -DENABLE_SSE41=OFF \
  -DENABLE_SSE42=OFF \
  -DENABLE_AVX=OFF \
  -DENABLE_AVX2=OFF \
  -DCMAKE_CXX_FLAGS=$OPTS \
  -DCMAKE_EXE_LINKER_FLAGS=$OPTS \
  -DCMAKE_CXX_FLAGS_DEBUG=$OPTS \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=$OPTS \
  -DCMAKE_C_FLAGS_RELWITHDEBINFO=$OPTS \
  -DCMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO=$OPTS \
  -DCMAKE_MODULE_LINKER_FLAGS_RELEASE=$OPTS \
  -DCMAKE_MODULE_LINKER_FLAGS_DEBUG=$OPTS \
  -DCMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO=$OPTS \
  -DCMAKE_SHARED_LINKER_FLAGS_RELEASE=$OPTS \
  -DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO=$OPTS \
  -DCMAKE_SHARED_LINKER_FLAGS_DEBUG=$OPTS \
  ../
# make opencv libs
echo emmake make
emmake make -j 4
# make bindings.bc
echo binding...
emcc \
  -I./ \
  -I../modules/core/include \
  -I../modules/imgproc/include \
  -I../modules/ts/include \
  -I../modules/video/include \
  ../../optflow_bindings.cpp \
  --bind \
  -O2 -s SIMD=1 \
  -o ./optflow_bindings.bc
# link
# The Object File order is important - https://github.com/kripken/emscripten/issues/2619
echo linking...
emcc \
  ./optflow_bindings.bc \
  ./lib/libopencv_video.a \
  ./lib/libopencv_imgproc.a \
  ./lib/libopencv_core.a \
  ./3rdparty/lib/libzlib.a \
  --bind \
  -O2 --llvm-lto 1 -s SIMD=1 \
  -o ./optflow_binded.bc
# compile to js
echo converting to js

emcc \
  optflow_binded.bc \
  --bind \
  -O2 --llvm-lto 3 -s SIMD=1 \
  -s TOTAL_MEMORY=33554432 \
  -s ASSERTIONS=0 \
  -o ../../cv.js
# -s ALLOW_MEMORY_GROWTH=1 \
# --memory-init-file 0 \
# -s MODULARIZE=1 \
# -s NO_FILESYSTEM=0 \
# -s ASSERTIONS=0 \
# -s AGGRESSIVE_VARIABLE_ELIMINATION=0 \
# -s NO_DYNAMIC_EXECUTION=0 \
# -s ALLOW_MEMORY_GROWTH=1 \
# -s TOTAL_MEMORY=33554432 \
#  --js-flags="--harmony_simd --wasm_simd_prototype" 