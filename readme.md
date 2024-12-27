cmake ../ -G "MinGW Makefiles" -DPRODUCTION_OPTIMIZATION=ON -DCMAKE_BUILD_TYPE=Release

maintain tools安装serial

记得使用清华源 --mirror https://mirrors.ustc.edu.cn/qtproject

cmake --build . -j4
or mingw32-make -j4