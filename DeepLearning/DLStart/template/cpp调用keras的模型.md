## cmake文件

```
cmake_minimum_required(VERSION 3.14)
project(cppdemo)

set(CMAKE_CXX_STANDARD 14)

include_directories("3rd_party/pocket/include")
include_directories("3rd_party/libsimdpp")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(cppdemo main.cpp)
target_link_libraries(cppdemo /home/zzh/CLionProjects/cppdemo/3rd_party/pocket/lib/libpocket-tensor.a )
```

## 头文件和库函数目录结构

![image-20191130195018133](img/image-20191130195018133.png)

## cpp代码

```
#include <iostream>
#include "pt_model.h"
#include "pt_tensor.h"


int main()
{
    // Initialize model:
    auto model = pt::Model::create("example.model");
    // REQUIRE(model);

    // Create input tensor:
    pt::Tensor in(2);
    in.setData({1, 0});

    // Run prediction:
    pt::Tensor out;
    bool success = model->predict(std::move(in), out);
    // REQUIRE(success);

    // Print output:
    std::cout << out << std::endl;
    return 0;
}
```



## 模型文件创建

```
# make_model.py:
# 引入pt.py文件后，到处训练完毕后的模型

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pt import export_model

test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')

model = Sequential()
model.add(Dense(1, input_dim=10))

model.compile(loss='mean_squared_error', optimizer='adamax')
model.fit(test_x, test_y, epochs=1)

print model.predict(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))

export_model(model, 'example.model')
```



## 模型文件的位置

模型文件应该与cpp的可执行性文件在同级目录