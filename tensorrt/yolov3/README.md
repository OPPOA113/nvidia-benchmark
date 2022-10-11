# yolov3
#data download
链接：https://pan.baidu.com/s/1UguUoJGSARhxn5XwPNeEcA 
提取码：bp93 
下载得到resnet50_yolov3_benchmark文件夹

The Pytorch implementation is [ultralytics/yolov3 archive branch](https://github.com/ultralytics/yolov3/tree/archive). It provides two trained weights of yolov3, `yolov3.weights` and `yolov3.pt`

This branch is using tensorrt7.2.1.6 API

## Config

- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h
- INT8/FP16/FP32 can be selected by the macro in yolov3.cpp
- GPU id can be selected by the macro in yolov3.cpp
- NMS thresh in yolov3.cpp
- BBox confidence thresh in yolov3.cpp

---

tensorrt版本测试说明：

1. 工程中，tensorrt插件实现了NMS的方式，所以在common.hpp文件中，`static int YOLOLAYER_OUTPUT_NUMBER = 2;`参数可设置是否采用tensorrt的后处理逻辑：为1 表示不采用tensorrt的后处理，为2则采用插件实现的后处理；

2. 在编译时，分别设置`YOLOLAYER_OUTPUT_NUMBER`=1，删除`libyololayer.so,yolov3.engine`，重新编译，测试 `only-forward`的时间，`YOLOLAYER_OUTPUT_NUMBER`=2，删除`libyololayer.so,yolov3.engine`，重新编译，测试`preprocess`和`foward+nms`的时间



---

## How to run

1. python3 environment

git clone https://github.com/amtf1683/tensorrt_benchmark.git

cd pytorch/yolov3

pip install -r requirements.txt 

2. generate yolov3.wts from pytorch implementation with yolov3.cfg and yolov3.weights, or use download .wts文件

resnet50_yolov3_benchmark/yolov3.wts 

// download its weights 'yolov3.pt' or 'yolov3.weights'

git clone https://github.com/amtf1683/tensorrt_benchmark.git

cp tensorrt_benchmark/tensorrt/yolov3/gen_wts.py tensorrt_benchmark/pytorch/yolov3

cd tensorrt_benchmark/pytorch/yolov3

python3 gen_wts.py yolov3.weights

// a file 'yolov3.wts' will be generated.
// the master branch of yolov3 should work, if not, you can checkout cf7a4d31d37788023a9186a1a143a2dab0275ead
```
3. put yolov3.wts into yolov3, build and run
```
mv yolov3.wts tensorrt_benchmark/tensorrt/yolov3/
cd tensorrt_benchmark/tensorrt/yolov3
mkdir build
cd build
cmake ..
make
cp -r resnet50_yolov3_benchmark/coco_val2017 yolov3/build/
mv coco_val2017  val2017
./yolov3 -s              // serialize model to plan file i.e. 'yolov3.engine'
./yolov3 -d  // deserialize plan file and run inference
./yolov3 -map //test coco_val2017 map0.5 or map0.7
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can get download my calibration images `coco_calib` from resnet50_yolov3_benchmark/coco_calib

2. cp -r coco_calib to tensorrt_benchmark/tensorrt/yolov3/build

3. set the macro `USE_INT8` in yolov3.cpp and make

4. serialize the model and test
