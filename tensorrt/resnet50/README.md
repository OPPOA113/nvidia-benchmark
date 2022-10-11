# resnet

#data download
链接：https://pan.baidu.com/s/1UguUoJGSARhxn5XwPNeEcA 
提取码：bp93 
下载得到resnet50_yolov3_benchmark文件夹

For the Pytorch implementation, you can refer to [tensorrt_benchmark/pytorch/resnet/resnet50](https://github.com/amtf1683/tensorrt_benchmark.git)


## TensorRT C++ API tensorrt7.2.1.6 CUDA Toolkit 11.1

```
// 1a. generate resnet50.wts from [tensorrt_benchmark/pytorch/resnet/resnet50](https://github.com/amtf1683/tensorrt_benchmark.git)
git clone https://github.com/amtf1683/tensorrt_benchmark.git

// 2. put resnet50.wts into resnet50(tensorrt_benchmark/tensorrt/resnet50)

// 3. build and run

cd resnet50

mkdir build
cd build
cmake ..
make
./resnet50 -s   // serialize model to plan file i.e. 'resnet50.engine'
./resnet50 -d   // deserialize plan file and run inference
./resnet50 -top //test top1 or top5

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For ILSVRC2012, you can get download my calibration images `images_resnet50_ILSVRC2012_val` from resnet50_yolov3_benchmark/images_resnet50_ILSVRC2012_val

2. cp -r images_resnet50_ILSVRC2012_val to resnet50/build
   mv images_resnet50_ILSVRC2012_val images 
   mkdir images_int8_simple and pick 1000 images as a quantitative calibration dataset from images 

3. set the macro `USE_INT8` in resnet50.cpp and make

4. serialize the model and test

