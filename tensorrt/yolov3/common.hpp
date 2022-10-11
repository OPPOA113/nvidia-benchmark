#ifndef YOLOV3_COMMON_H_
#define YOLOV3_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <cuda_runtime.h>
#include <cstdint>
#include<algorithm>
using namespace nvinfer1;
/*YOLOLAYER_OUTPUT_NUMBER = 1:The output bbox and score of the yolo layer are arranged in Detection format,so it can not use tensorrt nms plugin
YOLOLAYER_OUTPUT_NUMBER = 2:The output bbox and score of the yolo layer are stored separately, so it can use tensorrt nms plugin
***********************************************************************/
static int YOLOLAYER_OUTPUT_NUMBER = 1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");


    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // printf("len:%d\n",len);
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    // printf("scale: %d-%d\n",scale.type,scale.count);
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};
// printf("shift: %d-%d\n",shift.type,shift.count);
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};
    // printf("power: %d-%d\n",power.type,power.count);
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-5);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);

    lr->setAlpha(0.1);

    return lr;
}


IPluginV2Layer *addBatchedNMSLayer(INetworkDefinition *network, IPluginV2Layer *yolo, int num_classes, int top_k, int keep_top_k, float score_thresh, float iou_thresh, bool is_normalized = false, bool clip_boxes = false)
{
    auto creator = getPluginRegistry()->getPluginCreator("BatchedNMS_TRT", "1");
    // Set plugin fields and the field collection
    const bool share_location = true;
    const int background_id = -1;
    PluginField fields[9] = {
        PluginField{"shareLocation", &share_location,
                    PluginFieldType::kINT32, 1},
        PluginField{"backgroundLabelId", &background_id,
                    PluginFieldType::kINT32, 1},
        PluginField{"numClasses", &num_classes,
                    PluginFieldType::kINT32, 1},
        PluginField{"topK", &top_k, PluginFieldType::kINT32,
                    1},
        PluginField{"keepTopK", &keep_top_k,
                    PluginFieldType::kINT32, 1},
        PluginField{"scoreThreshold", &score_thresh,
                    PluginFieldType::kFLOAT32, 1},
        PluginField{"iouThreshold", &iou_thresh,
                    PluginFieldType::kFLOAT32, 1},
        PluginField{"isNormalized", &is_normalized,
                    PluginFieldType::kINT32, 1},
        PluginField{"clipBoxes", &clip_boxes,
                    PluginFieldType::kINT32, 1},
    };
    PluginFieldCollection pfc{9, fields};
    IPluginV2 *pluginObj = creator->createPlugin("batchednms", &pfc);
    ITensor *inputTensors[] = {yolo->getOutput(0), yolo->getOutput(1)};
    auto batchednmslayer = network->addPluginV2(inputTensors, 2, *pluginObj);
    batchednmslayer->setName("nms_layer");
    assert(batchednmslayer);
    return batchednmslayer;
}

#endif