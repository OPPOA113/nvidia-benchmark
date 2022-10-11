#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "calibrator.h"
#include "log_time.hpp"
#include "preprocess.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "logging.h"
#include "data_reader.hpp"

using namespace std;
using namespace nvinfer1;
using namespace Tn;

// #define USE_INT8//  // set USE_INT8 or USE_FP16 or USE_FP32
// #define BATCH_SIZE 1

#define MAX_IMAGE_WIDTH 1920
#define MAX_IMAGE_HEIGHT 1080
#define MAX_IMAGE_INPUT_SIZE_THRESH MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT // ensure it exceed the maximum size in the input images !
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static Logger gLogger;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;
static bool test_classify_top = false;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static std::string preprocess_device = "CPU";
const float EPSILON = 1e-6;
std::string data_root_path = "";
std::string model_name = "";

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
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
    // std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine_res50(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, string &type)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../" + model_name + ".wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.2.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.3.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.5.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.2.");

    IPoolingLayer* pool2 = network->addPoolingNd(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});
    
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);

    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 30);
    // config->setMaxWorkspaceSize(4 << 20);

    if(type == "USE_FP16"){
        config->setFlag(BuilderFlag::kFP16);
    }
    else if(type == "USE_INT8"){
        std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(64, INPUT_W, INPUT_H,
                                                                        std::string(data_root_path + "./ILSVRC2012_img_calib/").c_str(),
                                                                        std::string(model_name + "_int8calib.table").c_str(), INPUT_BLOB_NAME);
        config->setInt8Calibrator(calibrator);
    }

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
        ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine_res18(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, string &type)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../"+model_name+".wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

    IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});
    
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 30);

    if(type == "USE_FP16"){
        config->setFlag(BuilderFlag::kFP16);
    }
    else if(type == "USE_INT8"){
        std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(64, INPUT_W, INPUT_H,
                                                                        std::string(data_root_path + "./ILSVRC2012_img_calib/").c_str(),
                                                                        std::string(model_name + "_int8calib.table").c_str(),
                                                                        INPUT_BLOB_NAME);
        config->setInt8Calibrator(calibrator);
    }

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

// Create the engine using only the API and not any parser.
ICudaEngine* createEngine_res34(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, string &type) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shpae { 3, INPUT_H INPPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3,INPUT_H,INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../"+model_name+".wts");
    Weights emptywts{ DataType::kFLOAT,nullptr,0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{ 7,7 }, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 2,2 });
    conv1->setPaddingNd(DimsHW{ 3,3 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3,3 });
    assert(pool1);
    pool1->setStrideNd(DimsHW{ 2,2 });
    pool1->setPaddingNd(DimsHW{ 1,1 });

    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");
    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 64, 1, "layer1.2.");
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 64, 128, 2, "layer2.0.");
    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 128, 1, "layer2.1.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 128, 128, 1, "layer2.2.");
    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 128, 128, 1, "layer2.3.");
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 128, 256, 2, "layer3.0.");
    IActivationLayer* relu10 = basicBlock(network, weightMap, *relu9->getOutput(0), 256, 256, 1, "layer3.1.");
    IActivationLayer* relu11 = basicBlock(network, weightMap, *relu10->getOutput(0), 256, 256, 1, "layer3.2.");
    IActivationLayer* relu12 = basicBlock(network, weightMap, *relu11->getOutput(0), 256, 256, 1, "layer3.3.");
    IActivationLayer* relu13 = basicBlock(network, weightMap, *relu12->getOutput(0), 256, 256, 1, "layer3.4.");
    IActivationLayer* relu14 = basicBlock(network, weightMap, *relu13->getOutput(0), 256, 256, 1, "layer3.5.");
    IActivationLayer* relu15 = basicBlock(network, weightMap, *relu14->getOutput(0), 256, 512, 2, "layer4.0.");
    IActivationLayer* relu16 = basicBlock(network, weightMap, *relu15->getOutput(0), 512, 512, 1, "layer4.1.");
    IActivationLayer* relu17 = basicBlock(network, weightMap, *relu16->getOutput(0), 512, 512, 1, "layer4.2.");
    IPoolingLayer* pool2 = network->addPoolingNd(*relu17->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 7,7 });
    assert(pool2);
    pool2->setStrideNd(DimsHW{ 1,1 });
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 30);

    if(type == "USE_FP16"){
        config->setFlag(BuilderFlag::kFP16);
    }
    else if(type == "USE_INT8"){
        std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(64, INPUT_W, INPUT_H,
                                                                        std::string(data_root_path + "./ILSVRC2012_img_calib/").c_str(),
                                                                        std::string(model_name + "_int8calib.table").c_str(),
                                                                        INPUT_BLOB_NAME);
        config->setInt8Calibrator(calibrator);
    }

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}



void APIToModel(unsigned int maxBatchSize, std::string runType, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if(model_name == "resnet50"){
        createEngine_res50(maxBatchSize, builder, config, DataType::kFLOAT, runType);
    }
    else if(model_name == "resnet18"){
        createEngine_res18(maxBatchSize, builder, config, DataType::kFLOAT, runType);
    }
    else if(model_name == "resnet34"){
        createEngine_res34(maxBatchSize, builder, config, DataType::kFLOAT, runType);
    }
    
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInferencePreprocessCpu(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

//preprocess with gpu cuda
void doInference(IExecutionContext& context, cudaStream_t &stream, void** buffers, float* output, int outputIndex, int batchSize)
{
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

float getGPUMemoryAvailable(float& totoalM)
{
	size_t avaiavle = 0;
	size_t total = 0;
	cuMemGetInfo(&avaiavle, &total);
    totoalM = 1.0 * total / 1024 / 1024;
    return 1.0 * avaiavle / 1024 / 1024; // return used gpu M
}

#if 0
cv::Mat ImgResize_(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    //cv::resize(img, img, cv::Size(img_h, img_w));
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(0, 0, 0));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Mat ImgResize(cv::Mat& img, int img_h, int img_w) {
    int raw_h = img.rows;
    int raw_w = img.cols;

    int new_w = img.cols;
    int new_h = img.rows;
    if((( float )img_w / img.cols) < (( float )img_h / img.rows))
    {
        new_w = img_w;
        new_h = (img.rows * img_w) / img.cols;
    }
    else
    {
        new_h = img_h;
        new_w = (img.cols * img_h) / img.rows;
    }
    cv::Mat resize_img;
    cv::Mat dst_img;
    cv::resize(img, resize_img, cv::Size(new_w, new_h));

    int delta_h = (img_h - new_h) * 0.5f;
    int delta_w = (img_w - new_w) * 0.5f;
    cv::copyMakeBorder(resize_img, dst_img, delta_h, delta_h, delta_w, delta_w, cv::BORDER_CONSTANT);
    return dst_img;
}
#endif
void ImagePreprocess(cv::Mat img, float* input_data)
{
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    cv::Mat temp = channels[2];
    channels[2] = channels[0];
    channels[0] = temp;
    cv::merge(channels, img);
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = INPUT_W * INPUT_H;//img_h * img_w;

    //float mean[3] = {127.5, 127.5, 127.5};
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    for(int h = 0; h < INPUT_H; h++)
    {
        for(int w = 0; w < INPUT_W; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * INPUT_W + w] = (*img_data * 0.00392156862745098039 - mean[c])/std[c];
                img_data++;
            }
        }
    }
}

bool compare_float(const float& elem1, const float& elem2){
    return elem1 > elem2;
}

void softmax(float *x, int row, int column)
{
    for (int j = 0; j < row; ++j)
    {
        float max = 0.0;
        float sum = 0.0;
        for (int k = 0; k < column; ++k)
            if (max < x[k + j*column])
                max = x[k + j*column];
        for (int k = 0; k < column; ++k)
        {
            x[k + j*column] = exp(x[k + j*column] - max);    // prevent data overflow
            sum += x[k + j*column];
        }
        for (int k = 0; k < column; ++k) x[k + j*column] /= sum;
    }
}

int main(int argc, char** argv)
{
    // ./resnet -s batchsize /image_root_path/
    if (argc != 6) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./classifier resnet50 -s run_type batchsize /image_root_path/  // serialize model to plan file" << std::endl;
        std::cerr << "./classifier resnet50 -d run_type batchsize /image_root_path/  // deserialize plan file and run inference" << std::endl;
        std::cerr << "./classifier resnet50 -top run_type batchsize /image_root_path/  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    for (size_t i = 0; i < argc; i++)
    {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    model_name = std::string(argv[1]);
    std::string phase = std::string(argv[2]);
    std::string run_type = std::string(argv[3]);
    int BATCH_SIZE = atoi(argv[4]);
    data_root_path = std::string(argv[5]);

    if (phase == "-s") {
        IHostMemory* modelStream{nullptr};
        
        APIToModel(BATCH_SIZE, run_type, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(model_name+".engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (phase == "-d") {
        std::ifstream file(model_name + ".engine", std::ios::binary);
        preprocess_device = "GPU";
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else if (phase == "-top") {
        std::ifstream file(model_name + ".engine", std::ios::binary);
        preprocess_device = "CPU";
        test_classify_top = true;
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    //common
    
    std::map<string, string> val_label_map;
    std::vector<std::string> predict_right_vector_top1;
    std::vector<std::string> predict_error_vector_top1;
    std::vector<std::string> predict_right_vector_top5;
    std::vector<std::string> predict_error_vector_top5;
    GetRealLabelMap(data_root_path + "/"+"ILSVRC2012_val.txt", val_label_map);
    std::vector<std::string> file_names;
    if (read_files_in_dir(std::string(data_root_path + "/"+"images_resnet50_ILSVRC2012_val").c_str(), file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    int fcount = 0;
    float prob[OUTPUT_SIZE * BATCH_SIZE] = {0};
	float prob_bak[OUTPUT_SIZE * BATCH_SIZE] = {0};
    int batch_count=0;
    int warmup_batch=(int)(256/BATCH_SIZE);
    long long time_calc[2]={0};
    int coutNum=0;

    if(preprocess_device == "CPU")
	{
	    int img_size = BATCH_SIZE * INPUT_H * INPUT_W * 3;
	    float data[img_size];
	    float data_signel[INPUT_H * INPUT_W * 3];
        std::string file_name_current;
	    for(int f = 0; f < (int)file_names.size(); f++){
	        fcount++;
            if(f % 1000 == 0)
                std::cout << file_names[f] << "  " << f << std::endl;
            // 1. prepare image
	        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size())
                continue;
            for (int b = 0; b < fcount; b++) { 
	            cv::Mat img = cv::imread(data_root_path + "/images_resnet50_ILSVRC2012_val/" + file_names[f - fcount + 1 + b]);
                //cv::Mat img = cv::imread("images/ILSVRC2012_val_00000018.JPEG");
                file_name_current = file_names[f - fcount + 1 + b];
	            if (img.empty()) continue;
	            cv::Mat pr_img;
                //resize 等比例缩放后结果不对，因为训练的时候就是直接resize的
	            cv::resize(img, pr_img, cv::Size(224, 224));
	            ImagePreprocess(pr_img, data_signel);
	            for(int i = 0; i < 224*224*3; i++)
	            {
	                data[b*224*224*3+i] = data_signel[i];
	            }
	        }
	        // 2. inference
	        // TimerClock TC;
	        // TC.update();
	        doInferencePreprocessCpu(*context, data, prob, BATCH_SIZE);
	        // std::cout << "     [ETA] "<< TC.getTimerMicroSec() << "us" << std::endl;
            fcount = 0;
            if(BATCH_SIZE > 1 || test_classify_top == false)    //only batch_size == 1 calculate top1 or top5
            {
                continue;
            }

            // 3. calculator top
            softmax(prob, 1, OUTPUT_SIZE);
            memcpy(prob_bak, prob, OUTPUT_SIZE*4);
            // top1
            int prd_class_id = max_element(prob,prob+1000) - prob;
            int gt_class_id = atoi(val_label_map[file_name_current].c_str());
            if(gt_class_id == prd_class_id)
                predict_right_vector_top1.push_back(file_name_current);
            else
                predict_error_vector_top1.push_back(file_name_current);

            // top5
            sort(prob, prob+OUTPUT_SIZE, compare_float);
            std::vector<int> top5_id_vector;
            for(int i = 0; i < 1000; i++)
            {
                if(prob_bak[i] == prob[0] || prob_bak[i] == prob[1] || prob_bak[i] == prob[2] || 
                    prob_bak[i] == prob[3] || prob_bak[i] == prob[4])
                    top5_id_vector.push_back(i);
            }
            if(gt_class_id == top5_id_vector[0] || gt_class_id == top5_id_vector[1] ||
                gt_class_id == top5_id_vector[2] || gt_class_id == top5_id_vector[3] || 
                gt_class_id == top5_id_vector[4])
            {
                predict_right_vector_top5.push_back(file_name_current);
            }
            else
            {
                predict_error_vector_top5.push_back(file_name_current);
            }
            top5_id_vector.clear();

	    }
        // Release stream and buffers
        context->destroy();
        engine->destroy();
        runtime->destroy();
        std::cout << "right_size_cpu_top1:" << predict_right_vector_top1.size()
                  << "right_size_cpu_top5:" << predict_right_vector_top5.size() << std::endl;
        std::cout << "error_size_cpu_top1:" << predict_error_vector_top1.size()
                  << "error_size_cpu_top5:" << predict_error_vector_top5.size() << std::endl;
        //return 0;
    }
	else if(preprocess_device == "GPU")
	{
		void* buffers[2];
	    // In order to bind the buffers, we need to know the names of the input and output tensors.
	    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
	    // Create GPU buffers on device
	    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
	    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
	    // Create stream
	    cudaStream_t stream;
	    CUDA_CHECK(cudaStreamCreate(&stream));
	    uint8_t* img_host = nullptr;
	    uint8_t* img_device = nullptr;
	    // prepare input data cache in pinned memory 
	    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
	    // prepare input data cache in device memory
	    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

	    std::vector<long long> TimeCalculate;
        std::string file_name_current;
	    for(int f = 0; f < (int)file_names.size(); f++){
	        fcount++;
            if(f % 1000 == 0)
                std::cout << file_names[f] << "  " << f << std::endl;
	        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size())
                continue;

            for (int b = 0; b < fcount; b++) {

	            cv::Mat img = cv::imread(data_root_path + "/images_resnet50_ILSVRC2012_val/" + file_names[f - fcount + 1 + b]);
                // cv::Mat img = cv::imread("dog_00_640_480_0_rgb.jpg");
                file_name_current = file_names[f - fcount + 1 + b];
	            if (img.empty()) continue;
	            if(img.cols*img.rows > MAX_IMAGE_INPUT_SIZE_THRESH)
	            {
	                cv::resize(img, img, cv::Size(MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT));
	            }
	            //cuda preprocess
	            float* buffer_idx = (float*)buffers[inputIndex];
	            size_t  size_image = img.cols * img.rows * 3;
	            size_t  size_image_dst = INPUT_H * INPUT_W * 3;
	            //copy data to pinned memory
	            memcpy(img_host,img.data,size_image);
	            //copy data to device memory
	            CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
	            TimerClock TC_pre;
	            TC_pre.update();
	            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);     // gpu的预处理方式不正确！！！不能用于计算top  
	            // std::cout << "     [PRE] "<< TC_pre.getTimerMicroSec() << "us" << std::endl;
                if(batch_count > warmup_batch){
                    time_calc[0] += TC_pre.getTimerMicroSec();
                }
	            buffer_idx += size_image_dst;
	        }
            // 2. inference
	        TimerClock TC;
	        TC.update();
	        doInference(*context, stream, buffers, prob, outputIndex, BATCH_SIZE);
	        long long single_inference_time = TC.getTimerMicroSec();
            fcount = 0;
	        // std::cout << "     [ETA] "<< TC.getTimerMicroSec() << "us" << std::endl;
	        // std::cout << "     [ETA] "<< single_inference_time << "us" << std::endl;
            if(batch_count > warmup_batch){
                time_calc[1] += single_inference_time;
                if(coutNum <5){
                    float totalM = 0;
                    float avliable = getGPUMemoryAvailable(totalM);
                    std::cout << " used: " << totalM - avliable
                              << " M, percent: " << 100.0 * (totalM - avliable) / totalM
                              << "%" << std::endl;
                    coutNum++;
                }
            }
            batch_count++;

            if(BATCH_SIZE > 1 || test_classify_top == false)//only batch_size == 1 calculate top1 or top5
            {
                continue;
            }

            // 3. calculator top
            softmax(prob, 1, OUTPUT_SIZE);
            memcpy(prob_bak, prob, OUTPUT_SIZE*4);
            // top1
            int prd_class_id = max_element(prob,prob+1000) - prob;
            int gt_class_id = atoi(val_label_map[file_name_current].c_str());
            if(gt_class_id == prd_class_id)
                predict_right_vector_top1.push_back(file_name_current);
            else
                predict_error_vector_top1.push_back(file_name_current);

            // top5
            sort(prob, prob+OUTPUT_SIZE, compare_float);
            std::vector<int> top5_id_vector;
            for(int i = 0; i < 1000; i++)
            {
                if(prob_bak[i] == prob[0] || prob_bak[i] == prob[1] || prob_bak[i] == prob[2] || 
                    prob_bak[i] == prob[3] || prob_bak[i] == prob[4])
                    top5_id_vector.push_back(i);        // get top5 index
            }
            if(gt_class_id == top5_id_vector[0] || gt_class_id == top5_id_vector[1] ||
                gt_class_id == top5_id_vector[2] || gt_class_id == top5_id_vector[3] || 
                gt_class_id == top5_id_vector[4])
            {
                predict_right_vector_top5.push_back(file_name_current);
            }
            else
            {
                predict_error_vector_top5.push_back(file_name_current);
            }
            top5_id_vector.clear();

        }
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        std::cout<< " [PRE] " << time_calc[0] / (batch_count -warmup_batch) / 1000.0 << " ms" <<std::endl;
        double esclipe_time_par_batch = time_calc[1] / (batch_count - warmup_batch) / 1000.0;
        std::cout << "[ batchsize ] " << BATCH_SIZE << std::endl
                  << "[doInference] " << esclipe_time_par_batch << " ms" << std::endl
                  << "[ fps ] " << 1000.0 / esclipe_time_par_batch << std::endl
                  << "[ throughtout ] " << BATCH_SIZE * 1000.0 / esclipe_time_par_batch << std::endl
                  << std::endl;
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));

        context->destroy();
        engine->destroy();
        runtime->destroy();

        std::cout << "right_size_top1:" << predict_right_vector_top1.size() << "right_size_top5:" << predict_right_vector_top5.size() << std::endl;
        std::cout << "error_size_top1:" << predict_error_vector_top1.size() << "error_size_top5:" << predict_error_vector_top5.size()<< std::endl;

	}
	else
	{
        std::cout << "do nothing...." << std::endl;
	}

    float top1 = 1.0*predict_right_vector_top1.size()/(predict_right_vector_top1.size()+predict_error_vector_top1.size());
    printf("top1:%f\n", top1);
    float top5 = 1.0*predict_right_vector_top5.size()/(predict_right_vector_top5.size()+predict_error_vector_top5.size());
    printf("top5:%f\n", top5);

    return 0;
}
