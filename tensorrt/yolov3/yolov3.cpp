#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "NvInfer.h"
#include "utils.h"
#include "logging.h"
#include "yololayer.h"
#include "calibrator.h"
#include "preprocess.h"
#include "common.hpp"
#include "log_time.hpp"
#include "cuda_runtime_api.h"

#include <cuda.h>
#include"monitor.h"

//for map debug
#include "data_reader.hpp"
#include "eval_model.hpp"

// common 
#define USE_INT8   //  set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
// #define BATCH_SIZE 1
#define MAX_IMAGE_WIDTH 1920
#define MAX_IMAGE_HEIGHT 1080
//test interfence run time nms=0.45 bbox=0.25 test map nms=0.65 bbox=0.001
static float NMS_THRESH = 0.65;
static float CONF_THRESH = 0.001;
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
const char* INPUT_BLOB_NAME = "data";
static Logger gLogger;

//YOLOLAYER_OUTPUT_NUMBER=1
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  
const char* OUTPUT_BLOB_NAME = "prob";

//YOLOLAYER_OUTPUT_NUMBER=2
#define KEEP_TOPK 200
const char *OUTPUT_COUNTS = "count";
const char *OUTPUT_BOXES = "box";
const char *OUTPUT_SCORES = "score";
const char *OUTPUT_CLASSES = "class";
const int MAX_IMAGE_INPUT_SIZE_THRESH = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT; // ensure it exceed the maximum size in the input images !

std::string data_root_path = "";
std::string run_type = "USE_FP16";  // option USE_INT8 USE_FP16

using namespace nvinfer1;
using namespace Tn;

cv::Rect get_rect(int width, int height, float *bbox)
{
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (width * 1.0);
    float r_h = Yolo::INPUT_H / (height * 1.0);
    if (r_h > r_w)
    {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (Yolo::INPUT_H - r_w * height) / 2;
        b = bbox[3] - (Yolo::INPUT_H - r_w * height) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else
    {
        l = bbox[0] - (Yolo::INPUT_W - r_h * width) / 2;
        r = bbox[2] - (Yolo::INPUT_W - r_h * width) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

cv::Rect get_rect_signel_yolo_output(int width , int height, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (width * 1.0);
    float r_h = Yolo::INPUT_H / (height * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (Yolo::INPUT_H - r_w * height) / 2;
        b = bbox[1] + bbox[3]/2.f - (Yolo::INPUT_H - r_w * height) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (Yolo::INPUT_W - r_h * width) / 2;
        r = bbox[0] + bbox[2]/2.f - (Yolo::INPUT_W - r_h * width) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh, float bbox_conf_thresh) {
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + 7 * i + 4]*output[1+7*i+4+2] <= bbox_conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string rType) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("./yolov3.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    // printf("start add layer0\n");
    // Yeah I am stupid, I just want to expand the complete arch of darknet..
    auto lr0 = convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
    auto lr1 = convBnLeaky(network, weightMap, *lr0->getOutput(0), 64, 3, 2, 1, 1);
    auto lr2 = convBnLeaky(network, weightMap, *lr1->getOutput(0), 32, 1, 1, 0, 2);
    auto lr3 = convBnLeaky(network, weightMap, *lr2->getOutput(0), 64, 3, 1, 1, 3);
    // printf("start add layer\n");
    auto ew4 = network->addElementWise(*lr3->getOutput(0), *lr1->getOutput(0), ElementWiseOperation::kSUM);
    auto lr5 = convBnLeaky(network, weightMap, *ew4->getOutput(0), 128, 3, 2, 1, 5);
    auto lr6 = convBnLeaky(network, weightMap, *lr5->getOutput(0), 64, 1, 1, 0, 6);
    auto lr7 = convBnLeaky(network, weightMap, *lr6->getOutput(0), 128, 3, 1, 1, 7);
    auto ew8 = network->addElementWise(*lr7->getOutput(0), *lr5->getOutput(0), ElementWiseOperation::kSUM);
    auto lr9 = convBnLeaky(network, weightMap, *ew8->getOutput(0), 64, 1, 1, 0, 9);
    auto lr10 = convBnLeaky(network, weightMap, *lr9->getOutput(0), 128, 3, 1, 1, 10);
    auto ew11 = network->addElementWise(*lr10->getOutput(0), *ew8->getOutput(0), ElementWiseOperation::kSUM);
    auto lr12 = convBnLeaky(network, weightMap, *ew11->getOutput(0), 256, 3, 2, 1, 12);
    auto lr13 = convBnLeaky(network, weightMap, *lr12->getOutput(0), 128, 1, 1, 0, 13);
    auto lr14 = convBnLeaky(network, weightMap, *lr13->getOutput(0), 256, 3, 1, 1, 14);
    auto ew15 = network->addElementWise(*lr14->getOutput(0), *lr12->getOutput(0), ElementWiseOperation::kSUM);
    auto lr16 = convBnLeaky(network, weightMap, *ew15->getOutput(0), 128, 1, 1, 0, 16);
    auto lr17 = convBnLeaky(network, weightMap, *lr16->getOutput(0), 256, 3, 1, 1, 17);
    auto ew18 = network->addElementWise(*lr17->getOutput(0), *ew15->getOutput(0), ElementWiseOperation::kSUM);
    auto lr19 = convBnLeaky(network, weightMap, *ew18->getOutput(0), 128, 1, 1, 0, 19);
    auto lr20 = convBnLeaky(network, weightMap, *lr19->getOutput(0), 256, 3, 1, 1, 20);
    auto ew21 = network->addElementWise(*lr20->getOutput(0), *ew18->getOutput(0), ElementWiseOperation::kSUM);
    auto lr22 = convBnLeaky(network, weightMap, *ew21->getOutput(0), 128, 1, 1, 0, 22);
    auto lr23 = convBnLeaky(network, weightMap, *lr22->getOutput(0), 256, 3, 1, 1, 23);
    auto ew24 = network->addElementWise(*lr23->getOutput(0), *ew21->getOutput(0), ElementWiseOperation::kSUM);
    auto lr25 = convBnLeaky(network, weightMap, *ew24->getOutput(0), 128, 1, 1, 0, 25);
    auto lr26 = convBnLeaky(network, weightMap, *lr25->getOutput(0), 256, 3, 1, 1, 26);
    auto ew27 = network->addElementWise(*lr26->getOutput(0), *ew24->getOutput(0), ElementWiseOperation::kSUM);
    auto lr28 = convBnLeaky(network, weightMap, *ew27->getOutput(0), 128, 1, 1, 0, 28);
    auto lr29 = convBnLeaky(network, weightMap, *lr28->getOutput(0), 256, 3, 1, 1, 29);
    auto ew30 = network->addElementWise(*lr29->getOutput(0), *ew27->getOutput(0), ElementWiseOperation::kSUM);
    auto lr31 = convBnLeaky(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
    auto lr32 = convBnLeaky(network, weightMap, *lr31->getOutput(0), 256, 3, 1, 1, 32);
    auto ew33 = network->addElementWise(*lr32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
    auto lr34 = convBnLeaky(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
    auto lr35 = convBnLeaky(network, weightMap, *lr34->getOutput(0), 256, 3, 1, 1, 35);
    auto ew36 = network->addElementWise(*lr35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
    auto lr37 = convBnLeaky(network, weightMap, *ew36->getOutput(0), 512, 3, 2, 1, 37);
    auto lr38 = convBnLeaky(network, weightMap, *lr37->getOutput(0), 256, 1, 1, 0, 38);
    auto lr39 = convBnLeaky(network, weightMap, *lr38->getOutput(0), 512, 3, 1, 1, 39);
    auto ew40 = network->addElementWise(*lr39->getOutput(0), *lr37->getOutput(0), ElementWiseOperation::kSUM);
    auto lr41 = convBnLeaky(network, weightMap, *ew40->getOutput(0), 256, 1, 1, 0, 41);
    auto lr42 = convBnLeaky(network, weightMap, *lr41->getOutput(0), 512, 3, 1, 1, 42);
    auto ew43 = network->addElementWise(*lr42->getOutput(0), *ew40->getOutput(0), ElementWiseOperation::kSUM);
    auto lr44 = convBnLeaky(network, weightMap, *ew43->getOutput(0), 256, 1, 1, 0, 44);
    auto lr45 = convBnLeaky(network, weightMap, *lr44->getOutput(0), 512, 3, 1, 1, 45);
    auto ew46 = network->addElementWise(*lr45->getOutput(0), *ew43->getOutput(0), ElementWiseOperation::kSUM);
    auto lr47 = convBnLeaky(network, weightMap, *ew46->getOutput(0), 256, 1, 1, 0, 47);
    auto lr48 = convBnLeaky(network, weightMap, *lr47->getOutput(0), 512, 3, 1, 1, 48);
    auto ew49 = network->addElementWise(*lr48->getOutput(0), *ew46->getOutput(0), ElementWiseOperation::kSUM);
    auto lr50 = convBnLeaky(network, weightMap, *ew49->getOutput(0), 256, 1, 1, 0, 50);
    auto lr51 = convBnLeaky(network, weightMap, *lr50->getOutput(0), 512, 3, 1, 1, 51);
    auto ew52 = network->addElementWise(*lr51->getOutput(0), *ew49->getOutput(0), ElementWiseOperation::kSUM);
    auto lr53 = convBnLeaky(network, weightMap, *ew52->getOutput(0), 256, 1, 1, 0, 53);
    auto lr54 = convBnLeaky(network, weightMap, *lr53->getOutput(0), 512, 3, 1, 1, 54);
    auto ew55 = network->addElementWise(*lr54->getOutput(0), *ew52->getOutput(0), ElementWiseOperation::kSUM);
    auto lr56 = convBnLeaky(network, weightMap, *ew55->getOutput(0), 256, 1, 1, 0, 56);
    auto lr57 = convBnLeaky(network, weightMap, *lr56->getOutput(0), 512, 3, 1, 1, 57);
    auto ew58 = network->addElementWise(*lr57->getOutput(0), *ew55->getOutput(0), ElementWiseOperation::kSUM);
    auto lr59 = convBnLeaky(network, weightMap, *ew58->getOutput(0), 256, 1, 1, 0, 59);
    auto lr60 = convBnLeaky(network, weightMap, *lr59->getOutput(0), 512, 3, 1, 1, 60);
    auto ew61 = network->addElementWise(*lr60->getOutput(0), *ew58->getOutput(0), ElementWiseOperation::kSUM);
    auto lr62 = convBnLeaky(network, weightMap, *ew61->getOutput(0), 1024, 3, 2, 1, 62);
    auto lr63 = convBnLeaky(network, weightMap, *lr62->getOutput(0), 512, 1, 1, 0, 63);
    auto lr64 = convBnLeaky(network, weightMap, *lr63->getOutput(0), 1024, 3, 1, 1, 64);
    auto ew65 = network->addElementWise(*lr64->getOutput(0), *lr62->getOutput(0), ElementWiseOperation::kSUM);
    auto lr66 = convBnLeaky(network, weightMap, *ew65->getOutput(0), 512, 1, 1, 0, 66);
    auto lr67 = convBnLeaky(network, weightMap, *lr66->getOutput(0), 1024, 3, 1, 1, 67);
    auto ew68 = network->addElementWise(*lr67->getOutput(0), *ew65->getOutput(0), ElementWiseOperation::kSUM);
    auto lr69 = convBnLeaky(network, weightMap, *ew68->getOutput(0), 512, 1, 1, 0, 69);
    auto lr70 = convBnLeaky(network, weightMap, *lr69->getOutput(0), 1024, 3, 1, 1, 70);
    auto ew71 = network->addElementWise(*lr70->getOutput(0), *ew68->getOutput(0), ElementWiseOperation::kSUM);
    auto lr72 = convBnLeaky(network, weightMap, *ew71->getOutput(0), 512, 1, 1, 0, 72);
    auto lr73 = convBnLeaky(network, weightMap, *lr72->getOutput(0), 1024, 3, 1, 1, 73);
    auto ew74 = network->addElementWise(*lr73->getOutput(0), *ew71->getOutput(0), ElementWiseOperation::kSUM);
    auto lr75 = convBnLeaky(network, weightMap, *ew74->getOutput(0), 512, 1, 1, 0, 75);
    auto lr76 = convBnLeaky(network, weightMap, *lr75->getOutput(0), 1024, 3, 1, 1, 76);
    auto lr77 = convBnLeaky(network, weightMap, *lr76->getOutput(0), 512, 1, 1, 0, 77);
    auto lr78 = convBnLeaky(network, weightMap, *lr77->getOutput(0), 1024, 3, 1, 1, 78);
    auto lr79 = convBnLeaky(network, weightMap, *lr78->getOutput(0), 512, 1, 1, 0, 79);
    auto lr80 = convBnLeaky(network, weightMap, *lr79->getOutput(0), 1024, 3, 1, 1, 80);
    IConvolutionLayer* conv81 = network->addConvolutionNd(*lr80->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.81.Conv2d.weight"], weightMap["module_list.81.Conv2d.bias"]);
    assert(conv81);
    // 82 is yolo
    // printf("create backbone and one head.\n");
    auto l83 = lr79;
    auto lr84 = convBnLeaky(network, weightMap, *l83->getOutput(0), 256, 1, 1, 0, 84);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts85{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv85 = network->addDeconvolutionNd(*lr84->getOutput(0), 256, DimsHW{2, 2}, deconvwts85, emptywts);
    assert(deconv85);
    deconv85->setStrideNd(DimsHW{2, 2});
    deconv85->setNbGroups(256);
    weightMap["deconv85"] = deconvwts85;

    ITensor* inputTensors[] = {deconv85->getOutput(0), ew61->getOutput(0)};
    auto cat86 = network->addConcatenation(inputTensors, 2);
    auto lr87 = convBnLeaky(network, weightMap, *cat86->getOutput(0), 256, 1, 1, 0, 87);
    auto lr88 = convBnLeaky(network, weightMap, *lr87->getOutput(0), 512, 3, 1, 1, 88);
    auto lr89 = convBnLeaky(network, weightMap, *lr88->getOutput(0), 256, 1, 1, 0, 89);
    auto lr90 = convBnLeaky(network, weightMap, *lr89->getOutput(0), 512, 3, 1, 1, 90);
    auto lr91 = convBnLeaky(network, weightMap, *lr90->getOutput(0), 256, 1, 1, 0, 91);
    auto lr92 = convBnLeaky(network, weightMap, *lr91->getOutput(0), 512, 3, 1, 1, 92);
    IConvolutionLayer* conv93 = network->addConvolutionNd(*lr92->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.93.Conv2d.weight"], weightMap["module_list.93.Conv2d.bias"]);
    assert(conv93);
    // 94 is yolo

    auto l95 = lr91;
    auto lr96 = convBnLeaky(network, weightMap, *l95->getOutput(0), 128, 1, 1, 0, 96);
    Weights deconvwts97{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv97 = network->addDeconvolutionNd(*lr96->getOutput(0), 128, DimsHW{2, 2}, deconvwts97, emptywts);
    assert(deconv97);
    deconv97->setStrideNd(DimsHW{2, 2});
    deconv97->setNbGroups(128);
    ITensor* inputTensors1[] = {deconv97->getOutput(0), ew36->getOutput(0)};
    auto cat98 = network->addConcatenation(inputTensors1, 2);
    auto lr99 = convBnLeaky(network, weightMap, *cat98->getOutput(0), 128, 1, 1, 0, 99);
    auto lr100 = convBnLeaky(network, weightMap, *lr99->getOutput(0), 256, 3, 1, 1, 100);
    auto lr101 = convBnLeaky(network, weightMap, *lr100->getOutput(0), 128, 1, 1, 0, 101);
    auto lr102 = convBnLeaky(network, weightMap, *lr101->getOutput(0), 256, 3, 1, 1, 102);
    auto lr103 = convBnLeaky(network, weightMap, *lr102->getOutput(0), 128, 1, 1, 0, 103);
    auto lr104 = convBnLeaky(network, weightMap, *lr103->getOutput(0), 256, 3, 1, 1, 104);
    IConvolutionLayer* conv105 = network->addConvolutionNd(*lr104->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.105.Conv2d.weight"], weightMap["module_list.105.Conv2d.bias"]);
    assert(conv105);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");         // API 调用自定义插件方法
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {conv81->getOutput(0), conv93->getOutput(0), conv105->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    // printf("create addPluginV2. output layer\n");
	if(YOLOLAYER_OUTPUT_NUMBER == 2)
    {
		auto nms = addBatchedNMSLayer(network, yolo, Yolo::CLASS_NUM, Yolo::MAX_OUTPUT_BBOX_COUNT, KEEP_TOPK, CONF_THRESH, NMS_THRESH);

	    nms->getOutput(0)->setName(OUTPUT_COUNTS);
	    network->markOutput(*nms->getOutput(0));

	    nms->getOutput(1)->setName(OUTPUT_BOXES);
	    network->markOutput(*nms->getOutput(1));

	    nms->getOutput(2)->setName(OUTPUT_SCORES);
	    network->markOutput(*nms->getOutput(2));

	    nms->getOutput(3)->setName(OUTPUT_CLASSES);
	    network->markOutput(*nms->getOutput(3));
	}
	else if(YOLOLAYER_OUTPUT_NUMBER == 1)
	{
		yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    	network->markOutput(*yolo->getOutput(0));
	}
	else
	{
		yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    	network->markOutput(*yolo->getOutput(0));
	}

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1ULL << 30/* 16 * (1 << 20) */);  // 16MB
    if(rType=="USE_FP16"){
        config->setFlag(BuilderFlag::kFP16);
    }
    else if(rType=="USE_INT8"){
        std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        std::string table_name = "";
        if(YOLOLAYER_OUTPUT_NUMBER == 1){
            table_name = "yolo_output_1_int8calib.table";
        }else if(YOLOLAYER_OUTPUT_NUMBER == 2){
            table_name = "yolo_output_2_int8calib.table";
        }
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(maxBatchSize , INPUT_W, INPUT_H,
                                                                        std::string(data_root_path + "/" + "coco_calib/").c_str(), table_name.c_str(), INPUT_BLOB_NAME);
        config->setInt8Calibrator(calibrator);
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string rType) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, rType);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

float getGPUMemoryAvailable(float &totalM)
{
	size_t avaiavle = 0;
	size_t total = 0;
	cuMemGetInfo(&avaiavle, &total);
    totalM = 1.0 * total / 1024 / 1024;
    return 1.0 * avaiavle / 1024 / 1024;
}


void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, int *counts, float *boxes, float *scores, float *classes, int inputIndex, int countIndex, int bboxIndex, int scoreIndex, int classIndex, int batchSize)
{
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(counts, buffers[countIndex], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(boxes, buffers[bboxIndex], batchSize * KEEP_TOPK * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(scores, buffers[scoreIndex], batchSize * KEEP_TOPK * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(classes, buffers[classIndex], batchSize * KEEP_TOPK * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
}

void doInferenceYoloSingleOutput(IExecutionContext& context,cudaStream_t &stream,  void **buffers, int inputIndex, int outputIndex,float* input, float* output, int batchSize) {
    // const ICudaEngine& engine = context.getEngine();
    // // Pointers to input and output device buffers to pass to engine.
    // // Engine requires exactly IEngine::getNbBindings() number of buffers.
    // assert(engine.getNbBindings() == 2);
    // void* buffers[2];

    // // In order to bind the buffers, we need to know the names of the input and output tensors.
    // // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    // const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // // Create GPU buffers on device
    // CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // // Create stream
    // cudaStream_t stream;
    // CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    // CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // // Release stream and buffers
    // cudaStreamDestroy(stream);
    // CUDA_CHECK(cudaFree(buffers[inputIndex]));
    // CUDA_CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {

    if(argc !=5)
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov3 -s run_type bs /path // serialize model to plan file" << std::endl;
        std::cerr << "./yolov3 -d  run_type bs /path // deserialize plan file and run inference" << std::endl;
        std::cerr << "./yolov3 -map run_type bs /path // test_coco_val2017 map" << std::endl;
        return -1;
    }
    for (size_t i = 0; i < argc; i++)
    {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    cudaSetDevice(DEVICE);
	if(YOLOLAYER_OUTPUT_NUMBER == 2)
		initLibNvInferPlugins(&gLogger, "");

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    bool test_map = false;

    run_type = std::string(argv[2]);
    int BATCH_SIZE = atoi(argv[3]);
    data_root_path = std::string(argv[4]);

    printf("\n\nYOLOLAYER_OUTPUT_NUMBER:%d\n", YOLOLAYER_OUTPUT_NUMBER);
    printf("BATCH_SIZE:%d\n", BATCH_SIZE);

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream, run_type);
        assert(modelStream != nullptr);
        std::ofstream p("yolov3.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("yolov3.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else if (std::string(argv[1]) == "-map") {
        test_map = true;
        // YOLOLAYER_OUTPUT_NUMBER = 1;
        NMS_THRESH = 0.65;
        CONF_THRESH = 0.001;
        std::ifstream file("yolov3.engine", std::ios::binary);
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
    std::cout<< "load engine done\n";

    //labels.txt test map
    string listFile = data_root_path + "/" + "val2017_gt_labels.txt";
    list<string> fileNames;
    list<vector<Bbox>> groundTruth;
    if(listFile.length() > 0)
    {
        std::cout << "loading from eval list " << listFile << std::endl; 
        tie(fileNames, groundTruth) = readObjectLabelFileList(listFile);
    }
    std::cout << "fileNames.size():" << fileNames.size()
              << "groundTruth.size():" << groundTruth.size();
    assert(fileNames.size()==groundTruth.size());
    list<vector<Bbox>> outputs;
    int classNum = Yolo::CLASS_NUM;//
    int fcount = 0;
    std::cout << "YOLOLAYER_OUTPUT_NUMBER = "<< YOLOLAYER_OUTPUT_NUMBER<< std::endl; 
    int gpuMemNum=0;

	if(YOLOLAYER_OUTPUT_NUMBER == 2)    // 测试未成功
    {
        
        // prepare input data ---------------------------
        float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        int counts[BATCH_SIZE];
        float boxes[BATCH_SIZE * KEEP_TOPK * 4];
        float scores[BATCH_SIZE * KEEP_TOPK];
        float classes[BATCH_SIZE * KEEP_TOPK];
    
	    IRuntime* runtime = createInferRuntime(gLogger);
	    assert(runtime != nullptr);
	    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	    assert(engine != nullptr);
	    IExecutionContext* context = engine->createExecutionContext();
	    assert(context != nullptr);
	    delete[] trtModelStream;
	
		assert(engine->getNbBindings() == 5);
	    void *buffers[5];

	    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	    const int countIndex = engine->getBindingIndex(OUTPUT_COUNTS);
	    const int bboxIndex = engine->getBindingIndex(OUTPUT_BOXES);
	    const int scoreIndex = engine->getBindingIndex(OUTPUT_SCORES);
	    const int classIndex = engine->getBindingIndex(OUTPUT_CLASSES);

	    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
	    CUDA_CHECK(cudaMalloc(&buffers[countIndex], BATCH_SIZE * sizeof(int)));
	    CUDA_CHECK(cudaMalloc(&buffers[bboxIndex], BATCH_SIZE * KEEP_TOPK * 4 * sizeof(float)));
	    CUDA_CHECK(cudaMalloc(&buffers[scoreIndex], BATCH_SIZE * KEEP_TOPK * sizeof(float)));
	    CUDA_CHECK(cudaMalloc(&buffers[classIndex], BATCH_SIZE * KEEP_TOPK * sizeof(float)));

	    // Create stream
	    cudaStream_t stream;
	    CUDA_CHECK(cudaStreamCreate(&stream));
	    uint8_t* img_host = nullptr;
	    void* img_device = nullptr;
	    // prepare input data cache in pinned memory 
	    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
	    // img_host = (uint8_t*)malloc(MAX_IMAGE_INPUT_SIZE_THRESH * 3);
        // if(img_host==nullptr)
        //     exit(-1);
        // prepare input data cache in device memory
	    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
		int width_origin = 0;
	    int height_origin = 0;
	    auto iter = fileNames.begin();

        // static time
        int countBatch = 0;
        long long calTime[3] = {0};
        int warmup_batch = (int)(256 / BATCH_SIZE);
        std::cout << "fileNames.size(): "<< fileNames.size() << std::endl;
        
	    for (unsigned int i = 0;i < fileNames.size() ; ++i /*, ++iter */)
	    {
	        fcount++;
            if(i % 500 == 0)
                std::cout << *iter << "  " << i << std::endl;
	        if (fcount < BATCH_SIZE && i + 1 != (int)fileNames.size())
	            continue;

            int nbatch=0;
            //cuda preprocess
	        float* buffer_idx = (float*)buffers[inputIndex];

	        for (int b = 0; b < fcount; b++)
	        {
                const string &filename = data_root_path + "/" + *iter;
                if(filename.find(".jpg") == filename.npos){
                    continue;
                }
                // const string& filename = "dog_00_640_480_0_rgb.jpg";  //*iter;

                TimerClock TC_loadimg;
	            TC_loadimg.update();
	            cv::Mat img = cv::imread(filename);
                long long esp_time_load = TC_loadimg.getTimerMicroSec();
                if(countBatch>=warmup_batch){
                    calTime[2] += esp_time_load;
                }
	            if (img.empty()) continue;
				
                if(img.cols * img.rows > MAX_IMAGE_INPUT_SIZE_THRESH){
                    cv::resize(img, img, cv::Size(MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT));
                }
	            width_origin = img.cols;
	            height_origin = img.rows;
	            size_t  size_image = img.cols * img.rows * 3;
	            size_t  size_image_dst = INPUT_H * INPUT_W * 3;
	            //copy data to pinned memory
                memcpy(img_host, img.data, size_image);
                //copy data to device memory
	            CUDA_CHECK(cudaMemcpyAsync(&img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
	            // CUDA_CHECK(cudaMemcpyAsync(&buffers[inputIndex],img.data,size_image,cudaMemcpyHostToDevice,stream));
                
                // std::cout << "---------size_image"<<size_image << std::endl;
                // CUDA_CHECK(cudaMemcpy(img_device,img_host,size_image,cudaMemcpyHostToDevice));
                // cudaStreamSynchronize(stream);
	            // TC.update();
	            TimerClock TC_pre;
	            TC_pre.update();
	            preprocess_kernel_img((uint8_t*)img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);  
	            // preprocess_kernel_img((uint8_t*)&buffers[inputIndex], img.cols, img.rows, (float*)(&buffers[inputIndex]), INPUT_W, INPUT_H, stream);  
                long long esp_time = TC_pre.getTimerMicroSec();
	            // std::cout << "     [preprocess] "<< esp_time << "us" << std::endl; 
                if(countBatch>=warmup_batch){
                    calTime[0] += esp_time;
                }
	            buffer_idx += size_image_dst;

                ++iter;
                ++nbatch;
                // ++fcount;
                // i++;
                // if(nbatch==BATCH_SIZE || fcount==(int)fileNames.size()-1)
                // {
                //     break;
                // }
	        }

            if (nbatch != BATCH_SIZE) {
                continue;
            }

	        // Run inference

	        //auto start = std::chrono::system_clock::now();
	        if(nbatch > 1)
	        {
	            TimerClock TC;
	            TC.update();
				doInference(*context, stream, buffers, counts, boxes, scores, classes, inputIndex, countIndex, bboxIndex, scoreIndex, classIndex, nbatch);
	            long long esp_time = TC.getTimerMicroSec();
                // std::cout << "     [forward+nms] "<< TC.getTimerMicroSec() << "us" << std::endl;
                if(countBatch>=warmup_batch){
                    calTime[1] += esp_time;
                    if(gpuMemNum<5){
                        float totalM = 0;
                        float availableM = getGPUMemoryAvailable(totalM);
                        std::cout << "[used GPU memory]:" << totalM - availableM << std::endl;
                        gpuMemNum++;
                    }
                }
	        }
	        else if(nbatch == 1)
	        {
	            TimerClock TC;
	            TC.update();
	            doInference(*context, stream, buffers, counts, boxes, scores, classes, inputIndex, countIndex, bboxIndex, scoreIndex, classIndex, nbatch);
	            long long esp_time = TC.getTimerMicroSec();
                // std::cout << "     [forward+nms] "<< esp_time << "us" << std::endl;
                if(countBatch >= warmup_batch){
                    calTime[1] += esp_time;
                    if(gpuMemNum<5){
                        float totalM = 0;
                        float availableM = getGPUMemoryAvailable(totalM);
                        std::cout << "[used GPU memory]:" << totalM - availableM << std::endl;
                        gpuMemNum++;
                    }
                }

	            std::vector<Yolo::Detection> res;
	            vector<Bbox> boxes_temp;
	            for (int j = 0; j < counts[0]; j++)
	            {
	                float *curBbox = boxes + j * 4;
	                float *curScore = scores + j;
	                float *curClass = classes + j;
	                cv::Rect r = get_rect(width_origin, height_origin, curBbox);
	                Bbox bbox = 
	                { 
	                    int(*curClass),   //classId
	                    r.x,
	                    r.x+r.width,
	                    r.y,
	                    r.y+r.height,
	                    *curScore //score
	                };
	                boxes_temp.push_back(bbox);
	            }
	            outputs.emplace_back(boxes_temp);
	            boxes_temp.clear();
	        }
	        else
	        {
	            // return -1;
                break;
	        }
	        // fcount = 0;
            countBatch++;
	    }
        // printf static time
        std::cout << "nbatch:"<<(countBatch-warmup_batch) <<", BATCH_SIZE:"<< BATCH_SIZE << std::endl;
        std::cout << "avg [loadimg]: " << BATCH_SIZE * calTime[2] / ((countBatch-warmup_batch)*BATCH_SIZE )/ 1000.0 << " ms" << std::endl;
        std::cout << "avg [process]: " << BATCH_SIZE * calTime[0] / ((countBatch-warmup_batch)*BATCH_SIZE )/ 1000.0 << " ms" << std::endl;
        std::cout << " avg [forward+nms] "<< calTime[1] / (countBatch-warmup_batch) / 1000.0<< " ms" << std::endl;

        //释放资源
        

        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[countIndex]));
        CUDA_CHECK(cudaFree(buffers[bboxIndex]));
        CUDA_CHECK(cudaFree(buffers[scoreIndex]));
        CUDA_CHECK(cudaFree(buffers[classIndex])); 

        free(img_host);
        CUDA_CHECK(cudaFree(img_device));
    
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
	}
	else if(YOLOLAYER_OUTPUT_NUMBER == 1)
	{
	    IRuntime* runtime = createInferRuntime(gLogger);
	    assert(runtime != nullptr);
	    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	    assert(engine != nullptr);
	    IExecutionContext* context = engine->createExecutionContext();
	    assert(context != nullptr);
	    delete[] trtModelStream;
        std::cout<< "deserializeCudaEngine engine done\n";
//=========================
        //const ICudaEngine& engine = context.getEngine();
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine->getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
//=============================

        // prepare input data ---------------------------
	    float *data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	    float *prob = new float[BATCH_SIZE * OUTPUT_SIZE];

        int width_origin = 0;
	    int height_origin = 0;
	    auto iter = fileNames.begin();
        // static time
        int countBatch = 0;
        long long calTime = 0;
        int warmup_batch=(int)(256/BATCH_SIZE);
        std::cout << "fileNames.size(): "<< fileNames.size() << std::endl;

	    for (unsigned int i = 0; i < fileNames.size();  ++i /* ,++iter */)
	    {
	        fcount++;
            if(i % 500 == 0)
                std::cout << *iter << "  " << i << std::endl;
	        if (fcount < BATCH_SIZE && i + 1 != (int)fileNames.size())
	            continue;
	        
            int nbatch=0;
	        for (int b = 0; b < fcount; b++)
	        {
                const string& filename  = data_root_path + "/" + *iter;
                if(filename.find(".jpg") == filename.npos){continue;}

	            cv::Mat img = cv::imread(filename);
	            if (img.empty()) continue;
                width_origin  = img.cols;
                height_origin = img.rows;
	            cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);     // letterbox
	            for (int j = 0; j < INPUT_H * INPUT_W; j++) {               // bgr2rgb and div255
	                data[nbatch * 3 * INPUT_H * INPUT_W + j] = pr_img.at<cv::Vec3b>(j)[2] / 255.0;
	                data[nbatch * 3 * INPUT_H * INPUT_W + j + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(j)[1] / 255.0;
	                data[nbatch * 3 * INPUT_H * INPUT_W + j + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(j)[0] / 255.0;
	            }

                ++iter;
                ++nbatch;
	        }
            fcount = 0;
            if(nbatch!=BATCH_SIZE)
                continue;
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], data, nbatch * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
            
            // Run inference
	        //BATCH_SIZE > 1 only test latency BATCH_SIZE == 1 test latency and Map
	        if(nbatch > 1)
	        {
                std::vector<std::vector<Yolo::Detection>> batch_res(nbatch);
	            TimerClock TC;
	            TC.update();
	            doInferenceYoloSingleOutput(*context, stream, buffers, inputIndex, outputIndex, data, prob, nbatch);
                // std::cout << "     [only forward] "<< TC.getTimerMicroSec() << "us" << std::endl;
                if(countBatch > warmup_batch){
                    calTime +=TC.getTimerMicroSec();
                    if(gpuMemNum<5){
                        float totalM = 0;
                        float availableM = getGPUMemoryAvailable(totalM);
                        std::cout << "[used GPU memory]:" << totalM - availableM <<" M"<< std::endl;
                        gpuMemNum++;
                    }
                }
	            for (int b = 0; b < nbatch; b++) 
	            {
	                auto& res = batch_res[b];
                    //do nms on cpu
	                nms(res, &prob[b * OUTPUT_SIZE], NMS_THRESH, CONF_THRESH);
	            }
	            //std::cout << "     zzzzzzzzzz[ETA] "<< TC.getTimerMicroSec() << "us" << std::endl;
	            batch_res.clear();
	        }
	        else if(nbatch == 1)
	        {
                std::vector<Yolo::Detection> res;
	            TimerClock TC;
	            TC.update();
	            doInferenceYoloSingleOutput(*context, stream,buffers, inputIndex, outputIndex, data, prob, nbatch);
                // std::cout << "     [only forward] "<< TC.getTimerMicroSec() << "us" << std::endl;
                if(countBatch > warmup_batch){
                    calTime +=TC.getTimerMicroSec();
                    if(gpuMemNum<5){
                        float totalM = 0;
                        float availableM = getGPUMemoryAvailable(totalM);
                        std::cout << "[used GPU memory]:" << totalM - availableM  << " M "<< std::endl;
                        gpuMemNum++;
                    }
                }

	            //do nms on cpu
	            nms(res, prob, NMS_THRESH, CONF_THRESH);
	            // std::cout << "     [ETA] "<< TC.getTimerMicroSec() << "us" << std::endl;
	            //scale bbox to img 
	            float scale = min(float(INPUT_W)/width_origin,float(INPUT_H)/height_origin);
	            float scaleSize[] = {width_origin * scale,height_origin * scale};
                //save bbox to outputs for test map
	            vector<Bbox> boxes;
	            for(auto& item : res)
	            {
	                cv::Rect r = get_rect_signel_yolo_output(width_origin, height_origin, item.bbox);
	                Bbox bbox = 
	                { 
	                    item.class_id,   //classId
	                    r.x,
	                    r.x+r.width,
	                    r.y,
	                    r.y+r.height,
	                    item.det_confidence*item.det_confidence //score
	                };
	                boxes.push_back(bbox);
	            }
	            outputs.emplace_back(boxes);
	            memset(data, 0, BATCH_SIZE * 3 * INPUT_H * INPUT_W);
	            res.clear();
	            boxes.clear();
	        }
	        else
	        {
	            return -1;
	        }
            countBatch++;
	    }
        // printf static time
        std::cout << "time cal by nbatch:"<<(countBatch-warmup_batch) << std::endl;
        float eclpse_time_inference = calTime / (countBatch - warmup_batch) / 1000.0;
        std::cout << " [ avg forward time ] " << eclpse_time_inference << " ms" << std::endl
                  << "[ fps ] " << 1000.0 / eclpse_time_inference << std::endl
                  << "[ thoughtout ] " << BATCH_SIZE * 1000.0 / eclpse_time_inference << std::endl
                  << std::endl;

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));

        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();

        delete []data;
        delete []prob;

	}
	else
	{
		std::cout << "please set YOLOLAYER_OUTPUT_NUMBER as 1 or 2" << std::endl;
		return -1;
	}
    
    if(BATCH_SIZE > 1)
    {
        printf("do nothing....\n");
    }
    else
    {
        if(test_map == false)
        {
            return -1;
        }
        if(groundTruth.size() > 0)
        {
            //eval map
            evalMAPResult(outputs,groundTruth,classNum,0.5f);
        }
        else
        {
            std::cout << "val2017 groudTruth read have problem...." << std::endl;
        }
    }
    return 0;
}
