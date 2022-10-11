#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <iostream>
#include <vector>
#include "NvInfer.h"
#include "common.hpp"
namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 608;
    static constexpr int INPUT_W = 608;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {116,90,  156,198,  373,326}
    };
    static constexpr YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {30,61,  62,45,  59,119}
    };
    static constexpr YoloKernel yolo3 = {
        INPUT_W / 8,
        INPUT_H / 8,
        {10,13,  16,30,  33,23}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin:  public IPluginV2 //public IPluginV2IOExt ,
    {
        public:
            explicit YoloLayerPlugin();
            YoloLayerPlugin(const void* data, size_t length);

            // ~YoloLayerPlugin();

            // int getNbOutputs() const override    v7.2
            int32_t getNbOutputs() const noexcept override   // v8.2
            {
				if(YOLOLAYER_OUTPUT_NUMBER == 2)
                	return 2;
				else if(YOLOLAYER_OUTPUT_NUMBER == 1)
					return 1;
				else
					return 1;
            }

            // Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept;
            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

            int initialize() noexcept override;

            virtual void terminate() noexcept override{};

            virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override//{ return 0;}
            {
				if(YOLOLAYER_OUTPUT_NUMBER == 2)
                	return maxBatchSize * sizeof(int);
				else if(YOLOLAYER_OUTPUT_NUMBER == 1)
					return 0;                
            }

            virtual int enqueue(int batchSize, const void *const *inputs, void * const* outputs,
                                void *workspace, cudaStream_t stream) noexcept override;

            virtual size_t getSerializationSize() const noexcept override;

            void serialize(void* buffer) const noexcept override;

            /* bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, 
                                        int nbInputs, int nbOutputs) const noexcept override{
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            } */
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
            void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims,
                                     int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept override;

            const char* getPluginType() const noexcept override;

            const char* getPluginVersion() const noexcept;

            void destroy() noexcept override;

            // IPluginV2IOExt* clone() const noexcept override;
            IPluginV2* clone() const noexcept override;

            void setPluginNamespace(const char* pluginNamespace) noexcept override;

            const char* getPluginNamespace() const noexcept override;

            /* DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const noexcept override;

            bool isOutputBroadcastAcrossBatch(int outputIndex,
                                              const bool *inputIsBroadcasted, int nbInputs) const noexcept override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

            void attachToContext(
                cudnnContext *cudnnContext, cublasContext *cublasContext,
                IGpuAllocator *gpuAllocator) noexcept override;

            void configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out,
                                 int nbOutput) noexcept override;

            void detachFromContext() noexcept override; */

        private:
            void forwardGpu(const float *const *inputs, void *const*output, void *workspace,
                            cudaStream_t stream, int batchSize = 1);
            void forwardGpuSingleOutput(const float *const *inputs, float *output,
                                        cudaStream_t stream, int batchSize = 1);
            int mClassCount;
            int mKernelCount;
            std::vector<Yolo::YoloKernel> mYoloKernel;
            int mThreadCount = 256;
            const char* mPluginNamespace;
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            YoloPluginCreator();

            ~YoloPluginCreator() noexcept = default;

            const char* getPluginName() const noexcept;

            const char* getPluginVersion() const noexcept;

            const PluginFieldCollection* getFieldNames() noexcept override;

            IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

            IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

            void setPluginNamespace(const char* libNamespace) noexcept
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const noexcept
            {
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
}

#endif 
