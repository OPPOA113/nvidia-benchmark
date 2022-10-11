#include "yololayer.h"
#include "utils.h"
#include <assert.h>

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin()
    {
        mClassCount = CLASS_NUM;
        mYoloKernel.clear();
        mYoloKernel.push_back(yolo1);
        mYoloKernel.push_back(yolo2);
        mYoloKernel.push_back(yolo3);

        mKernelCount = mYoloKernel.size();
    }
    
    /* YoloLayerPlugin::~YoloLayerPlugin()
    {
    } */

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(mYoloKernel.data(),d,kernelSize);
        d += kernelSize;

        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void* buffer) const noexcept
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(d,mYoloKernel.data(),kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }
    bool YoloLayerPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
	{
		return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
	}
    void YoloLayerPlugin::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, 
                            int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept
	{

	}
    
    size_t YoloLayerPlugin::getSerializationSize() const noexcept
    {  
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount)  + sizeof(Yolo::YoloKernel) * mYoloKernel.size();
    }

    int YoloLayerPlugin::initialize() noexcept
    { 
        return 0;
    }
    
    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
    {
        //output the result to channel
        if(YOLOLAYER_OUTPUT_NUMBER == 2)
        {
            assert(index < 2);
            //output the result to channel
            if (index == 0)
            {
                return Dims3(MAX_OUTPUT_BBOX_COUNT, 1, 4);
            }
            return DimsHW(MAX_OUTPUT_BBOX_COUNT, mClassCount);
        }
        else if(YOLOLAYER_OUTPUT_NUMBER == 1)
        {
            int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);
            return Dims3(totalsize + 1, 1, 1);
        }
        else
        {
            int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);
            return Dims3(totalsize + 1, 1, 1);
        }
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const noexcept
    {
        return mPluginNamespace;
    }

   /*  // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)noexcept
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() noexcept{} */

    const char* YoloLayerPlugin::getPluginType() const noexcept
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const noexcept
    {
        return "1";
    }

    void YoloLayerPlugin::destroy() noexcept
    {
        delete this;
    }

    // Clone the plugin
    // IPluginV2IOExt* YoloLayerPlugin::clone() const noexcept
    IPluginV2* YoloLayerPlugin::clone() const noexcept
    {
        YoloLayerPlugin *p = new YoloLayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

	__device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };		

	__global__ void CalDetectionYoloSingleOuput(const float *input, float *output,int noElements, 
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes,int outputElem) 
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;

            float *res_count = output + bnIdx*outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= MAX_OUTPUT_BBOX_COUNT) return;
            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
            det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
            det->bbox[2] = expf(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            det->bbox[3] = expf(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }
    
    __global__ void CalDetection(const float *input, float *bboxData, float *scoreData, int *countData, int noElements, 
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;
        //printf("idx_1:%d noElements:%d\n", idx, noElements);
        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        //printf("idx:%d total_grid:%d bnIdx:%d info_len_i:%d\n", idx, total_grid, bnIdx, info_len_i);
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);
        for (int k = 0; k < 3; ++k) {
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH) continue;
            int *res_count = countData + bnIdx;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= MAX_OUTPUT_BBOX_COUNT) return;

            float *curBbox = bboxData + bnIdx * MAX_OUTPUT_BBOX_COUNT * 4 + count * 4;
            float *curScore = scoreData + bnIdx * MAX_OUTPUT_BBOX_COUNT * classes + count * classes;

            //float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            //if (box_prob < IGNORE_THRESH) continue;
            for (int i = 5; i < info_len_i; ++i)
            {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                curScore[i - 5] = p * box_prob;
            }

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            float cx = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
            float cy = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;

            float w = expf(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            float h = expf(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];

            // float w = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            // w = w * w * anchors[2 * k];
            // float h = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            // h = h * h * anchors[2 * k + 1];
            // cx,cy,w,h to x1,y1,x2,y2
            curBbox[0] = cx - 0.5 * w;
            curBbox[1] = cy - 0.5 * h;
            curBbox[2] = cx + 0.5 * w;
            curBbox[3] = cy + 0.5 * h;
        }
    }

    void YoloLayerPlugin::forwardGpu(const float *const * inputs, void*const* output, void* workspace, cudaStream_t stream, int batchSize) {
        void* devAnchor;
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));
        float *bboxData = (float *)output[0];
        float *scoreData = (float *)output[1];
        int *countData = (int *)workspace;
        CUDA_CHECK(cudaMemset(countData, 0, sizeof(int) * batchSize));
        CUDA_CHECK(cudaMemset(bboxData, 0, sizeof(float) * MAX_OUTPUT_BBOX_COUNT * 4 * batchSize));
        CUDA_CHECK(cudaMemset(scoreData, 0, sizeof(float) * MAX_OUTPUT_BBOX_COUNT * mClassCount * batchSize));
        int numElem = 0;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            CUDA_CHECK(cudaMemcpy(devAnchor, yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
            CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],bboxData, scoreData, countData, numElem, yolo.width, yolo.height, (float *)devAnchor, mClassCount);
        }
        CUDA_CHECK(cudaFree(devAnchor));
    }
    
    void YoloLayerPlugin::forwardGpuSingleOutput(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        void* devAnchor;
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));

        int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        for(int idx = 0 ; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            CUDA_CHECK(cudaMemcpy(devAnchor, yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
            CalDetectionYoloSingleOuput<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],output, numElem, yolo.width, yolo.height, (float *)devAnchor, mClassCount ,outputElem);
        }

        CUDA_CHECK(cudaFree(devAnchor));
    }


    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void*const*  outputs, 
                                void* workspace, cudaStream_t stream) noexcept
    {
        if(YOLOLAYER_OUTPUT_NUMBER == 2)
            forwardGpu((const float *const *)inputs, outputs, workspace, stream, batchSize);
        else if(YOLOLAYER_OUTPUT_NUMBER == 1)
            forwardGpuSingleOutput((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        else
            forwardGpuSingleOutput((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const noexcept
    {
            return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const noexcept
    {
            return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames()noexcept
    {
            return &mFC;
    }

    IPluginV2* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)noexcept
    {
        YoloLayerPlugin* obj = new YoloLayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)noexcept
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
