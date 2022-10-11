#include <iostream>
#include <iterator>
#include <fstream>
#include <opencv2/dnn/dnn.hpp>
#include "calibrator.h"
#include "cuda_runtime_api.h"
#include "utils.h"

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache)
    : batchsize_(batchsize)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , input_blob_name_(input_blob_name)
    , read_cache_(read_cache)
{
    input_count_ = 3 * input_w * input_h * batchsize;
    CUDA_CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    read_files_in_dir(img_dir, img_files_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    CUDA_CHECK(cudaFree(device_input_));
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    return batchsize_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (img_idx_ + batchsize_ > (int)img_files_.size()) {
        return false;
    }

    // std::vector<cv::Mat> input_imgs_;
    std::vector<float> input_imgs_(input_count_, 0);
    float input_data[3*224*224] = {0};
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        //cv::Mat pr_img = preprocess_img(temp, input_w_, input_h_);
        cv::Mat pr_img;
        cv::resize(temp, pr_img, cv::Size(224, 224));

        std::vector<cv::Mat> channels;
        cv::split(pr_img, channels);
        cv::Mat temp_mat = channels[2];
        channels[2] = channels[0];
        channels[0] = temp_mat;
        cv::merge(channels, pr_img);
        pr_img.convertTo(pr_img, CV_32FC3);
        float* img_data = ( float* )pr_img.data;
        int hw = 224 * 224;//img_h * img_w;

        //float mean[3] = {127.5, 127.5, 127.5};
        float mean[3] = {0.485, 0.456, 0.406};
        float std[3] = {0.229, 0.224, 0.225};
        for(int h = 0; h < 224; h++)
        {
            for(int w = 0; w < 224; w++)
            {
                for(int c = 0; c < 3; c++)
                {
                    input_data[c * hw + h * 224 + w] = (*img_data * 0.00392156862745098039 - mean[c])/std[c];
                    img_data++;
                }
            }
        }
        for(int num_nub = 0; num_nub < 3*224*224; num_nub++)
        {
            input_imgs_[(i-img_idx_)*3*224*224+num_nub] = input_data[num_nub];
        }
        //input_imgs_.push_back(pr_img);
    }
    img_idx_ += batchsize_;
    // cv::Mat blob = cv::dnn::blobFromImages(input_imgs_, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);
    //cv::Mat blob = cv::dnn::blobFromImages(input_imgs_, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(123.68, 116.28, 103.53), true, false);
    
    // CUDA_CHECK(cudaMemcpy(device_input_, blob.ptr<float>(0), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input_, input_imgs_.data(), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], input_blob_name_));
    bindings[0] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept
{
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

