#ifndef _EVAL_MODEL_HPP_
#define _EVAL_MODEL_HPP_

#include <vector>
#include <list>
#include <string>
#include "data_reader.hpp"

namespace Tn
{
    float evalTopResult(std::list<std::vector<float>>& result,std::list<int>& groundTruth,int* Tp = nullptr,int* FP = nullptr,int topK = 1);
    float evalMAPResult(const std::list<std::vector<Bbox>>& bboxesList,const std::list<std::vector<Bbox>> & truthboxesList,int classNum,float iouThresh);
}

#endif