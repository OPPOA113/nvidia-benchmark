
### benchmark测试

+ 算法模型性能测试
    + 依赖[仓库](https://github.com/wang-xinyu/tensorrtx)
    + resnet50\retinaface\yolov3\yolov5
    + 强化知识点：
        + 配合依赖仓库，tensorrt api使用 和 量化方式
        + 配合依赖仓库，tensorrt python接口使用 
        + yolov5 python接口 tensorrt的多线程处理方式 `./tensorrt/yolov5/yolov5_trt_multi.py`