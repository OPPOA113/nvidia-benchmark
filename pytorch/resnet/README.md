# PyTorchx

Popular deep learning networks are implemented with pytorch in this project. And then weights files are exported for tensorrt implementation.

## Test Environments
1. Python 3.8.12
2. cuda 11.1
3. PyTorch 1.8.1
4. torchvision 0.9.1
5. timm 0.5.4
## Run
cp gluon_resnet50_v1b-0ebe02e2.pth ~/.cache/torch/hub/checkpoints/  (需要梯子才能下载下来，所以直接下载好放这里)
python3 gluoncv_resnet50.py

```
