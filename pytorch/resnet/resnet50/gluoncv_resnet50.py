
import torch
import struct
from torchsummary import summary
import timm

# net = torch.hub.load('rwightman/pytorch-pretrained-gluonresnet', 'gluon_resnet50_v1b', pretrained=True)
net=timm.create_model("gluon_resnet50_v1b",pretrained=True)
net = net.to('cuda:0')
print("==================")
net.eval()
tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
out = net(tmp)
summary(net, (3,224,224))
#return
f = open("resnet50.wts", 'w')
f.write("{}\n".format(len(net.state_dict().keys())))
for k,v in net.state_dict().items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")