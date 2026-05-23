import torch
import timm

model = timm.create_model(
    "mambaout_tiny",
    pretrained=False,
    num_classes=100
)

x = torch.randn(2, 3, 224, 224)

y = model(x)

print(y.shape)