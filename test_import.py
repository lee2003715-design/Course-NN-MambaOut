import torch
import timm

print(torch.__version__)
print("CUDA:", torch.cuda.is_available())

model = timm.create_model("mambaout_tiny", pretrained=False)

print("Model loaded!")