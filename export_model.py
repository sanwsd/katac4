import torch
from model import Net

model_path = './weights/b3c128nbt_2025-05-24_20-47-22/katac4_b3c128nbt_30000.pth'

device = torch.device('cpu')
net = Net().eval()
net.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

model = torch.jit.script(net)
model = torch.jit.freeze(model)

torch.jit.save(model, './saiblo/model.pt')
