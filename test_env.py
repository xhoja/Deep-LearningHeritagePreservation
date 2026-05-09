import torch
import numpy as np

x = np.random.rand(4, 3, 224, 224).astype(np.float32)
t = torch.from_numpy(x).to('mps')
print('tensor shape:', t.shape)
print('device:', t.device)
print('OK')
