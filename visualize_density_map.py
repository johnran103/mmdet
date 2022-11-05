import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

img = np.load('test_density_map.npy')
img_tensor = torch.from_numpy(img)
img_tensor = F.interpolate(img_tensor[None, None, :, :], size=(731, 1360)).squeeze(0).squeeze(0)
print(f'sum of img_tensor {img_tensor.sum()}')
img_array = img_tensor.numpy()
img_Image = Image.fromarray((img_array * 1000).astype(np.uint8))
img_Image.save('./test_density_map.jpg')

img = np.load('9999984_00000_d_0000101.npy')
img_tensor = torch.from_numpy(img)
img_tensor = F.interpolate(img_tensor[None, None, :, :], size=(731, 1360)).squeeze(0).squeeze(0)
print(f'sum of img_tensor {img_tensor.sum()}')
img_array = img_tensor.numpy()
img_Image = Image.fromarray((img_array * 100000).astype(np.uint8))
img_Image.save('./gt_density_map.jpg')