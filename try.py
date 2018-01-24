from cppn import CPPN
from KS_lib import KSimage
import torch

model = CPPN(batch_size=1, scale=3.0, c_dim=3)
if torch.cuda.is_available():
    model = model.cuda()

img = model.generate()
# img = img.transpose(0,2,3,1)
KSimage.imshow(img[0,:,:,:])
print('lol')