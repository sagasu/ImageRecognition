from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt

from PIL import Image

help(untar_data)

path = untar_data(URLs.PETS)
print(path)
print(path.ls())

path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size= 224)
data.normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(7,6)) #rows=3, figsize=(7,6)
#plt.imshow(data)
print(data)