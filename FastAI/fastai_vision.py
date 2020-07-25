from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
#import sys

from PIL import Image


# execute this file from run_vision_parallel!!!

help(untar_data)

path = untar_data(URLs.PETS)
print(path)
print(path.ls())

path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
#pat = re.compile(r'\\([^\\]+)_\d+.jpg$')
#pat = r'\\([^\\]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=32, padding_mode='border')
#data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size= 224, padding_mode='zeros', num_workers=0)

# im = data.dls[0].dl.dataset.x.items
# plt.imshow(im)
# plt.show()

#sys.exit()

#print(sys.last_value)
#print(sys.last_traceback)
data.normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(7,6)) #rows=3, figsize=(7,6)

#plt.imshow(data)
print(data.classes)
print(data.c)

# learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save('stage-1')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))