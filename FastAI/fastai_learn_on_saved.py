from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt

from PIL import Image

# execute this file from run_vision_parallel!!!


path = untar_data(URLs.PETS)


path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.load('stage-1')
learn.lr_find()
plt.show()

learn.recorder.plot()
plt.show()

# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_top_losses(9, figsize=(15,11))
# plt.show()

# interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
# plt.show()

# interp.most_confused(min_val=2)
# plt.show()
