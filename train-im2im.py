import os
import numpy as np
from anet.options import Options
from anet.data.examples import GenericTransformedImages
from anet.data.file_loader import ImageLoader

from anet.networks import UnetGenerator, get_dssim_l1_loss
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from anet.data.examples import TransformedTubulin001
from anet.data.utils import make_generator, download_with_url
from anet.utils import export_model_to_js

# import importlib
# importlib.reload(UnetGenerator)


opt = Options().parse()
opt.input_channels = [('bf1', {'filter':'BF-10um.png', 'loader':ImageLoader()}), ('bf2', {'filter':'BF+10um.png', 'loader':ImageLoader()}), ('nuclei', {'filter':'DAPI-MaxProj.png', 'loader':ImageLoader()})]
opt.target_channels = [('mebrane', {'filter':'GFP.png', 'loader':ImageLoader()})]
opt.input_nc = len(opt.input_channels)
opt.target_nc = len(opt.target_channels)

if not os.path.exists(os.path.join(opt.work_dir, 'train')):
    print('Downloading dataset...')
    os.makedirs(opt.work_dir, exist_ok=True)
    download_with_url('https://kth.box.com/shared/static/r6kjgvdkcuehssxipaxqxfflmz8t65u1.zip', os.path.join(opt.work_dir, 'SegmentationTrainingProcessed_CG_20200109-offset-corrected.zip'), unzip=True)

model = UnetGenerator(input_size=opt.input_size, input_channels=opt.input_nc, target_channels=opt.target_nc, base_filter=16)

if opt.load_from is not None:
    model.load_weights(opt.load_from)

DSSIM_L1 = get_dssim_l1_loss()
model.compile(optimizer='adam',
              loss=DSSIM_L1,
              metrics=['mse', DSSIM_L1])

sources = GenericTransformedImages(opt)

# d = make_generator(sources['train'])
# x,y = next(d)
# print(x.shape, y.shape)

tensorboard = TensorBoard(log_dir=os.path.join(opt.checkpoints_dir, 'logs'), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True)
checkpointer = ModelCheckpoint(filepath=os.path.join(opt.checkpoints_dir, 'weights.hdf5'), verbose=1, save_best_only=True)
model.fit_generator(make_generator(sources['train'], batch_size=opt.batch_size),
                    validation_data=make_generator(sources['valid'], batch_size=opt.batch_size),
                    validation_steps=4, steps_per_epoch=200, epochs=1000, verbose=2, callbacks=[tensorboard, checkpointer])

export_model_to_js(model, opt.work_dir+'/__js_model__')
