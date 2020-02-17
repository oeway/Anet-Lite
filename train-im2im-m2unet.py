import os
import numpy as np
from anet.options import Options
from anet.data.examples import GenericTransformedImages
from anet.data.file_loader import ImageLoader

from anet.networks import MobileUNet, MobileUNetFS, get_dssim_l1_loss
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from anet.data.examples import TransformedTubulin001
from anet.data.utils import make_generator, download_with_url
from anet.utils import export_model_to_js, UpdateUI
import tensorflow as tf

opt = Options().parse()
opt.input_scale = 0.5
opt.target_scale = 0.5
opt.input_channels = [('EM1', {'filter':'EM.png', 'loader':ImageLoader(scale=opt.input_scale)}), ('EM2', {'filter':'EM.png', 'loader':ImageLoader(scale=opt.input_scale)}), ('EM3', {'filter':'EM.png', 'loader':ImageLoader(scale=opt.input_scale)})]
opt.target_channels = [('mask', {'filter':'Mask.png', 'loader':ImageLoader(scale=opt.target_scale)})]
opt.input_nc = len(opt.input_channels)
opt.target_nc = len(opt.target_channels)
opt.batch_size = 10

if not os.path.exists(os.path.join(opt.work_dir, 'train')):
    print('Downloading dataset...')
    os.makedirs(opt.work_dir, exist_ok=True)
    download_with_url('https://kth.box.com/shared/static/r6kjgvdkcuehssxipaxqxfflmz8t65u1.zip', os.path.join(opt.work_dir, 'SegmentationTrainingProcessed_CG_20200109-offset-corrected.zip'), unzip=True)

model = generate_m2_unet() #MobileUNet(input_size=opt.input_size, input_channels=opt.input_nc, target_channels=opt.target_nc)#.build()
model.summary()

if opt.load_from is not None:
    model.load_weights(opt.load_from)

# DSSIM_L1 = get_dssim_l1_loss()
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

# model._layers[1].batch_input_shape = [None, 3, None, None]

# import tensorflowjs as tfjs
# tfjs.converters.save_keras_model(model, os.path.join(opt.checkpoints_dir, 'mobile-unet-best-tfjs'))

sources = GenericTransformedImages(opt)

tensorboard = TensorBoard(log_dir=os.path.join(opt.checkpoints_dir, 'logs'), histogram_freq=0, write_graph=True, write_grads=False, write_images=True)
checkpointer = ModelCheckpoint(filepath=os.path.join(opt.checkpoints_dir, 'weights.hdf5'), verbose=1, save_best_only=True)
updateUI = UpdateUI(1000, make_generator(sources['valid'], batch_size=opt.batch_size), opt.save_dir)
updateUI.test(model)
model.fit_generator(make_generator(sources['train'], batch_size=opt.batch_size),
                    validation_data=make_generator(sources['valid'], batch_size=opt.batch_size),
                    validation_steps=4, steps_per_epoch=10, epochs=1000, verbose=2, use_multiprocessing=False, workers=1, callbacks=[tensorboard, checkpointer, updateUI])

export_model_to_js(model, opt.work_dir+'/__js_model__')
