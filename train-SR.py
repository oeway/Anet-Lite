import os
import numpy as np
from anet.networks import UnetGenerator, get_dssim_l1_loss
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from anet.data.examples import TransformedLRSR
from anet.options import Options
from anet.data.utils import make_generator
from anet.utils import export_model_to_js
# import importlib
# importlib.reload(UnetGenerator)

opt = Options().parse()
model = UnetGenerator(input_size=opt.input_size, input_channels=opt.input_nc, target_channels=opt.target_nc, base_filter=16)

if opt.load_from is not None:
    model.load_weights(opt.load_from)

DSSIM_L1 = get_dssim_l1_loss()
model.compile(optimizer='adam',
              loss=DSSIM_L1,
              metrics=['mse', DSSIM_L1])

sources = TransformedLRSR(opt)

tensorboard = TensorBoard(log_dir=os.path.join(opt.checkpoints_dir, 'logs'), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True)
checkpointer = ModelCheckpoint(filepath=os.path.join(opt.checkpoints_dir, 'weights.hdf5'), verbose=1, save_best_only=True)
model.fit_generator(make_generator(sources['train'], batch_size=opt.batch_size),
                    validation_data=make_generator(sources['test'], batch_size=opt.batch_size),
                    validation_steps=4, steps_per_epoch=200, epochs=1000, verbose=2, callbacks=[checkpointer, tensorboard])

export_model_to_js(model, opt.work_dir+'/__js_model__')
