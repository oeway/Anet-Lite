import numpy as np
from anet.networks import UnetGenerator
from keras.losses import mean_absolute_error
from keras.callbacks import ModelCheckpoint
from anet.data.examples import TransformedTubulin001
from anet.options import Options
from anet.data.utils import make_generator

opt = Options().parse(['--work_dir=./results/__test__', '--batch_size=2'])
def test_train():
    model = UnetGenerator(image_size=opt.input_size, input_channels=opt.input_nc, target_channels=opt.target_nc, filters_base=16)
    model.compile(optimizer='adam',
                  loss=mean_absolute_error,
                  metrics=[mean_absolute_error])
    sources = TransformedTubulin001(opt)
    d = make_generator(sources['train'], batch_size=opt.batch_size)
    x, y = next(d)
    assert x.shape == (opt.batch_size, opt.input_size, opt.input_size, opt.input_nc)
    assert y.shape == (opt.batch_size, opt.target_size, opt.target_size, opt.target_nc)
    model.fit_generator(make_generator(sources['train'], batch_size=opt.batch_size),
                        validation_data=make_generator(sources['test'], batch_size=opt.batch_size),
                        validation_steps=1, steps_per_epoch=1, epochs=1, verbose=2, callbacks=[])
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, opt.work_dir+'/__js_model__')
