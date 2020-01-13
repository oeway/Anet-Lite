import os
import numpy as np
from anet.options import Options
from anet.data.examples import GenericTransformedImages
from anet.data.file_loader import ImageLoader

from anet.networks import UnetGenerator, get_dssim_l1_loss
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from anet.data.examples import TransformedTubulin001
from anet.data.utils import make_generator
from anet.utils import export_model_to_js

import tensorflow as tf
import json

from tensorflow.keras import backend as K
# import importlib
# importlib.reload(UnetGenerator)

opt = Options().parse()
# opt.work_dir = '/Users/weiouyang/ImJoyWorkspace/default/unet_data/train'
opt.input_channels = [('cell', {'filter':'cells*.png', 'loader':ImageLoader()})]
opt.target_channels = [('mask', {'filter':'mask_edge*.png', 'loader':ImageLoader()})]

model = UnetGenerator(input_size=opt.input_size, input_channels=opt.input_nc, target_channels=opt.target_nc, base_filter=16)

if opt.load_from is not None:
    model.load_weights(opt.load_from)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


# Create, compile and train model...

tf.identity(tf.get_default_graph().get_tensor_by_name(model.outputs[0].op.name+':0'), name="unet_output")
frozen_graph = freeze_session(K.get_session(),
                              output_names=['unet_output'])

config = json.load(open(os.path.join('config_template.json'), 'r'))

config['label'] = 'Unet_{}x{}_{}_{}'.format(opt.input_size, opt.input_size, len(opt.input_channels), len(opt.target_channels))
config['model_name'] = config['label']

config['inputs'][0]['key'] = 'unet_input'
config['inputs'][0]['channels'] = [ ch[0] for ch in opt.input_channels]
config['inputs'][0]['shape'] = [1, opt.input_size, opt.input_size, len(opt.input_channels)]
config['inputs'][0]['size'] = opt.input_size

config['outputs'][0]['key'] = 'unet_output'
config['outputs'][0]['channels'] = [ ch[0] for ch in opt.target_channels]
config['outputs'][0]['shape'] = [1, opt.input_size, opt.input_size, len(opt.target_channels)]
config['outputs'][0]['size'] = opt.input_size

with open(os.path.join( opt.work_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

tf.train.write_graph(frozen_graph, opt.work_dir, "tensorflow_model.pb", as_text=False)
