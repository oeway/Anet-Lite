import sys
import os
from PIL import Image

from tensorflow.keras.callbacks import Callback
def export_model_to_js(model, path, remove_input_size=True):
    import tensorflowjs as tfjs
    input_shape_bk = None
    if remove_input_size:
        # make input image size adaptive
        input_layer = model.get_layer(index=0)
        input_shape_bk = input_layer.batch_input_shape
        input_layer.batch_input_shape = (None, None, None, input_shape_bk[3])
    tfjs.converters.save_keras_model(model, path)
    # recover shape
    if remove_input_size and input_shape_bk and input_layer:
        input_layer.batch_input_shape = input_shape_bk


def save_tensors(tensor_list, label, titles, output_dir):
    image_list = [tensor.reshape(tensor.shape[-3], tensor.shape[-2], -1) for tensor in tensor_list]
    displays = {}
    for i in range(len(image_list)):
        ims = image_list[i]
        for j in range(ims.shape[2]):
            im = ims[:, :, j]
            mi = im.min()
            im = Image.fromarray(((im - mi) / (im.max() - mi) * 255).astype('uint8'))
            os.makedirs(os.path.join(output_dir, label), exist_ok=True)
            im.save(os.path.join(output_dir, label, titles[i]+str(j) + '.png'))

class UpdateUI(Callback):
    def __init__(self, total_epoch, gen, output_dir):
        self.total_epoch = total_epoch
        self.epoch = 0
        self.logs = {}
        self.step = 0
        self.gen = gen
        self.output_dir = output_dir
    
    def on_batch_end(self, batch, logs):
        self.logs = logs
        print('training epoch:'+str(self.epoch)+'/'+str(self.total_epoch) + ' ' + str(logs))
        sys.stdout.flush()
        #print('onStep', self.step, {'mse': np.asscalar(logs['mean_squared_error']), 'dssim_l1': np.asscalar(logs['DSSIM_L1'])})
        self.step += 1
    
    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
        self.logs = logs
        #api.showProgress(self.epoch/self.total_epoch*100)
        print('training epoch:'+str(self.epoch)+'/'+str(self.total_epoch) + ' '+ str(logs))
        xbatch, ybatch = next(self.gen)
        ypbatch = self.model.predict(xbatch[:1, :, :, :], batch_size=1)
        tensor_list = [ypbatch, xbatch[:1, :, :, :], ybatch[:1, :, :, :]]
        label = 'Step '+ str(self.step)
        titles = ["output", 'input', 'target']
        save_tensors(tensor_list, label, titles, self.output_dir)
