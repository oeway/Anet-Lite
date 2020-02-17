import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.isTrain = False

    def initialize(self):
        self.parser.add_argument('--work_dir', type=str, required=True, help='work directory')
        self.parser.add_argument('--folder', type=str, required=True, help='subfolder to save outputs')
        self.parser.add_argument('--phase', type=str, default='train', help='training or testing')
        self.parser.add_argument('--load_from', type=str, default=None, help='load weights from path')
        self.parser.add_argument('--save_dir', type=str, default=None, help='path for save outputs and configs')
        self.parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
        self.parser.add_argument('--data_root', type=str, default=None, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_scale', type=int, default=1, help='scale images when loading')
        self.parser.add_argument('--target_scale', type=int, default=1, help='scale images when loading')
        self.parser.add_argument('--input_size', type=int, default=256, help='size of the input image')
        self.parser.add_argument('--target_size', type=int, default=256, help='size of the target image')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--target_nc', type=int, default=1, help='# of target image channels')
        self.parser.add_argument('--model', type=str, default='a_net',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--checkpoints_dir', type=str, default=None, help='models are saved here')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--dim_ordering', type=str, default='channels_last', help='dim ordering, for tensorflow, should be channels_last, for pytorch should be channels_first')
        self.parser.add_argument('--input_norm', type=str, default=None, help='normalization method for input image')
        self.parser.add_argument('--target_norm', type=str, default=None, help='normalization method for target image')

        self.initialized = True

    def parse(self, *args):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(*args)
        self.opt.isTrain = self.isTrain   # train or test

        if not os.path.exists(self.opt.work_dir):
            os.makedirs(self.opt.work_dir)

        if self.opt.save_dir is None:
            self.opt.save_dir = os.path.join(self.opt.work_dir, self.opt.folder, 'outputs')
            if not os.path.exists(self.opt.save_dir):
                os.makedirs(self.opt.save_dir)

        if self.opt.checkpoints_dir is None:
            self.opt.checkpoints_dir = os.path.join(self.opt.work_dir, self.opt.folder, '__model__')
            if not os.path.exists(self.opt.checkpoints_dir):
                os.makedirs(self.opt.checkpoints_dir)

        if self.opt.data_root is None:
            self.opt.data_root = self.opt.work_dir

        if self.opt.load_from is not None:
            if os.path.isdir(self.opt.load_from):
                self.opt.load_from = os.path.join(self.opt.load_from, '__model__', 'weights.hdf5')
        return self.opt
