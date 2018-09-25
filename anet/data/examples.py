import os
import random
from anet.data.datasets import TUBULIN
from anet.data.image_utils import *
from anet.data.folder_dataset import FolderDataset
EPS = 0.0001

class TransformedTubulin001():
    def __init__(self, opt):
        self.tags = ['microtubule', 'simulation']
        self.iRot = RandomRotate()
        self.iMerge = Merge()
        self.iSplit = Split([0, 1], [1, 2])
        self.irCropTrain = RandomCropNumpy(size=(opt.input_size+100, opt.input_size+100))
        self.ioCropTrain = CenterCropNumpy(size=[opt.input_size, opt.input_size])
        self.iCropTest = CenterCropNumpy(size=(1024, 1024))
        self.iElastic = ElasticTransform(alpha=1000, sigma=40)
        self.iBlur = GaussianBlurring(sigma=1.5)
        self.iPoisson = PoissonSubsampling(peak=['lognormal', -0.5, 0.001])
        self.iBG = AddGaussianPoissonNoise(sigma=25, peak=0.06)
        self.train_count = 0
        self.test_count = 0
        self.dim_ordering = opt.dim_ordering
        self.repeat = 1
        self.opt = opt

    def __getitem__(self, key):
        if key == 'train':
            source_train = TUBULIN('./datasets', train=True, download=True, transform=self.transform_train, repeat=self.repeat)
            return source_train
        elif key == 'test':
            source_test = TUBULIN('./datasets', train=False, download=True, transform=self.transform_test, repeat=self.repeat)
            return source_test
        else:
            raise Exception('only train and test are supported.')

    def transform_train(self, imageIO):
        img = self.iMerge(imageIO.copy())
        img = self.irCropTrain(img)
        img = self.iRot(img)
        img = self.ioCropTrain(img)
        img = self.iElastic(img)
        iIm, oIm = self.iSplit(img)
        iIm, oIm = self.iBlur(iIm), self.iBlur(oIm)

        imgin, imgout = self.iBG(self.iPoisson(iIm)), oIm
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'

        path = str(self.train_count)
        self.train_count += 1
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageIO):
        img = self.iMerge(imageIO.copy())
        img = self.irCropTrain(img)
        # img = iRot(img)
        img = self.ioCropTrain(img)
        iIm, oIm = self.iSplit(img)
        iIm, oIm = self.iBlur(iIm), self.iBlur(oIm)
        imgin, imgout = self.iBG(self.iPoisson(iIm)), oIm
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        path = str(self.test_count)
        self.test_count += 1
        return {'A': imgin, 'B': imgout, 'path': path}

class TransformedTubulin002(TransformedTubulin001):
    def __init__(self, opt):
        super(TransformedTubulin002, self).__init__(opt)
        self.wfBlur = GaussianBlurring(sigma=['uniform', 6, 8])
        self.wfNoise = AddGaussianNoise(mean=0, sigma=['uniform', 0.5, 1.5])

    def transform_train(self, imageIO):
        img = self.iMerge(imageIO.copy())
        img = self.irCropTrain(img)
        img = self.iRot(img)
        img = self.ioCropTrain(img)
        img = self.iElastic(img)
        iIm, oIm = self.iSplit(img)
        iIm, oIm = self.iBlur(iIm), self.iBlur(oIm)

        wf = self.wfBlur(iIm)[:, :, 0]
        wf = scipy.misc.imresize(wf, self.opt.input_scale)
        wf = self.wfNoise(wf[:,:,None])

        imgin, imgout = wf, oIm/255.0
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'

        path = str(self.train_count)
        self.train_count += 1
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageIO):
        img = self.iMerge(imageIO.copy())
        img = self.irCropTrain(img)
        # img = iRot(img)
        img = self.ioCropTrain(img)
        iIm, oIm = self.iSplit(img)
        iIm, oIm = self.iBlur(iIm), self.iBlur(oIm)
        wf = self.wfBlur(iIm)[:, :, 0]
        wf = scipy.misc.imresize(wf, self.opt.input_scale)
        wf = self.wfNoise(wf[:,:,None])
        imgin, imgout = wf, oIm/255.0
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        path = str(self.test_count)
        self.test_count += 1
        return {'A': imgin, 'B': imgout, 'path': path}

class DriftingTubulin001(TransformedTubulin001):
    def __init__(self, opt):
        super(DriftingTubulin001, self).__init__(opt)
        self.wfBlur = GaussianBlurring(sigma=['uniform', 6, 8])
        self.wfNoise = AddGaussianNoise(mean=0, sigma=['uniform', 0.5, 1.5])
        self.shift_range = 16 

    def transform_train(self, imageIO):
        img = self.iMerge(imageIO.copy())
        img = self.irCropTrain(img)
        img = self.iRot(img)
        img = self.ioCropTrain(img)
        img = self.iElastic(img)
        iIm, oIm = self.iSplit(img)
        iIm, oIm = self.iBlur(iIm), self.iBlur(oIm)
        wf = self.wfBlur(iIm)[:, :, 0]
        wf = scipy.ndimage.interpolation.shift(wf, [np.random.uniform(-self.shift_range, self.shift_range), np.random.uniform(-self.shift_range, self.shift_range)])
        wf = scipy.misc.imresize(wf, self.opt.input_scale)
        wf = self.wfNoise(wf[:,:,None])

        imgin, imgout = wf, oIm/255.0
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'

        path = str(self.train_count)
        self.train_count += 1
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageIO):
        img = self.iMerge(imageIO.copy())
        img = self.irCropTrain(img)
        # img = iRot(img)
        img = self.ioCropTrain(img)
        iIm, oIm = self.iSplit(img)
        iIm, oIm = self.iBlur(iIm), self.iBlur(oIm)
        wf = self.wfBlur(iIm)[:, :, 0]
        wf = scipy.ndimage.interpolation.shift(wf, [np.random.uniform(-self.shift_range, self.shift_range), np.random.uniform(-self.shift_range, self.shift_range)])
        wf = scipy.misc.imresize(wf, self.opt.input_scale)
        wf = self.wfNoise(wf[:,:,None])
        imgin, imgout = wf, oIm/255.0
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        path = str(self.test_count)
        self.test_count += 1
        return {'A': imgin, 'B': imgout, 'path': path}


class SubFolderWFImagesLoader(FileLoader):
    def __init__(self, drift_correction=False, scale_LR=True):
        self.__cache = {}
        self.ext = 'tif'
        self.drift_correction = drift_correction
        self.scale_LR = scale_LR

    def load(self, path):
        if path not in self.__cache:
            self.cache(path)
        return self.__cache[path]

    def cache(self, path):
        Bs = [os.path.join(path, p) for p in os.listdir(path) if p == 'Histograms.tif']
        LRs = [os.path.join(path, p) for p in os.listdir(path) if p == 'WF_TMR_calibrated.tif']
        ImgBs, PathBs, ImgLRs, PathLRs= [], [], [], []
        for p in Bs:
            img = np.array(Image.open(p))
            img = np.expand_dims(img, axis=2) if img.ndim == 2 else img
            ImgBs.append(img)
            PathBs.append(p)

        for p in LRs:
            try:
                imgStack = Image.open(p)
                indexes = [i for i in range(imgStack.n_frames)]
                random.shuffle(indexes)
                c = min(len(indexes), 20)
                for i in indexes[:c]:
                    imgStack.seek(i)
                    img = np.array(imgStack)
                    dtype = img.dtype
                    assert img.ndim == 2
                    if self.drift_correction:
                        import imreg_dft as ird
                        from skimage import exposure
                        b = ImgBs[0][:, :, 0]
                        b = exposure.equalize_hist(b)
                        b = scipy.ndimage.filters.gaussian_filter(b, sigma=(6, 6))
                        b = scipy.misc.imresize(b, img.shape[:2])
                        ts = ird.translation(b, img)
                        tvec = ts["tvec"]
                        # the Transformed IMaGe.
                        img = ird.transform_img(img, tvec=tvec)
                    if self.scale_LR == True:
                        img = scipy.misc.imresize(img, ImgBs[0].shape[:2])
                    elif type(self.scale_LR) is list:
                        img = scipy.misc.imresize(img, self.scale_LR)
                    img = np.expand_dims(img, axis=2)
                    img = img.astype(dtype)
                    ImgLRs.append(img)
                    PathLRs.append(p)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print('error when reading file ', p)
                import traceback, sys
                traceback.print_exc(file=sys.stdout)

        self.__cache[path] = { 'B': ImgBs, 'A':ImgLRs, 'path': path, 'pathB': PathBs, 'pathA': PathLRs}
        return True

    def __call__(self, path):
        if path not in self.__cache:
            self.cache(path)
        return self.__cache[path].copy()


class TransformedLRSR():
    def __init__(self, opt):
        train_crop_size1 = opt.input_size * 2
        train_crop_size2 = opt.input_size + 200
        train_crop_size3 = opt.input_size
        test_size = opt.input_size

        self.input_clip = (0, 5)
        self.output_clip = (2, 100)

        # prepare the transforms
        self.iMerge = Merge()
        self.iElastic = ElasticTransform(alpha=1000, sigma=40)
        self.iSplit = Split([0, 1], [1, 2])
        self.iRot = RandomRotate()
        self.iRCropTrain = RandomCropNumpy(size=(train_crop_size2, train_crop_size2))
        self.iCropFTrain = CenterCropNumpy(size=(train_crop_size1, train_crop_size1))
        self.iCropTrain = CenterCropNumpy(size=(train_crop_size3, train_crop_size3))
        self.iCropTest = CenterCropNumpy(size=(test_size, test_size))
        self.ptrain = './datasets/Christian-TMR-IF-v0.1/train'
        self.ptest = './datasets/Christian-TMR-IF-v0.1/test'
        self.dim_ordering = opt.dim_ordering
        self.opt = opt
        self.repeat = 30
        self.folder_filter = '*'
        self.drift_correction = False
        self.scale_LR = True

    def __getitem__(self, key):
        if key == 'train':
            imgfolderLoader = SubFolderWFImagesLoader(drift_correction=self.drift_correction, scale_LR=self.scale_LR)
            source_train = FolderDataset(self.ptrain,
                              channels = {'image': {'filter': self.folder_filter, 'loader': imgfolderLoader} },
                             transform = self.transform_train,
                             recursive=False,
                             repeat=self.repeat)
            return source_train
        elif key == 'test':
            imgfolderLoader = SubFolderWFImagesLoader(drift_correction=self.drift_correction, scale_LR=self.scale_LR)
            source_test = FolderDataset(self.ptest,
                              channels = {'image': {'filter': self.folder_filter, 'loader': imgfolderLoader} },
                             transform = self.transform_test,
                             recursive=False,
                             repeat=self.repeat)
            return source_test
        else:
            raise Exception('only train and test are supported.')

    def transform_train(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        img = self.iMerge([histin, histout])
        img = self.iRCropTrain(img)
        img = self.iRot(img)
        img = self.iElastic(img)
        histin, histout = self.iSplit(img)
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTrain(histin), self.iCropTrain(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTest(histin), self.iCropTest(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

    
class TransformedLRSR002(TransformedLRSR):
    def transform_train(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        img = self.iMerge([histin, histout])
        img = self.iRCropTrain(img)
        img = self.iRot(img)
        img = self.iElastic(img)
        histin, histout = self.iSplit(img)
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTrain(histin), self.iCropTrain(histout)
        imgin = scipy.misc.imresize(imgin[:, :, 0], (self.opt.input_size//4, self.opt.input_size//4))[:, :, None]
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTest(histin), self.iCropTest(histout)
        imgin = scipy.misc.imresize(imgin[:, :, 0], (self.opt.input_size//4, self.opt.input_size//4))[:, :, None]
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}