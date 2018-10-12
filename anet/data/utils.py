import os
import numpy as np
import hashlib
import json

EPS = 0.0001

def make_generator(source, batch_size=1, batch_num=None):
    x, y = [], []
    count = 0
    while batch_num is None or count<batch_num:
        for d in source:
            x.append(d['A'])
            y.append(d['B'])
            if len(x) >= batch_size:
                x = np.stack(x, axis=0)
                y = np.stack(y, axis=0)
                m, s = x.mean(), x.std()+EPS
                x = (x-m)/s
                y = y/(y.max()+EPS)
                yield x, y
                x, y = [], []
                count += 1

def make_test_generator(source, batch_size=1, batch_num=None):
    x, path = [], []
    count = 0
    for d in source:
        x.append(d['A'])
        path.append(d['path'])
        if len(x) >= batch_size:
            x = np.stack(x, axis=0)
            m, s = x.mean(), x.std()+EPS
            x = (x-m)/s
            yield x, path
            x, path = [], []
            count += 1
    return

def check_integrity(fpath, md5c):
    if not os.path.isfile(fpath):
        return False
    md5 = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
    print('file md5: ' + md5)
    if md5 != md5c:
        return False
    return True


def download_url(url, fpath, md5c=None):
    from six.moves import urllib
    # downloads file
    if md5c and os.path.isfile(fpath) and hashlib.md5(open(fpath, 'rb').read()).hexdigest() == md5c:
        print('Using downloaded file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath)
    print('Done!')


def get_id_for_dict(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode('utf-8')).hexdigest()

def calculate_mean_std(data_source, count=1000, label='A'):
    '''
    from data.datasets import calculate_mean_std
    mean, std = calculate_mean_std(source_train, count=200)
    '''
    xList = []
    for i, d in enumerate(data_source):
        xList.append(d[label])
        if i>count:
            break
    if type(xList[0]) is np.ndarray:
        arr = np.stack(xList)
        return arr.mean(axis=(0,1, 2)), arr.std(axis=(0,1, 2))
    else:
        import torch
        trr = torch.stack(xList)
        trr = trr.transpose(1, 3).contiguous()
        return trr.view(-1, trr.size(3)).mean(dim=0), trr.view(-1, trr.size(3)).std(dim=0)
