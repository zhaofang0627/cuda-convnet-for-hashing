'''
Created on Nov 6, 2013

@author: wuzifeng
'''

import os
from collections import OrderedDict
import subprocess as sp
from math import sqrt

import numpy as np
import numpy.random as nr
import pylab as pl
import xml.etree.ElementTree as etree
import Image
from scipy import linalg as la

from python_util.util import *
from python_util.data import LabeledDataProvider
import convdata as dataprovider

# dataset-specific functions

def voc_read_bboxes(bbox_path):
    '''
    parse the corresponding object bounding box given a xml file
    each row is an bounding box of the following format:
    [
    ... box1
    [x_min, y_min, x_max, y_max]
    ... box_n
    ]
    '''
    if not os.path.exists(bbox_path):
        return np.array([])
    tree = etree.parse(bbox_path)
    root = tree.getroot()
    objs = root.findall('object')
    nobjs = len(objs)
    if len(objs) == 0:
        return np.array([])
    boxes = np.zeros((nobjs, 4), dtype=int)
    for i in xrange(nobjs):
        o = objs[i]        
        bndbox = o[-1] # the last property of object
        assert(bndbox.tag == 'bndbox')
        boxes[i, 0] = int(bndbox[0].text)
        boxes[i, 1] = int(bndbox[1].text)
        boxes[i, 2] = int(bndbox[2].text)
        boxes[i, 3] = int(bndbox[3].text)
    return boxes

def siftflow_segment(dataDir):
    fileList = np.sort(os.listdir(dataDir))
    for item in fileList:
        if item.startswith('data_batch_'):
            batchPath = os.path.join(dataDir, item)
            dic = unpickle(batchPath)
            dic['segmentations'] = []
            for im in dic['data']:
                tf1, tf2 = './temp-in.ppm', './temp-out.ppm'
                Image.fromarray(im).save(tf1)
                os.system('./segment 0.5 500 20 %s %s > /dev/null' % (tf1, tf2))
                imSeg = np.array(Image.open(tf2))
                pixelList = list(imSeg.T.swapaxes(1,2).reshape((3,-1)))
                pixelList = zip(pixelList[0],pixelList[1],pixelList[2])
                pixelDic = {}
                for i,pixel in enumerate(pixelList):
                    if pixelDic.has_key(pixel):
                        pixelDic[pixel].append(i)
                    else:
                        pixelDic[pixel] = [i]
                seg = np.zeros(im.shape[:2], np.int32).flatten()
                for i,p in enumerate(pixelDic.keys()):
                    seg[np.array(pixelDic[p])] = i + 1
                dic['segmentations'].append(seg.reshape(im.shape[:2]))
            pickle(batchPath, dic)

# generic functions

def crop_array(im, box, padType=0):
    #w, h = imsize
    h, w = im.shape[:2]
    # crop area
    ymin, xmin, ymax, xmax = box
    padding = xmin < 0 or ymin < 0 or xmax > w-1 or ymax > h-1
    if padding:
        # dist to boundary
        padx_min = int(max(-xmin, 0))
        padx_max = int(max(xmax - w, 0))
        pady_min = int(max(-ymin, 0))
        pady_max = int(max(ymax - h, 0))
        if padType == 0:
            assert(False)
        elif padType == 1:
            # crop with boundary padding
            x_ind = [0] * padx_min + range(max(xmin,0),min(xmax,w)) + [-1] * padx_max
            y_ind = [0] * pady_min + range(max(ymin,0),min(ymax,h)) + [-1] * pady_max
            nim = im[y_ind, x_ind, :]
        return nim
    else:
        return im[ymin: ymax, xmin:xmax, :]

def is_empty_box(box):
    return not ((box[3]>box[1]) and (box[2]>box[0])) 

def jitter_pca(pic, pca):
    assert(pic.shape[0] == pca[0].shape[0] == len(pca[1]))
    dim = pic.shape[0]
    return (pic.T + np.dot(pca[0], (nr.normal(0.0,0.01,dim)*pca[1]).T)).T

def resize_image(im, size, dtype, resample):
    assert(im.ndim == 3 and len(size) == 2)
    channels = im.shape[2]
    toSize = list(size)
    toSize.append(channels)
    nim = np.zeros(toSize, dtype)
    for i in xrange(channels):
        nim[:,:,i] = Image.fromarray(im[:,:,i]).resize(size, resample)
    return nim

def rotate_image(im, angle, resample):
    assert(im.ndim == 3)
    for i in xrange(im.shape[2]):
        im[:,:,i] = Image.fromarray(im[:,:,i]).rotate(angle, resample)
    return im

def rotate_labels(labels, angle):
    assert(labels.shape == (3,3))
    if angle in range(0,360,90):
        res = np.array(Image.fromarray(labels).rotate(angle, Image.NEAREST))
    else:
        assert(angle in range(45,360,90))
        pos = [[(0,1),(0,2),(1,2),(0,0),(1,1),(2,2),(1,0),(2,0),(2,1)],
               [(1,2),(2,2),(2,1),(0,2),(1,1),(2,0),(0,1),(0,0),(1,0)],
               [(2,1),(2,0),(1,0),(2,2),(1,1),(0,0),(1,2),(0,2),(0,1)],
               [(1,0),(0,0),(0,1),(2,0),(1,1),(0,2),(2,1),(2,2),(1,2)]]
        i = int(angle / 90)
        y,x = zip(*pos[i])
        res = labels[y,x].reshape((3,3))
    return res

def show_boxes(im, boxes, clr=(0, 1, 0), dft=None):  
    delay = 0#.05
    if delay:
        pl.ion()
    if not hasattr(show_boxes, "sv_idx"):
        show_boxes.sv_idx = 0
    pl.cla() 
    pl.imshow(im)
    box = np.array(boxes)
    if box.size > 0:
        box = np.reshape(box, (box.size/4, 4))
    
    lw = 3
    n_inst = box.shape[0]
    for i in xrange(n_inst):
        if i > 0:
            clr = (1, 1, 0)
            lw = 3
        if dft != None:
            clr = (0, 0, 1) if dft[i] else (0, 1, 0)
        xcen, ycen = 0.5 * (box[i, 2] + box[i, 0]), 0.5 * (box[i, 3] + box[i, 1])
        if xcen == box[i, 0] or ycen == box[i, 1]:
            continue
        pl.plot(xcen, ycen, '+', color=clr, markersize=10, markeredgewidth=lw)
        pl.plot([box[i, 0], box[i, 2], box[i, 2], box[i, 0], box[i, 0]],
                [box[i, 1], box[i, 1], box[i, 3], box[i, 3], box[i, 1]],
                color=clr, linewidth=lw, linestyle='-')  
    pl.axis('equal')
    pl.axis('off')
    pl.draw()
    
    #pl.savefig('./loc_results/%05d.png' % show_boxes.sv_idx, bbox_inches=0)
    #show_boxes.sv_idx += 1
    if delay:
        pl.pause(delay)
    else:
        pl.show()

def show_image(pic, scale):
    pl.imshow(np.uint8(np.maximum(0, np.minimum(255,50+pic.reshape(3,scale,scale).T.swapaxes(0,1)))))
    pl.show()

def train_pca(data):
    """
    returns: eigen vectors and eigen values
    pass in: data as 2D NumPy array
    """
    import cmath
    mn = np.mean(data, axis=0)
    # mean center the data
    data -= mn
    # calculate the covariance matrix
    C = np.cov(data.T)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eig(C)
    # sorted them by eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    assert(np.all([x.imag==0 for x in evals]))
    return evecs, np.array([cmath.polar(x)[0] for x in evals])

def flaten_img(img):
    """
    flaten a 2D img into a 1D vector 
    """
    # (h,w,c) -> (c,h,w)
    return np.array(img, dtype=np.single).T.swapaxes(1,2).ravel('C')

def img_from_vector(vector, size = None, channel_num = 3):
    """
    reshape a img from a 1D vector into 2D form
    size: int or float
    (h, w)
    """ 
    if size is None:
        size = int(sqrt(vector.size/channel_num))
    if isinstance(size, int) or isinstance(size, float):
        size = (channel_num, size, size)
    elif isinstance(tuple(size), tuple) and len(size)==2:
        size =  (channel_num,)+tuple(size)
    else:
        raise TypeError, 'size is in incorrect type'
    # (c,h,w) -> (h,w,c)
    # not sure if uint8
    return vector.reshape(size).swapaxes(0,2).swapaxes(0,1)


def resize_vimg(vimg, size, channel_num = 3):
    """
    resize a img (in 1D vector form)
    return the resized img vector 
    """
    img = img_from_vector(vimg, channel_num = channel_num).astype(np.uint8)
    img = Image.fromarray(img)\
                    .resize((size,size), Image.ANTIALIAS)
    img = flaten_img(np.asarray(img))
    return img


class FIFOCache(OrderedDict):
    """
    from [stackoverflow](http://stackoverflow.com/a/2437645/1813988)
    """
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


def get_tp_curve(scores, labels, step=-0.01, true_lable=1):
    tpfn = (labels==true_lable).sum()
    # recall from 0 to 1
    threshold_list = np.arange(1, step, step)
    positive = []
    true_positive = []
    for threshold in threshold_list:
        index = (scores[:,true_lable]>=threshold)
        positive.append(float(index.sum()))
        true_positive.append(float((labels[index]==true_lable).sum()))
    return threshold_list, np.array(positive), \
        np.array(true_positive), tpfn


def get_top_tp(scores, labels, top_list=None, true_lable=1, 
               relative=True):
    if top_list is None:
        if not relative:
            x_list = top_list
            top_list = np.arange(5,1501,5)
        else:
            num_sample = labels.shape[0]
            step = 0.005
            x_list = np.arange(step, 1.0+step, step)
            top_list = x_list*num_sample
    tpfn = (labels==true_lable).sum()
    rank = np.argsort(scores[:, true_lable], 
                      kind = 'mergesort', axis=0)[::-1]
    top_tp = np.array([(labels[rank[:top+1]]==true_lable).sum()
                        for top in top_list])
    return x_list, top_list, top_tp, tpfn


def auc(prec, recall):
    """
    area under precision-recall curve
    """
    num_points = recall.shape[0]
    step = np.abs(recall[1:] - recall[:num_points-1])
    return 0.5*((prec[1:] + prec[:num_points-1])*step).sum()


def init_dp_cache(data_path, cache_size=2, 
            data_file_name_fmt='data_batch_%08d'):
    """
    return a FIFO cache for getting batch
    """
    dp = dataprovider.\
         LabeledDataProvider(data_path, 
                             data_file_name_fmt=data_file_name_fmt)
    datadic_cache = FIFOCache(size_limit = cache_size)
    setattr(datadic_cache, 'dp', dp)
    return datadic_cache


def get_batch(datadic_cache, batchnum):
    """
    get batch from cache according to the batchnum
    """
    # load original batchdata for process
    if batchnum not in datadic_cache:
        datadic_cache[batchnum] = datadic_cache.dp.get_batch(batchnum)
    datadic = datadic_cache[batchnum]
    return datadic
