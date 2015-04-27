'''
Created on Sep 2, 2014

@author: zhaofang

'''

from python_util.data import *
import numpy.random as nr
import numpy as n
import random as r
from random import shuffle
from time import time
from threading import Thread
from math import sqrt
import sys
from PIL import Image
from StringIO import StringIO
from time import time
import itertools as it
import copy
    
class RetrBatchLoaderThread(Thread):
    def __init__(self, dp, batch_num, label_offset, list_out):
        Thread.__init__(self)
        self.list_out = list_out
        self.label_offset = label_offset
        self.dp = dp
        self.batch_num = batch_num
        
    @staticmethod
    def load_jpeg_batch(rawdics, dp, label_offset):
        if type(rawdics) != list:
            rawdics = [rawdics]
        if len(rawdics) == 3:
            nc_total = len(rawdics[0]['data'])
    
            jpeg_strs = rawdics[0]['data']
            labels = rawdics[0]['labels']
            jpeg1_strs = rawdics[1]['data']
            labels1 = rawdics[1]['labels']
            jpeg2_strs = rawdics[2]['data']
            labels2 = rawdics[2]['labels']
            
            img_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
            lab_mat = n.vstack(labels)
            
            img1_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
            lab1_mat = n.vstack(labels1)

            img2_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
            lab2_mat = n.vstack(labels2)
            
            topic_diff = n.zeros((nc_total, 1), dtype=n.float32)
            
            dp.convnet.libmodel.decodeJpeg(jpeg_strs, img_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview) 
            dp.convnet.libmodel.decodeJpeg(jpeg1_strs, img1_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)       
            dp.convnet.libmodel.decodeJpeg(jpeg2_strs, img2_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
            
            for c in xrange(nc_total):
                #topic_diff[c,0] = (lab_mat[c,:]*lab1_mat[c,:]).sum() - (lab_mat[c,:]*lab2_mat[c,:]).sum()
                topic_diff[c,0] = rawdics[1]['label_codes'][c]
            #topic_diff = n.tile(topic_diff, (dp.data_mult, 1))
    
            return {'data': img_mat[:nc_total * dp.data_mult,:],
                    'labmat': lab_mat[:nc_total * dp.data_mult,:],
                    'data1': img1_mat[:nc_total * dp.data_mult,:],
                    'data2': img2_mat[:nc_total * dp.data_mult,:],
                    'topicdiff': topic_diff[:nc_total,:]}
        else:
            nc_total = len(rawdics[0]['data'])
    
            jpeg_strs = rawdics[0]['data']
            labels = rawdics[0]['labels']
            
            img_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
            lab_mat = n.vstack(labels)
            
            topic_diff = n.zeros((nc_total, 1), dtype=n.float32)
            
            dp.convnet.libmodel.decodeJpeg(jpeg_strs, img_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
    
            return {'data': img_mat[:nc_total * dp.data_mult,:],
                    'labmat': lab_mat[:nc_total * dp.data_mult,:],
                    'topicdiff': topic_diff[:nc_total,:]}
    
    def run(self):
        rawdics = self.dp.get_batch(self.batch_num)
        p = RetrBatchLoaderThread.load_jpeg_batch(rawdics,
                                                  self.dp,
                                                  self.label_offset)
        self.list_out.append(p)


class NUSWideDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = int(sqrt(self.batch_meta['num_vis'] / self.num_colors))
        self.mini = dp_params['minibatch_size']
        self.inner_size = dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.img_size
        self.inner_pixels = self.inner_size **2
        self.border_size = (self.img_size - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test #or (dp_params['write_features'] == 'hash')
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.data_mean = self.batch_meta['data_mean'].reshape((1,3*self.inner_size**2)).astype(n.single)
        self.batch_size = self.batch_meta['batch_size']
        # tran range: 0-859; test rang:860-881
        self.num_batch = 430 #430 #1620 #len(self.batch_range)
        self.train_size = self.batch_size * self.num_batch
        self.label_offset = 0 if 'label_offset' not in self.batch_meta else self.batch_meta['label_offset']
        self.scalar_mean = dp_params['scalar_mean'] 
        # Maintain pointers to previously-returned data matrices so they don't get garbage collected.
        self.data = [None, None] # These are pointers to previously-returned data matrices

        self.loader_thread = None
        self.convnet = dp_params['convnet']
            
        self.batches_generated, self.loaders_started = 0, 0

        if self.scalar_mean >= 0:
            self.data_mean = self.scalar_mean
            
    def showimg(self, img):
        from matplotlib import pylab as pl
        pixels = img.shape[0] / 3
        size = int(sqrt(pixels))
        img = img.reshape((3,size,size)).swapaxes(0,2).swapaxes(0,1)
        pl.imshow(img, interpolation='nearest')
        pl.show()
            
    def get_data_dims(self, idx=0):
        if idx == 0 or idx == 1 or idx == 2:
            return self.inner_size**2 * 3
#         if idx == 3 or idx == 4 or idx == 5:
#             return self.get_num_classes()
        #return self.num_topics
        return 1

    def start_loader(self, batch_idx):
        self.load_data = []
        self.loader_thread = RetrBatchLoaderThread(self,
                                                   self.batch_range[batch_idx],
                                                   self.label_offset,
                                                   self.load_data)
        self.loader_thread.start()

    def set_labels(self, datadic):
        pass
    
    def get_data_from_loader(self):
        if self.loader_thread is None:
            self.start_loader(self.batch_idx)
            self.loader_thread.join()
            self.data[self.d_idx] = self.load_data[0]

            self.start_loader(self.get_next_batch_idx())
        else:
            # Set the argument to join to 0 to re-enable batch reuse
            self.loader_thread.join()   
            if not self.loader_thread.is_alive():
                self.data[self.d_idx] = self.load_data[0]
                self.start_loader(self.get_next_batch_idx())
            #else:
            #    print "Re-using batch"
        self.advance_batch()
    
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        epoch, batchnum = self.curr_epoch, self.curr_batchnum

        self.get_data_from_loader()

        # Subtract mean
        self.data[self.d_idx]['data'] -= self.data_mean
        self.data[self.d_idx]['data1'] -= self.data_mean
        self.data[self.d_idx]['data2'] -= self.data_mean
        
        self.batches_generated += 1
        
#         return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['topic'].T]
        
        return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['data1'].T, self.data[self.d_idx]['data2'].T, 
                                 self.data[self.d_idx]['topicdiff'].T, self.data[self.d_idx]['labmat'].T]
    
    def get_batch(self, batch_num):
        qtgt = {'data': [], 'label_codes': [], 'labels': [], 'imageId': []}
        tgts = []
        tgts += [{'data': [], 'label_codes': [], 'labels': [], 'imageId': []}]
        tgts += [{'data': [], 'label_codes': [], 'labels': [], 'imageId': []}]
        tgts += [{'data': [], 'label_codes': [], 'labels': [], 'imageId': []}]
        #balance
        for i in xrange(self.batch_size):
            d_mat = [self.convnet.data_list[batch_num*self.batch_size + i]]
            l_mat = [self.convnet.label_mat[batch_num*self.batch_size + i]]
            rep_num = int((l_mat[0]*self.convnet.cls_weight).max())
            qtgt['data'] += d_mat*rep_num
            qtgt['labels'] += l_mat*rep_num
        qtgt['labels'] = n.vstack(qtgt['labels'])
        
        rel = n.dot(qtgt['labels'], self.convnet.label_mat[:self.train_size].T)
        
        for m in xrange(rel.shape[0]):
            idxmat2 = n.where(rel[m]==0)[0]
            ind_p2 = idxmat2[nr.randint(idxmat2.shape[0])]
            currel_max = rel[m].max()
            if currel_max > 1:
                currel = copy.copy(rel[m])
                idxmat1 = n.where(currel==currel_max)[0]
                ind_p1a = idxmat1[nr.randint(idxmat1.shape[0])]
                currel[idxmat1] = 0
                idxmat1 = currel.nonzero()[0]
                ind_p1b = idxmat1[nr.randint(idxmat1.shape[0])]
                
                tgts[0]['data'] += [qtgt['data'][m]]*3
                tgts[0]['labels'] += [qtgt['labels'][m]]*3
                ndcg_c = (pow(2.0,rel[m][ind_p1a])-1)/n.log(2) + (pow(2.0,rel[m][ind_p1b])-1)/n.log(3)
                
                tgts[1]['data'] += [self.convnet.data_list[ind_p1a]]*2
                tgts[1]['labels'] += [self.convnet.label_mat[ind_p1a]]*2
                
                tgts[2]['data'] += [self.convnet.data_list[ind_p1b]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p1b]]
                tgts[2]['data'] += [self.convnet.data_list[ind_p2]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p2]]
                
                tgts[1]['data'] += [self.convnet.data_list[ind_p1b]]
                tgts[1]['labels'] += [self.convnet.label_mat[ind_p1b]]

                tgts[2]['data'] += [self.convnet.data_list[ind_p2]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p2]]
                
                tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1a])-pow(2.0,rel[m][ind_p1b]))/ndcg_c]
                tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1a])-pow(2.0,rel[m][ind_p2]))/ndcg_c]
                tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1b])-pow(2.0,rel[m][ind_p2]))/ndcg_c]
            else:
                idxmat1 = rel[m].nonzero()[0]
                ind_p1 = idxmat1[nr.randint(idxmat1.shape[0])]
                tgts[0]['data'] += [qtgt['data'][m]]
                tgts[0]['labels'] += [qtgt['labels'][m]]
                
                tgts[1]['data'] += [self.convnet.data_list[ind_p1]]
                tgts[1]['labels'] += [self.convnet.label_mat[ind_p1]]
                
                tgts[2]['data'] += [self.convnet.data_list[ind_p2]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p2]]
                
                tgts[1]['label_codes'] += [n.log(2)]

        return tgts


class NUSWideTestDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 3
        self.img_size = int(sqrt(self.batch_meta['num_vis'] / self.num_colors))
        self.mini = dp_params['minibatch_size']
        self.inner_size = dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.img_size
        self.inner_pixels = self.inner_size **2
        self.border_size = (self.img_size - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.data_mean = self.batch_meta['data_mean'].reshape((1,3*self.inner_size**2)).astype(n.single)
        self.batch_size = self.batch_meta['batch_size']
        self.label_offset = 0 if 'label_offset' not in self.batch_meta else self.batch_meta['label_offset']
        self.scalar_mean = dp_params['scalar_mean'] 
        # Maintain pointers to previously-returned data matrices so they don't get garbage collected.
        self.data = [None, None] # These are pointers to previously-returned data matrices

        self.loader_thread = None
        self.convnet = dp_params['convnet']
            
        self.batches_generated, self.loaders_started = 0, 0

        if self.scalar_mean >= 0:
            self.data_mean = self.scalar_mean
            
    def showimg(self, img):
        from matplotlib import pylab as pl
        pixels = img.shape[0] / 3
        size = int(sqrt(pixels))
        img = img.reshape((3,size,size)).swapaxes(0,2).swapaxes(0,1)
        pl.imshow(img, interpolation='nearest')
        pl.show()
            
    def get_data_dims(self, idx=0):
        if idx == 0 or idx == 1 or idx == 2:
            return self.inner_size**2 * 3
#         if idx == 3 or idx == 4 or idx == 5:
#             return self.get_num_classes()
        #return self.num_topics
        return 1

    def start_loader(self, batch_idx):
        self.load_data = []
        self.loader_thread = RetrBatchLoaderThread(self,
                                                   self.batch_range[batch_idx],
                                                   self.label_offset,
                                                   self.load_data)
        self.loader_thread.start()

    def set_labels(self, datadic):
        pass
    
    def get_data_from_loader(self):
        if self.loader_thread is None:
            self.start_loader(self.batch_idx)
            self.loader_thread.join()
            self.data[self.d_idx] = self.load_data[0]

            self.start_loader(self.get_next_batch_idx())
        else:
            # Set the argument to join to 0 to re-enable batch reuse
            self.loader_thread.join()
            if not self.loader_thread.is_alive():
                self.data[self.d_idx] = self.load_data[0]
                self.start_loader(self.get_next_batch_idx())
            #else:
            #    print "Re-using batch"
        self.advance_batch()
    
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        epoch, batchnum = self.curr_epoch, self.curr_batchnum

        self.get_data_from_loader()

        # Subtract mean
        self.data[self.d_idx]['data'] -= self.data_mean
        
        self.batches_generated += 1
        
        return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['data'].T, self.data[self.d_idx]['data'].T, 
                                 self.data[self.d_idx]['topicdiff'].T, self.data[self.d_idx]['labmat'].T]
    
    def get_batch(self, batch_num):
        fname = self.get_data_file_name(batch_num)
        if os.path.isdir(fname): # batch in sub-batches
            sub_batches = sorted(os.listdir(fname), key=alphanum_key)
            num_sub_batches = len(sub_batches)
            tgts = [[] for i in xrange(num_sub_batches)]

            for i in xrange(num_sub_batches):
                tgts[i] += [unpickle(os.path.join(fname, sub_batches[i]))]
            # Only for the case of num_sub_batches == 1
            return tgts[0]
        return unpickle(self.get_data_file_name(batch_num))
        
    