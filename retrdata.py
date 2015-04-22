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
            #lab_mat = n.zeros((nc_total, dp.get_num_classes()), dtype=n.float32)
            lab_mat = n.vstack(labels)
            #topic_mat = n.zeros((nc_total, dp.num_topics), dtype=n.float32)
            img1_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
            #lab1_mat = n.zeros((nc_total, dp.get_num_classes()), dtype=n.float32)
            lab1_mat = n.vstack(labels1)
            #topic1_mat = n.zeros((nc_total, dp.num_topics), dtype=n.float32)
            img2_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
            #lab2_mat = n.zeros((nc_total, dp.get_num_classes()), dtype=n.float32)
            lab2_mat = n.vstack(labels2)
            #topic2_mat = n.zeros((nc_total, dp.num_topics), dtype=n.float32)
            topic_diff = n.zeros((nc_total, 1), dtype=n.float32)
            
            dp.convnet.libmodel.decodeJpeg(jpeg_strs, img_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
            #lab_vec = n.tile(n.asarray([(l[nr.randint(len(l))] if len(l) > 0 else -1) + label_offset for l in labels], dtype=n.single).reshape((nc_total, 1)), (dp.data_mult,1))
#             for c in xrange(nc_total):
#                 lab_mat[c,:] = labels[c]
#                 #topic_mat[c,:] = (rawdics[0]['label_codes'][c]>0.5).astype(float)
#                 #lab_mat[c, [z + label_offset for z in labels[c]]] = 1
#             lab_mat = n.tile(lab_mat, (dp.data_mult, 1))
            
            dp.convnet.libmodel.decodeJpeg(jpeg1_strs, img1_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
            #lab1_vec = n.tile(n.asarray([(l[nr.randint(len(l))] if len(l) > 0 else -1) + label_offset for l in labels1], dtype=n.single).reshape((nc_total, 1)), (dp.data_mult,1))
#             for c in xrange(nc_total):
#                 lab1_mat[c,:] = labels1[c]
#                 #topic1_mat[c,:] = (rawdics[1]['label_codes'][c]>0.5).astype(float)
#                 #lab1_mat[c, [z + label_offset for z in labels1[c]]] = 1
#             lab1_mat = n.tile(lab1_mat, (dp.data_mult, 1))
            
            dp.convnet.libmodel.decodeJpeg(jpeg2_strs, img2_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
            #lab2_vec = n.tile(n.asarray([(l[nr.randint(len(l))] if len(l) > 0 else -1) + label_offset for l in labels2], dtype=n.single).reshape((nc_total, 1)), (dp.data_mult,1))
#             for c in xrange(nc_total):
#                 lab2_mat[c,:] = labels2[c]
#                 #topic2_mat[c,:] = (rawdics[2]['label_codes'][c]>0.5).astype(float)
#                 #lab2_mat[c, [z + label_offset for z in labels2[c]]] = 1
#             lab2_mat = n.tile(lab2_mat, (dp.data_mult, 1))
            
            for c in xrange(nc_total):
                #topic_diff[c,0] = (topic_mat[c,:]*topic1_mat[c,:]).sum() - (topic_mat[c,:]*topic2_mat[c,:]).sum()
                topic_diff[c,0] = (lab_mat[c,:]*lab1_mat[c,:]).sum() - (lab_mat[c,:]*lab2_mat[c,:]).sum()
                #topic_diff[c,0] = rawdics[1]['label_codes'][c]
            #topic_diff = n.tile(topic_diff, (dp.data_mult, 1))
    
            return {'data': img_mat[:nc_total * dp.data_mult,:],
                    'labmat': lab_mat[:nc_total * dp.data_mult,:],
                    #'topic': topic_mat[:nc_total,:],
                    'data1': img1_mat[:nc_total * dp.data_mult,:],
                    #'labmat1': lab1_mat[:nc_total * dp.data_mult,:],
                    #'topic1': topic1_mat[:nc_total,:],
                    'data2': img2_mat[:nc_total * dp.data_mult,:],
                    #'labmat2': lab2_mat[:nc_total * dp.data_mult,:],
                    #'topic2': topic2_mat[:nc_total,:],
                    'topicdiff': topic_diff[:nc_total,:]}
        else:
            nc_total = len(rawdics[0]['data'])
    
            jpeg_strs = rawdics[0]['data']
            labels = rawdics[0]['labels']
            
            img_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
            lab_mat = n.zeros((nc_total, dp.get_num_classes()), dtype=n.float32)
            topic_mat = n.zeros((nc_total, dp.num_topics), dtype=n.float32)
            
            dp.convnet.libmodel.decodeJpeg(jpeg_strs, img_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
            #lab_vec = n.tile(n.asarray([(l[nr.randint(len(l))] if len(l) > 0 else -1) + label_offset for l in labels], dtype=n.single).reshape((nc_total, 1)), (dp.data_mult,1))
            for c in xrange(nc_total):
                lab_mat[c,:] = labels[c]
                #topic_mat[c,:] = (rawdics[0]['label_codes'][c]>0.5).astype(float)
                #lab_mat[c, [z + label_offset for z in labels[c]]] = 1
            lab_mat = n.tile(lab_mat, (dp.data_mult, 1))
    
            return {'data': img_mat[:nc_total * dp.data_mult,:],
                    'labmat': lab_mat[:nc_total * dp.data_mult,:],
                    'topic': topic_mat[:nc_total,:]}
    
    def run(self):
        rawdics = self.dp.get_batch(self.batch_num)
        p = RetrBatchLoaderThread.load_jpeg_batch(rawdics,
                                                  self.dp,
                                                  self.label_offset)
        self.list_out.append(p)

class MIRFlickrDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_topics = 64
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
        self.num_batch = 1934 #1620 #len(self.batch_range)
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
#         return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['data1'].T, self.data[self.d_idx]['data2'].T, 
#                                  self.data[self.d_idx]['topic'].T, self.data[self.d_idx]['topic1'].T, self.data[self.d_idx]['topic2'].T, 
#                                  self.data[self.d_idx]['labmat'].T, self.data[self.d_idx]['labmat1'].T, self.data[self.d_idx]['labmat2'].T]
    
    def get_batch(self, batch_num):
        fname = self.get_data_file_name(batch_num)
        if os.path.isdir(fname): # batch in sub-batches
            sub_batches = sorted(os.listdir(fname), key=alphanum_key)
            num_sub_batches = len(sub_batches)
            tgts = [[] for i in xrange(num_sub_batches)]

            for i in xrange(num_sub_batches):
                tgts[i] += [unpickle(os.path.join(fname, sub_batches[i]))]
                tgts[i] += [{'data': [], 'labels': [], 'imageId': []}]
                tgts[i] += [{'data': [], 'labels': [], 'imageId': []}]
                
                batch1_idx = nr.randint(self.num_batch)
                while True:
                    batch2_idx = nr.randint(self.num_batch)
                    if not batch2_idx == batch1_idx:
                        break
                fname1 = self.get_data_file_name(batch1_idx)
                fname2 = self.get_data_file_name(batch2_idx)
                sub_batches1 = sorted(os.listdir(fname1), key=alphanum_key)
                sub_batches2 = sorted(os.listdir(fname2), key=alphanum_key)
                dic = unpickle(os.path.join(fname1, sub_batches1[0]))
                dic_l = zip(dic['data'], dic['imageId'])
                topic_l = dic['labels']
                dic = unpickle(os.path.join(fname2, sub_batches2[0]))
                dic_l += zip(dic['data'], dic['imageId'])
                topic_l += dic['labels']
                topic_p = (n.vstack(topic_l)>0.5).astype(float)
                topic_q = (n.vstack(tgts[i][0]['labels'])>0.5).astype(float)
                rel = n.dot(topic_q, topic_p.T)
                #print n.where(rel.sum(axis=1)==0)[0]
                rel_norm1 = rel / n.tile(rel.sum(axis=1).reshape(rel.shape[0],1), (1,rel.shape[1]))
                rel += 100
                rel_norm2 = rel / n.tile(rel.sum(axis=1).reshape(rel.shape[0],1), (1,rel.shape[1]))
                ind_p1 = []
                ind_p2 = []
                for m in xrange(rel.shape[0]):
                    ind_p1.append(n.random.multinomial(1,rel_norm1[m],size=1).reshape(rel.shape[1],).nonzero()[0][0])
                    ind_p2.append(n.random.multinomial(1,rel_norm2[m],size=1).reshape(rel.shape[1],).nonzero()[0][0])
                tgts[i][1]['data'] = list(dic_l[ind][0] for ind in ind_p1)
                tgts[i][1]['labels'] = list(topic_l[ind] for ind in ind_p1)
                #tgts[i][1]['label_codes'] = list(dic_l[ind][1] for ind in ind_p1)
                tgts[i][1]['imageId'] = list(dic_l[ind][1] for ind in ind_p1)
                tgts[i][2]['data'] = list(dic_l[ind][0] for ind in ind_p2)
                tgts[i][2]['labels'] = list(topic_l[ind] for ind in ind_p2)
                #tgts[i][2]['label_codes'] = list(dic_l[ind][1] for ind in ind_p2)
                tgts[i][2]['imageId'] = list(dic_l[ind][1] for ind in ind_p2)
            # Only for the case of num_sub_batches == 1
            return tgts[0]
        return unpickle(self.get_data_file_name(batch_num))
        
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
        mean = self.data_mean.reshape((data.shape[0],1)) if data.flags.f_contiguous or self.scalar_mean else self.data_mean.reshape((data.shape[0],1))
        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)


class MIRFlickrTransferDataProvider(MIRFlickrDataProvider):
    def get_data_dims(self, idx=0):
        if idx == 0: #or idx == 1 or idx == 2:
            return self.inner_size**2 * 3
        if idx == 1:
            return self.get_num_classes()
        return self.num_topics
        #return 1
        
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        epoch, batchnum = self.curr_epoch, self.curr_batchnum

        self.get_data_from_loader()

        # Subtract mean
        self.data[self.d_idx]['data'] -= self.data_mean
        
        self.batches_generated += 1
        
        return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['labmat'].T, self.data[self.d_idx]['topic'].T]
        
#         return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['data1'].T, self.data[self.d_idx]['data2'].T, 
#                                  self.data[self.d_idx]['topicdiff'].T, self.data[self.d_idx]['labmat'].T]

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


class NUSWideDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_topics = 64
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
        self.batch_size = self.batch_meta['batch_size']/2
        # tran range: 0-859; test rang:860-881
        self.num_batch = 52*2 #430 #1620 #len(self.batch_range)
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
#         return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['data1'].T, self.data[self.d_idx]['data2'].T, 
#                                  self.data[self.d_idx]['topic'].T, self.data[self.d_idx]['topic1'].T, self.data[self.d_idx]['topic2'].T, 
#                                  self.data[self.d_idx]['labmat'].T, self.data[self.d_idx]['labmat1'].T, self.data[self.d_idx]['labmat2'].T]
    
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
        #print n.where(rel.sum(axis=1)==0)[0]
        #rel_norm1 = rel.astype(n.float64) / n.tile(rel.sum(axis=1).reshape(rel.shape[0],1), (1,rel.shape[1]))
        #rel += 1
        #rel_norm2 = rel.astype(n.float64) / n.tile(rel.sum(axis=1).reshape(rel.shape[0],1), (1,rel.shape[1]))
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
                #idxmat = n.delete(range(currel.shape[0]),n.concatenate((idxmat1,idxmat2)))
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
#                 tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1a])-pow(2.0,rel[m][ind_p1b]))/pow(2.0,rel[m][ind_p1a])]
#                 tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1a])-pow(2.0,rel[m][ind_p2]))/pow(2.0,rel[m][ind_p1a])]
#                 tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1b])-pow(2.0,rel[m][ind_p2]))/pow(2.0,rel[m][ind_p1a])]
            else:
                idxmat1 = rel[m].nonzero()[0]
                ind_p1 = idxmat1[nr.randint(idxmat1.shape[0])]
                tgts[0]['data'] += [qtgt['data'][m]]
                tgts[0]['labels'] += [qtgt['labels'][m]]
                #ndcg_c = (pow(2.0,rel[ind_p1])-1)/n.log(2)
                tgts[1]['data'] += [self.convnet.data_list[ind_p1]]
                tgts[1]['labels'] += [self.convnet.label_mat[ind_p1]]
                
                tgts[2]['data'] += [self.convnet.data_list[ind_p2]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p2]]
                
                tgts[1]['label_codes'] += [n.log(2)]
#                 tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1])-pow(2.0,rel[m][ind_p2]))/pow(2.0,rel[m][ind_p1])]

        # Only for the case of num_sub_batches == 1
        return tgts
        
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
        mean = self.data_mean.reshape((data.shape[0],1)) if data.flags.f_contiguous or self.scalar_mean else self.data_mean.reshape((data.shape[0],1))
        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)


class NUSWideOrderDataProvider(NUSWideDataProvider):
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
        #print n.where(rel.sum(axis=1)==0)[0]
        #rel_norm1 = rel.astype(n.float64) / n.tile(rel.sum(axis=1).reshape(rel.shape[0],1), (1,rel.shape[1]))
        #rel += 1
        #rel_norm2 = rel.astype(n.float64) / n.tile(rel.sum(axis=1).reshape(rel.shape[0],1), (1,rel.shape[1]))
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
                #idxmat = n.delete(range(currel.shape[0]),n.concatenate((idxmat1,idxmat2)))
                ind_p1b = idxmat1[nr.randint(idxmat1.shape[0])]
                tgts[0]['data'] += [qtgt['data'][m]]*4
                tgts[0]['labels'] += [qtgt['labels'][m]]*4
                ndcg_c = (pow(2.0,rel[m][ind_p1a])-1)/n.log(2) + (pow(2.0,rel[m][ind_p1b])-1)/n.log(3)
                
                tgts[1]['data'] += [self.convnet.data_list[ind_p1a]]*2
                tgts[1]['labels'] += [self.convnet.label_mat[ind_p1a]]*2
                tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1a])-1)/ndcg_c]*2
                
                tgts[2]['data'] += [self.convnet.data_list[ind_p1b]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p1b]]
                tgts[2]['data'] += [self.convnet.data_list[ind_p2]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p2]]
                
                tgts[1]['data'] += [self.convnet.data_list[ind_p1b]]*2
                tgts[1]['labels'] += [self.convnet.label_mat[ind_p1b]]*2
                tgts[1]['label_codes'] += [(pow(2.0,rel[m][ind_p1b])-1)/ndcg_c]*2
                
                tgts[2]['data'] += [self.convnet.data_list[ind_p1a]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p1a]]
                tgts[2]['data'] += [self.convnet.data_list[ind_p2]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p2]]
            else:
                idxmat1 = rel[m].nonzero()[0]
                ind_p1 = idxmat1[nr.randint(idxmat1.shape[0])]
                tgts[0]['data'] += [qtgt['data'][m]]
                tgts[0]['labels'] += [qtgt['labels'][m]]
                ndcg_c = (pow(2.0,rel[m][ind_p1])-1)/n.log(2)
                
                tgts[1]['data'] += [self.convnet.data_list[ind_p1]]
                tgts[1]['labels'] += [self.convnet.label_mat[ind_p1]]
                tgts[1]['label_codes'] += [n.log(2)]
                
                tgts[2]['data'] += [self.convnet.data_list[ind_p2]]
                tgts[2]['labels'] += [self.convnet.label_mat[ind_p2]]
        # Only for the case of num_sub_batches == 1
        return tgts


class NUSWideTestDataProvider(NUSWideDataProvider):
    def get_batch(self, batch_num):
        tgts = []

        tgts += [{'data': [], 'label_codes': [], 'labels': [], 'imageId': []}]
        tgts += [{'data': [], 'label_codes': [], 'labels': [], 'imageId': []}]
        tgts += [{'data': [], 'label_codes': [], 'labels': [], 'imageId': []}]
        #balance
        for i in xrange(self.batch_size):
            d_mat = [self.convnet.data_list[batch_num*self.batch_size + i]]
            l_mat = [self.convnet.label_mat[batch_num*self.batch_size + i]]
            rep_num = int(l_mat[0].sum()>0)
            tgts[0]['data'] += d_mat*rep_num
            tgts[0]['labels'] += l_mat*rep_num
        tgts[0]['labels'] = n.vstack(tgts[0]['labels'])
        
        rel = n.dot(tgts[0]['labels'], self.convnet.label_mat.T)
        #print n.where(rel.sum(axis=1)==0)[0]
        #rel_norm1 = rel.astype(n.float64) / n.tile(rel.sum(axis=1).reshape(rel.shape[0],1), (1,rel.shape[1]))
        #rel += 1
        #rel_norm2 = rel.astype(n.float64) / n.tile(rel.sum(axis=1).reshape(rel.shape[0],1), (1,rel.shape[1]))
        ind_p1 = []
        ind_p2 = []
        for m in xrange(rel.shape[0]):
            idxmat = rel[m].nonzero()[0]
            ind_p1.append(idxmat[nr.randint(idxmat.shape[0])])
            #idxmat = n.where(rel[m]==0)[0]
            #ind_p2.append(idxmat[nr.randint(idxmat.shape[0])])
            ind_p2.append(nr.randint(rel[m].shape[0]))
            #ind_p1.append(n.random.multinomial(1,rel_norm1[m],size=1).argmax())
            #ind_p2.append(n.random.multinomial(1,rel_norm2[m],size=1).argmax())
        tgts[1]['data'] = [self.convnet.data_list[ind] for ind in ind_p1]
        tgts[1]['labels'] = self.convnet.label_mat[ind_p1]
        #tgts[1]['label_codes'] = list(dic_l[ind][1] for ind in ind_p1)
        #tgts[1]['imageId'] = list(dic_l[ind][2] for ind in ind_p1)
        tgts[2]['data'] = [self.convnet.data_list[ind] for ind in ind_p2]
        tgts[2]['labels'] = self.convnet.label_mat[ind_p2]
        #tgts[2]['label_codes'] = list(dic_l[ind][1] for ind in ind_p2)
        #tgts[2]['imageId'] = list(dic_l[ind][2] for ind in ind_p2)
        # Only for the case of num_sub_batches == 1
        return tgts


class USAADataProvider(LabeledDataProvider):
        