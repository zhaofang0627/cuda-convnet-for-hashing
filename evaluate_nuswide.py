'''
Created on Sep 20, 2014

@author: zhaofang
'''

import numpy as n
import numpy.random as nr
import os
from python_util.util import *
from PIL import Image
from matplotlib import pylab as pl
import matplotlib.gridspec as gridspec

def get_plottable_data(data, inner_size):
    return n.require(data.T.reshape(data.shape[1], 3, inner_size, inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

if __name__ == "__main__":
    lib_name = "cudaconvnet._ConvNet"
    libmodel = __import__(lib_name,fromlist=['_ConvNet'])
    img_size = 224
    inner_size = 224
    num_colors = 3
    inner_pixels = inner_size **2
    multiview = 0
    test = 1
    num_view = 10
    test_start = 430
    test_end = 440
    num_train_batch = 430
    num_test_batch = 11
    num_batch = num_train_batch + num_test_batch
    batch_size = 512
    num_bits = 64
    k = 100
    feat_path = "./cuda-workspace/retr-model/nuswide/metric/64bits/hash-nuswide-64bits"
    data_path = "./dataset/NUS_WIDE/batchdata"
    meta = unpickle(os.path.join(data_path, 'batches.meta'))
    label_names = n.array(meta['label_names'])
    
    hash_codes = n.empty((batch_size*num_batch, num_bits), dtype=n.float32)
    lab_mat = n.zeros((batch_size*num_batch, 81), dtype=n.float32)
    for i in xrange(num_batch):
        file_name = "data_batch_"+str(i)
        feat_dic = unpickle(os.path.join(feat_path, file_name))
        for c in xrange(batch_size):
            hash_codes[i*batch_size+c,:] = feat_dic['data'][c,:]
            lab_mat[i*batch_size+c,:] = feat_dic['labels'][c,:]
    hash_codes[hash_codes>0.5] = 1
    hash_codes[hash_codes<=0.5] = -1
    q_hash_codes = hash_codes[test_start*batch_size:(test_end+1)*batch_size,:]
    q_lab_mat = lab_mat[test_start*batch_size:(test_end+1)*batch_size,:]
    p_lab_mat = lab_mat[:num_train_batch*batch_size,:]
    nzidx = q_lab_mat.sum(axis=1).nonzero()[0]
    q_lab_mat = q_lab_mat[nzidx]
    dis = (num_bits - n.dot(q_hash_codes[nzidx], hash_codes[:num_train_batch*batch_size,:].T))*0.5
    result_idx = n.argsort(dis)
    return_topk_idx = result_idx[:,:k]
    #del dis
    #del result_idx
     
    # NDCG
    expndcg_topk = 0
    for q_rnd in xrange(return_topk_idx.shape[0]):
        r_lab_mat = p_lab_mat[return_topk_idx[q_rnd]]
        dis_sorted = dis[q_rnd][return_topk_idx[q_rnd]]
        rel_level = n.dot(q_lab_mat[q_rnd],r_lab_mat.T)
        unique_dis = list(set(dis_sorted))
        unique_dis.sort()
        ndcg_topk = 0
        NRAND = 100
        for num_rand in xrange(NRAND):
            idx = n.array(range(1,k+1))
            for i in range(len(unique_dis)):
                rand_temp = idx[dis_sorted==unique_dis[i]]
                nr.shuffle(rand_temp)
                idx[dis_sorted==unique_dis[i]] = rand_temp
            ndcg_topk += ((pow(2.0,rel_level)-1)/n.log(idx+1)).sum()
        ndcg_topk /= NRAND
         
        truth_rel_level = -n.sort(-n.dot(q_lab_mat[q_rnd],p_lab_mat.T))
        truth_rel_level = truth_rel_level[:k]

        ndcg_c = ((pow(2.0,truth_rel_level)-1)/n.log(range(2,k+2))).sum()
        ndcg_topk /= ndcg_c
        expndcg_topk += ndcg_topk
    expndcg_topk = expndcg_topk / return_topk_idx.shape[0]
    print "NDCG of top %d samples: %f," % (return_topk_idx.shape[1], expndcg_topk)
     
    # ACG
    avgacg_topk = 0
    for q_rnd in xrange(return_topk_idx.shape[0]):
        r_lab_mat = p_lab_mat[return_topk_idx[q_rnd]]
        if r_lab_mat.shape[0] > 0:
            acg_topk = n.dot(q_lab_mat[q_rnd],r_lab_mat.T).sum()
            acg_topk = acg_topk/r_lab_mat.shape[0]
            avgacg_topk += acg_topk
    avgacg_topk = avgacg_topk/return_topk_idx.shape[0]
    print "ACG of top %d samples: %f," % (return_topk_idx.shape[1], avgacg_topk)
    
    # Weighted mAP
    weighted_map = 0
    for q_rnd in xrange(return_topk_idx.shape[0]):
        r_lab_mat = p_lab_mat[result_idx[q_rnd,:5000]]
        rel = n.dot(q_lab_mat[q_rnd],r_lab_mat.T)
        rel_idx = n.where(rel>0)[0]
        if rel_idx.shape[0] == 0:
            continue
        avgacg_topk = 0
        for p in rel_idx:
            acg_topk = rel[:p+1].sum()
            acg_topk = acg_topk/(p+1)
            avgacg_topk += acg_topk
        avgacg_topk /= rel_idx.shape[0]
        weighted_map += avgacg_topk
    weighted_map = weighted_map/return_topk_idx.shape[0]
    print "Weighted mAP: %f," % weighted_map
    
    # Precision
    avgprec_topk = 0
    for q_rnd in xrange(return_topk_idx.shape[0]):
        r_lab_mat = p_lab_mat[return_topk_idx[q_rnd]]
        prec_topk = (n.dot(q_lab_mat[q_rnd],r_lab_mat.T)>0).astype(float).sum()
        prec_topk = prec_topk/return_topk_idx.shape[1]
        avgprec_topk += prec_topk
    avgprec_topk = avgprec_topk/return_topk_idx.shape[0]
    print "Precision of top %d samples: %f, \naveraged over %d queries." % (return_topk_idx.shape[1], avgprec_topk, return_topk_idx.shape[0])
    for i in xrange(1):
        query_rnd = nr.randint(return_topk_idx.shape[0])
        return_top7_idx = return_topk_idx[query_rnd,:7]
        fig = pl.figure(i, figsize=(12,9))
        gs = gridspec.GridSpec(2, 4)
        for row in xrange(2):
            for col in xrange(4):
                img_mat = n.empty((1, inner_pixels * num_colors), dtype=n.float32)
                if row * 4 + col == 0:
                    batch_idx = nzidx[query_rnd]/batch_size + test_start
                    img_idx = nzidx[query_rnd]%batch_size
                    file_dir = "data_batch_"+str(batch_idx)
                    file_name = "data_batch_"+str(batch_idx)+".0"
                else:
                    batch_idx = return_top7_idx[row * 4 + col - 1]/batch_size
                    img_idx = return_top7_idx[row * 4 + col - 1]%batch_size
                    file_dir = "data_batch_"+str(batch_idx)
                    file_name = "data_batch_"+str(batch_idx)+".0"
                rawdics = unpickle(os.path.join(data_path, file_dir, file_name))
                jpeg_strs = [rawdics['data'][img_idx]]
                libmodel.decodeJpeg(jpeg_strs, img_mat, img_size, inner_size, test, multiview)
                img_mat = get_plottable_data(img_mat.T, inner_size)
                pl.subplot(gs[row * 4 + col])
                name_list = [l+' ' for l in list(label_names[rawdics['labels'][img_idx].nonzero()[0]])]
                name_list.insert(len(name_list)/2, '\n')
                pl.xlabel(''.join(name_list))
                pl.imshow(img_mat[0,:,:,:], interpolation='lanczos')
        pl.show()
    print 'done!'
    
