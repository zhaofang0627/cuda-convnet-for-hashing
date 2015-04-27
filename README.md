# Deep semantic ranking based hashing
This algorithm is described in [Deep Semantic Ranking Based Hashing for Multi-Label Image Retrieval](http://arxiv.org/abs/1501.06272).
## Main Classes And Functions
- LocRankCostLayer in [cudaconvnet/src/layer.cu](https://github.com/zhaofang0627/cuda-convnet-for-hashing/blob/master/cudaconvnet/src/layer.cu) implements ranking cost layer

- computeLocRankCost in [cudaconvnet/src/layer_kernels.cu](https://github.com/zhaofang0627/cuda-convnet-for-hashing/blob/master/cudaconvnet/src/layer_kernels.cu) computes ranking cost

- computeLocRankGrad in [cudaconvnet/src/layer_kernels.cu](https://github.com/zhaofang0627/cuda-convnet-for-hashing/blob/master/cudaconvnet/src/layer_kernels.cu) computes gradients of ranking cost

- [retrdata.py](https://github.com/zhaofang0627/cuda-convnet-for-hashing/blob/master/retrdata.py) implements data providers for training and test

- [evaluate_nuswide.py](https://github.com/zhaofang0627/cuda-convnet-for-hashing/blob/master/evaluate_nuswide.py) computes ndcg, acg and weighted mAP.
