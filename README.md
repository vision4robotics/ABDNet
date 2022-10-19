# Adversarial Blur-Deblur Network for Robust UAV Tracking
### Haobo Zuo, Changhong Fu, Sihang Li, Kunhan L, Yiming Li, Chen Feng
## About Code
### 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2. Please install related libraries before running this code:

      pip install -r requirements.txt
### 2. Test
Download pretrained model: [AFRTmodel](https://pan.baidu.com/s/1xXs60LeQehvCwKJo1zwzrg)(code: huat) and put it into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit.git) to set test_dataset.

       python test.py 
	        --dataset UAV123                #dataset_name
	        --snapshot snapshot/AFRTmodel.pth  # tracker_name
	
The testing result will be saved in the `results/dataset_name/tracker_name` directory.
### 3. Train
#### Prepare training datasets

Download the datasets：

[VID](https://image-net.org/challenges/LSVRC/2017/)
 
[COCO](https://cocodataset.org/#home)

[GOT-10K](http://got-10k.aitestunion.com/downloads)

[LaSOT](http://vision.cs.stonybrook.edu/~lasot/)

#### Train a model

To train the AFRT model, run `train.py` with the desired configs:

       cd tools
       python train.py

### 4. Evaluation
We provide the tracking [results](https://pan.baidu.com/s/1d8P3O9V3I6jqDqgG2LG5Ng)(code: 6q8m) of UAV123@10fps, UAV123, and UAVTrack112_L. If you want to evaluate the tracker, please put those results into `results` directory.

        python eval.py 	                          \
	         --tracker_path ./results          \ # result path
	         --dataset UAV123                  \ # dataset_name
	         --tracker_prefix 'AFRTmodel'   # tracker_name
### 5. Contact
If you have any questions, please contact me.

Haobo Zuo

Email: <1951684@tongji.edu.cn>
## Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot.git). We would like to express our sincere thanks to the contributors.
