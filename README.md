# MCCNet
   This project provides the code and results for 'Multi-Content Complementation Network for Salient Object Detection in Optical Remote Sensing Images', IEEE TGRS, accepted, 2021.
 
 
# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/MCCNet/blob/main/images/MCCNet.png">
   </div>
   
# Multi-Content Complementation Module (MCCM)
   <div align=center>
   <img src=https://github.com/MathLee/MCCNet/blob/main/images/MCCM.png width=52% />
   </div> 
   
   
# Requirements
   python2.7
   
   pytorch 0.4.0
   

# Usage

Modify the pathes of [VGG backbone](https://pan.baidu.com/s/1YQxKZ-y2C4EsqrgKNI7qrw) (code: ego5) and datasets, then run train_MCCNet.py or test_MCCNet.py.


# Saliency maps
   We provide saliency maps of [all compared methods](https://pan.baidu.com/s/1TP6An1VWygGUy4uvojL0bg) (code: 5np3) and [our MCCNet](https://pan.baidu.com/s/10JIKL2Q48RvBGeT2pmPfDA) (code: 3pvq) on ORSSD and EORSSD datasets.
   
   In addition, we also provide saliency maps of our MCCNet on the recently published [ORSI-4199](https://pan.baidu.com/s/1dz-GeELIqMdzKlPvzETixA) (code: 413m) dataset.
   
   ![Image](https://github.com/MathLee/MCCNet/blob/main/images/table.png)
   
   
# Pre-trained model
[ORSSD](https://pan.baidu.com/s/1LdUE8F11r61r8wk3Y9wPLA) (code: awqr)

[EORSSD](https://pan.baidu.com/s/14LrEt1LW5QmZvkhsgbKgfg) (code: wm3p)

[ORSI-4199](https://pan.baidu.com/s/1hmANQp9cslyPuDE-3NlqAg) (code: 336a)

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](http://dpfan.net/d3netbenchmark/) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2021_MCCNet,
                author = {Gongyang Li and Zhi Liu and Weisi Lin and Haibin Ling},
                title = {Multi-Content Complementation Network for Salient Object Detection in Optical Remote Sensing Images},
                journal = {IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING},
                year = {2021},
                volume = {},
                pages = {},}
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
