# Source code of the method LWAU shown in "Layer-Wise Adaptive Updating for Few-Shot Image Classification"

URL: https://arxiv.org/abs/2007.08129

LWAU is evaluated with two backbones:Conv-4 and ResNet12  

## Performance
The performances on MiniImagenet is shown as the following Table.  
The upper part of the table shows the meta-learning based few-shot learning methods with Conv-4 backbone.  
The lower part shows the methods with ResNet12 backbone.  
![](https://github.com/qyxqyx/LWAU/raw/master/performance_miniimagenet.png)  

## Learning efficiency
Except for improving the meta-learner's few-shot learning performance, LWAU can also greatly speed up the meta-learner's learning on the support set.  
This is because when learning on novel few-shot learning tasks, frezing LWAU meta-learner's bottom layers will not damage the meta-learner's performance.  
The comparison between LWAU and MAML when their bottom layers are frozen is shown in the following Figure.  
![](https://github.com/qyxqyx/LWAU/raw/master/freeze.png)  

## Spare representation
At last, LWAU extracts sparser image representations.  
![](https://github.com/qyxqyx/LWAU/raw/master/representation.png)  

## Please refer to our paper to get more detail of LWAU.  

If LWAU is helpful for your work, please cite our paper. Thanks!  

@article{qin2020layer,  
  title={Layer-Wise Adaptive Updating for Few-Shot Image Classification},    
  author={Qin, Yunxiao and Zhang, Weiguo and Wang, Zezheng and Zhao, Chenxu and Shi, Jingping},  
  journal={arXiv preprint arXiv:2007.08129},  
  year={2020}  
}

