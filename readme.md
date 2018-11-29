[PyTorch](http://pytorch.org/) Project for NeurIPS 2018 paper: 

**Connectionist Temporal Classification with Maximum Entropy Regularization**
Hu Liu, Sheng Jin and Changshui Zhang. *Neural Information Processing Systems (NeurIPS)*, 2018. 

## Abstract
Connectionist Temporal Classification (CTC) is an objective function for end-to-end sequence learning, which adopts dynamic programming algorithms to directly learn the mapping between sequences. CTC has shown promising results in many sequence learning applications including speech recognition and scene text recognition. However, CTC tends to produce highly peaky and overconfident distributions, which is a symptom of overfitting. To remedy this, we propose a regularization method based on maximum conditional entropy which penalizes peaky distributions and encourages exploration. We also introduce an entropy-based pruning method to dramatically reduce the number of CTC feasible paths by ruling out unreasonable alignments. Experiments on scene text recognition show that our proposed methods consistently improve over the CTC baseline without the need to adjust training settings. Code has been made publicly available at: https://github.com/liuhu-bigeye/enctc.crnn.

## Requirements
* python2
* pytorch
* lmdb

## Datasets
* ICDAR-2003 (IC03) \[2\]
* ICDAR-2013 (IC13) \[3\]
* IIIT5k-word (IIIT5k) \[4\]
* Street View Text (SVT) \[5\]
* Synth90k \[6\]
* Synth5K (randomly sampled from Synth90k) 

## Usage
### Train
`zsh shs/seg_ent_fb/seg_5k.sh`
### Test
`CUDA_VISIBLE_DEVICES=0 python test.py --crnn_path model_dir --valroot data/svt1/testset.lmdb`

## Citation
If you use this code in your project, please consider citing this paper.
```
@inproceedings{liu2018connectionist,
  title={Connectionist Temporal Classification with Maximum Entropy Regularization},
  author={Liu, Hu and Jin, Sheng and Zhang, Changshui},
  booktitle={Advances in Neural Information Processing Systems},
  pages = {837--847},
  year={2018}
}
```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/liuhu-bigeye/enctc.crnn/issues).

## References

\[1\] B. Shi, X. Bai and C. Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *Transactions on Pattern Analysis Machine Intelligence (TPAMI)*, 39(11):2298-2304, 2016  
CRNN [pytorch implementation](https://github.com/meijieru/crnn.pytorch)

\[2\] S. Lucas et al. Icdar 2003 robust reading competitions: entries, results, and future directions. *International Journal of Document Analysis and Recognition (IJDAR)*, 7(2-3):105-122, 2005

\[3\] D. Karatzas et al. Icdar 2013 robust reading competition. *International Conference on Document Analysis and Recognition (ICDAR)*, pages 1484-1493, 2013

\[4\] A. Mishra, K. Alahari, and C.V. Jawahar. Scene text recognition using higher order language priors. *British Machine Vision Conference (BMVC)*, 2012

\[5\] K. Wang, B. Babenko, and S. Belongie. End-to-end scene text recognition. *International Conference on Computer Vision (ICCV)*, pages 1457-1464, 2011

\[6\] M. Jaderberg et al. Synthetic data and artificial neural networks for natural scene text recognition. *arXiv preprint arXiv:1406.2227*, 2014

