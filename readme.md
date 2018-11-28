[PyTorch](http://pytorch.org/) Project for NeurIPS 2018 paper: 

**Connectionist Temporal Classification with Maximum Entropy Regularization**
Hu Liu, Sheng Jin and Changshui Zhang. *Neural Information Processing Systems (NeurIPS)*, 2018. 


## Requirements
* python2
* pytorch
* lmdb

## Datasets
* ICDAR-2003 (IC03)
* ICDAR-2013 (IC13)
* IIIT5k-word (IIIT5k) 
* Street View Text (SVT)
* Synth5K (sampled from Synth90k)
* Synth90k (so large, we do not include in this repository)

## Usage
### Train
`zsh shs/seg_ent_fb/seg_5k.sh`
### Test
`CUDA_VISIBLE_DEVICES=0 python test.py --crnn_path model_dir --valroot data/svt1/testset.lmdb`

## Citation
If you use this code in your project, please cite this paper.

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/liuhu-bigeye/enctc.crnn/issues).

## References
