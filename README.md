Code is for reproducing experiment results (model comparison part)
================================
## Requirements:
1. Python 3.6.5
2. Pytorch 0.4.1
3. torchvision 0.2.1
4. tqdm 4.26.0
5. numpy, scipy

## Model Evaluation is Done on Datasets:
Algorithm evaluation in our experiment section is done with dataset MNIST and FashionMNIST.

## Classification Task Code is at [Classification Repos](https://github.com/FirstHandScientist/classification-EM-GM)

## Usage Instruction:

- Train model with:
```
$ python train.py <hparams> 
```
If hparams.Mixture.naive = True in configuration file <hparams>, then GenMM is going to be trained. If hparams.naive is setting False, LatMM is going to be trained.
    
- Training example:
```
python train.py hparams/LatMM/fashion-mnist5.json fashion-mnist directory_to_dataset
```         

For generating samples and latent space interpolation, the directory path to a trained model (either GenMM or LatMM) should be specificed for Infer.pre_trained in configuration file <hparams>. To generate samples or do latent space interpolation, run:
```
python infer_mix.py <hparams> <dataset_root> <mode>
````
For option <mode>, it can either be Generating or Interpolation.


## Model implementation and evaluation:

For generator implementation in our GenMM and LatMM, the gnerators are implemented as flow-based models. The specific glow structure among flow models is used. The glow structure in [glow-pytorch](https://github.com/chaiyujin/glow-pytorch) as pytorch implemention of paper "Glow: Generative Flow with Invertible 1×1 Convolutions" is adopted.

For implementation of our model/algorithm evaluation regarding different metrics, we use the [generative model evaluation framework](https://github.com/xuqiantong/GAN-Metrics)
    

