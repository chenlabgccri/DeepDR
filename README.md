# DeepDR: Predicting drug response of tumors from integrated genomic profiles by deep neural networks

## Introduction
*DeepDR* is a Python implementation to predict drug response of unscreened cell lines and tumors by mutation and gene expression profiles. It has a transfer learning design to learn from tumor genomics using unlabeled tumor samples (i.e., without screening data) and to be further fine-tuned and optimized using labeled cell line screening data.

## Model overview
<img align="center" src="./sketch/overview.png?raw=true" alt="drawing" width="600">

## Main codes
**PretrainAE.py:**
This code pretrains an autoencoder using unlabeled gene mutation or gene expression data of tumors. Output (model parameters saved as a pickle file) is used by TrainModel.py to initialize the IC50 predictor.

**TrainModel.py:**
This code trains an IC50 predictor using labeled cell-line screening data, predicts new samples (e.g., unscreened cell lines or tumors), and output the model and prediction results.

Instructions are provided within individual codes.

## Input genomic and IC50 data
For details regarding input/output data preparation and sources, please refer to the Methods section of our [paper](https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-018-0460-9#Sec2).

## Environment
The functionality of our codes was developed and tested on Python 3.5 with Keras 1.2.2 and TensorFlow 1.4 backend on an x86_64-pc-linux-gnu platform.

## Reference
Chiu YC, Chen HIH, Zhang T, Zhang S, Gorthi A, Wang LJ, Huang Y, Chen Y.
**"DeepDEP: deep learning of a cancer dependency map using cancer genomics."**
*BMC Med Genomics*. 2019 Jan 31;12(Suppl 1):18. doi: 10.1186/s12920-018-0460-9.
