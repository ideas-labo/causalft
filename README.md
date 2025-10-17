# Causally Perturbed Fairness Testing

<!-- ABOUT THE PROJECT -->

## About The Project

To mitigate unfair and unethical discrimination over sensitive features (e.g., gender, age, or race), fairness testing plays an integral role in engineering deep learning systems. A key challenge therein is how to effectively reveal fairness bugs under an intractable sample size using perturbation. Much current work has been focusing on designing the test sample generators, ignoring the valuable knowledge about data characteristics that can help guide the perturbation and hence limiting their full potential. In this paper, we seek to bridge such a gap by proposing a generic framework of causally perturbed fairness testing, dubbed  CausalFT. Through causal inference, the key idea of CausalFT is to extract the most directly and causally relevant non-sensitive feature to its sensitive counterpart, which can jointly influence the prediction of the label. Such a casual relationship is then seamlessly injected into the perturbation to guide a test sample generator. Unlike existing generator-level work,  CausalFT serves as a higher-level framework that can be paired with diverse base generators. Extensive experiments on 324 cases confirm that CausalFT can considerably improve arbitrary base generators in revealing fairness bugs over 94% of the cases with acceptable extra runtime overhead. Compared with a state-of-the-art approach that ranks the non-sensitive features solely based on correlation, CausalFT performs significantly better on 65% cases while being much more efficient.
## Table of Content

### Prerequisites

The codes have been tested with **Python 3.7** and **Tensorflow 2.x**

### Documents

**causalgraph**: 
    
    contains causal analysis of the data set and calculation of the causal effect,the comparison method FairRF in the paper.

**generators**:  

    contains three generator in the paper.
    
**models**:  

    contains utility functions to build DNN models.
    
**data_model**:  

    contains the trained model structure for each dataset.
**model_evaluation**:  

    contains functions to evaluate adequacy metric and fairness metric.     

**dataset**:  

    performance datasets of 8 fairness dataset as specified in the paper.

### Running
**1. find top1 non-senstive feature**:

    #choose the dataset and sensitive feature
    python causalgraph/Causal effect.py 'adult' 'sex'
    
**2. train**:

    #choose the dataset 
    python models/model_train.py 'adult'
    
**3. test generator**:

    #choose the dataset, sensitive feature and chosen non-senstive feature
    python generators/ADF.py
    
**4.evaluation**:

    python models_evaluation/fairness_evaluation.py



