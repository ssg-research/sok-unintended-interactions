# SoK: Unintended Interactions among Machine Learning Defenses and Risks

This paper will appear in IEEE Symposium on Security and Privacy 2024. This repo contains code to run experiments used in the paper.

## Structure

The repository includes the following folders:

- Cases/: code for generating the plots depitcing the relationship between overfitting and memorization
- FairInv/: code for data reconstruction attacks on models trained with group fairness
- PropInfExpl/: code for distribution inference against model explanations

See README in individual folders on how to run the code.

## Environment Setup

You need __conda__. Create a virtual environment and install requirements:

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate sok
```



## Generating Figure 1 (Relation between Memorization and Overfitting)

Run the code from inside the folder: Cases 

```bash
cd Cases
```

To generate the plots,

```bash
python sok_cases.py
```

## Interaction between Group Fairness and Data Reconstruction (Section 5.1)

Run all the code from inside the folder: FairInv 

```bash
cd FairInv
```

### Identifying the nature of the interaction

We evaluate the loss on training a model with and without group fairness constraint.

```bash
python -m src.inversion_attack
```

### Impact of Model Capacity

We test the loss across four different models of different capacities.

```bash
python -m src.test_model_size --modelnum [1,2,3,4]
```

### Impact of NUmber of input Atrributes

We test the loss across four different models of different capacities.

```bash
python -m src.test_input_size --input_size [10,20,30]
```


## Interaction between Model Explanations and Distribution Inference (Section 5.2)

Run all the code from inside the folder: PropInfExpl 

```bash
cd PropInfExpl
```

### Training Models 

To train models on datasets with different ratios varying from 0.1 to 0.9, use

```bash
./generate_models
```

This will also train models for 0.1 and 0.5 for different number of attributes: [15,25,35]. 
This will also train models with different capacity and layers.
Link to models: https://drive.google.com/file/d/1amm-MXcKF4jKmwoNqjn_PFzGDySgEtSO/view?usp=sharing

### Identifying the nature of the interaction

For this, we evaluate the inference risk across different ratios and different model explanation algorithms.

```bash
python -m src.test_inference_attack --dataset CENSUS --filter sex --explanations {smoothgrad,DeepLift,IntegratedGradients,GradientShap} --ratio1 0.5 --ratio2 [0.1,....,0.9]
```


### Influence of Different Number of Features

We evaluate the attack performance for ratios 0.1 and 0.5 using:

```bash
python -m src.test_dimensionality --dataset CENSUS --explanations {smoothgrad,DeepLift,IntegratedGradients,GradientShap} --filter sex --num_features [15,25,35,42]
```

### Influence of Model Capacity

```bash
python -m src.test_capacity --dataset CENSUS --explanations {smoothgrad,DeepLift,IntegratedGradients,GradientShap} --filter sex --model_number [1,2,3,4]
```

To run all evaluations,

```bash
./generate_output
```