---
license: apache-2.0
base_model: google/vit-base-patch16-224-in21k
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: finetuned-dermnet
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# finetuned-dermnet

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2333
- Accuracy: 0.6384

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 2.7983        | 0.12  | 100  | 2.7116          | 0.2039   |
| 2.5909        | 0.24  | 200  | 2.5220          | 0.2798   |
| 2.4805        | 0.36  | 300  | 2.4179          | 0.3033   |
| 2.409         | 0.48  | 400  | 2.2938          | 0.3440   |
| 2.3541        | 0.6   | 500  | 2.2198          | 0.3676   |
| 2.1934        | 0.73  | 600  | 2.1468          | 0.3852   |
| 2.2037        | 0.85  | 700  | 2.1230          | 0.3835   |
| 2.0442        | 0.97  | 800  | 2.0604          | 0.3822   |
| 1.8488        | 1.09  | 900  | 2.0152          | 0.4092   |
| 1.8789        | 1.21  | 1000 | 1.9458          | 0.4293   |
| 1.8709        | 1.33  | 1100 | 1.9382          | 0.4323   |
| 1.8705        | 1.45  | 1200 | 1.8454          | 0.4537   |
| 1.7757        | 1.57  | 1300 | 1.7779          | 0.4747   |
| 1.783         | 1.69  | 1400 | 1.7073          | 0.5021   |
| 1.8145        | 1.81  | 1500 | 1.6875          | 0.4961   |
| 1.9551        | 1.93  | 1600 | 1.6558          | 0.5077   |
| 1.4622        | 2.06  | 1700 | 1.5840          | 0.5266   |
| 1.4823        | 2.18  | 1800 | 1.5623          | 0.5308   |
| 1.347         | 2.3   | 1900 | 1.5829          | 0.5219   |
| 1.4082        | 2.42  | 2000 | 1.5152          | 0.5424   |
| 1.2099        | 2.54  | 2100 | 1.4378          | 0.5741   |
| 1.3461        | 2.66  | 2200 | 1.4326          | 0.5711   |
| 1.3242        | 2.78  | 2300 | 1.4148          | 0.5814   |
| 1.1142        | 2.9   | 2400 | 1.3684          | 0.5968   |
| 1.0141        | 3.02  | 2500 | 1.3351          | 0.6024   |
| 0.9045        | 3.14  | 2600 | 1.3330          | 0.6041   |
| 0.9571        | 3.26  | 2700 | 1.2983          | 0.6165   |
| 1.117         | 3.39  | 2800 | 1.2881          | 0.6230   |
| 0.9663        | 3.51  | 2900 | 1.2864          | 0.6221   |
| 1.0363        | 3.63  | 3000 | 1.2612          | 0.6285   |
| 1.0026        | 3.75  | 3100 | 1.2538          | 0.6345   |
| 1.2539        | 3.87  | 3200 | 1.2434          | 0.6397   |
| 0.8415        | 3.99  | 3300 | 1.2333          | 0.6384   |


### Framework versions

- Transformers 4.38.1
- Pytorch 2.1.2
- Datasets 2.18.0
- Tokenizers 0.15.2
