# BioOntoBERT - BERT + Ontology

This repository contains the code and instructions to pre-train the BioOntoBERT model using the Onto2Sen-generated corpus from biomedical ontologies. The pre-trained BioOntoBERT model is then fine-tuned on the MedMCQA dataset, leading to improved accuracy over baseline BERT models including PubMedBERT for biomedical multiple-choice question-answering tasks with just 0.7% of pre-training data used for PubMedBERT.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Pre-training](#pre-training)
- [Fine-tuning](#fine-tuning)
- [Results](#results)

## Introduction

BioOntoBERT is a domain-specific language model tailored for the biomedical domain. It is pre-trained on a large corpus generated from biomedical ontologies using the Onto2Sen methodology, which helps capture domain-specific context and semantics. This pre-trained model is then fine-tuned on the MedMCQA dataset, a benchmark for biomedical question answering, to improve its performance on this specific task.

## Requirements

- Python (>=3.6)
- PyTorch (>=1.6)
- Transformers library (Hugging Face)
- CUDA (optional but recommended for faster training)
- MedMCQA dataset

Install the required packages using:

```bash
pip install torch torchvision transformers
```

## Usage
Follow the below steps to pre-train BioOntoBERT on the Onto2Sen biomedical corpus and fine-tune it on the MedMCQA dataset:

## Pre-training
- Data Preparation: Prepare the Onto2Sen-generated biomedical corpus in text format for pre-training.

- Model Configuration: Modify the pre-training configuration in pretrain_config.json to set hyperparameters, paths, and other settings.

- Run Pre-training: Execute the pre-training script


## Fine-tuning
- Data Preparation: Obtain the MedMCQA dataset and preprocess it for fine-tuning.

- Model Configuration: Adjust the fine-tuning configuration in finetune_config.json according to your hardware and preferences.

- Run Fine-tuning: Start fine-tuning the pre-trained BioOntoBERT model


## Results

![Accuracy comparision of different BERT-based models with BioOntoBERT](results.png)

The above table shows how efficiently BioOntoBERT is outperforming other pre-training BERT models with just 158MB of pre-training data from Biomedical ontologies
