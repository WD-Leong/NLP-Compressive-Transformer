# NLP-Compressive-Transformer

This repository contains my implementation of the [Compressive Transformer](https://arxiv.org/abs/1911.05507), with a general architecture shown in Fig. 1. This implementation has some minor differences from that in the [paper](https://arxiv.org/abs/1911.05507) in that it learns a softmax projection to form the compressed memory instead of applying the methods listed in the paper.

![Compressive_Transformer_Image_1](Compressive_Transformer.JPG)
Fig. 1: Compressive Transformer Model as provided in the [Compressive Transformer](https://arxiv.org/abs/1911.05507) paper.

## Training
To train the model using the [Reddit Jokes](https://github.com/taivop/joke-dataset) dataset, run
```
python process_reddit_jokes_subwords.py
python train_reddit_jokes_subword_tf_ver2_gpt_compressive.py
```
and 
```
python infer_reddit_jokes_subword_tf_ver2_gpt_compressive.py
```
to perform inference.
