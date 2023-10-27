Code for Event Detection with Partial Annotation

====

We develop a model that treats ED as a trigger localization problem (similar to a Machine Reading Comprehension model). The runing steps are:

1. Run `mrc_preprocessing.py` for preprocessing.
2. Run `mrc_train.py` for training with different ratios of labeled data.
3. Run `mrc_evaluation.py` for evaluation a particular model.

We also provide code for the model SeqBERT for training, which has a similar training steps.