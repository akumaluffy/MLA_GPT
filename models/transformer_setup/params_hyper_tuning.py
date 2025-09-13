'''
Hyperparameter Tuning Plan

Values to be used for hyperparameter testing

'''

# Suggested Hyperparameter values for tuning

# Model Architecture Hyperparameters

#1
layer_vals = [8, 10, 12] 

#2
attention_head_vals = [8, 10, 12] 

# Embedding

if attention_head_vals in [8, 12]:
    n_embd = 768
else:
    n_embd = 760