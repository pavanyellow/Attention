"""
A pure Python implementation of the attention mechanism with no PyTorch or NumPy dependencies.
"""

import random 
import math

context_length = 10
model_dim = 64
attention_head_dimension = model_dim

logits= [[random.uniform(-0.5,0.5) for _ in range(model_dim)] for _ in range(context_length)]
query_matrix = [[random.uniform(-0.5,0.5) for _ in range(attention_head_dimension)] for _ in range(model_dim)]
key_matrix = [[random.uniform(-0.5,0.5) for _ in range(attention_head_dimension)] for _ in range(model_dim)]
value_matrix = [[random.uniform(-0.5,0.5) for _ in range(attention_head_dimension)] for _ in range(model_dim)]
projection_matrix = [[random.uniform(-0.5,0.5) for _ in range(model_dim)] for _ in range(attention_head_dimension)]


def apply_layer(input_vector : 'list[float]', layer_to_apply : 'list[list[float]]') -> 'list[float]':
    # This is equivalent to applying a MLP layer on a vector
    return simple_matmul([input_vector], layer_to_apply)[0]

def simple_matmul(X : 'list[list[float]]' ,Y : 'list[list[float]]'):
    a,b = len(X), len(X[0])
    c,d = len(Y), len(Y[0])

    if b!=c:
        raise Exception(f"Can't multiply matrices of sizes ({a},{b}) and ({c},{d})")
    
    output_mat = [[0 for _ in range(d)] for _ in range(a)]

    for i in range(a):
        for j in range(b):
            for k in range(d):
                output_mat[i][k] += X[i][j]*Y[j][k]
        
    return output_mat

def vector_dot_product(X,Y):
    # Both are vectors
    a,b = len(X), len(Y)
    if a!=b:
        raise Exception(f"Vectors are of different size {a},{b}")
    return sum([X[i] * Y[i] for i in range(a)])

def apply_soft_max(pattern : 'list[float]', token_index) -> 'list[float]':
    exponentiated_array = [math.exp(i) for i in pattern]
    for i in range(token_index+1, context_length):
        # Masking
        exponentiated_array[i] = 0

    partition_function = sum(exponentiated_array)
    return [i/partition_function for i in exponentiated_array]

def calculate_attention_pattern(logits):
    # Returns a square matrix of size context_length*context_length
    attention_pattern = [[0 for _ in range(context_length)] for _ in range(context_length)]
    for destination_token in range(context_length):
        destination_token_logits = logits[destination_token]
        query_vector = apply_layer(destination_token_logits, query_matrix)
                
        for source_token in range(destination_token+1):
            source_token_logits= logits[source_token]
            key_vector = apply_layer(source_token_logits, key_matrix)
            attention_value : float = vector_dot_product(query_vector, key_vector)/math.sqrt(attention_head_dimension)
            attention_pattern[destination_token][source_token] = attention_value

    return [apply_soft_max(attention_pattern[token_index], token_index) for token_index in range(context_length)]

        
def get_attention_output(logits, attention_pattern):
    attention_output = [[0 for _ in range(model_dim)] for _ in range(context_length)]
    for destination_token in range(context_length):   
        for source_token in range(destination_token+1):
            source_token_logits= logits[source_token]
            value_vector = apply_layer(source_token_logits, value_matrix)
            projection_vector = apply_layer(value_vector,projection_matrix)
            attention_value : float = attention_pattern[destination_token][source_token]
            for dim in range(model_dim):
                attention_output[destination_token][dim] += attention_value*(projection_vector[dim])
        
    return attention_output


attention_pattern= calculate_attention_pattern(logits)
attention_output = get_attention_output(logits, attention_pattern) # This is added to the residual stream ie logits

for attention in attention_pattern:
    print(' '.join(['{:.2f}'.format(item) for item in attention]))


from attention import *

def test_attention(attention_pattern):
    # Finally a test to check if the attention implementation is correct by comparing with reference implementation
    # https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L67
    import torch
    import torch.nn as nn
    import math

    logits_reshaped = torch.tensor(logits).unsqueeze(0)
    query = nn.Linear(model_dim, attention_head_dimension, bias=False)
    query.weight.data = torch.tensor(query_matrix).T
    key = nn.Linear(model_dim, attention_head_dimension, bias=False)
    key.weight.data = torch.tensor(key_matrix).T

    q = query(logits_reshaped)
    k = key(logits_reshaped)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(torch.tril(att) == 0, float('-inf'))
    att = torch.nn.functional.softmax(att, dim=-1)

    assert torch.all(att.squeeze(0).round(decimals=3) == torch.tensor(attention_pattern).round(decimals=3))

test_attention(attention_pattern)
