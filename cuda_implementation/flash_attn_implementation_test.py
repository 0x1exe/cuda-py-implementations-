import torch
import pandas as pd
import torch.nn.functional as F

def load_csv_to_tensor(file_path):
    df = pd.read_csv(file_path, header=None)  
    tensor = torch.tensor(df.values)  
    return tensor

Q = load_csv_to_tensor('query_output.csv')
K = load_csv_to_tensor('key_output.csv')
V = load_csv_to_tensor('value_output.csv')
expected_O = load_csv_to_tensor('output_output.csv')

dk = Q.size(-1)  
scores = torch.matmul(Q, K.T) / (dk ** 0.5)  

attention_weights = F.softmax(scores, dim=-1)

computed_O = torch.matmul(attention_weights, V)

is_close = torch.allclose(computed_O, expected_O, atol=1e-5)

print("Attention Weights:")
print(attention_weights)
print("Computed Output O:")
print(computed_O)
print("Expected Output O:")
print(expected_O)
print("Do the computed output and expected output match (within tolerance)?", is_close)

print("Raw scores (Q @ K.T):")
print(scores)
print("Max score (for stability):", torch.max(scores))