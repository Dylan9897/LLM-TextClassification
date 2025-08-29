import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, input_seq_len, output_seq_len, feature_dim):
        super(TemporalAttention, self).__init__()
        # Define a linear layer to compute attention scores for each time step.
        # We assume the same attention mechanism is applied across all features.
        self.attention_linear = nn.Linear(feature_dim, 1)
        # Output sequence length must be defined in advance.
        self.output_seq_len = output_seq_len
        # Optionally, you can add a transformation layer if needed.
        self.transform = nn.Linear(input_seq_len, output_seq_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(f"checkpoint for TemporalAttention's transform layers params is:")
        # print("Weight:", self.transform.weight)
        # print("Bias:", self.transform.bias)
        batch_size, seq_len, feature_dim = x.size()
        
        # Compute attention scores (batch_size, seq_len, 1)
        attention_scores = self.attention_linear(x).squeeze(-1)  # Remove the last dimension
        
        # Apply softmax along the sequence length to get attention weights (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Reshape attention weights to match the input dimensions for multiplication (batch_size, seq_len, 1)
        attention_weights = attention_weights.unsqueeze(-1)
        
        # Expand attention weights to match the input dimensions (batch_size, seq_len, feature_dim)
        attention_weights = attention_weights.expand_as(x)
        
        # Apply the attention weights to the input features (batch_size, seq_len, feature_dim)
        weighted_input = x * attention_weights
        
        # Sum over the sequence length dimension to get the attended features (batch_size, feature_dim)
        attended_features = weighted_input.sum(dim=1)
        
        # Transform the attended features to match the desired output sequence length (batch_size, output_seq_len, feature_dim)
        output = attended_features.unsqueeze(1).expand(batch_size, self.output_seq_len, feature_dim)
        output = output.permute(0, 2, 1)
        return output
    
if __name__ == '__main__':
    # Example usage
    batch_size = 2
    input_seq_len = 1536
    output_seq_len = 7
    feature_dim = 512

    x = torch.randn(batch_size, input_seq_len, feature_dim)  # Example input tensor with shape (2, 1536, 512)
    attention_layer = TemporalAttention(input_seq_len=input_seq_len, output_seq_len=output_seq_len, feature_dim=feature_dim)
    output = attention_layer(x)  # Output tensor with shape (2, 7, 512)

    print(output.shape)