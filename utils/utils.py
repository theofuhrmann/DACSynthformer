import torch
import os
import dac

#from DACTransformer.DACTransformer import TransformerDecoder
#from DACTransformer.CondQueryTransformer import ClassConditionedTransformer

def save_model(model, inf_context_length, filepath):
    torch.save({
        'inf_context_length': inf_context_length,
        
        'model_state_dict': model.state_dict(),
        'embed_size': model.embed_size,
        'num_layers': model.num_layers, # len(model.layers),
        'num_heads': model.num_heads, #  model.layers[0].attention.num_heads,
        'forward_expansion': model.forward_expansion, # model.layers[0].feed_forward[0].out_features // model.embed_size,
        'dropout': model.dropout, # model.layers[0].dropout.p,
        'max_len': model.max_len, # model.position_embedding.num_embeddings,
        'num_codebooks': model.num_codebooks, 
        'vocab_size': model.vocab_size,
        'cond_size': model.cond_size,
        'num_classes': model.num_classes,
    }, filepath)

#-----------------------------------------------------

def load_model(filepath, TransformerClass, device='cuda'):
    checkpoint = torch.load(filepath, map_location=device)  
    inf_context_length = checkpoint['inf_context_length'] # This is used to set the context length for the inference model
    
    model =  TransformerClass(
        embed_size=checkpoint['embed_size'],
        num_layers=checkpoint['num_layers'],
        num_heads=checkpoint['num_heads'],
        forward_expansion=checkpoint['forward_expansion'],
        dropout=checkpoint['dropout'],
        # This should be the training conext size size it affect the rotary positional encoding, not the conext length itself.
        max_len=checkpoint['max_len'],
        num_codebooks=checkpoint['num_codebooks'],
        vocab_size=checkpoint['vocab_size'],
        cond_size= checkpoint['cond_size'],
        num_classes = checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model, inf_context_length, checkpoint['vocab_size'], checkpoint['num_codebooks'], checkpoint['cond_size'] # used for consructing the initial input window to the model
    
#----------------------------------------------------------------------    
# I don't really know how to set these DACFile params, but they are necessary for model.decompress(dacfile) to work
# 
def writeDACFile(fname, codeseq) :
    with torch.no_grad(): 
        dac_file = dac.DACFile(
                codes=codeseq.cpu(),
                chunk_length=codeseq.shape[2],
                original_length=int(codeseq.shape[2]*512), 
                input_db= torch.tensor(-20), #np.array([-20], dtype=np.float32),
                channels=1,
                sample_rate=44100,
                padding=True,
                dac_version='1.0.0'
            )
        
        # Save to disk
        directory = os.path.dirname(fname)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'Just so ya know, I had to create the path to save the file')
        dac_file.save(fname + ".dac")
        

def generate_mask(sz, max_lookback):
    """
    Generates a mask for the attention mechanism that incorporates causal structure with a limited lookback window.

    Args:
    sz (int): Size of the square matrix (number of time steps/sequence length).
    max_lookback (int): Maximum number of steps a position can look back in the sequence.

    Returns:
    torch.Tensor: The attention mask.
    """
    # Full mask with all positions set to -inf initially
    mask = torch.full((sz, sz), float('-inf'))

    # Fill the band of allowed positions with 0s
    for i in range(sz):
        start = max(0, i - max_lookback)  # Start of the lookback window
        end = i + 1  # End of the lookback window, non-inclusive
        mask[i, start:end] = 0.0

    return mask

# -----------------------------------------------------------------------

def interpolate_vectors(v, s):
    assert len(v) == len(s), "List of vectors and list of time indexes must be of the same length."
    
    n = len(v[0])  # Length of each vector
    m = s[-1] + 1  # Last element of s plus one
    result = torch.zeros((1, m, n))  # Initialize the result tensor with zeros
    
    v_tensors = [torch.tensor(vec) for vec in v]
    
    for i in range(len(s) - 1):
        start_idx = s[i]
        end_idx = s[i + 1]
        start_vec = v_tensors[i]
        end_vec = v_tensors[i + 1]
        
        for j in range(start_idx, end_idx):
            t = (j - start_idx) / (end_idx - start_idx)
            result[0, j, :] = (1 - t) * start_vec + t * end_vec
    
    # Set the last vector directly
    result[0, s[-1], :] = v_tensors[-1]
    
    return result


# # Example usage
# v = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
# s = [0, 2, 4]
# result_tensor = interpolate_vectors(v, s)
# print(result_tensor)


def sample_top_n(logits, n):
    """
    Select top `n` logits for each token, apply softmax, and sample token indices.
    
    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, 1, num_tokens, vocab_size)
        n (int): Number of top logits to consider per token.
        
    Returns:
        torch.Tensor: Tensor of sampled token indices with shape (batch_size, num_tokens)
    """
    # Ensure the input tensor shape is (batch_size, num_tokens, vocab_size)
    batch_size, num_tokens, vocab_size = logits.shape

    # Find the top n logits and their indices along the vocabulary dimension
    top_n_logits, top_n_indices = torch.topk(logits, n, dim=-1)  # Shape: (batch_size, 1, num_tokens, n)

    # Apply softmax to the top n logits
    top_n_probs = torch.softmax(top_n_logits, dim=-1)  # Shape: (batch_size, 1, num_tokens, n)

    # Sample from the top n probabilities for each token
    sampled_indices = torch.multinomial(top_n_probs.view(-1, n), 1).squeeze(-1)  # Shape: (batch_size * num_tokens)

    # Map the sampled indices back to the original vocabulary indices
    sampled_vocab_indices = top_n_indices.view(-1, n).gather(1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    # Reshape back to (batch_size, num_tokens)
    sampled_vocab_indices = sampled_vocab_indices.view(batch_size, num_tokens)

    return sampled_vocab_indices
    