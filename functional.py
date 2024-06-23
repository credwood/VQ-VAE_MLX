"""
Functions missing from MLX which are
analogous to those in torch.nn.functional
"""
import mlx.core as mx




#------------- Normalization -------------#

def l2normalize(t, ord=2, axis=-1):
    out = mx.linalg.norm(t, ord=ord, axis=axis, keepdims=True)
    return mx.divide(t, out.max(axis=-1, keepdims=True))

def l2norm(t, ord=None, axis=None, keepdims=False):
    return mx.linalg.norm(t, ord=ord, axis=axis, keepdims=keepdims)

#------------- Embedding -------------#

def custom_batched_embedding(input_indices, batched_weight_matrix):
    """
    Custom batched embedding function.
    
    Args:
        input_indices (mx.array): Tensor containing indices with shape (batch_size, ...).
        batched_weight_matrix (mx.array): Batched weight matrix with shape (batch_size, num_codes, embedding_dim).

    Returns:
        mx.array: Embedding output with shape (*input_indices.shape, embedding_dim).
    """
    
    batch_size, embedding_dim = input_indices.shape[0], batched_weight_matrix.shape[-1]
    input_shape = input_indices.shape
    flattened_indices = input_indices.reshape(batch_size, -1)
    embeddings = mx.zeros((batch_size, flattened_indices.shape[1], embedding_dim))
    
    for i in range(batch_size):
        embeddings[i] = batched_weight_matrix[flattened_indices[i]]
    
    output_shape = (*input_shape, embedding_dim)
    embeddings = embeddings.reshape(output_shape)
    
    return embeddings


def embedding(input, weight):
    assert len(input.shape) == 2, "vectorized embedding only supported for input with shape [batch_size, num_embeddings]"
    return custom_batched_embedding(input, weight)
    

def one_hot(indices, num_classes):
    """
    Custom one_hot function without using scatter_.

    Args:
        indices (mx.array): Tensor containing indices.
        num_classes (int): Number of classes for the one-hot encoding.

    Returns:
        mx.array: One-hot encoded tensor with shape (*indices.shape, num_classes).
    """    
    
    input_shape = indices.shape
    one_hot = mx.zeros((*input_shape, num_classes))
    flattened_one_hot = one_hot.reshape(-1, num_classes)
    flattened_indices = indices.reshape(-1)
    flattened_one_hot[mx.arange(flattened_indices.size), flattened_indices] = 1
    one_hot = flattened_one_hot.reshape(*input_shape, num_classes)
    
    return one_hot

