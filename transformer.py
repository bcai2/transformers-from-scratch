# %%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from typing import Optional, Tuple

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """Initialize multi-head attention module.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            
        Notes:
            - d_model must be divisible by num_heads
            - Each head will have dimension d_model // num_heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        
    def split_heads(self, x: t.Tensor) -> t.Tensor:
        """Split the last dimension into (num_heads, d_k).
        
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Tensor of shape (batch_size, num_heads, seq_length, d_k)
        """
        return einops.rearrange(x, 'batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k')
        
    def forward(self, query: t.Tensor, key: t.Tensor, value: t.Tensor, 
                mask: Optional[t.Tensor] = None) -> t.Tensor:
        """Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, query_len, d_model)
            key: Key tensor of shape (batch_size, key_len, d_model)
            value: Value tensor of shape (batch_size, value_len, d_model)
            mask: Optional mask tensor of shape (batch_size, num_heads, query_len, key_len)
            
        Returns:
            Output tensor of shape (batch_size, query_len, d_model)
        """
        Q_heads = self.split_heads(self.W_Q(query))
        K_heads = self.split_heads(self.W_K(key))
        V_heads = self.split_heads(self.W_V(value))
        attn_out, _ = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        return self.W_out(einops.rearrange(attn_out, 'batch n_heads query_len d_k -> batch query_len (n_heads d_k)'))

# %%
def scaled_dot_product_attention(query: t.Tensor, key: t.Tensor, value: t.Tensor,
                               mask: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
    """Compute scaled dot-product attention.
    
    Args:
        query: Query tensor of shape (batch_size, num_heads, query_len, d_k)
        key: Key tensor of shape (batch_size, num_heads, key_len, d_k)
        value: Value tensor of shape (batch_size, num_heads, value_len, d_k)
        mask: Optional mask tensor of shape (batch_size, num_heads, query_len, key_len)
        
    Returns:
        Tuple of:
            - Output tensor of shape (batch_size, num_heads, query_len, d_k)
            - Attention weights of shape (batch_size, num_heads, query_len, key_len)
    """
    d_k = query.shape[-1]
    QK_product = einops.einsum(query, key, 'batch_size num_heads query_len d_k, batch_size num_heads key_len d_k -> batch_size num_heads query_len key_len')
    QK_product /= d_k**0.5
    if mask:
        QK_product[~mask] = -t.inf
    attn_pattern = t.softmax(QK_product, dim=-1)
    attn_out = einops.einsum(attn_pattern, value, 'batch n_heads query_len key_len, batch n_heads key_len d_k -> batch n_heads query_len d_k') # key_len == value_len
    return (attn_out, attn_pattern)

# %%
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize position-wise feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        # Initialize the two linear transformations and dropout
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self.model = nn.Sequential(
            self.W_1,
            nn.ReLU(),
            self.W_2,
            self.dropout
        )
                
    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        return self.model(x)


# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super().__init__()
        
        odd_pos_mask = (t.arange(max_seq_length) % 2 == 1).unsqueeze(-1)
        pe = t.arange(max_seq_length)[:, None] / t.pow(10000, t.arange(d_model) / d_model)[None, :] # [seq_len d_model]
        pe[~odd_pos_mask] = t.sin(pe[~odd_pos_mask])
        pe[odd_pos_mask] = t.cos(pe[~odd_pos_mask])
        self.register_buffer('pe', pe)

        # Create positional encoding matrix
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: t.Tensor) -> t.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        return self.dropout(x + self.pe[:x.size(1)])


# %%
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize transformer encoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        # Initialize attention, feed-forward, normalization and dropout layers
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: t.Tensor, mask: Optional[t.Tensor] = None) -> t.Tensor:
        """Process input through encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        attn_out = self.self_attn(x, x, x, mask)
        resid_mid = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(resid_mid)
        resid_out = self.norm2(resid_mid + self.dropout(ff_out))
        return resid_out


# %%
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_seq_length: int = 5000, 
                 dropout: float = 0.1):
        """Initialize transformer encoder.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Hidden dimension of feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        # Initialize embedding, positional encoding, and encoder layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.pe = PositionalEncoding(d_model, max_seq_length, dropout)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: t.Tensor, mask: Optional[t.Tensor] = None) -> t.Tensor:
        """Process input through transformer encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x


# %%
def create_padding_mask(seq: t.Tensor, pad_idx: int = 0) -> t.Tensor:
    """Create padding mask for attention.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_length)
        pad_idx: Index used for padding
        
    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_length)
    """
    padding_mask = (seq != pad_idx)
    padding_mask = einops.rearrange(padding_mask, 'batch seq_len -> batch 1 1 seq_len')
    return padding_mask


# %%
def create_causal_mask(size: int) -> t.Tensor:
    """Create causal mask for decoder self-attention.
    
    Args:
        size: Size of the square mask
        
    Returns:
        Causal mask tensor of shape (1, 1, size, size)
    """
    mask = (t.arange(size)[:, None] >= t.arange(size)[None, :])
    return einops.rearrange(mask, 'query_len key_len -> 1 1 query_len key_len')


# %%
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize transformer decoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
            
        Note:
            Decoder layer has 3 main components:
            1. Masked self-attention (prevents attending to future tokens)
            2. Cross-attention (attends to encoder output)
            3. Feed-forward network
        """
        super().__init__()
        # 1. Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. Feed-forward
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor, 
            enc_output: t.Tensor,
            look_ahead_mask: Optional[t.Tensor] = None,
            padding_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """Process input through decoder layer.
        
        Args:
            x: Input tensor from previous layer, shape (batch_size, target_seq_len, d_model)
            enc_output: Encoder output tensor, shape (batch_size, input_seq_len, d_model)
            look_ahead_mask: Mask for masked self-attention (prevents attending to future tokens)
            padding_mask: Mask for encoder-decoder attention (prevents attending to padding)
            
        Returns:
            Output tensor of shape (batch_size, target_seq_len, d_model)
        """
        # 1. Masked self-attention
        attn1 = self.self_attn(x, x, x, look_ahead_mask)
        out1 = self.norm1(x + self.dropout(attn1))
        
        # 2. Cross-attention
        attn2 = self.cross_attn(out1, enc_output, enc_output, padding_mask)
        out2 = self.norm2(out1 + self.dropout(attn2))
        
        # 3. Feed-forward
        ff_out = self.feed_forward(out2)
        out3 = self.norm3(out2 + self.dropout(ff_out))
        
        return out3

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 d_ff: int,
                 max_seq_length: int = 5000,
                 dropout: float = 0.1):
        """Initialize transformer decoder.
        
        Args:
            vocab_size: Size of target vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Hidden dimension of feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: t.Tensor,
                enc_output: t.Tensor,
                look_ahead_mask: Optional[t.Tensor] = None,
                padding_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """Process input through decoder.
        
        Args:
            x: Target sequence, shape (batch_size, target_seq_len)
            enc_output: Encoder output, shape (batch_size, input_seq_len, d_model)
            look_ahead_mask: Mask for masked self-attention
            padding_mask: Mask for encoder-decoder attention
            
        Returns:
            Output tensor of shape (batch_size, target_seq_len, vocab_size)
        """
        seq_len = x.size(1)
        
        # 1. Embedding + Positional encoding
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 2. Pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)
            
        # 3. Final linear layer to get logits
        output = self.final_layer(x)  # (batch_size, target_seq_len, vocab_size)
        
        return output

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_length: int = 5000,
                 dropout: float = 0.1):
        """Initialize full transformer model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            d_ff: Hidden dimension of feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
    def forward(self,
                src: t.Tensor,
                tgt: t.Tensor,
                src_mask: Optional[t.Tensor] = None,
                tgt_mask: Optional[t.Tensor] = None,
                memory_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """Process source and target sequences.
        
        Args:
            src: Source sequence, shape (batch_size, src_seq_len)
            tgt: Target sequence, shape (batch_size, tgt_seq_len)
            src_mask: Mask for source sequence padding
            tgt_mask: Look-ahead mask for target sequence
            memory_mask: Mask for encoder-decoder attention
            
        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 1. Encode source sequence
        enc_output = self.encoder(src, src_mask)
        
        # 2. Decode target sequence using encoder output
        output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        
        return output

# %%
def create_transformer_encoder(vocab_size: int = 1000, d_model: int = 512, 
                               num_heads: int = 8, num_layers: int = 6, 
                               d_ff: int = 2048) -> TransformerEncoder:
    """Create a transformer encoder model.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of encoder layers
        d_ff: Hidden dimension of feed-forward network
        
    Returns:
        Initialized TransformerEncoder model
    """
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )
    return model
