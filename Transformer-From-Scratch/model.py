import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        """Input embedding layer

        Args:
            d_model (int): dimension of vector embedding
            vocab_size (int): vocabulary size
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model) # seq_len, d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape (seq_len, d_model)
        pos_emd = torch.zeros(self.seq_len, self.d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0 / self.d_model)))

        # Apply the sin to even pos and cos to odd pos
        pos_emd[:, 0::2] =  torch.sin(position * div_term)
        pos_emd[:, 1::2] =  torch.cos(position * div_term)

        pos_emd = pos_emd.unsqueeze(0)  # (1, seq_len, d_model)
        # since pos encoding are fixed and are not learned - but we need to store it in the model
        self.register_buffer('pos_emd', pos_emd)    # Saves it as part of the model state but not as a parameter

    def forward(self, x):
        # x : seq_len, d_model - seq_len is max seq_len
        # :x.shape[1] - cur sequence is lesser than max_seq - so handle that
        x = x + self.pos_emd[:, :x.shape[1], :].detach()  # unlearnable parameter
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))    # Multiplied - learnable
        self.bias = nn.Parameter(torch.ones(1))    # Added - learnable

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)   # calc mean acorss the last dimension
        std = x.std(dim = -1, keepdim=True) # calc std acorss the last dimension
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # W1 B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)    # W2 B2

    def forward(self, x):
        # batch, seq_len, d_model -> batch, seq_len, d_ff -> batch, seq_len, d_model
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h:int, dropout:float):
        """
        Args:
            h (int): number of attention heads
        """
        super().__init__()
        self.d_model = d_model
        self.h = h

        # Divide d_model into h number of heads
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h # each head's embedding dimension

        # Query, Key, Value, Output matrix
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        self.w_o = nn.Linear(d_model, d_model)  # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]   # get the shape of the embedding of each
        
        # batch, num heads, seq_len, d_k -> batch, num heads, seq_len, seq_len
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)   # apply -1e9 to attention scores tensor whereever the mask=0
        attention_scores = attention_scores.softmax(dim=-1) # bathc, num heads, seq_len

        if dropout:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # second output is for visualization

    def forward(self, q, k, v, mask):
        """
        Args:
            q (torch.tensor): query
            k (torch.tensor): key
            v (torch.tensor): value
            mask (torch.tensor): mask to mask the attention
        """
        # fancy way of mat mul with bias addition Q = qWqᵀ
        # nn.linear does the mat mul
        query = self.w_q(q) # batch, seq_len, d_model
        key = self.w_k(k)   # batch, seq_len, d_model
        value = self.w_v(v) # batch, seq_len, d_model

        # split query to different heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)    #batch, seq_len, num heads, embedding size per head
        query = query.transpose(1, 2)   # batch, num heads, seq_len, embedding size per head

        # similarly split key and value to different heads
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)  # output, attention scores

        # Concat individual heads
        # batch, h, seq_len, d_k -> batch, seq_len, h, d_k
        x = x.transpose(1, 2)
        # batch, seq_len, h, d_k -> batch, seq_len, d_model
        # if .view() is called directly, PyTorch might complain or give incorrect results — 
        # because view() needs the tensor to be stored in a contiguous block of memory.
        # contiguous() - Ensures the tensor is safely reinterpreted in memory as a 3D tensor combining all heads back into
        x = x.contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)  # batch, seq_len, d_model

class ResidualConnection(nn.Module):
    """
    Skip connection between Add & Norm (layer norm) and input
    """
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Args:
            x : inputs
            sublayer : previous layer (multi head attention)
        """
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, multi_head_self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float):
        super().__init__()
        self.multi_head_self_attention_block = multi_head_self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])  # create list of two residual connections per encoder block

    def forward(self, x, src_mask):
        # Residual Connection expects input tensor x and function (which takes one argument)
        # we need to create first residual connection with input tensor x and multihead self attention
        # but since multi head self attention expects 4 values, we can't use it directly
        # so we create a lambda function with one argument that internally creates multi head self attention
        # just making things compatible
        x = self.residual_connections[0](x, lambda x: self.multi_head_self_attention_block(x, x, x, src_mask))  # first skip connection - input and multi head attention
        x = self.residual_connections[1](x, self.feed_forward_block)  # second skip connection - input and feed forward attention
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        """
        Args:
            layers (nn.ModuleList): number of encoder blocks
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, multi_head_self_attention_block: MultiHeadAttentionBlock, multi_head_cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float):
        super().__init__()
        self.multi_head_self_attention_block = multi_head_self_attention_block
        self.multi_head_cross_attention_block = multi_head_cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])  # create list of three residual connections per decoder block

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x (_type_): input of decoder
            encoder_output (_type_): for cross attention
            src_mask (_type_): mask for encoder (since bidirectional attention)
            tgt_mask (_type_): mask for decoder (since causal attention)
        """
        x = self.residual_connections[0](x, lambda x: self.multi_head_self_attention_block(x, x, x, tgt_mask))  # first skip connection - input and multi head self attention
        x = self.residual_connections[1](x, lambda x: self.multi_head_cross_attention_block(x, encoder_output, encoder_output, src_mask))  # second skip connection - input and multi head cross attention
        x = self.residual_connections[2](x, self.feed_forward_block)  # third skip connection - input and feed forward attention
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        """
        Args:
            layers (nn.ModuleList): number of decoder blocks
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # batch, seq_len, d_model -> batch, seq_len, vocab_size
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder:Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos

        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos

        self.projection_layer = projection_layer

    # instead of one forward method, we can have encode, decode methods separately - helpful when inferencing and output visualization

    def encode(self, src, src_mask):
        src = self.src_embed(src)   # embedding
        src = self.src_pos(src) # positional embedding
        return self.encoder(src, src_mask)  # encode
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)   # embedding
        tgt = self.tgt_pos(tgt) # positional embedding
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)    # decode
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len:int, tgt_seq_len:int, d_model:int = 512, N:int=6, h:int=8, dropout:float=0.1, d_ff:int=2048)->Transformer:
    """build the complete transformer for translation

    Args:
        src_vocab_size (int): source language vocabulary size
        tgt_vocab_size (int): target language vocabulary size
        src_seq_len (int): source language max sequence length
        tgt_seq_len (int): target language max sequence length
        d_model (int, optional): embedding dimension. Defaults to 512.
        N (int, optional): number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): number of attention heads. Defaults to 8.
        dropout (float, optional): dropout value. Defaults to 0.1.
        d_ff (int, optional): feed forward layer dimension. Defaults to 2048.

    Returns:
        Transformer: transformer model
    """
    # create embeding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional embedding layers
    src_pos_embed = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos_embed = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []
    for i in range(N):
        encoder_multihead_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_multihead_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create decoder blocks
    decoder_blocks = []
    for i in range(N):
        decoder_multihead_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_multihead_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_multihead_self_attention_block, decoder_multihead_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) # project to target vocabulary only (not source vocabulary)

    # create transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_embed, tgt_pos_embed, projection_layer)

    # Initialize Parameters (Xavier Initialization)
    for p in transformer.parameters():  # loops over all parameters
        if p.dim() > 1: # skip 1D param, bias. Only initialize weights, not biases
            nn.init.xavier_uniform_(p)  # assign xavier uniform weight initialization

    return transformer