import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, n_encoders, n_decoders, dmodel, vocab_size):
        super().__init__()
        assert(n_encoders == n_decoders)
        self.encoder = Encoder(n_encoders = n_encoders, dmodel = dmodel, vocab_size = vocab_size)
        self.decoder = Decoder(n_decoders = n_decoders, dmodel = dmodel)
        self.linear = nn.Linear()

    def forward(self, x) -> torch.Tensor:
        # passed in input is of shape [batch_size, padded_seq_len]
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.linear(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dmodel, heads = 8):
        self.dk = dmodel // heads
        self.dmodel = dmodel
        self.heads = heads
        assert(self.dk * heads == dmodel)
        self.Q = nn.Linear(dmodel, self.dk, bias = False)
        self.K = nn.Linear(dmodel, self.dk, bias = False)
        self.V = nn.Linear(dmodel, self.dk, bias = False)
        self.linear = nn.Linear(dmodel, dmodel)

    def forward(self, keys, queries, values, mask = None) -> torch.Tensor:
        # keys, queries, and values are all of dimension [n, pad_seq_len, dmodel]
        # for every sequence in the batch, and for every token in the sequence, we want to represent that token in h different heads
        # thus, we have the shape [n, len, dmodel] -> [n, len, heads, dmodel]

        # RESHAPE OPERATION HERE
        keys = keys.reshape(keys.shape[0], keys.shape[1], self.heads, keys.shape[2])
        queries = queries.reshape(queries.shape[0], queries.shape[1], self.heads, queries.shape[2])
        values = values.reshape(values.shape[0], values.shape[1], self.heads, values.shape[2])
        # now the shape of the keys, queries, and values are all [n, len, heads, dmodel]

        keys = self.K(keys)
        queries = self.Q(queries)
        values = self.V(values)
        # after matmul, keys, queries, and values are now of shape [n, len, heads, dk]

        # softmax(QK_T / √dk)
        qk = queries * torch.transpose(keys, 2, 3)
        qk /= (self.dk ** -2)
        if(mask != None):
            # apply mask
            pass
        qk = nn.Softmax(qk)
        # qk is of shape [n, len, heads, heads]

        head_outs = (qk * values).reshape(head_outs.shape[0], head_outs.shape[1], self.dmodel) # head_outs is now of shape [n, len, dmodel]

        out = self.linear(head_outs)
        
        return out

class EncoderBlock(nn.Module):
    def __init__(self, dmodel, dff, dropout):
        self.attention = SelfAttention()
        self.ff = nn.Sequential(
            nn.Linear(dmodel, dff),
            F.relu(),
            nn.Linear(dff, dmodel)
        )
        self.layerNorm1 = nn.LayerNorm(dmodel)
        self.layerNorm2 = nn.LayerNorm(dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        # x is of shape [n, padded_seq_len, dmodel]
        x = self.dropout(self.layerNorm1(x + self.attention(x, x, x))) # x is of shape [n, len, dmodel] when returned
        x = self.dropout(self.layerNorm2(x + self.ff(x))) # x is of shape [n, len, dmodel] when returned
        return x

class Encoder(nn.Module):
    def __init__(self, n_encoders, dmodel, vocab_size, max_seq_len, dff = 2048, dropout = 0.1):
        self.positionalEmbedding = nn.Embedding(max_seq_len, dmodel) # converts [n, padded_seq_len] to [n, padded_seq_len, dmodel]
        self.wordEmbedding = nn.Embedding(vocab_size, dmodel, dff = dff, dropout = dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(dmodel = dmodel, dff = dff, dropout = dropout) for _ in range(n_encoders)
        ])
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_seq_len

    def forward(self, x) -> torch.Tensor:
        # passed in input x is of shape [n, padded_seq_len]
        # get positions to do positional embedding
        positions = torch.arange(0, x.shape[1]).expand(x.shape[0], x.shape[1])
        x = self.dropout(self.positionalEmbedding(positions) + self.wordEmbedding(x)) # x is now shape [n, padded_seq_len, dmodel]
        for encoderBlock in self.layers:
            x = encoderBlock(x)

        # x = x.linear(x) -> add this in later
        return x

class DecoderBlock(nn.Module):
    def __init__(self):
        self.maskedAttention = SelfAttention()
        self.attention = SelfAttention()
        self.linear1 = nn.Linear()
        self.linear2 = nn.Linear()

class Decoder(nn.Module):
    def __init__(self, n_decoders):
        self.layers = []
        for i in range(n_decoders):
            self.layers.append(DecoderBlock())


class CNN(nn.Module):
    def __init__(self, dropout = 0.5):
        super().__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(64)
        )
        self.linear = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(168960, 500),
            nn.Dropout(p = dropout),
            nn.Linear(500, 100),
            nn.Dropout(p = dropout),
            nn.Linear(100, 4)
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

# Accuracy - 83.158%
class CIFAR_CNN(nn.Module):
    # shape is (batch_size, rgb, dim1, dim2)
    def __init__(self):
        super().__init__()
        # start with (4, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(1152, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (4, 3, 32, 32)
        x = self.conv1(x) # (4, 16, 30, 30)
        x = self.pool(F.relu(x)) # (4, 16, 15, 15)
        x = self.conv2(x) # (4, 32, 13, 13)
        x = self.pool(F.relu(x)) # (4, 32, 6, 6)
        x = torch.flatten(x, 1) # (4, 32*6*6) = (4, 1152)
        x = F.relu(self.fc1(x)) # (4, 120)
        x = F.relu(self.fc2(x)) # (4, 84)
        x = self.fc3(x) # (4, 10)
        return x

"""
Transformer Notes
    Overall Architecture
        Seeks to map a sequence of input representations x = (x1, x2, x3, ..., xn) to a sequence of continuous representations z = (z1, z2, z3, ..., zn)
        Given z, the decoder then maps z = (z1, z2, z3, ..., zn) to a representation y = (y1, y2, y3, ..., ym)
            Note also that the decoder attempts to perform this operation one token at a time
            Meaning first the decoder will compute y1, then it will compute y2, and so on until ym
            The model consumes the previously generated representations for each sequential representation
                The model consumes y1 to make y2, consumes y2 to make y3, etc. 

    Flow of Information
        Inputs -> Input embeddings
        Input embeddings -> positional encoding
        positional encodings -> encoder
            Repeat for each encoder block:
                Multi head attention
                Add and normalize with residuals
                Feed forward layer
                Add and normalize with residuals
                Pass to equal index decoder block
        encoder -> output embeddings
        output embeddings -> positional encodings
        positional encodings -> decoder
            Repeat for each decoder block:
                Masked multi-headed attention
                Add and normalize with residuals
                Multihead attention with equal encoder block index representations
                Add and normalize with residuals
                Feed forward
                Add and normalize with residuals
        decoder -> linear layer
        linear layer -> softmax output probabilities

    NOTE: All sublayers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512 (or variable)
    Encoder Stack: More Detail
        The encoder is composed of N = 6 (or variable number) identical layers
        Each layer (encoder block) has 2 sublayers:
            Multihead self attention
            Fully connected feed-forward network

    Deocder Stack: More Detail
        The decoder also has N = 6 (or variable) identical layers
        Each layer (decoder block) has 3 sublayers:
            Masked multiheaded attention (over the output from the encoder stack)
                We mask it to prevent positions from the encoder from impacting subsequent positions
                We also offset the output embeddings by one position
                    These two things guarantee that the predictions for position i can only depend on known outputs less than position i
            Standard multihead attention
            Fully connected feedforward network
    
    NOTE: In the encoder and decoder, residual connections exist around EVERY sublayer, followed by layer normalization

    Self Attention: More Detail
        Can be described as mapping a query and a set of key-value pairs to an output
        The query, keys, values, and output are all vectors
        The output is the weighted sum of the values
            The weight matrix is computed by multiplying the query with the keys
            (QK^T)V
        Scaled Dot Product Attention
            Matrix multiply Q and K_T
            scale the Q and K
            Mask (only in decoder, optional)
            Apply Softmax
            Matrix multiply result with V
        Multi-Head Attention
            Perform scaled dot product h times with h unique initializations of the Q, K, V matrices
            Concatenate all h output matrices into one big out matrix
            Pass out matrix through linear layer to produce final output
        Scaled Dot Product Attention: More Detail
            The input consists of queries and keys of dimension dk, and values of dimension dv
            We compute the dot product of each query with all of the keys, and divide each by √dk 
            Afterwards, we apply the softmax function to obtain the weights for the values
            In practice, we compute attention on all queries simultaneously, packed into a matrix Q
            To compute all of the dot products, we matrix multiply by the key matrix K_Transpose
            After dividing by √dk and applying softmax, we matrix multiply this weight matrix with the packed V matrix
        Multi-Head Attention: More Detail
            We don't necessarily want to perform attention with dmodel dimensional keys, values, and queries
            Instead we can project each of these values into dk, dk, and dv dimensions
            Thus, each attention head has to have unique transformation matrices for Q, K, and V
                In this case, the matrix WQ has dimensions [dmodel, dk]
                The matrix WK has dimensions [dmodel, dk]
                The matrix WV has dimensions [dmodel, dv]
                Since we perform WQ x WK_T, this gives us dimensions [dmodel, dk] x [dk, dmodel] = [dmodel, dmodel]
                Now we perform matrix multiplication with the values, producing [dmodel, dmodel] x [dmodel, dv] = [dmodel, dv]
                The output for each scaled dot product attention head is [dmodel, dv]
                Next, we concatenate all of these matrices together for h heads, producing a [dmodel, h*dv] sized matrix
                We matrix multiply this with the matrix W0 of shape [hdv, dmodel] producing [dmodel, hdv] x [hdv, dmodel] = [dmodel, dmodel]
                Thus, the output of self attention is [dmodel, dmodel]
                The paper cites using dk = dv = dmodel / h
        Where do Queries, Keys, and Values come from?
            In the encoder, the queries, keys, and values all come from the output of the previous encoder layer
            In the standard decoder attention (non masking), the keys and values come from the matching encoder stack, and the queries come from the previous decoder block
            In masked self attention, we take the keys, queries, and values from previous decoder blocks
                However, we mask out information from the QxK_T product immediately before applying softmax to preserve the autoregressive property
    Feed Forward Layer: More Detail
        The feed forward layer consists of 2 linear transformations
        One transformation has an input of dmodel and an output of dff -> [dmodel, dff]
        The next has transformation [dff, dmodel] to reproduce the original output shape
"""