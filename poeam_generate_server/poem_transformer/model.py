import math
import torch
from torch import nn
from torch import Tensor, device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoetryNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        device: device,
        *,
        embed_size: int = 512,
        n_head: int = 8,
        n_layer: int = 4,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.device = device
        self.embed_size = embed_size
        self.sq = math.sqrt(embed_size)
        self.transformer = nn.Transformer(
            embed_size,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            batch_first=True,
            dim_feedforward=hidden_size,
        )
        self.linear = nn.Linear(embed_size, vocab_size)
        self.positional_encoding = PositionalEncoding(embed_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        src_emb = self.embed(src) * self.sq
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.embed(tgt) * self.sq
        tgt_emb = self.positional_encoding(tgt_emb)

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        out = self.linear(out)
        return out

    def encode(self, src: Tensor) -> Tensor:
        embeded = self.embed(src) * self.sq
        return self.transformer.encoder(self.positional_encoding(embeded))

    def decode(self, tgt: Tensor, memory: Tensor) -> Tensor:
        embeded = self.embed(tgt) * self.sq
        return self.linear(self.transformer.decoder(self.positional_encoding(embeded), memory))
