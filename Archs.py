import talos
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class UnpairedTransformer(talos.TalosModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        max_seq_length: int = 200,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        """
        Initialize a Transformer model for unpaired sequence transduction.
        
        Args:
            vocab_size: Size of vocabulary (same for source and target)
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function for the feedforward network
            max_seq_length: Maximum sequence length
            pad_idx: Index of padding token
            bos_idx: Beginning of sequence token index
            eos_idx: End of sequence token index
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._reset_parameters()
    
    # def _reset_parameters(self):
    #     """Initialize parameters using Xavier uniform initialization"""
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ):
        """
        Forward pass of the transformer model for unpaired sequence-to-sequence.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            src_mask: Mask for source sequence
            src_key_padding_mask: Mask for source padding
            max_len: Maximum length of generated sequence (default: src_seq_len * 1.5)
            
        Returns:
            output_tokens: Generated output sequence tokens
            output_probs: Output probabilities for each position
        """
        batch_size = src.size(0)
        device = src.device
        
        if max_len is None:
            max_len = min(int(src.size(1) * 1.5), self.max_seq_length)
        
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == self.pad_idx)
        
        memory = self.encode(src, src_mask, src_key_padding_mask)
        memory_key_padding_mask = src_key_padding_mask
        
        tgt = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)
        
        all_probs = []
        all_logits = []
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        #Autoregressive thing
        for i in range(max_len - 1):
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            output = self.decode(tgt, memory, tgt_mask, memory_key_padding_mask)
            
            next_token_logits = output[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
            
            all_probs.append(next_token_probs)
            all_logits.append(next_token_logits)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            finished = finished | (next_token.squeeze(-1) == self.eos_idx)
            if finished.all():
                break
        
        output_probs = torch.stack(all_probs, dim=1) if all_probs else None
        output_logits = torch.stack(all_logits, dim=1) if all_logits else None
        
        return tgt, output_logits
    
    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        """Encode source sequence"""
        src_emb = self.positional_encoding(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(
            src_emb, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        """Decode target sequence given memory"""
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt_emb = self.positional_encoding(self.embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer_decoder(
            tgt_emb, 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        output = self.output_projection(output)
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(talos.TalosModule):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)





class TransformerDiscriminator(talos.TalosModule):
    """
    Transformer-based discriminator model to distinguish between real and generated sequences.
    Used for adversarial training in unpaired translation.
    """
    
    def __init__(
        self, 
        vocab_size, 
        d_model=512, 
        nhead=8, 
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=200,
        pad_idx=0
    ):
        super().__init__()
        
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the discriminator
        
        Args:
            x: Input sequence tensor of shape (batch_size, seq_len)
            
        Returns:
            scores: Discrimination scores (real=1, fake=0)
        """
        padding_mask = (x == self.pad_idx)
        
        x_emb = self.positional_encoding(self.embedding(x) * math.sqrt(self.embedding.embedding_dim))
        
        encoded = self.transformer_encoder(x_emb, src_key_padding_mask=padding_mask)
        
        mask = (x != self.pad_idx).float().unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        scores = self.classifier(pooled)
        
        return scores







class CycleTransformer(talos.TalosModule):
    """
    Cycle-consistent transformer for unpaired sequence translation.
    Contains two UnpairedTransformer models for A→B and B→A translation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        max_seq_length: int = 200,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        
        self.ab_transformer = UnpairedTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_length=max_seq_length,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )
        
        self.ba_transformer = UnpairedTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_length=max_seq_length,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )
        
        self.discriminator = self.add_discriminator(d_model, nhead)
    
    def forward_ab(self, src_a):
        """Forward pass from language A to B"""
        return self.ab_transformer(src_a)
    
    def forward_ba(self, src_b):
        """Forward pass from language B to A"""
        return self.ba_transformer(src_b)
    
    def cycle_forward(self, src_a):
        """Cycle forward pass: A → B → A"""
        tgt_b, _ = self.forward_ab(src_a)
        
        tgt_a_cycle, _ = self.forward_ba(tgt_b)
        
        return tgt_b, tgt_a_cycle
    
    def add_discriminator(self, d_model, nhead, num_layers=4):
        """Add a transformer-based discriminator for adversarial training"""
        self.discriminator = TransformerDiscriminator(
            vocab_size=self.ab_transformer.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pad_idx=self.ab_transformer.pad_idx
        )
        return self.discriminator





def create_cycle_transformer(
    vocab_size,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation="relu",
    max_seq_length=200,
    pad_idx=0,
    bos_idx=1,
    eos_idx=2,
):
    """Factory function to create a cycle transformer model with specific parameters"""
    model = CycleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        max_seq_length=max_seq_length,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    )
    
    return model




def adversarial_loss(d_fake):
    """
    Args:
        d_fake: Discriminator output for generated data (logits), shape (B,)
    Returns:
        BCE loss for generator (want D(fake) → 1)
    """
    target_real = torch.ones_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_fake, target_real)

def discriminator_loss(d_real, d_fake):
    """
    Args:
        d_real: D output on real Swahili tokens
        d_fake: D output on generated Swahili tokens (detached)
    Returns:
        BCE loss for discriminator
    """
    
    real_targets = torch.ones_like(d_real)
    fake_targets = torch.zeros_like(d_fake)
    
    loss_real = F.binary_cross_entropy_with_logits(d_real, real_targets)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_targets)
    
    return (loss_real + loss_fake) / 2

def cyclic_loss(predicted_logits, original_token_ids, pad_token_id=None):
    """
    Computes cross-entropy reconstruction loss between predicted logits and target token IDs.
    
    Args:
        logits: Tensor of shape (B, T_pred, V) — raw output logits.
        targets: Tensor of shape (B, T_gold) — target token IDs.
        pad_token_id: Optional int — if set, pads are ignored in loss.
    
    Returns:
        Scalar loss tensor.
    """
    
    logits = predicted_logits
    targets = original_token_ids
    
    B, T_pred, V = logits.shape
    T_gold = targets.shape[1]
    
    if T_gold > T_pred:
        targets = targets[:, :T_pred]
    elif T_gold < T_pred:
        pad = torch.full((B, T_pred - T_gold), fill_value=pad_token_id or 0, device=targets.device)
        targets = torch.cat([targets, pad], dim=1)

    logits = logits.transpose(1, 2)  # (B, V, T)
    
    return F.cross_entropy(
        logits, targets,
        ignore_index=pad_token_id if pad_token_id is not None else -100
    )






