import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


class Embeddings(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id

        self.wte = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.wpe = nn.Embedding(max_position_embeddings, hidden_size, padding_idx=pad_token_id)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        mask = input_ids.ne(self.pad_token_id).int()
        position_ids = torch.cumsum(mask, dim=1).type_as(mask) * mask

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        embeddings = inputs_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Pocket(nn.Module):

    def __init__(self, config, device):
        super(Pocket, self).__init__()
        self.device = device
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.pad_id = config.pad_id
        self.mask_id = config.mask_id

        self.embeddings = Embeddings(config.vocab_size, config.d_model, config.n_positions, config.embd_pdrop, self.pad_id)

        encoder_layers = nn.TransformerEncoderLayer(config.d_model, config.nhead, config.d_hid, config.resid_pdrop, batch_first=True)
        encoder_norm = LayerNorm(config.d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.nlayers, encoder_norm)

        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, labels=None):
        input_shape = input_ids.size()

        inputs_embeds = self.embeddings(input_ids)

        padding_mask = (input_ids == self.pad_id).float()
        attn_mask = nn.Transformer.generate_square_subsequent_mask(input_shape[-1], device=self.device)
        hidden_states = self.transformer_encoder(inputs_embeds, attn_mask, padding_mask)
        logits = self.linear(hidden_states)

        # Get the index of the mask in each row
        target_indices = torch.eq(input_ids, self.mask_id).int().argmax(-1)
        # Extract the logits for the predicted token in each sequence
        pooled_logits = logits[torch.arange(logits.shape[0], device=self.device), target_indices]

        if labels is not None:
            # Calculate the cross-entropy loss
            loss = F.cross_entropy(pooled_logits, labels, label_smoothing=0.1)
            return pooled_logits, loss

        return pooled_logits
