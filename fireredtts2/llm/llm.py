import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from huggingface_hub import PyTorchModelHubMixin
from fireredtts2.llm.modules import FLAVORS
import logging

logger = logging.getLogger(__name__)

def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


# Does multinomial sampling without a cuda synchronization
def _multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk_and_return_indices(logits: torch.Tensor, topk: int, temperature: float, eos_only_argmax: bool = False, codebook_eos_token_id: int = 0):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    topk_values = torch.topk(logits, topk)[0]
    indices_to_remove = logits < topk_values[..., -1, None]
    
    # If eos_only_argmax is True, only allow EOS if it's the argmax
    if eos_only_argmax:
        # Get the argmax token index
        argmax_token = torch.argmax(logits, dim=-1, keepdim=True)
        # If EOS is not the argmax, filter it out
        eos_not_argmax = (argmax_token != codebook_eos_token_id)
        indices_to_remove[..., codebook_eos_token_id] = torch.where(
            eos_not_argmax.squeeze(-1),
            torch.tensor(True, dtype=torch.bool, device=logits.device),
            indices_to_remove[..., codebook_eos_token_id]
        )
    
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    indices_to_keep = torch.nonzero(~indices_to_remove, as_tuple=False)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token, indices_to_keep, topk_values

def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    sample_token, _, _ = sample_topk_and_return_indices(logits, topk, temperature)
    return sample_token


def sample_top_nsigma(logits: torch.Tensor, n: float, temperature: float):
    """_summary_

    Args:
        logits (torch.Tensor): _description_
        n (float): _description_
        temperature (float): _description_

    Returns:
        _type_: _description_
    """
    logits = logits / temperature
    threshold = logits.max(dim=-1, keepdim=True).values - n * logits.std(
        dim=-1, keepdim=True
    )
    logits[logits < threshold] = float("-inf")
    # scores_processed = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    decoder_loss_weight: float
    use_text_loss: bool
    decoder_sampling_ratio: float = 0.125  # default 1/8 sampling for decoder training


class Model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.text_head = nn.Linear(backbone_dim, config.text_vocab_size, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )

        self.decoder_loss_weight = config.decoder_loss_weight
        self.use_text_loss = config.use_text_loss
        self.decoder_sampling_ratio = config.decoder_sampling_ratio

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(
                max_batch_size,
                dtype,
                decoder_max_seq_len=self.config.audio_num_codebooks,
            )

        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(self.backbone.max_seq_len, device),
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, device),
        )

    def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor):
        """
        Forward pass for Sesame's CSM model.
        This will be added to the model with `model.forward = types.MethodType(forward, model)`

        Args:
            tokens: (batch_size, seq_len, n_codebooks+1)
            tokens_mask: (batch_size, seq_len, n_codebooks+1)
        """

        dtype = next(self.parameters()).dtype
        bsz, seq_len, _ = tokens.size()
        device = tokens.device

        # embed tokens
        embeds = self._embed_tokens(tokens)  # (bsz,seq_len,17,2048)

        # get targets and codebook embeddings corresponding to audio tokens
        audio_mask = tokens_mask[:, :, 0]  # [bsz, seq_len]
        audio_positions = audio_mask.nonzero(as_tuple=False)
        target_tokens = tokens[audio_mask][:, :-1]  # [audio_len, n_codebooks]
        # [audio_len, n_codebooks, embed_dim]
        c_embeds = embeds[:, :, :-1, :][audio_mask]

        # get targets corresponding to text tokens
        text_mask = tokens_mask[:, :, -1]
        text_target_mask = torch.roll(input=text_mask, shifts=1, dims=1)
        text_target_tokens = tokens[text_target_mask][:, -1]

        # retain just non-padding embeddings
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)

        # backbone forward pass
        # [bsz, seq_len]
        padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]
        # [seq_len, seq_len]
        backbone_attn_mask = _create_causal_mask(seq_len, device)
        # [bsz, seq_len, seq_len]
        padding_3d = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)
        backbone_attn_mask = backbone_attn_mask.unsqueeze(0) * padding_3d
        backbone_attn_mask = backbone_attn_mask | torch.eye(
            seq_len, device=device
        ).bool().unsqueeze(0).expand(bsz, -1, -1)
        input_pos = (
            torch.arange(0, seq_len).unsqueeze(0).expand(bsz, seq_len).long().to(device)
        )
        h = self.backbone(h, input_pos=input_pos, mask=backbone_attn_mask).to(
            dtype=dtype
        )

        # get backbone embeddings used for audio codebook prediction predict first codebook and compute loss
        audio_mask_shifted = torch.roll(audio_mask, -1, 1)  # shift audio mask to the right by 1
        audio_h = h[audio_mask_shifted]  # [audio_len, embed_dim]
        c0_logits = self.codebook0_head(audio_h)  # [audio_len, audio_vocab_size]
        c0_target = target_tokens[:, 0]  # [audio_len]
        c0_loss = F.cross_entropy(c0_logits, c0_target)
        backbone_logits = torch.zeros(
            bsz,
            seq_len,
            self.config.audio_vocab_size,
            device=device,
            dtype=c0_logits.dtype,
        )
        if audio_mask_shifted.any():
            backbone_logits[audio_mask_shifted] = c0_logits

        # predict text loss
        text_h = h[text_mask]
        text_logits = self.text_head(text_h)
        text_loss = F.cross_entropy(text_logits, text_target_tokens, ignore_index=0)
        full_text_logits = torch.zeros(
            bsz,
            seq_len,
            self.config.text_vocab_size,
            device=device,
            dtype=text_logits.dtype,
        )
        if text_mask.any():
            full_text_logits[text_mask] = text_logits

        # "compute amortization" (train decoder on random subset of audio tokens)
        # decoder_sampling_ratio: 1/8 (default) for efficiency, 1.0 for determinism
        num_samples = int(c_embeds.size(0) * self.decoder_sampling_ratio)
        indices = torch.randperm(c_embeds.size(0))[:num_samples]
        # [audio_len//16, n_codebooks-1, embed_dim]
        c_embeds = c_embeds[indices][:, :-1, :]
        audio_h = audio_h[indices]  # [audio_len//16, embed_dim]
        target_tokens = target_tokens[indices][:, 1:]  # [audio_len//16, n_codebooks-1]

        # concatenate backbone embeddings and codebook embeddings for decoder input
        # [audio_len//16, n_codebooks, embed_dim]
        decoder_embeds = torch.cat([audio_h.unsqueeze(1), c_embeds], dim=1)
        N, n_codebooks, _ = decoder_embeds.size()
        c_pos = (
            torch.arange(0, n_codebooks)
            .unsqueeze(0)
            .expand(N, n_codebooks)
            .long()
            .to(device)
        )

        decoder_causal_mask = _create_causal_mask(
            decoder_embeds.size(1), device
        ).expand(N, -1, -1)
        decoder_h = self.decoder(
            self.projection(decoder_embeds), input_pos=c_pos, mask=decoder_causal_mask
        ).to(dtype=dtype)
        c_logits = torch.einsum("bsd,sdv->bsv", decoder_h[:, 1:, :], self.audio_head)

        per_token_loss = F.cross_entropy(
            c_logits.reshape(-1, c_logits.size(-1)),
            target_tokens.reshape(-1),
            reduction="none",
        )
        c_loss = per_token_loss.mean()
        per_token_loss = per_token_loss.view(c_logits.size(0), c_logits.size(1))
        depth_decoder_logits = torch.zeros(
            bsz,
            seq_len,
            self.config.audio_num_codebooks - 1,
            self.config.audio_vocab_size,
            device=device,
            dtype=c_logits.dtype,
        )
        depth_decoder_loss_per_token = torch.zeros(
            bsz,
            seq_len,
            self.config.audio_num_codebooks - 1,
            device=device,
            dtype=per_token_loss.dtype,
        )
        depth_decoder_sample_mask = torch.zeros(
            bsz,
            seq_len,
            device=device,
            dtype=torch.bool,
        )
        if indices.numel() > 0:
            sampled_positions = audio_positions[indices]
            depth_decoder_logits[
                sampled_positions[:, 0], sampled_positions[:, 1]
            ] = c_logits
            depth_decoder_loss_per_token[
                sampled_positions[:, 0], sampled_positions[:, 1]
            ] = per_token_loss
            depth_decoder_sample_mask[sampled_positions[:, 0], sampled_positions[:, 1]] = True

        if self.use_text_loss:
            loss = (
                2
                * (
                    (1 - self.decoder_loss_weight) * c0_loss
                    + self.decoder_loss_weight * c_loss
                )
                + 0.01 * text_loss
            )
        else:
            loss = 2 * (
                (1 - self.decoder_loss_weight) * c0_loss
                + self.decoder_loss_weight * c_loss
            )
        return (
            loss,
            text_loss,
            c0_loss,
            c_loss,
            full_text_logits,
            backbone_logits,
            depth_decoder_logits,
            depth_decoder_loss_per_token,
            depth_decoder_sample_mask,
        )

    def generate_frame_and_logits(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        depth_decoder_temperature: float = 0.75,
        depth_decoder_topk: int = 10,
        eos_only_argmax: bool = False,
        codebook_eos_token_id: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate one audio frame with optional logits return.
        
        Debugging kwargs:
            last_text_eos_pos: (int) - Position of last text EOS token in sequence (for logging).

        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            temperature: Sampling temperature for backbone generation.
            topk: Top-k sampling for backbone generation.
            depth_decoder_temperature: Temperature for depth decoder sampling.
            depth_decoder_topk: Top-k for depth decoder sampling.
            eos_only_argmax: If True, only sample EOS if it has the highest probability.
                This prevents premature EOS sampling when it's not the most likely next token.
            codebook_eos_token_id: Token ID representing EOS in the codebook (default: 0).

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        # Debugging kwargs
        last_text_eos_pos = kwargs.get("last_text_eos_pos", None)
        distance_from_text_eos = None
        if logger.isEnabledFor(logging.DEBUG) and last_text_eos_pos is not None:
            logger.debug(f"last_text_eos_pos: {last_text_eos_pos}")
            distance_from_text_eos = input_pos[..., -1] - last_text_eos_pos

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(
            dtype=dtype
        )

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample, c0_indices_to_keep, topk_values = sample_topk_and_return_indices(c0_logits, topk, temperature, eos_only_argmax, codebook_eos_token_id)
        c0_embed = self._embed_audio(0, c0_sample)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )

        # If token 0 is in topk, then print logit value, and softmax value only if DEBUG mode is enabled
        if logger.isEnabledFor(logging.DEBUG) and (c0_logits[..., 0] >= topk_values[..., -1]).any():
            # Current position in the sequence
            curr_pos = input_pos[..., -1]
        
            
            # Handle batched tensors properly
            for batch_idx in range(c0_logits.shape[0]):
                if c0_logits[batch_idx, 0] >= topk_values[batch_idx, -1]:
                    softmax_probs = torch.nn.functional.softmax(c0_logits[batch_idx], dim=-1)

                    # Get token indices for this specific batch
                    batch_mask = c0_indices_to_keep[:, 0] == batch_idx
                    batch_token_indices = c0_indices_to_keep[batch_mask, 1]

                    # Sort these token indices by their logit values
                    sorted_order = torch.argsort(c0_logits[batch_idx, batch_token_indices], descending=True)
                    sorted_indices = batch_token_indices[sorted_order]

                    logger.info(f"Batch {batch_idx} - Current position: {curr_pos[batch_idx]}")
                    if distance_from_text_eos is not None:
                        logger.info(f"Batch {batch_idx} - Distance from last text EOS position: {distance_from_text_eos}")

                    logger.info(f"Batch {batch_idx} - Token 0           -- {c0_logits[batch_idx, 0]}        -- {softmax_probs[0]}")
                    logger.info(f"Batch {batch_idx} - max (Token {sorted_indices[0]}) -- {c0_logits[batch_idx, sorted_indices[0]]}        -- {softmax_probs[sorted_indices[0]]}")
                    logger.info(f"Batch {batch_idx} - min (Token {sorted_indices[-1]}) -- {c0_logits[batch_idx, sorted_indices[-1]]}        -- {softmax_probs[sorted_indices[-1]]}")
                    logger.info(f"Batch {batch_idx} - median (Token {sorted_indices[len(sorted_indices) // 2]}) -- {c0_logits[batch_idx, sorted_indices[len(sorted_indices) // 2]]}        -- {softmax_probs[sorted_indices[len(sorted_indices) // 2]]}")
                    sampled_token = c0_sample[batch_idx].item()
                    logger.info(f"Batch {batch_idx} - c0_sample (Token {c0_sample[batch_idx].item()}):        -- {c0_logits[batch_idx, sampled_token]}           -- {softmax_probs[c0_sample[batch_idx]]}")

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            ).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, depth_decoder_topk, depth_decoder_temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample, c0_logits, ci_logits
    
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor, 
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        depth_decoder_temperature: float = 0.75,
        depth_decoder_topk: int = 10,
        **kwargs
    ) -> torch.Tensor:
        curr_sample, _, _ = self.generate_frame_and_logits(
            tokens=tokens,
            tokens_mask=tokens_mask,
            input_pos=input_pos,
            temperature=temperature,
            topk=topk,
            depth_decoder_temperature=depth_decoder_temperature,
            depth_decoder_topk=depth_decoder_topk,
            **kwargs
        )
        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
