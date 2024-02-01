import torch
import torch.nn as nn
from torch.nn import LayerNorm 
from scaling import BiasNorm,  penalize_abs_values_gt, ScaledLinear, Whiten
from zipformer2 import _whitening_schedule
from torch import Tensor
import math

from icefall.utils import make_pad_mask
import argparse
import random

def add_phi_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title = "CIF pooler (Phi) related options")
    
    group.add_argument(
        "--phi-type",
        type=str,
        default="att",
    )
    
    group.add_argument(
        "--phi-arch",
        type=str,
        default="vanilla",
    )
    
    group.add_argument(
        "--phi-norm-type",
        type=str,
        default="layernorm"
    )

def get_phi_model(params) -> nn.Module:
    if params.phi_arch == "vanilla":
        phi = VanillaPooler(
            d_model=params.encoder_out_dim,
            phi_type=params.phi_type,
            norm_type=params.phi_norm_type,
        )
    else:
        raise TypeError(f"--phi-arch {params.phi_arch} not recognised")
    
    return phi


class RelPositionalEncoding(torch.nn.Module):
    # Modified from https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        num_heads: int = 8,
        max_len: int = 5000,
    ) -> None:
        """Construct a PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.pe = None
        self.num_heads = num_heads
        self.extend_pe(torch.tensor(0.0).expand(1, 1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        x_size = x.size(2)
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(2) >= x_size * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size, self.d_model)
        pe_negative = torch.zeros(x_size, self.d_model)
        position = torch.arange(0, x_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive2 = torch.flip(pe_positive, [0])
        pe_negative = pe_negative[1:]
        pe = torch.cat([pe_positive2, pe_negative], dim=0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)
        S = self.pe.size(0)
        self.pe = self.pe.reshape(1, S, self.num_heads, -1).permute(0, 2, 1, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.

        Args:
            x (Tensor): Input tensor (batch, num_heads, S)

        Returns:
            Tensor: Encoded tensor (batch, num_heads, left_context_len + 2*time-1, `*`).

        """
        self.extend_pe(x)
        x_size = x.size(2)
        left_from = (self.pe.size(2) -  x_size) // 2 
        pos_emb = self.pe[
            :,
            :,
            left_from : left_from  # noqa E203
            + x_size,
        ]
        return self.dropout(pos_emb)

class Phi(nn.Module):
    def __init__(self):
        super().__init__()
        
    def _get_firing_points(self, alphas : Tensor, src_key_padding_mask : Tensor):
        fire_cumsum = alphas.cumsum(dim=1)
        out_lens = fire_cumsum[:, -1].round().to(torch.int32).clamp(min=1)
        T1 = out_lens.max()
        
        # All feature vectors which should be masked are added to T1 in scatter_add_, and truncated in x_out
        firing_points = (fire_cumsum-0.5).round().to(torch.int64)
        firing_points = firing_points.roll(shifts=1, dims=1) 
        firing_points[...,0] = 0
        mask = torch.logical_or(src_key_padding_mask, firing_points == out_lens[..., None])
        firing_points = firing_points.masked_fill(mask, T1)
        return firing_points, out_lens, T1
    
    def forward(self, src : Tensor, src_key_padding_mask: Tensor, alphas : Tensor) -> Tensor:
        """

        Args:
            src (Tensor): (B, T, C)
            src_key_padding_mask (Tensor): (B, T)
            alphas (Tensor): (B, T)

        Raises:
            NotImplementedError: Implement in child classes

        Returns:
            Tensor: x_out (B, T1, C)
            Tensor: x_out_lens (B, )
        """
        raise NotImplementedError
    
    def _collapse(self, src : Tensor, firing_points : Tensor, T1 : int) -> Tensor:
        B, T, C = src.shape
        x_out : Tensor = torch.zeros((B, T1+1, C), dtype=src.dtype, device=src.device)
        x_out.scatter_add_(
            dim=1,
            index=firing_points[...,None].expand_as(src),
            src=src
        )
        # Remove the extra time slot added in _get_firing_points
        return x_out[:,:-1]

class AttnPhi(Phi):
    def __init__(
        self, 
        d_model,
        dropout = 0.1,
        num_heads = 8,
    ):
        """Attention pooling via query

        Args:
            d_model (int): Dimension of the model backbone
            dropout (float, optional): Dropout probability for attention matrix and positional embedding. Defaults to 0.1.
            num_heads (int, optional): Number of heads. Defaults to 8.
        """
        super(AttnPhi, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        assert not (d_model % self.num_heads), (
            f"d_model {d_model} must be divisible by num_heads {num_heads}."
        )
        self.d_att = d_model // self.num_heads 
        self.query = nn.Parameter(torch.randn(self.num_heads, self.d_att) * self.d_att ** -0.5)        
        self.pos = RelPositionalEncoding(d_model, dropout, num_heads)
    
    def _get_pos(self, src : Tensor):
        B, h, T, C = src.shape
        pos_emb : Tensor = self.pos(src) * (self.d_model ** -0.5)
        
        return pos_emb
    
    def forward(self, src: Tensor, src_key_padding_mask : Tensor, alphas : Tensor):
        """Calculates firing points and compresses input `src` into `x_out`, i.e. C. 

        Args:
            src (Tensor): Input
            src_key_padding_mask (Tensor): _description_
            alphas (Tensor): Weights as predicted by Omega function

        Returns:
            Tensor: Compressed representation `x_out`, i.e. C.
            Tensor: Length of compressed representation
        """
        firing_points, out_lens, T1 = self._get_firing_points(alphas[..., 0], src_key_padding_mask)
        
        B, T, C = src.shape
        key = src.reshape(B, T, self.num_heads, self.d_att).permute(0, 2, 1, 3) # (B, h, T, d_att)
        
        scores = (self.query[None, :, None, :] * key).sum(dim=-1) # (B, h, T)
        
        if not torch.jit.is_scripting():
            if self.training and random.random() < 0.1:
                # This is a harder way of limiting the attention scores to not be too large.
                # It incurs a penalty if any of them has an absolute value greater than 50.0.
                # this should be outside the normal range of the attention scores.  We use
                # this mechanism instead of, say, a limit on entropy, because once the entropy
                # gets very small gradients through the softmax can become very small, and
                # some mechanisms like that become ineffective.
                scores = penalize_abs_values_gt(
                    scores, limit=25.0, penalty=1.0e-04
                )
        
        weights = self._ragged_softmax(scores, firing_points[:, None, :].expand_as(scores), T1)
        weights = self.dropout(weights)
        
        value = key + self._get_pos(key) # (B, h, T, d_att)
        x_out = value * weights[..., None]
        x_out = x_out.permute(0, 2, 1, 3).reshape(B, T, -1) # (B, T, C)
        x_out = self._collapse(x_out, firing_points, T1) # (B, T, C) -> (B, M, C)
        return x_out, out_lens
        
        
    def _ragged_softmax(self, x: Tensor, firing_points: Tensor, T1: int):
        # ragged softmax via logsumexp trick

        B = x.size(0)
        x_max = x.max()
        x_normed = x - x_max
        x_exp = torch.exp(x_normed)
        x_sumexp : Tensor= torch.zeros((B, self.num_heads, T1+1), dtype=x.dtype, device=x.device) # (B, T1+1)
        x_sumexp.scatter_add_(dim=-1, index=firing_points, src=x_exp)
        x_logsumexp = torch.log(x_sumexp) + x_max # (B, h, T1+1)
        x_denom = x_logsumexp.gather(dim=-1, index=firing_points) # (B, h, T1+1) -> (B, h, T)
        weights = torch.exp(x - x_denom)
        return weights

class MiniQAttnPhi(Phi):
    def __init__(
        self, 
        d_model : int,
        d_att : int,
        dropout = 0.1,
        num_heads = 8,
    ):
        """Attention pooling via query

        Args:
            d_model (int): Dimension of the model backbone
            dropout (float, optional): Dropout probability for attention matrix and positional embedding. Defaults to 0.1.
            num_heads (int, optional): Number of heads. Defaults to 8.
        """
        super(MiniQAttnPhi, self).__init__()
        self.d_model = d_model
        self.d_att = d_att
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        assert not (d_model % self.num_heads), (
            f"d_model {d_model} must be divisible by num_heads {num_heads}."
        )
        d_per_head = d_model // self.num_heads 
        
        self.proj_k = ScaledLinear(
            in_features=d_per_head,
            out_features=d_att,
        )
        
        self.query = nn.Parameter(torch.randn(self.num_heads, self.d_att) * self.d_att ** -0.5)        
        self.pos = RelPositionalEncoding(d_model, dropout, num_heads)
    
    def _get_pos(self, src : Tensor):
        pos_emb : Tensor = self.pos(src) * (self.d_model ** -0.5)
        return pos_emb
    
    def forward(self, src: Tensor, src_key_padding_mask : Tensor, alphas : Tensor):
        """Calculates firing points and compresses input `src` into `x_out`, i.e. C. 

        Args:
            src (Tensor): Input
            src_key_padding_mask (Tensor): _description_
            alphas (Tensor): Weights as predicted by Omega function

        Returns:
            Tensor: Compressed representation `x_out`, i.e. C.
            Tensor: Length of compressed representation
        """
        firing_points, out_lens, T1 = self._get_firing_points(alphas[..., 0], src_key_padding_mask)
        
        B, T, C = src.shape
        key = src.reshape(B, T, self.num_heads, self.d_att).permute(0, 2, 1, 3) # (B, h, T, d_per_head)
        key : Tensor = self.proj_k(key) # (B, h, T, d_att)

        scores = (self.query[None, :, None, :] * key).sum(dim=-1) # (B, h, T)
        
        if not torch.jit.is_scripting():
            if self.training and random.random() < 0.1:
                # This is a harder way of limiting the attention scores to not be too large.
                # It incurs a penalty if any of them has an absolute value greater than 50.0.
                # this should be outside the normal range of the attention scores.  We use
                # this mechanism instead of, say, a limit on entropy, because once the entropy
                # gets very small gradients through the softmax can become very small, and
                # some mechanisms like that become ineffective.
                scores = penalize_abs_values_gt(
                    scores, limit=25.0, penalty=1.0e-04
                )
        
        weights = self._ragged_softmax(scores, firing_points[:, None, :].expand_as(scores), T1)
        weights = self.dropout(weights)

        value = value + self._get_pos(value) # (B, h, T, d_att)
        x_out = value * weights[..., None]
        x_out = x_out.permute(0, 2, 1, 3).reshape(B, T, -1) # (B, T, C)
        x_out = self._collapse(x_out, firing_points, T1) # (B, T, C) -> (B, M, C)
        return x_out, out_lens
        
        
    def _ragged_softmax(self, x: Tensor, firing_points: Tensor, T1: int):
        # ragged softmax via logsumexp trick

        B = x.size(0)
        x_max = x.max()
        x_normed = x - x_max
        x_exp = torch.exp(x_normed)
        x_sumexp : Tensor= torch.zeros((B, self.num_heads, T1+1), dtype=x.dtype, device=x.device) # (B, T1+1)
        x_sumexp.scatter_add_(dim=-1, index=firing_points, src=x_exp)
        x_logsumexp = torch.log(x_sumexp) + x_max # (B, h, T1+1)
        x_denom = x_logsumexp.gather(dim=-1, index=firing_points) # (B, h, T1+1) -> (B, h, T)
        weights = torch.exp(x - x_denom)
        return weights


class OriCIFPhi(Phi):
    def __init__(
        self, 
        tail_thres = 0.5, 
        beta = 1.0, 
        eps = 1e-4,
    ):
        """ A fast parallel implementation of continuous integrate-and-fire (CIF)
        https://arxiv.org/abs/1905.11235

        Args:
            beta (float): the threshold used for determine firing.
            tail_thres (float): the threshold for determine firing for tail handling.
            eps (float, optional): Epsilon to prevent underflow for divisions.
                Default: 1e-4
        """
        super(OriCIFPhi, self).__init__()
        self.tail_thres = tail_thres
        self.beta = beta 
        self.eps = eps
    
    def forward(
        self, 
        src : Tensor, 
        src_key_padding_mask : Tensor,
        alphas : Tensor, 
        # target_lengths : Tensor = None,
    ):
        r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
        https://arxiv.org/abs/1905.11235

        Args:
            src (Tensor): (B, S, C) Input features to be integrated.
            src_lens (Tensor): (B,) Input feature lengths
            alpha (Tensor): (B, S, 1) Weights corresponding to each elements in the
                input. It is expected to be 0 < x < 1.0. Assumes that alphas are 
                adjusted to target_n.
            target_lengths (Tensor, optional): (B,) Desired length of the targets
                for each sample in the minibatch.

        Returns -> Dict[str, List[Optional[Tensor]]]: Key/values described below.
            cif_out (Tensor): (B, T, C) The output integrated from the source.
            cif_lengths (Tensor): (B,) The output length for each element in batch.
            alpha_sum (Tensor): (B,) The sum of alpha for each element in batch.
                Can be used to compute the quantity loss.
            delays (Tensor): (B, T) The expected delay (in terms of source tokens) for
                each target tokens in the batch.
            tail_weights (Tensor, optional): (B,) During inference, return the tail.
        """
        alphas = alphas.squeeze(-1)
        
        B, S, C = src.size()
        assert tuple(alphas.size()) == (B, S), f"{alphas.size()} != {(B, S)}"
        
        alphas = alphas.masked_fill(src_key_padding_mask, 0)

        """
        Done in `model.py`, `_adjust_alphas`. 
        """
        # if target_lengths is not None:
        #     feat_lengths = target_lengths.long()
        #     desired_sum = self.beta * target_lengths.type_as(src) + self.eps
        #     alpha_sum = alphas.sum(1)
        #     alphas = alphas * (desired_sum / alpha_sum).unsqueeze(1)
        #     T = feat_lengths.max()
        # else:
        #     alpha_sum = alphas.sum(1)
        #     feat_lengths = (alpha_sum / self.beta).floor().long()
        #     T = feat_lengths.max()

        # Assumes that alphas have been adjusted.
        alpha_sum = alphas.sum(1)
        
        feat_lengths = (alpha_sum / self.beta).floor().long()
        
        T = feat_lengths.max()
        
        # aggregate and integrate
        csum = alphas.cumsum(-1)
        

        with torch.no_grad():
            # indices used for scattering
            right_idx = (csum / self.beta).round().long().clip(max=T)
            left_idx = right_idx.roll(1, dims=1)
            left_idx[:, 0] = 0
            

            # count # of fires from each source
            fire_num = right_idx - left_idx
            extra_weights = (fire_num - 1).clip(min=0)
            

        # The extra entry in last dim is for
        output : Tensor = src.new_zeros((B, T + 1, C))
        
        # delay : Tensor = src.new_zeros((B, T + 1))
        # source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(src)
        zero = alphas.new_zeros((1,))

        # right scatter
        fire_mask = fire_num > 0
        
        right_weight = torch.where(
            fire_mask,
            csum - right_idx.type_as(alphas) * self.beta,
            zero
        ).type_as(src)
        
        # assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
        output.scatter_add_(
            1,
            right_idx.unsqueeze(-1).expand(-1, -1, C),
            right_weight.unsqueeze(-1) * src
        )
        
        # delay.scatter_add_(
        #     1,
        #     right_idx,
        #     right_weight * source_range / self.beta
        # )

        # left scatter
        left_weight = (
            alphas - right_weight - extra_weights.type_as(alphas) * self.beta
        ).type_as(src)
        
        output.scatter_add_(
            1,
            left_idx.unsqueeze(-1).expand(-1, -1, C),
            left_weight.unsqueeze(-1) * src
        )
        
        # delay.scatter_add_(
        #     1,
        #     left_idx,
        #     left_weight * source_range / self.beta
        # )

        # extra scatters
        if extra_weights.ge(0).any():
            extra_steps = extra_weights.max().item()
            tgt_idx = left_idx
            src_feats = src * self.beta
            
            for _ in range(extra_steps):
                tgt_idx = (tgt_idx + 1).clip(max=T)
                # (B, S, 1)
                src_mask = (extra_weights > 0)
                output.scatter_add_(
                    1,
                    tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                    src_feats * src_mask.unsqueeze(2)
                )
                # delay.scatter_add_(
                #     1,
                #     tgt_idx,
                #     source_range * src_mask
                # )
                extra_weights -= 1
                

        # tail handling
        # if target_lengths is not None:
        #     # training time -> ignore tail
        #     output = output[:, :T, :]
        #     # delay = delay[:, :T]
        # else:
        # find out contribution to output tail
        # note: w/o scaling, extra weight is all 0
        zero = right_weight.new_zeros((1,))
        r_mask = right_idx == feat_lengths.unsqueeze(1)
        tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
        l_mask = left_idx == feat_lengths.unsqueeze(1)
        tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

        # a size (B,) mask that extends position that passed threshold.
        extend_mask = tail_weights >= self.tail_thres

        # extend 1 fire and upscale the weights
        if extend_mask.any():
            # (B, T, C), may have infs so need the mask
            upscale = (
                torch.ones_like(output)
                .scatter(
                    1,
                    feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                    self.beta / tail_weights.masked_fill(~extend_mask, self.beta).view(B, 1, 1).expand(-1, -1, C),
                )
                .detach()
            )
            output *= upscale
            feat_lengths += extend_mask.long()
            T = feat_lengths.max()
        output = output[:, :T, :]
        # delay = delay[:, :T]

        # a size (B, T) mask to erase weights
        tail_mask = torch.arange(T, device=output.device).unsqueeze(0) >= feat_lengths.unsqueeze(1)
        output[tail_mask] = 0

        
        return output, feat_lengths.clamp(min=1)

class VanillaPooler(nn.Module):
    def __init__(
        self,
        d_model,
        phi_type="att",
        norm_type="layernorm",
        dropout=0.1,
    ):
        super().__init__()
        
        if norm_type == "layernorm":
            self.norm_final = LayerNorm(d_model)
        elif norm_type == "biasnorm":
            self.norm_final = BiasNorm(d_model)
        else:
            raise TypeError(f"--phi-norm-type {norm_type} not recognised.")
        
        phi_args = phi_type.split(";")[1] if ";" in phi_type else ""
        
        if "att" in phi_type:
            head = phi_args.split(",")
            head = int(head)
            self.pooler : Phi = AttnPhi(d_model, dropout, num_heads=head)
        elif "miniatt" in phi_type:
            head, d_att = phi_args.split(",")
            head = int(head)
            d_att = int(d_att)
            self.pooler : Phi = MiniQAttnPhi(d_model, d_att, dropout, head)
        elif "ori" == phi_type:
            self.pooler : Phi = OriCIFPhi()
        else:
            raise TypeError(f"--phi-type {phi_type} not recognised.")
    
    def forward(self, src: Tensor, src_lens : Tensor, alphas : Tensor, feature_mask : Tensor):
        """

        Args:
            src (Tensor): Encoder output, (B, T, C)
            src_lens (Tensor): Encoder output length, (B,)
            alphas (Tensor): Alphas obtained from Omega function

        Returns:
            Tensor: Collapsed feature output
            Tensor: Collapsed feature length 
        """
        src_key_padding_mask = make_pad_mask(src_lens)
        out, out_lens = self.pooler(src, src_key_padding_mask, alphas)
        out = self.norm_final(out)
        out = out * feature_mask
        return out, out_lens