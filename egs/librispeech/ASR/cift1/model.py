import argparse
from typing import Optional, Tuple
import logging

import k2
import torch 
from torch import Tensor
import torch.nn as nn
# from encoder_interface import EncoderInterface

from icefall.utils import add_sos, make_pad_mask

from scaling import ScaledLinear
from zipformer2 import Zipformer2, get_zipformer_model, add_zipformer_arguments
from subsampling import ConvNeXt, get_encoder_embed_model, add_encoder_embed_arguments
from decoder import Decoder, get_decoder_model, add_decoder_arguments
from joiner import Joiner, get_joiner_model, add_joiner_arguments
from phi import Phi, get_phi_model, add_phi_arguments
from omega import Omega, get_omega_model, add_omega_arguments

def add_model_argument(parser : argparse.ArgumentParser):
    add_zipformer_arguments(parser)
    add_encoder_embed_arguments(parser)
    add_decoder_arguments(parser)
    add_joiner_arguments(parser)
    add_phi_arguments(parser)
    add_omega_arguments(parser)
    
    group = parser.add_argument_group("CifTModel related options")

    group.add_argument(
        "--detach-alphas",
        type=int,
        default=1,
    )
    
    group.add_argument(
        "--use-ctc",
        type=int,
        default=0,
        help="If 1, use CTC head."
    )
    
    group.add_argument(
        "--prune-range",
        type=int,
        default=5,
    )

def get_model(params) -> "CifTModel":
    def _to_int_tuple(s : str):
        return tuple(map(int, s.split(",")))
    params.encoder_dim = _to_int_tuple(params.encoder_dim)
    params.encoder_out_dim = max(params.encoder_dim)
    params.encoder_in_dim = params.encoder_dim[0]
    encoder = get_zipformer_model(params)
    encoder_embed = get_encoder_embed_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)
    phi = get_phi_model(params)
    omega = get_omega_model(params)
    
    return CifTModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        phi=phi,
        omega=omega,
        prune_range=params.prune_range,
        encoder_dim=params.encoder_out_dim,
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_ctc=params.use_ctc,
        detach_alphas=params.detach_alphas,
    )
    

class CifTModel(nn.Module):
    def __init__(
        self,
        encoder_embed: ConvNeXt,
        encoder: Zipformer2,
        decoder: Decoder,
        joiner: Joiner,
        phi: Phi,
        omega: Omega,
        prune_range : int,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_ctc: int = 0,
        detach_alphas = 1,
    ):
        """A joint CTC & CifTransducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        
        # assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder


        # Modules for Transducer head
        assert decoder is not None
        assert hasattr(decoder, "blank_id")
        assert joiner is not None

        self.decoder = decoder
        self.joiner = joiner

        self.simple_am_proj = ScaledLinear(
            encoder_dim, vocab_size, initial_scale=0.25
        )
        self.simple_lm_proj = ScaledLinear(
            decoder_dim, vocab_size, initial_scale=0.25
        )

        self.phi = phi
        self.omega = omega
        self.prune_range = prune_range
        self.detach_alphas = detach_alphas

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (B, T, C).
          x_lens:
            A 1-D tensor of shape (B,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (B, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (B,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(B, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=encoder_out_lens,
            target_lengths=target_lengths,
            reduction="sum",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=self.prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="none",
            )

        return simple_loss, pruned_loss

    def _compute_qua_loss(self, alphas : Tensor, target_lens : Tensor) -> Tensor:
        pred_n = alphas.sum(dim=1).squeeze(-1)
        return nn.functional.l1_loss(pred_n, target_lens, reduction="sum")

    def _adjust_alphas(self, alphas : Tensor, target_lens : Tensor) -> Tensor:
        if not self.training:
            return alphas
        
        if self.detach_alphas:
            alphas = alphas.detach().clone()
        
        pred_n = alphas.sum(dim=1).squeeze(-1) # pred_n : (B, )
        alphas *= (target_lens / (pred_n + 1e-10))[..., None, None]

        return alphas

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        y: k2.RaggedTensor,
        target_lens: Tensor, 
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, qua_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        # Compute CIF
        ratio : Tensor = (target_lens / (y_lens+1e-10)).clamp(min=( 1 / self.prune_range -3))
        target_lens = (y_lens * ratio).round() #.to(torch.int32)
        
        alphas = self.omega(encoder_out, encoder_out_lens)
                
        qua_loss = self._compute_qua_loss(alphas, target_lens.to(alphas))
        alphas = self._adjust_alphas(alphas, target_lens)
        encoder_out, encoder_out_lens = self.phi(encoder_out, encoder_out_lens, alphas)

        # Compute transducer loss
        simple_loss, pruned_loss = self.forward_transducer(
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            y=y.to(x.device),
            y_lens=y_lens,
            am_scale=am_scale,
            lm_scale=lm_scale,
        )

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)



        # Error handling for pruned_loss 

        if ((pruned_loss - pruned_loss) != 0.0).any():
            if self.training:
                pruned_input = {
                    "pruned_loss": pruned_loss,
                    "encoder_out": encoder_out,
                    "y": y,
                    "alphas": alphas,
                    "encoder_out_lens": encoder_out_lens
                }
                torch.save(pruned_input, "pruned_bad_case.pt")
                raise Exception(
                    "Bad case encountered with pruned loss. Saved to pruned_bad_case.pt"
                )
            logging.warning(
                f"Evaluation: ignoring inf in pruned_loss for loss calculation."
            )
            pruned_loss = pruned_loss.masked_fill(~pruned_loss.isfinite(), 0.)

        pruned_loss = torch.sum(pruned_loss)

        return simple_loss, pruned_loss, qua_loss, ctc_loss
