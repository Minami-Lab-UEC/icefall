import warnings
from typing import Dict, List, Tuple

import torch
from k2.ragged import RaggedShape, RaggedTensor, create_ragged_shape2
from model import CifTModel
from torch import Tensor

# Warning. This is not the same beam_search.py as the zipformer recipe.

class Hypothesis:
    def __init__(self, ys: List[int], log_prob: Tensor, T: int, t: int, s: int):
        # The predicted tokens so far.
        # Newly predicted tokens are appended to `ys`.
        self.ys = ys
        # The log prob of ys.
        self.log_prob = log_prob
        self.T = T
        self.t = t
        self.s = s

    @property
    def key(self) -> str:
        return f"{self.t}:" + "_".join(map(str, self.ys))

    @property
    def done(self) -> bool:
        return self.t >= self.T


class HypothesisList(object):
    def __init__(
        self,
        b: int,
        T: int,
        V: Tensor,
        move_t_syms: List[int],
        starting_ys: List[int],
        beam: int,
        device: torch.device,
        max_s_per_t=0,
    ):
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        self.b = b
        self.T = T
        starting_point = Hypothesis(
            ys=starting_ys,
            log_prob=torch.zeros([1], dtype=torch.float32, device=device),
            t=0, 
            T=T,
            s=0,
        )
        self.active_hyps: Dict[str, Hypothesis] = {starting_point.key: starting_point}

        self.done_hyps: Dict[str, Hypothesis] = {}
        self.move_t_syms = move_t_syms
        self.V = V
        self.beam = beam
        self.device = device
        self.max_s_per_t = max_s_per_t
        self.data = {**self.active_hyps, **self.done_hyps}

    @property
    def done(self) -> bool:
        return not self.active_hyps

    def advance(self, next_lprobs: Tensor):
        new_active_hyps: Dict[str, Hypothesis] = {}
        new_done_hyps: Dict[str, Hypothesis] = {}
        active_hyps_list: List[Hypothesis] = [hyp for hyp in self.active_hyps.values()]
        done_hyps_list: List[Hypothesis] = [hyp for hyp in self.done_hyps.values()]

        next_lprobs_len = next_lprobs.size(0)

        if self.done_hyps:
            next_lprobs = torch.concat(
                [next_lprobs] + [hyp.log_prob for hyp in done_hyps_list]
            )

        topk_log_probs, topk_indexes = next_lprobs.topk(self.beam)
        topk_token_indexes = (topk_indexes % self.V).tolist()

        done_mask: Tensor = topk_indexes >= next_lprobs_len


        for is_done, hyp_idx, new_token, new_log_prob in zip(
            done_mask, topk_indexes, topk_token_indexes, topk_log_probs
        ):

            if not is_done:
                assert new_token >= 0, new_token
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0
                    # (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior,
                    # use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').

                    this_hyp_idx = hyp_idx // self.V
                    hyp : Hypothesis = active_hyps_list[this_hyp_idx]
                new_ys: List[int] = hyp.ys
                t = hyp.t
                s = hyp.s

                # ------------------------------------------------------------------------------------------------------
                # This part is different from beam_search_5.py

                if new_token in self.move_t_syms:
                    t += 1 
                    s = 0
                else:
                    new_ys = new_ys + [new_token]
                    if s == self.max_s_per_t:
                        t += 1
                        s = 0
                    else:
                        # move in u direction
                        s += 1
                # -------------------------------------------------------------------------------------------------------

                # get original probability
                log_prob = new_log_prob 
                hyp = Hypothesis(
                    ys=new_ys, log_prob=log_prob[None], T=self.T, t=t, s=s
                )
                which_hyps = new_done_hyps if hyp.done else new_active_hyps
            else:
                hyp = done_hyps_list[hyp_idx - next_lprobs_len]
                which_hyps = new_done_hyps

            key = hyp.key
            if key in which_hyps:
                which_hyps[key].log_prob = torch.logaddexp(
                    which_hyps[key].log_prob, hyp.log_prob
                )
            else:
                which_hyps[key] = hyp

        self.active_hyps = new_active_hyps
        self.done_hyps = new_done_hyps
        self.data = {**self.active_hyps, **self.done_hyps}

    def get_most_probable(
        self, length_norm: bool = False, use_field: bool = False
    ) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(
                self.data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys)
            )
        else:
            return max(self.data.values(), key=lambda hyp: hyp.log_prob)

    def __contains__(self, key: str):
        return key in self.data

    def __iter__(self):
        return iter(self.data.values())

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        s = []
        for key in self.data.values():
            s.append(key)
        return ", ".join(s)



class BatchHypothesisList:
    def __init__(
        self,
        B: int,
        T: Tensor,
        context_size: int,
        V: int,
        starting_sym: int,
        beam=4,
        move_t_syms=[0],
        device=torch.device("cpu"),
        max_s_per_t=5,
    ):
        assert T.size(0) == B, (T.size(0), B)
        V = torch.tensor([V], device=device)
        self.active_hypotheses = [
            HypothesisList(
                b=i,
                T=t.cpu(),
                move_t_syms=move_t_syms,
                V=V,
                beam=beam,
                device=device,
                max_s_per_t=max_s_per_t,
                starting_ys=[starting_sym] * context_size,
            )
            for i, t in enumerate(T)
        ]
        self.done_hypotheses: List[HypothesisList] = []
        self.V = V
        self.device = device
        assert context_size >= 0, context_size
        self.context_size = context_size
        self.steps_taken = 0
        self.max_s_per_t = max_s_per_t

    @property
    def done(self):
        return not self.active_hypotheses

    @property
    def active_log_probs(self):
        ans = torch.concat(
            [
                hyp.log_prob
                for hyps in self.active_hypotheses
                for hyp in hyps.active_hyps.values()
            ]
        )
        return ans

    @property
    def active_ys(self) -> List:
        return torch.tensor(
            [
                hyp.ys[-self.context_size :]
                for hyps in self.active_hypotheses
                for hyp in hyps.active_hyps.values()
            ],
            dtype=torch.int64,
            device=self.device,
        )

    @property
    def active_shape(self) -> RaggedShape:
        """Return a ragged shape with axes [utt][num_hyps].

        Args:
        hyps:
            len(hyps) == batch_size. It contains the current hypothesis for
            each utterance in the batch.
        Returns:
        Return a ragged shape with 2 axes [utt][num_hyps]. Note that
        the shape is on CPU.
        """
        num_hyps = [len(h.active_hyps) for h in self.active_hypotheses]

        # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
        # to get exclusive sum later.
        num_hyps.insert(0, 0)

        num_hyps = torch.tensor(num_hyps, device=self.device)
        row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
        ans = create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=row_splits[-1].item()
        )
        return ans 

    @property
    def BTs(self) -> Tuple[List[int]]:
        ans = [
            (hyplist.b, hyp.t)
            for hyplist in self.active_hypotheses
            for hyp in hyplist.active_hyps.values()
        ]

        return tuple(zip(*ans))

    def advance(self, ragged_log_probs: RaggedTensor):
        assert len(self.active_hypotheses) == ragged_log_probs.dim0, len(
            self.active_hypotheses, ragged_log_probs
        )

        # NOTE: Although k2.RaggedTensor does not implement iter and will not stop iteration by itself,
        # this works because zip is limited by self.hypotheses which is same in length as ragged_log_probs.dim0.
        active_hypotheses = []
        for hypotheses, log_probs in zip(self.active_hypotheses, ragged_log_probs):
            hypotheses.advance(log_probs)

            if hypotheses.done:
                self.done_hypotheses.append(hypotheses)
            else:
                active_hypotheses.append(hypotheses)

        self.active_hypotheses = active_hypotheses
        self.steps_taken += 1

        if self.steps_taken >= 10000:
            self.active_hypotheses = []

    def get_most_probable(self, length_norm=True, use_field=False) -> List[List[int]]:
        self.done_hypotheses = sorted(
            self.done_hypotheses, key=lambda hyplist: hyplist.b
        )
        best_hyps = [
            hyplist.get_most_probable(length_norm=length_norm, use_field=use_field)
            for hyplist in self.done_hypotheses
        ]
        best_hyps = [h.ys[self.context_size :] for h in best_hyps]

        return best_hyps


def beam_search(
    model: CifTModel,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam=4,
    max_s_per_t=5,
    temperature: float = 1.0,
) -> List[List[int]]:
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    gaussian_diag_cov = (
        [float(v) for v in gaussian_diag_cov.split(",")] if gaussian_diag_cov else None
    )

    vocab_size = model.joiner.output_linear.out_features
    unk_id = getattr(model, "unk_id", -1)
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    B = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens

    batchHypotheses = BatchHypothesisList(
        B=B,
        T=encoder_out_lens,
        context_size=context_size,
        V=vocab_size,
        beam=beam,
        starting_sym=blank_id,
        move_t_syms=[model.decoder.blank_id, unk_id],
        device=device,
        max_s_per_t=max_s_per_t,
    )

    encoder_out = model.joiner.encoder_proj(encoder_out)

    while not batchHypotheses.done:
        bs, ts = batchHypotheses.BTs
        current_encoder_out = encoder_out[
            bs,
            ts,
            None,
            None,
            :,
        ]  # (num_hyps, 1, 1, joiner_dim)
        ys_log_probs = batchHypotheses.active_log_probs  # (num_hyps, 1)

        ys = batchHypotheses.active_ys
        decoder_out = model.decoder(ys, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(
            decoder_out
        )  # (num_hyps, 1, 1, joiner_dim)

        logits: torch.Tensor = model.joiner(
            current_encoder_out, decoder_out, project_input=False
        )  # (num_hyps, 1, 1, V)

        logits = logits[:,0,0,:]

        log_probs = (logits / temperature).log_softmax(dim=-1)  # (num_hyps, V)

        log_probs += ys_log_probs[:, None]

        log_probs = log_probs.reshape(-1)

        row_splits = batchHypotheses.active_shape.row_splits(1) * vocab_size
        log_probs_shape = create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = RaggedTensor(shape=log_probs_shape, value=log_probs)

        batchHypotheses.advance(ragged_log_probs)

    return batchHypotheses.get_most_probable(length_norm=True)

def greedy_search(
    model: CifTModel,
    encoder_out: torch.Tensor,
    max_sym_per_frame: int,
) -> List[int]:
    """Greedy search for a single utterance.
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      max_sym_per_frame:
        Maximum number of symbols per frame. If it is set to 0, the WER
        would be 100%.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    unk_id = getattr(model, "unk_id", blank_id)

    device = next(model.parameters()).device

    decoder_input = torch.tensor(
        [-1] * (context_size - 1) + [blank_id], device=device, dtype=torch.int64
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    encoder_out = model.joiner.encoder_proj(encoder_out)

    T = encoder_out.size(1)
    t = 0
    hyp = [blank_id] * context_size

    # timestamp[i] is the frame index after subsampling
    # on which hyp[i] is decoded
    timestamp = []

    # Maximum symbols per utterance.
    max_sym_per_utt = 1000

    # symbols per frame
    sym_per_frame = 0

    # symbols per utterance decoded so far
    sym_per_utt = 0

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue

        current_encoder_out = encoder_out[:, t:t+1, None, :] 
        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits is (1, 1, 1, vocab_size)

        y = logits.argmax().item()
        if y not in (blank_id, unk_id):
            hyp.append(y)
            timestamp.append(t)
            decoder_input = torch.tensor([hyp[-context_size:]], device=device).reshape(
                1, context_size
            )

            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

            sym_per_utt += 1
            sym_per_frame += 1

        else:
            sym_per_frame = 0
            t += 1

    hyp = hyp[context_size:]  # remove blanks

    return hyp
