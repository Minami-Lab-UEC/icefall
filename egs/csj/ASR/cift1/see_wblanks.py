#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:
(1) greedy search
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --lang data/lang_char \
    --decoding-method greedy_search

(2) beam search (not recommended)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method beam_search \
    --lang data/lang_char \
    --beam-size 4

(3) modified beam search
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method modified_beam_search \
    --lang data/lang_char \
    --beam-size 4

(4) fast beam search (one best)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --lang data/lang_char \
    --max-states 64

(5) fast beam search (nbest)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --lang data/lang_char \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --lang data/lang_char \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --lang data/lang_char \
    --max-states 64
"""


import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from asr_datamodule import CSJAsrDataModule
from tokenizer import Tokenizer
from train import get_params
from model import add_model_arguments, get_model, CifTModel

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
    add_sos,
)
from icefall import LmScorer
from targetlens import TargetLength
import k2

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        default="zipformer/exp",
        help="The experiment dir",
    )
    
    parser.add_argument(
        "--res-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding-method is greedy_search""",
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--pad-feature",
        type=int,
        default=0,
    )


    add_model_arguments(parser)

    return parser

def forced_alignment(
    logits : torch.Tensor,
    tokens : List[int],
    blank_id = 0,
) -> List:
    logits = logits[0].cpu()
    T, U, V = logits.shape    
    
    # 1. Build trellis
    trellis = torch.zeros((T, U))
    trellis[1:, 0] = logits[:-1, 0, 0].cumsum(dim=0)
    trellis[0, 1:] = logits[0, range(U-1), tokens].cumsum(dim=0)
    for t in range(1, T):
        for u in range(1, U):
            trellis[t, u] = torch.maximum(
                trellis[t-1, u] + logits[t-1, u, blank_id],
                trellis[t, u-1] + logits[t, u-1, tokens[u-1]],
            )
    
    # 2. Backtrack
    t = T-1
    u = U-1
    path = [(t, u, blank_id)]
    for step in range(T+U, 2, -1):
        if not t:
            u -= 1
            path.append((t, u, tokens[u]))
        elif not u:
            t -= 1
            path.append((t, u, blank_id))
        elif trellis[t-1, u] > trellis[t, u-1]:
            t -= 1
            path.append((t, u, blank_id))
        else:
            u -= 1
            path.append((t, u, tokens[u]))
    assert (t, u) == (0, 0), (t, u)

    return path[::-1]   
    
    

def best_path_search(
    model: CifTModel,
    encoder_out: torch.Tensor,
    tokens: List[int],
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

    device = next(model.parameters()).device

    y = k2.RaggedTensor([tokens]).to(device)
    sos_y = add_sos(y, sos_id=0)
    sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

    decoder_out = model.decoder(sos_y_padded)

    full_logits = model.joiner(encoder_out.unsqueeze(2), decoder_out.unsqueeze(1))

    forced_alignment_path = forced_alignment(full_logits, tokens)
    tokens_w_blank = [t for *_, t in forced_alignment_path]
    return tokens_w_blank


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: Tokenizer,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      LM:
        A neural network language model.
      ngram_lm:
        A ngram language model
      ngram_lm_scale:
        The scale for the ngram language model.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    if params.pad_feature:
        feature_lens += params.pad_feature
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, params.pad_feature),
            value=LOG_EPS,
        )
    encoder_out, encoder_out_lens, feature_mask = model.forward_encoder(feature, feature_lens)
    alphas = model.omega(encoder_out, encoder_out_lens)
    encoder_out2, encoder_out_lens2 = model.phi(encoder_out, encoder_out_lens, alphas, feature_mask)
    batch_size = encoder_out2.size(0)

    hyps_w_blank = []
    for i in range(batch_size):
        hyp_w_blank = best_path_search(
            model=model,
            encoder_out=encoder_out2[i, None, : encoder_out_lens2[i]],
            tokens=sp.encode(batch["supervisions"]["text"][i]),
        )
        hyp_w_blank = sp.decode(hyp_w_blank)
        hyp_w_blank = hyp_w_blank.replace("<blk>", "∅")
        hyp_w_blank = ' '.join(list(hyp_w_blank))
        hyps_w_blank.append(hyp_w_blank)

    return {"forced_alignment_search": hyps_w_blank}



def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: Tokenizer,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding-method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    log_interval = 50

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
        )

        for name, hyps_w_blank in hyps_dict.items():
            this_batch = []
            assert len(hyps_w_blank) == len(cut_ids)
            for cut_id, hyp_w_blank in zip(cut_ids, hyps_w_blank):
                this_batch.append((cut_id, hyp_w_blank))

            results[name].extend(this_batch)

        num_cuts += len(cut_ids)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    for key, results in results_dict.items():
        wblank_path = (
            params.res_dir / f"wblank-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        with open(wblank_path, "w", encoding="utf8") as f1:
            for cut_id, hyp_w_blank in results:
                print(f"{cut_id}:\thyp={hyp_w_blank}", file=f1)

        logging.info(f"The forced alignment paths are stored in {wblank_path}")


@torch.no_grad()
def main():
    parser = get_parser()
    CSJAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    Tokenizer.add_arguments(parser)
    TargetLength.add_targetlength_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    if not params.res_dir:
        params.res_dir = params.exp_dir / "see_blanks"


    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"-chunk-{params.chunk_size}"
        params.suffix += f"-left-context-{params.left_context_frames}"


    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available() and params.gpu is not None:
        device = torch.device("cuda", params.gpu)

    logging.info(f"Device: {device}")

    sp = Tokenizer.load(params.lang)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()


    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    csj_corpus = CSJAsrDataModule(args)

    for subdir in ["eval1", "eval2", "eval3", "excluded", "valid"]:
        results_dict = decode_dataset(
            dl=csj_corpus.test_dataloaders(getattr(csj_corpus, f"{subdir}_cuts")()),
            params=params,
            model=model,
            sp=sp,
        )

        save_results(
            params=params,
            test_set_name=subdir,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
