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
from asr_datamodule import LibriSpeechAsrDataModule
from beam_search import (
    beam_search,
    greedy_search,
)
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
    store_transcripts,
    str2bool,
    write_error_stats,
)
from targetlens import TargetLength
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
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )
    
    parser.add_argument(
        "--res-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - modified_beam_search_LODR
          - fast_beam_search
          - fast_beam_search_nbest
          - fast_beam_search_nbest_oracle
          - fast_beam_search_nbest_LG
        If you use fast_beam_search_nbest_LG, you have to specify
        `--lang-dir`, which should contain `LG.pt`.
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding-method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding-method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--use-shallow-fusion",
        type=str2bool,
        default=False,
        help="""Use neural network LM for shallow fusion.
        If you want to use LODR, you will also need to set this to true
        """,
    )

    parser.add_argument(
        "--lm-type",
        type=str,
        default="rnn",
        help="Type of NN lm",
        choices=["rnn", "transformer"],
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.3,
        help="""The scale of the neural network LM
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    parser.add_argument(
        "--tokens-ngram",
        type=int,
        default=2,
        help="""The order of the ngram lm.
        """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        default=500,
        help="ID of the backoff symbol in the ngram LM",
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=2,
        help="""
        The bonus score of each token for the context biasing words/phrases.
        Used only when --decoding-method is modified_beam_search and
        modified_beam_search_LODR.
        """,
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--context-file",
        type=str,
        default="",
        help="""
        The path of the context biasing lists, one word/phrase each line
        Used only when --decoding-method is modified_beam_search and
        modified_beam_search_LODR.
        """,
    )

    parser.add_argument(
        "--pad-feature",
        type=int,
        default=0,
    )

    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: CifTModel,
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

    if params.decoding_method == "greedy_search":
        hyps = []
        for i in range(batch_size):
            hyp = greedy_search(
                model=model,
                encoder_out=encoder_out2[i, None, : encoder_out_lens2[i]],
                max_sym_per_frame=params.max_sym_per_frame,
            )
            hyps.append(sp.text2word(sp.decode(hyp)))
    else:
        hyps = beam_search(
            model=model,
            encoder_out=encoder_out2,
            encoder_out_lens=encoder_out_lens2,
            beam=params.beam_size,
            max_s_per_t=params.max_sym_per_frame,
            # gaussian_diag_cov=params.gaussian_diag_cov,
        )
        hyps = [sp.text2word(h) for h in sp.decode(hyps)]

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    else:
        return {f"beam_size_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: CifTModel,
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

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = sp.text2word(ref_text)
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)

        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)

    return test_set_wers

@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    Tokenizer.add_arguments(parser)
    # Not used, but just to ensure that the experiment arguments are parsed.
    TargetLength.add_targetlength_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
    )
    if not params.res_dir:
        params.res_dir = params.exp_dir / params.decoding_method


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

    if "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
        if params.decoding_method in (
            "modified_beam_search",
            "modified_beam_search_LODR",
        ):
            if params.has_contexts:
                params.suffix += f"-context-score-{params.context_score}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

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
    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()
    dev_clean_cuts = librispeech.dev_clean_cuts()
    dev_other_cuts = librispeech.dev_other_cuts()


    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)
    dev_clean_dl = librispeech.test_dataloaders(dev_clean_cuts)
    dev_other_dl = librispeech.test_dataloaders(dev_other_cuts)
    
    test_sets = ["test-clean", "test-other", "dev-clean", "dev-other"]
    test_dls = [test_clean_dl, test_other_dl, dev_clean_dl, dev_other_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
        )

        tot_err = save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

        with (
            params.res_dir
            / (
                f"{test_set}-{params.beam_size}"
                f"_{params.avg}_{params.epoch}_{params.max_sym_per_frame}.cer"
            )
        ).open("w") as fout:
            if len(tot_err) == 1:
                fout.write(f"{tot_err[0][1]}")
            else:
                fout.write("\n".join(f"{k}\t{v}") for k, v in tot_err)

    logging.info("Done!")


if __name__ == "__main__":
    main()
