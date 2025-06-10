#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pathlib
import torch
import math
import pandas as pd

from esm import Alphabet, FastaBatchedDataset, pretrained, MSATransformer

def create_parser():
    parser = argparse.ArgumentParser(
        description="Compute log-likelihood (sum of log pAA) for each sequence in FASTA using an ESM model."
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (e.g. esm2_t36_3B_UR50D).",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file with sequences.",
    )
    parser.add_argument(
        "output_csv",
        type=pathlib.Path,
        help="Path to CSV file where we save log-likelihood scores.",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")

    return parser

def compute_log_likelihood(logits: torch.Tensor, tokens: torch.Tensor, seq_len: int) -> float:
    """
    logits: [B, L, vocab_size]
    tokens: [B, L]
    seq_len: длина реальной белковой последовательности (без BOS/EOS)
    Возвращает сумму log P(верный aa) по всем позициям (без BOS/EOS).
    """
    # logits -> log_probs
    log_probs = torch.log_softmax(logits, dim=-1)  # размер [B, L, vocab_size]

    total_ll = 0.0

    for pos in range(seq_len):
        true_aa_idx = tokens[0, pos+1].item()
        total_ll += log_probs[0, pos+1, true_aa_idx].item()

    return total_ll

def run(args):
    print(f"Loading model: {args.model_location}")
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()

    if isinstance(model, MSATransformer):
        raise ValueError("calc_likelihood.py does not handle MSA Transformer models.")

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    else:
        print("Running on CPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {len(dataset)} sequences from {args.fasta_file}")


    results = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx+1} / {len(batches)} with {toks.size(0)} seqs")

            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[], need_head_weights=False)
            logits = out["logits"]
            logits = logits.cpu()
            toks = toks.cpu()

            for i, label in enumerate(labels):
                seq_str = strs[i]
                seq_len = len(seq_str)
                ll_value = compute_log_likelihood(logits[i:i+1], toks[i:i+1], seq_len)
                results.append((label, seq_str, ll_value))

    df = pd.DataFrame(results, columns=["name", "sequence", "log_likelihood"])
    df.to_csv(args.output_csv, index=False)
    print(f"Saved log-likelihood to {args.output_csv}")

def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
