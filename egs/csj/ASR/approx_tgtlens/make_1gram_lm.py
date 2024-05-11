import argparse
import io
import math
import os
import re
import sys
from collections import defaultdict
import sentencepiece as spm

parser = argparse.ArgumentParser(
    description="""
    Generate kneser-ney language model as arpa format. By default,
    it will read the corpus from standard input, and output to standard output.
    """
)
parser.add_argument("-text", type=str, default=None, help="Path to the corpus file")
parser.add_argument(
    "-lm", type=str, default=None, help="Path to output arpa file for language models"
)
parser.add_argument(
    "-verbose", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="Verbose level"
)
parser.add_argument("-bpe", type=str, default=None, help="Only use if need to use BPE")
args = parser.parse_args()

# For encoding-agnostic scripts, we assume byte stream as input.
# Need to be very careful about the use of strip() and split()
# in this case, because there is a latin-1 whitespace character
# (nbsp) which is part of the unicode encoding range.
# Ref: kaldi/egs/wsj/s5/utils/lang/bpe/prepend_words.py @ 69cd717
default_encoding = "utf-8"

strip_chars = " \t\r\n"
whitespace = re.compile("[ \t]+")

class UnigramCounts:
    def __init__(self, bos_symbol="<s>", eos_symbol="</s>", bpe=None):
        self.ngram_order = 1
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol
        
        self.counts = defaultdict(int)
        
        if bpe:
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(bpe)
            self.split = lambda x : tokenizer.encode(x, out_type = str)
        else:
            self.split = lambda x : whitespace.split(x)
    
    def add_raw_counts_from_line(self, line : str):
        if line == "":
            words = [self.eos_symbol]
        else:
            words = self.split(line) + [self.eos_symbol]
        
        for word in words:
            self.counts[word] += 1
    
    def add_raw_counts_from_file(self, filename):
        lines_processed = 0
        
        with open(filename, encoding=default_encoding) as fp:
            for line in fp:
                line = line.strip(strip_chars)
                self.add_raw_counts_from_line(line)
                lines_processed += 1
        
        if lines_processed == 0 or args.verbose > 0:
            print(
                "make_phone_lm.py: processed {0} lines of input".format(
                    lines_processed
                ),
                file=sys.stderr,
            )
        print(f"{lines_processed} lines processed.")

    def print_as_arpa(self, fout=io.TextIOWrapper(sys.stdout.buffer, encoding="latin-1")):
        
        self.counts[self.bos_symbol] = 0
        print("\\data\\", file=fout)
        print(f"ngram 1={len(self.counts)}", file=fout)
        print("", file=fout)
        print("\\1-grams:", file=fout)
        total_sum = sum(self.counts.values())

        for word, count in self.counts.items():
            prob = count / total_sum
            if prob == 0:
                prob = 1e-99
            logprob = math.log10(prob)
            
            print(f"{logprob:.7f}\t{word}", file=fout)
        print("", file=fout)
        print("\\end\\", file=fout)

if __name__ == "__main__":
    ngram_counts = UnigramCounts(bpe=args.bpe)

    assert os.path.isfile(args.text)
    ngram_counts.add_raw_counts_from_file(args.text)

    if args.lm is None:
        ngram_counts.print_as_arpa()
    else:
        with open(args.lm, "w", encoding=default_encoding) as f:
            ngram_counts.print_as_arpa(fout=f)
