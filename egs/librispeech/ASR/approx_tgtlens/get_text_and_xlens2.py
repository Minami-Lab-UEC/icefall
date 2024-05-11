from lhotse import CutSet
from lhotse.utils import supervision_to_frames
import argparse
from pathlib import Path
import sentencepiece as spm
from dataclasses import dataclass

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--train-cuts",
    type=Path
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=Path
)
parser.add_argument(
    "--bpe",
    type=str,
)
args = parser.parse_args()

@dataclass
class Utt:
    text : str
    tok_text : str
    frame : int
    token : int
    word : int

train_set = CutSet.from_file(args.train_cuts)
sp = spm.SentencePieceProcessor()
sp.load(args.bpe)

utts : list[Utt] = []

for cut in train_set:
    if "sp" in cut.id:
        continue
    t = cut.supervisions[0].text
    if t:
        # ftext.write(f"{t}\n")
        # fword_count.write(f"{t.count(' ')+1}\n")
        _, xlens = supervision_to_frames(
            cut.supervisions[0], cut.frame_shift, cut.sampling_rate, 
            max_frames=cut.num_frames
        )
        xlens = (xlens - 7) // 2
        xlens = (xlens + 1) // 2
        # fframe_count.write(f"{xlens}\n")
        tokenized_list = sp.Encode(t, out_type=str)
        # ftoken_count.write(f"{len(tokenized_text)}\n")
        tokenized_text = ' '.join(tokenized_list)
        # ftoken_text.write(f"{tokenized_text}\n")
        utts.append(Utt(
            text=t,
            word=t.count(' ') + 1,
            frame=xlens,
            token=len(tokenized_list),
            tok_text=tokenized_text
        ))

utts = sorted(utts, key=lambda x:x.text)

with open(args.output_dir / "texts.txt", 'w') as ftext, \
    open(args.output_dir / "tokenized_texts.txt", 'w') as ftoken_text, \
    open(args.output_dir / "frame_count.txt", 'w') as fframe_count, \
    open(args.output_dir / "token_count.txt", 'w') as ftoken_count, \
    open(args.output_dir / "word_count.txt", 'w') as fword_count:
    for utt in utts:
        print(utt.text, file=ftext)
        print(utt.tok_text, file=ftoken_text)
        print(utt.token, file=ftoken_count)
        print(utt.frame, file=fframe_count)
        print(utt.word, file=fword_count)
        
        
            


