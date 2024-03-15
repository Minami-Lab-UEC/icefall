from lhotse import CutSet
from lhotse.utils import supervision_to_frames
import argparse
from pathlib import Path
import sentencepiece as spm

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


train_set = CutSet.from_file(args.train_cuts)
sp = spm.SentencePieceProcessor()
sp.load(args.bpe)

with open(args.output_dir / "texts.txt", 'w') as ftext, \
    open(args.output_dir / "tokenized_texts.txt", 'w') as ftoken_text, \
    open(args.output_dir / "frame_count.txt", 'w') as fframe_count, \
    open(args.output_dir / "token_count.txt", 'w') as ftoken_count, \
    open(args.output_dir / "word_count.txt", 'w') as fword_count:
    for cut in train_set:
        if "sp" in cut.id:
            continue
        t = cut.supervisions[0].text
        if t:
            ftext.write(f"{t}\n")
            fword_count.write(f"{t.count(' ')+1}\n")
            _, xlens = supervision_to_frames(
                cut.supervisions[0], cut.frame_shift, cut.sampling_rate, 
                max_frames=cut.num_frames
            )
            xlens = (xlens - 7) // 2
            xlens = (xlens + 1) // 2
            fframe_count.write(f"{xlens}\n")
            tokenized_text = sp.Encode(t, out_type=str)
            ftoken_count.write(f"{len(tokenized_text)}\n")
            tokenized_text = ' '.join(tokenized_text)
            ftoken_text.write(f"{tokenized_text}\n")
            


