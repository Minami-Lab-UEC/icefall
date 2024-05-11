from lhotse import CutSet
from lhotse.utils import supervision_to_frames
import argparse
from pathlib import Path

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

args = parser.parse_args()


train_set = CutSet.from_file(args.train_cuts)


with open(args.output_dir / "texts.txt", 'w') as ftext, \
    open(args.output_dir / "tokenized_texts.txt", 'w') as ftoken_text, \
    open(args.output_dir / "frame_count.txt", 'w') as fframe_count, \
    open(args.output_dir / "token_count.txt", 'w') as ftoken_count, \
    open(args.output_dir / "word_count.txt", 'w') as fword_count:
    for cut in train_set:
        if "sp" in cut.id:
            continue
        if "R" in cut.id:
            continue
        _, xlens = supervision_to_frames(
            cut.supervisions[0], cut.frame_shift, cut.sampling_rate, 
            max_frames=cut.num_frames
        )
        xlens = (xlens - 7) // 2
        xlens = (xlens + 1) // 2
        fframe_count.write(f"{xlens}\n")
        
        wakati_text = cut.supervisions[0].custom["fluent"].strip()

        ftext.write(f"{wakati_text}\n")
        if wakati_text:
            fword_count.write(f"{wakati_text.count(' ')+1}\n")
            words = wakati_text.split(" ")
            tokens = list(wakati_text.replace(" ", ""))
            ftoken_count.write(f"{len(tokens)}\n")
            nospace_tokens = ' '.join(tokens)
            ftoken_text.write(f"{nospace_tokens}\n")
        else:
            fword_count.write("0\n")
            ftoken_count.write("0\n")
            ftoken_text.write(f"\n")

