from lhotse import CutSet
from kenlmwrapper import Model
from pathlib import Path
import sentencepiece as spm

subparts = [
    "test-clean",
    "test-other",
    "dev-clean",
    "dev-other"
]
fbank_cut_dir = Path("data/fbank")

targetlengths = [
    ("3gram", 0.36376589909469687),
    ("2gram", 0.28089988989667980),
    ("1gram", 0.21428022865333540),
    ("4gram", 0.45542393511953849),
    ("5gram", 0.50731459230278320),
]

sp = spm.SentencePieceProcessor()
sp.load("approx_tgtlens/data/unigram_500.model")

out_dir = Path("approx_tgtlens/data/naive")
out_dir.mkdir(parents=True, exist_ok=True)
for t, _ in targetlengths:
    (out_dir / t).mkdir(parents=True, exist_ok=True)

for subpart in subparts:
    cuts = CutSet.from_file(fbank_cut_dir / f"librispeech_cuts_{subpart}.jsonl.gz")
    lines = []
    for c in cuts:
        line : str = c.supervisions[0].text
        line = ' '.join(sp.Encode(line, out_type=str))
        # line = ' '.join(list(line.replace(" ", "")))
        lines.append((c.id, line))
    
    lines = sorted(lines, key=lambda x:x[0])
    
    for targetlength, slope in targetlengths:
        # res_lines = []
        model = Model(f"approx_tgtlens/data/{targetlength}.arpa")
        out_file = out_dir / targetlength / (subpart + ".txt") # Path(f"approx_tgtlens/data/groundtruth/{subpart}_{targetlength}.txt").open("w")
        out_file = out_file.open('w')
        unit = 1/slope
        for cid, line in lines:
            out = []
            accum_logprob = 0
            for token, (lprob, ctxt, oov) in zip(line.split(" "), model.full_scores(line, bos=False, eos=False)):
                # half_lprob = lprob / 2

                accum_logprob -= lprob
                while accum_logprob >= unit:
                    accum_logprob -= unit
                    out.append("∅")
                out.append(token)
                # accum_logprob -= half_lprob
                # while accum_logprob >= unit:
                #     accum_logprob -= unit
                #     out.append("∅")
            if accum_logprob > 0:
                out.append("∅")
            out = ' '.join(out)
            out_file.write(f"{cid}:\tref={out}\n")
        out_file.close()

