from lhotse import CutSet
from kenlmwrapper import Model
from pathlib import Path

subparts = [
    "eval1",
    "eval2",
    "eval3",
    "excluded",
    "valid"
]
fbank_cut_dir = Path("data/fbank")

targetlengths = [
    ("3gram", 0.55819859528638738),
    ("2gram", 0.41650609460494159),
    ("1gram", 0.27111056595534722),
    ("4gram", 0.66557475148552303),
    ("5gram", 0.72205136160654204),
]

out_dir = Path("approx_tgtlens/data_noR_wSosEos/naive")
out_dir.mkdir(parents=True, exist_ok=True)
for t, _ in targetlengths:
    (out_dir / t).mkdir(parents=True, exist_ok=True)

for subpart in subparts:
    cuts = CutSet.from_file(fbank_cut_dir / f"csj_cuts_{subpart}.jsonl.gz")
    lines = []
    for c in cuts:
        line : str = c.supervisions[0].custom["fluent"]
        line = ' '.join(list(line.replace(" ", "")))
        lines.append((c.id, line))
    
    lines = sorted(lines, key=lambda x:x[0])
    
    for targetlength, slope in targetlengths:
        # res_lines = []
        model = Model(f"approx_tgtlens/data_noR_wSosEos/{targetlength}.arpa")
        out_file = out_dir / targetlength / (subpart + ".txt") # Path(f"approx_tgtlens/data/groundtruth/{subpart}_{targetlength}.txt").open("w")
        out_file = out_file.open('w')
        unit = 1/slope
        for cid, line in lines:
            out = []
            accum_logprob = 0
            for token, (lprob, ctxt, oov) in zip(line.split(" "), model.full_scores(line)):
                accum_logprob -= lprob
                out.append(token)
                while accum_logprob >= unit:
                    accum_logprob -= unit
                    out.append("∅")
            if accum_logprob > 0:
                out.append("∅")
            out = ' '.join(out)
            out_file.write(f"{cid}:\tref={out}\n")
        out_file.close()

