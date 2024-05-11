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


out_dir = Path("approx_tgtlens/transcripts")
out_dir.mkdir(parents=True, exist_ok=True)

for subpart in subparts:
    cuts = CutSet.from_file(fbank_cut_dir / f"csj_cuts_{subpart}.jsonl.gz")
    lines = []
    for c in cuts:
        line : str = c.supervisions[0].custom["fluent"].strip()
        line = ' '.join(line.split())
        lines.append((c.id, line))
    
    lines = sorted(lines, key=lambda x:x[0])
    
    out_file = out_dir / (subpart + ".txt")
    out_file = out_file.open('w')
    
    out_file.writelines(
        f"{cid}:\t{line}\n" for cid,line in lines
    )
    
    out_file.close()
