from lhotse import CutSet
from pathlib import Path

fbank_dir = Path("data/fbank")

train_set = CutSet.from_file(fbank_dir / "csj_cuts_train.jsonl.gz")

train_set = filter(lambda c: "_sp" not in c.id, train_set)

train_set : CutSet = CutSet.from_items(train_set)

train_set.to_file(fbank_dir / "csj_cuts_train_nosp.jsonl.gz")
