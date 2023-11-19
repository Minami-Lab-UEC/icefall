import argparse
from collections import defaultdict
from pathlib import Path 


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--res-dir", "-i", type=Path,
    )
    parser.add_argument(
        "--output", "-o", type=Path,
    )

    return parser.parse_args()

def main():
    args = get_args()

    assert args.res_dir.is_dir(), f"{args.res_dir} cannot be found."
    assert args.output.parent.is_dir(), f"Please create parent for {args.output}."
    results = defaultdict(lambda:defaultdict(dict))

    for res in args.res_dir.glob("*/*cer"):
        evalset, param = res.stem.split("-")
        results[param][res.suffix][evalset] = res.read_text()

    with args.output.open("w") as fout:
        for param, v in results.items():
            fout.write(param.replace("_", ","))
            for suffix in [".cer"]:
                for evalset in ["eval1", "eval2", "eval3", "excluded", "valid"]:
                # for evalset in ["eval1", "eval2", "eval3"]:
                    fout.write("," + v[suffix][evalset])
            fout.write("\n")

if __name__ == "__main__":
    main()