from pathlib import Path 
from collections import defaultdict
import argparse

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
    # res_dir : Path = args.res_dir
    # res_dir.exists()
    assert args.res_dir.is_dir(), f"{args.res_dir} cannot be found."
    assert args.output.parent.is_dir(), f"Please create parent for {args.output}."
    # res_dir = Path(args.res_dir)
    results = defaultdict(lambda:defaultdict(dict))

    for res in args.res_dir.glob("*/*cer"):
        *evalset, param = res.stem.split("-")
        evalset = "-".join(evalset)
        results[param][res.suffix][evalset] = res.read_text()
        
    with args.output.open("w") as fout:
        for param, v in results.items():
            fout.write(param.replace("_", ","))
            for suffix in [".cer"]:
                for evalset in ["test-clean", "test-other", "dev-clean", "dev-other"]:
                    try:
                        fout.write("," + v[suffix][evalset])
                    except:
                        fout.write(",-1")
            fout.write("\n")
            
if __name__ == "__main__":
    main()