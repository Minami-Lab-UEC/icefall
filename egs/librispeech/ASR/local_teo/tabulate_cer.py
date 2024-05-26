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
    parser.add_argument(
        "--subparts",
        nargs="+",
        type=str,
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
    subparts = args.subparts

    for res in args.res_dir.glob("*/*.ci"):
        *evalset, param = res.stem.split("-")
        evalset = "-".join(evalset)
        results[param][res.suffix][evalset] = res.read_text()
        
    with args.output.open("w") as fout:
        for param, v in results.items():
            fout.write(param.replace("_", ","))
            for suffix in [".ci"]:
                for evalset in subparts:
                    try:
                        fout.write("," + v[suffix][evalset])
                    except:
                        fout.write(",-1")
            fout.write("\n")
            
if __name__ == "__main__":
    main()