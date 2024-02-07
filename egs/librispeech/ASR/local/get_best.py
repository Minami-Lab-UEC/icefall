from pathlib import Path
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--res-file", "-i", type=Path,
    )
    parser.add_argument(
        "--output", "-o", type=Path,
    )
    
    parser.add_argument(
        "--top", "-t", type=int,
    )

    return parser.parse_args()

def main():
    args = get_args()
    # res_dir : Path = args.res_dir
    # res_dir.exists()
    assert args.res_file.exists(), f"{args.res_file} cannot be found."
    assert args.output.parent.is_dir(), f"Please create parent for {args.output}."
    # res_dir = Path(args.res_dir)
    test = {}
    val = {}
    with args.res_file.open('r') as fin:
        for line in fin:
            line = line.strip().split(',')
            key = ','.join(line[:-4])
            ers = [float(l) for l in line[-4:]]
            test[key] = (ers[-4], ers[-3])
            val[key] = (ers[-2], ers[-1])
    
    # val = sorted(val.items(), key=lambda x: sum(x[1]))
    sorted_keys = sorted(val.items(), key=lambda x: sum(x[1]))
    
    with args.output.open('w') as fout:
        for i, (top_key, top_val) in enumerate(sorted_keys):
            if i < args.top:
                fout.write(
                    f"{top_key},{','.join(str(s) for s in test[top_key])},{','.join(str(s) for s in val[top_key])}\n"
                )

if __name__ == "__main__":
    main()