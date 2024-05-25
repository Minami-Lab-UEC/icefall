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
    assert args.res_file.exists(), f"{args.res_file} cannot be found."
    assert args.output.parent.is_dir(), f"Please create parent for {args.output}."

    test = {}
    val = {}
    val_cer = {}
    with args.res_file.open('r') as fin:
        for line in fin:
            line = line.strip().split(',')
            key = ','.join(line[:-4])
            ers = line[-4:] # [float(l) for l in line[-4:]]
            test[key] = (ers[0], ers[1])
            val[key] = (ers[2], ers[3])
            val_cer[key] = float(ers[2].split("±")[0])+float(ers[3].split("±")[0])
    

    sorted_keys = sorted(val_cer.items(), key=lambda x: x[1])
    
    with args.output.open('w') as fout:
        for i, (top_key, top_val) in enumerate(sorted_keys):
            if i < args.top:
                fout.write(
                    f"{top_key},{','.join(str(s) for s in test[top_key])},{','.join(str(s) for s in val[top_key])}\n"
                )

if __name__ == "__main__":
    main()