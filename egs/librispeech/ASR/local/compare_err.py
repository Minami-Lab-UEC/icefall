import argparse
import re
from pathlib import Path
from collections import Counter
from typing import Dict
from functools import reduce

tagger = re.compile(r"\((.*?)\)")

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--errs", type=Path, nargs="+"
    )
    parser.add_argument(
        "--out", "-o", type=Path
    )

    return parser.parse_args()

def get_errors(err_file : Path) -> Dict[str, Counter]:
    utts = {}
    
    with err_file.open() as fin:
        skip1 = True
        for line in fin:
            if skip1:
                skip1 = "PER-UTT DETAILS: corr or (ref->hyp)" not in line
                continue
            line = line.strip()
            if not line:
                break
            utt_id, line = line.split(":")
            err_counter = Counter(tagger.findall(line))
            
            corr_line = line
            for c in err_counter:
                hyp = c.split("->")[1]
                corr_line = corr_line.replace('(' + c + ')', "*" if hyp == "*" else f"({hyp})")
            # corr_lines.append(corr_line.replace(' ',''))
            utts[utt_id] = err_counter, corr_line.strip() #.replace(" ",'')
    return utts


def counter2str(x : Counter):
    return "\"" + "\n".join(f"({k})" for k,v in x.items() for _ in range(v)) + "\""

def main():
    args = get_args()
    
    errors = []
    for errfile in args.errs:
        errors.append(get_errors(errfile))
    
    assert len(set(len(f) for f in errors)) == 1, [len(f) for f in errors]
    
    chars = [chr(65+i) for i in range(len(args.errs))]
    with args.out.open("w") as fout:
        fout.write("\r\n".join(f"{i}: {f}" for i, f in zip(chars, args.errs)) )
        # fout.write("\r\n@@@@@@@\r\n")
        fout.write("\r\nspk id,common,")
        fout.write(','.join(chars) + ",")
        fout.write(','.join(f"ori {i}" for i in chars))
        fout.write("\r\n")
        for error in zip(*[f.items() for f in errors]):
            sgids, err = list(zip(*error))
            # err - list(zip(*err))
            err, line = list(zip(*err))
            assert len(set(sgids)) == 1, sgids
            common = reduce((lambda a,b:a&b), err)
            err = list(map(lambda x: x-common, err))
            fout.write(f"{sgids[0].replace('_', ',')},{counter2str(common)},")
            fout.write(",".join(counter2str(e) for e in err))
            fout.write("," + ",".join(line))
            fout.write("\r\n")
    
if __name__ == "__main__":
    main()
    
    