import kenlm
import argparse     

class TargetLength:
    def __init__(self):
        pass
    
    @staticmethod
    def load(target_len_from : str) -> "TargetLength":
        if target_len_from == "num_tokens":
            return FromTokenCount()
        elif target_len_from == "num_words":
            return FromWordCount()
        
        assert target_len_from.endswith(".arpa"), target_len_from
        with open(target_len_from) as fin:
            next(fin)
            next(fin)
            third_line = next(fin)
        if third_line != "\n":
            return KenlmNgramLM(target_len_from)
        else:
            return UnigramLM(target_len_from)   
    
    @staticmethod
    def add_targetlength_arguments(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Target Length related options")
        group.add_argument(
            "--targetlen-from",
            type=str,
            default="data/lang_bpe_500/librispeech_train_500_3gram.arpa",
        )
        
        group.add_argument(
            "--ent2awe-slope",
            type=float,
            default=0.5,
        )
        
        group.add_argument(
            "--ent2awe-intercept",
            type=float,
            default=0.
        )
    
    def score(self, line : str, bos = False, eos = False) -> float:
        "Assumes that the string has spaces between each token, not each word."
        raise NotImplementedError

class UnigramLM(TargetLength):
    def __init__(self, arpa_str : str):
        self.logprobs = {}
        with open(arpa_str) as fin:
            next(fin)
            next(fin)
            next(fin)
            next(fin)
            for line in fin:
                if line == "\n":
                    break
                logprob, word = line.strip().split("\t")
                self.logprobs[word] = float(logprob)
        self.bos_symbol = "<s>"
        self.eos_symbol = "</s>"

    def score(self, line : str, bos = False, eos = False) -> float:
        words = line.split(" ")
        logprob = 0
        if bos:
            logprob += self.logprobs[self.bos_symbol]
        
        for word in words:
            logprob += self.logprobs[word]
        
        if eos:
            logprob += self.logprobs[self.eos_symbol]
        
        return logprob

class KenlmNgramLM(TargetLength):
    def __init__(self, arpa_str : str):
        self.model = kenlm.Model(arpa_str)

    def score(self, line : str, bos = False, eos = False) -> float:
        return self.model.score(line, bos=bos, eos=eos)

class FromWordCount(TargetLength):
    def __init__(self, *args, **kwargs):
        pass

    def score(self, line : str, bos = False, eos = False) -> float:
        tgtlen = line.count("▁") # + 1
        # Negative to align with the entropy scores from ngrams which are negative.
        return -tgtlen

class FromTokenCount(TargetLength):
    def __init__(self, *args, **kwargs):
        pass

    def score(self, line : str, bos = False, eos = False) -> float:
        tgtlen = line.count(" ") + 1
        # Negative to align with the entropy scores from ngrams which are negative.
        return -tgtlen
