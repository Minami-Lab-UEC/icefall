import kenlm

def Model(arpa_str : str):
    with open(arpa_str) as fin:
        next(fin)
        next(fin)
        third_line = next(fin)
    if third_line != "\n":
        return kenlm.Model(arpa_str)
    else:
        return UnigramLM(arpa_str)        

class UnigramLM:
    def __init__(self, arpa_str : str):#, bos_symbol = "<s>", eos_symbol = "</s>"):
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
        
    
    def score(self, line : str, bos = True, eos = True) -> float:
        words = line.split(" ")
        logprob = 0
        if bos:
            logprob += self.logprobs[self.bos_symbol]
        
        for word in words:
            logprob += self.logprobs[word]
        
        if eos:
            logprob += self.logprobs[self.eos_symbol]
        
        return logprob

    def full_scores(self, line : str, bos = True, eos = True):
        words = line.split(" ")
        
        for word in words:
            prob = self.logprobs.get(word, -99)
            yield (prob, 1, prob == -99)
        
        if eos:
            prob = self.logprobs.get(self.eos_symbol, -99)
            yield (prob, 1, prob == -99)