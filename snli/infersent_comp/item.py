from sentence import Sentence

class Item:
    @staticmethod
    def parse_training_item(string):
        #print "Parsing: " + string.strip()
        fields=string.strip().split('\t')
        prem=Sentence.parse(fields[1])
        hyp=Sentence.parse(fields[2])
        return(Item(prem,hyp,fields[0],fields[0],False))
    
    def __init__(self, premise, hypothesis, correct, label, test):
        self.premise = premise
        self.hypothesis = hypothesis
        self.correct = correct
        self.label = label
        self.test = test

    def hash_key(self):
        return(str(self.premise)+str(self.hypothesis)+str(self.correct)+str(self.label)+str(self.test))
        
    def get_premise(self):
        return(self.premise)

    def get_hypothesis(self):
        return(self.hypothesis)
        
    def distance(self, other):
        return(self.premise.distance(other.premise) +
        self.hypothesis.distance(other.hypothesis))

    def diff(self, other):
        if not (Sentence.diff(self.premise, other.premise) == []): return (Sentence.diff(self.premise, other.premise), "premise")
        if not (Sentence.diff(self.hypothesis, other.hypothesis) == []): return (Sentence.diff(self.hypothesis, other.hypothesis), "hypothesis")

    def is_target(self):
        return((self.premise.adjective == "rose" or
            self.hypothesis.adjective == "rose") and
            (self.premise.noun == "minor" or
            self.hypothesis.noun == "minor") and
            (self.premise.verb == "swim" or
            self.hypothesis.verb == "swim"))

    def csv(self):
        str(self.premise)+","+str(self.hypothesis)+","+str(self.correct)+","+str(self.label)+","+str(self.test)
