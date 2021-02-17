import re
 
template=re.compile(r'(some|all)\s*(feathery|parrot|bird like|slippery|fish like|red|rose)?\s+(aves|anapsida|pig|minor|pumba|wart hog|parrot|macaw|reptilia|caretta)\s*(that|who)?\s*(hum|sing)?\s*(not)?\s*(squawk|chirp|walk|swim|move|breathe|eat|sleep|screech)')
    
class Sentence:

    @staticmethod
    def parse(string):
        #print "Parsing: " + string.strip()
        m=re.match(template, string)
        return Sentence(m.group(1),
                        m.group(2),
                        m.group(3),
                        m.group(4),
                        m.group(5),
                        m.group(6),
                        m.group(7))

    def distance(self, other):
        distance = 0
        #elems = set([self.quantifier, other.quantifier, self.adjective, other.adjective, self.noun, other.noun, self.comp, other.comp, self.relative, other.relative, self.negation, other.negation, self.verb, other.verb])
        #return len(elems) // 7
        if not(self.quantifier==other.quantifier):
                distance = distance + 1
        if not(self.adjective==other.adjective):
                distance = distance + 1
        if not(self.noun==other.noun):
                distance = distance + 1
        if not(self.comp==other.comp):
                distance = distance + 1
        if not(self.relative==other.relative):
                distance = distance + 1
        if not(self.negation==other.negation):
                distance = distance + 1
        if not(self.verb==other.verb):
                distance = distance + 1
        return(distance)


    def diff(self, other):
        diff = []
        if not(self.quantifier==other.quantifier):
                diff = [self.quantifier, other.quantifier]
        if not(self.adjective==other.adjective):
                diff = [self.adjective, other.adjective]
        if not(self.noun==other.noun):
                diff = [self.noun, other.noun]
        if not(self.relative==other.relative):
                diff = [self.comp+' ' +self.relative, other.comp+' ' +other.relative]
        if not(self.negation==other.negation):
                diff = [self.negation, other.negation]
        if not(self.verb==other.verb):
                diff = [self.verb, other.verb]
        return(diff)
        
    def __init__(self, quantifier, adjective, noun, comp, relative, negation, verb):
        self.quantifier = quantifier
        self.adjective = adjective
        self.noun = noun
        self.comp=comp
        self.relative=relative
        self.negation=negation
        self.verb=verb

    def __str__(self):
        return("(S (Q "+ str(self.quantifier) + ")" +
        "(A " + str(self.adjective) + ")" +
        "(N " + str(self.noun) + ")" +
        "(C " + str(self.comp) + ")" +
        "(R " + str(self.relative) + ")" +
        "(Neg " + str(self.negation) + ")" +
        "(V " + str(self.verb) + "))")
