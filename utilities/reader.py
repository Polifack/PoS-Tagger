from conllu import parse_tree_incr
from models.conll_node import ConllNode
from models.dependency_tree import DependencyTree

def parse_conllu(in_file):
    '''
    Given an input CONLL file parses it and returns the Dependency Trees
    in the form of a list of ConllNodes
    '''
    
    treebank = []
    in_file = open(in_file)

    for i, token_tree in enumerate(parse_tree_incr(in_file)):
        print("[*] Reading conll-U "+str(i), end="\r")

        dt = DependencyTree()
        data = token_tree.serialize().split('\n')
        dependency_start_idx = 0

        # skip info lines
        for line in data:
            if line[0]!="#":
                break
            if "# text" in line:
                sentence = line.split("# text = ")[1]
            dependency_start_idx+=1
        
        data = data[dependency_start_idx:]

        # data lines
        for line in data:
            # check if not valid line
            if (len(line)<=1) or len(line.split('\t'))<10 or line[0] == "#":
                continue
            dt.append_string(line)
        treebank.append(dt)
    
    print("[*] Reading conll-U",i,": Done")
    return treebank
