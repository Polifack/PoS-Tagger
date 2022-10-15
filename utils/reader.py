from conllu import parse_tree_incr
from models.conll_node import ConllNode

def parse_conllu(in_file):
    '''
    Given an input CONLL file parses it and returns the Dependency Trees
    in the form of a list of ConllNodes
    '''
    conll_node_list=[]
    
    for token_tree in parse_tree_incr(in_file):
        nodes = []
        postags = []
        words = []
        
        nodes.append(ConllNode.dummy_root())
        data = token_tree.serialize().split('\n')
        dependency_start_idx = 0
        for line in data:
            if line[0]!="#":
                break
            if "# text" in line:
                sentence = line.split("# text = ")[1]
            dependency_start_idx+=1
        
        data = data[dependency_start_idx:]

        for line in data:
            # check if not valid line
            if (len(line)<=1) or len(line.split('\t'))<10 or line[0] == "#":
                continue
            
            conll_node = ConllNode.from_string(line)
            nodes.append(conll_node)
        
        conll_node_list.append(nodes)
    
    return conll_node_list