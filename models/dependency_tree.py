from .conll_node import ConllNode

class DependencyTree:
    ''' 
    Class representing a dependency tree as a set of CONLL-u nodes.
    '''
    def __init__(self, nodes=[]):
        self.nodes = nodes

    def append(self, conll_node):
        if not isinstance(conll_node, ConllNode):
            print("[!] Error: append must have as argument an instance of ConllNode")
            print("    If you want to append a conll node from string use dep_tree.append_string(line) instead")
            return
        self.nodes.append(conll_node)

    def append_string(self, node_str):

        if not isinstance(node_str, str):
            print("[!] Error: append_string must have as argument an instance of string")
            print("    If you want to append a conll node directly use dep_tree.append(conll_node) instead")
            return
        self.nodes.append(ConllNode.from_string(node_str))


    def get_upos(self):
        return [e.upos for e in self.nodes]
    
    def get_xpos(self):
        return [e.xpos for e in self.nodes]

    def get_lemma(self):
        return [e.lemma for e in self.nodes]

    def get_feats(self):
        return [e.feats for e in self.nodes]

    def get_relations(self):
        return [(e.id, e.head, e.relation) for e in self.nodes]
    
    def get_sentence(self):
        sent_array = [(s.form)for s in self.nodes]
        return " ".join(sent_array)
    
    def __repr__(self):
        return (str(n) for n in self.nodes)
