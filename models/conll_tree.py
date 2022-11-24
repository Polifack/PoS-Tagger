from . import ConllNode

class ConllTree:
  def __init__(self, nodes):
      self.nodes = nodes
  
  def get_edges(self):
      '''
      Return sentence dependency edges as a tuple 
      shaped as ((d,h),r) where d is the dependant of the relation,
      h the head of the relation and r the relationship type
      '''
      return list(map((lambda x :((x.id, x.head), x.relation)), self.nodes))
  
  def get_arcs(self):
      '''
      Return sentence dependency edges as a tuple 
      shaped as (d,h) where d is the dependant of the relation,
      and h the head of the relation.
      '''
      return list(map((lambda x :(x.id, x.head)), self.nodes))

  def get_relations(self):
      '''
      Return a list of relationships betwee nodes
      '''
      return list(map((lambda x :x.relation), self.nodes))
  
  def get_sentence(self):
      '''
      Return the sentence as a string
      '''
      return " ".join(list(map((lambda x :x.form), self.nodes)))

  def get_words(self):
      '''
      Returns the words of the sentence as a list
      '''
      return list(map((lambda x :x.form), self.nodes))

  def get_indexes(self):
      '''
      Returns a list of integers representing the words of the 
      dependency tree
      '''
      return list(map((lambda x :x.id), self.nodes))

  def get_postags(self):
      '''
      Returns the part of speech tags of the tree
      '''
      return list(map((lambda x :x.upos), self.nodes))

  def is_projective(self):
    '''
    Returns a boolean indicating if the dependency tree
    is projective (i.e. no edges are crossing)
    '''
    arcs = self.get_arcs()
    for (i,j) in arcs:
      for (k,l) in arcs:
        if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
          return False
    return True
  
  def __repr__(self):
      return "".join(str(e) for e in self.nodes)+"\n"
  
  @staticmethod
  def from_string(conll_str):
      '''
      Create a ConllTree from a dependency tree conll-u string.
      '''
      data = conll_str.split('\n')
      dependency_tree_start_index = 0
      for line in data:
          if line[0]!="#":
              break
          dependency_tree_start_index+=1
      data = data[dependency_tree_start_index:]
      nodes = []
      nodes.append(ConllNode.dummy_root())
      for line in data:
          # check if not valid line (empty or not enough fields)
          if (len(line)<=1) or len(line.split('\t'))<10:
              continue 
          # check if node is a contraction (multiexp lines are marked with .)
          # check if node is an omited word (empty nodes are marked with .)
          # check if node is a comment (comments are marked with #)
          if ("-" or "." or "#") in line.split('\t')[0]:
              continue
          
          conll_node = ConllNode.from_string(line)
          nodes.append(conll_node)
      
      return ConllTree(nodes)
  
  @staticmethod
  def read_conllu_file(file_path):
      '''
      Read a conllu file and return a list of ConllTree objects.
      '''
      with open(file_path, 'r') as f:
          data = f.read()
      data = data.split('\n\n')[:-1]
      return list(map((lambda x :ConllTree.from_string(x)), data))

  @staticmethod
  def write_conllu_file(file_path, trees):
      '''
      Write a list of ConllTree objects to a conllu file.
      '''
      with open(file_path, 'w') as f:
          f.write("".join(str(e) for e in trees))