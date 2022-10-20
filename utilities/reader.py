from conllu import parse_tree_incr

def parse_conllu(in_file):
    sentences = []
    postags = [] 
    words = []

    in_file = open(in_file)

    for i, token_tree in enumerate(parse_tree_incr(in_file)):
        print("[*] Reading conll-U "+str(i), end="\r")

        current_tags = []
        current_words = []
        
        data = token_tree.serialize().split('\n')
        dependency_start_idx = 0
        # skip info lines
        for line in data:
            if line[0]!="#":
                break
            if "# text" in line:
                sentence = line.split("# text = ")[1]
                sentences.append(sentence)
            dependency_start_idx+=1
        
        data = data[dependency_start_idx:]

        # data lines
        for line in data:
            # check if not valid line
            if (len(line)<=1) or len(line.split('\t'))<10 or line[0] == "#":
                continue
            wid,form,lemma,upos,xpos,feats,head,deprel,deps,misc = line.split('\t')
            current_tags.append(upos)
            current_words.append(form)

        postags.append(current_tags)
        words.append(current_words)    
    
    print("[*] Reading conll-U",i,": Done")
    return sentences, postags, words
