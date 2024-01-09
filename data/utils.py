import numpy as np

def load_data(n, seq_length):
    import collections
    f = open(n)
    lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]
    lines = [l for l in lines if len(l[0])<seq_length-2]
    smiles = [l[0] for l in lines]

    total_string = ''
    for s in smiles:
        total_string+=s
    counter = collections.Counter(total_string)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, counts = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))

    chars+=('E',) #End of smiles
    chars+=('X',) #Start of smiles
    vocab['E'] = len(chars)-2
    vocab['X'] = len(chars)-1

    length = np.array([len(s)+1 for s in smiles])
    smiles_input = [('X'+s).ljust(seq_length, 'E') for s in smiles]
    smiles_output = [s.ljust(seq_length, 'E') for s in smiles]
    smiles_input = np.array([np.array(list(map(vocab.get, s)))for s in smiles_input])
    smiles_output = np.array([np.array(list(map(vocab.get, s)))for s in smiles_output])
    prop = np.array([l[1:] for l in lines])
    return smiles_input, smiles_output, chars, vocab, prop, length