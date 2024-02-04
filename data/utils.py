import numpy as np
import torch
from torch.utils import data



def load_data(path, train_split, batch_size, seq_length):
    import collections
    f = open(path)
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
    word_to_idx = dict(zip(chars, range(3,len(chars)+3)))


    chars+=('X',) #Start of smiles
    chars+=('E',) #End of smiles
    chars+=('Q',) #Padding
    word_to_idx['X'] = 1
    word_to_idx['E'] = 2
    word_to_idx['Q'] = 0

    idx_to_word = {v: k for k, v in word_to_idx.items()}


    length = np.array([len(s)+3 for s in smiles])

    smiles_input = [('X'+s+'E').ljust(seq_length, 'Q') for s in smiles]
    smiles_output = [('X'+s+'E').ljust(seq_length, 'Q') for s in smiles]

    smiles_input = np.array([np.array(list(map(word_to_idx.get, s)))for s in smiles_input])
    smiles_output = np.array([np.array(list(map(word_to_idx.get, s)))for s in smiles_output])
    prop = torch.tensor(np.array([l[1:] for l in lines]).astype(float))
    prop[:,0] = prop[:,0]/prop[:,0].max()
    prop[:,1] = prop[:,1]/prop[:,1].max()
    prop[:,2] = prop[:,2]/prop[:,2].max()

    class MoleculeDataset(data.Dataset):
        def __init__(self, inputs, props, targets):
            self.inputs = inputs
            self.props = props
            self.targets = targets

        def __len__(self):
            # Return the size of the dataset
            return len(self.targets)

        def __getitem__(self, index):
            # Retrieve inputs and targets at the given index
            X = self.inputs[index]
            c = self.props[index]
            y = self.targets[index]

            return (X, c), y

    all_data = len(smiles_input)
    x_train = torch.tensor(smiles_input[0:int(all_data*train_split)])
    x_test = torch.tensor(smiles_input[int(all_data*train_split):])

    c_train = prop[0:int(all_data*train_split)]
    c_test = prop[int(all_data*train_split):]

    y_train = torch.tensor(smiles_input[0:int(all_data*train_split)])
    y_test = torch.tensor(smiles_input[int(all_data*train_split):])

    train_set = MoleculeDataset(x_train, c_train, y_train)
    test_set = MoleculeDataset(x_test, c_test, y_test)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, word_to_idx, idx_to_word, prop, length