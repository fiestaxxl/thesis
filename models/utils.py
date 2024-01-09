import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

atom_types_number = 10
# Number of all possible bond types
bond_types_number = atom_types_number * (atom_types_number + 1) // 2 + 1

def integer_symmetric_matrix(v, h, w):
    """
    Generate a matrix of bond types' embeddings
    params: v, h, w - int
    """
    m = torch.randint(v, size=(h, w))
    m = torch.tril(m, 0) + torch.tril(m, -1).T
    return m

bond_types = integer_symmetric_matrix(bond_types_number, atom_types_number, atom_types_number)


def shortest_path_sequence(path, n_atoms, max_dist, atoms_types):
    """
    Find sequences of the atoms in the shortest paths
    """
    bonds = np.zeros((n_atoms, n_atoms, max_dist), dtype=np.int64)
    atoms_to = np.broadcast_to(np.arange(n_atoms), (n_atoms, n_atoms))
    atoms_from = atoms_to.T
    atoms_inner = np.ones((n_atoms, n_atoms), dtype=np.int64)
    mask = path != -9999
    for k in range(1, max_dist+1):
        atoms_inner[mask] = path[atoms_from[mask], atoms_to[mask]]
        bonds[:, :, k-1][mask] = bond_types[
            atoms_types[atoms_to[mask]],
            atoms_types[atoms_inner[mask]]
        ]
        mask *= atoms_inner != atoms_from
        atoms_to = atoms_inner.copy()
        atoms_inner[...] = 0
    return bonds.T

def shortest_path(atoms, atoms_from_to):
    """
    Compute the shortest paths
    """
    n = len(atoms)
    atoms_from, atoms_to = atoms_from_to
    row = atoms_from.numpy()
    col = atoms_to.numpy()
    is_bond = torch.ones_like(atoms_from).numpy()
    graph = csr_matrix((is_bond, (row, col)), shape=(n, n))
    dist_matrix, path = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
    spatial_pos = torch.from_numpy((dist_matrix)).long()
    max_dist = np.amax(dist_matrix).astype(np.int64)
    return spatial_pos, path, max_dist

def shortest_path_distance(atoms, atoms_from_to):
    """
    Compute the shortest paths and the coresponding sequences of the atoms
    """
    n = len(atoms)
    if atoms_from_to.shape[1] == 0:
        spatial_pos = torch.zeros((n, n)).long()
        edge_input = torch.zeros((1, n, n)).long()
        max_dist = 1
    else:
        spatial_pos, path, max_dist = shortest_path(atoms, atoms_from_to)
        edge_input = shortest_path_sequence(path, n, max_dist, atoms)
        edge_input = torch.from_numpy(edge_input).long()
    return spatial_pos, edge_input, max_dist