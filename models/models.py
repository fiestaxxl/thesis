import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.utils import shortest_path_distance, atom_types_number, bond_types_number


class GaussianLayer(nn.Module):
    """
    Gaussian Basis Kernel functions
    """
    def __init__(self, n_kernels=128):
        """
        n_kernels - the number of Gaussian Basis kernels (K)
        """
        super().__init__()
        self.means = nn.Embedding(1, n_kernels)
        self.stds = nn.Embedding(1, n_kernels)
        self.gamma = nn.Embedding(bond_types_number, 1, padding_idx=0)
        self.beta = nn.Embedding(bond_types_number, 1, padding_idx=0)

        # nn.init.uniform_(self.means.weight, 0, 3)
        # nn.init.uniform_(self.stds.weight, 0, 3)
        # nn.init.constant_(self.bias.weight, 0)
        # nn.init.constant_(self.mul.weight, 1)

    def gaussian(self, x, mean, std):
        """
        Calculate a value of a gaussian function at x
        """
        norm_const = 1 / (torch.sqrt(2 * torch.tensor(torch.pi)) * std)
        exp_value = torch.exp(-0.5 * (((x - mean) / std) ** 2))
        return norm_const * exp_value

    def forward(self, x, edge_types):
        """
        If n is the number of the atoms in a particular molecule:
        x - tensor of pairwise distances between the atoms with size [n, n],
        edge_types - integer tensor of pairwise bond types between the atoms with size [n, n]
        """
        gamma = self.gamma(edge_types).squeeze(-1)            # [n, n, 1] -> [n, n]
        beta = self.beta(edge_types).squeeze(-1)              # [n, n, 1] -> [n, n]
        x = gamma * x + beta                                  # [n, n]
        means = self.means.weight.float().view(-1)            # [1, K] -> [K]
        stds = self.stds.weight.float().view(-1).abs() + 1e-2 # [1, K] -> [K]
        psi = self.gaussian(x[..., None], means, stds)             # [n, n, K]
        return psi


class BondEncoding2D(nn.Module):
    """
    Compute PhiSTD, PhiEdge
    """
    def __init__(self, n_heads, n_spatial):
        super().__init__()
        self.n_heads = n_heads
        self.spd_encoder = nn.Embedding(n_spatial, n_heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(bond_types_number, n_heads, padding_idx=0)
        self.edge_dis_encoder = nn.Embedding(n_spatial * n_heads * n_heads, 1)
        # self.apply(lambda module: init_params(module, n_layers=n_layers))



    def forward(self, atoms, atoms_from_to):
        # spatial_pos [n, n]
        # edge_input [n, n, max_dist]
        spatial_pos, edge_input, max_dist = shortest_path_distance(atoms, atoms_from_to)
        n = len(atoms)
        phi_spd = self.spd_encoder(spatial_pos).permute(2, 0, 1)
        edge_input = self.edge_encoder(edge_input)
        edge_input_flat = edge_input.permute(2, 0, 1, 3).reshape(max_dist, -1, self.n_heads)
        edge_weights = self.edge_dis_encoder.weight.reshape(-1, self.n_heads, self.n_heads)[:max_dist]
        edge_input_flat = torch.bmm(edge_input_flat, edge_weights) # [max_dist, n x n, n_heads]
        edge_input = edge_input_flat.reshape(max_dist, n, n, self.n_heads).permute(1, 2, 0, 3)
        spatial_pos_ = spatial_pos.clone()
        spatial_pos_[spatial_pos_ == 0] = 1
        spatial_pos_ = spatial_pos_.float().unsqueeze(-1) # [n, n, 1]
        phi_edge = (edge_input.sum(-2) / spatial_pos_).permute(2, 0, 1) # [n_heads, n, n]
        return phi_spd, phi_edge

class AtomEncoding2D(nn.Module):
    """
    Compute PsiDegree
    """
    def __init__(self, max_degree, atom_feature_dim):
        super().__init__()
        # self.n_heads = n_heads
        self.atom_feature_dim = atom_feature_dim
        self.atom_encoder = nn.Embedding(atom_types_number, atom_feature_dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(max_degree * 100, atom_feature_dim, padding_idx=0)
        # self.atom_encoder = nn.Embedding(atom_types_number, atom_feature_dim * n_heads, padding_idx=0)
        # self.degree_encoder = nn.Embedding(max_degree, atom_feature_dim * n_heads, padding_idx=0)
        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, atoms, degrees):
        # atoms - [n]
        # degrees - [n]
        psi_atoms = self.atom_encoder(atoms)      # [n, d]
        psi_degree = self.degree_encoder(degrees) # [n, d]
        psi_degree = psi_degree + psi_atoms
        # psi_degree = (psi_degree + psi_atoms).reshape(-1, self.n_heads, self.atom_feature_dim) # [n, n_heads, d]
        # psi_degree = psi_degree.permute(1, 0, 2)
        return psi_degree

class AtomEncoding3D(nn.Module): #TODO
    """
    Compute PsiSum3DDistance
    """
    def __init__(self, n_kernels, atom_feature_dim):
        super().__init__()
        # self.n_heads = n_heads
        self.atom_feature_dim = atom_feature_dim
        self.W_3d = nn.Linear(n_kernels, atom_feature_dim, bias=False)
        # self.W_3d = nn.Linear(n_kernels, n_heads * atom_feature_dim, bias=False)
        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, psi_3d):
        # psi_3d - [n, n, n_kernels]
        # [n, n, n_kernels] -> [n, n_kernels] -> [n, d]
        phi_3d_sum = self.W_3d(psi_3d.sum(-2))
        # phi_3d_sum = phi_3d_sum.reshape(-1, self.n_heads, self.atom_feature_dim)
        # [n, n_heads, d] -> [n_heads, n, d]
        # phi_3d_sum = phi_3d_sum.permute(1, 0, 2)
        return phi_3d_sum

class NonLinear(nn.Sequential):
    def __init__(self, input_size, output_size):
        super().__init__(
            nn.Linear(input_size, input_size, bias=False),
            nn.GELU(),
            nn.Linear(input_size, output_size, bias=False)
        )

class BondEncoding3D(nn.Module):
    """
    Compute Phi3D
    """
    def __init__(self, n_heads, n_kernels):
        super().__init__()
        self.gaussian_kernels = GaussianLayer(n_kernels)
        self.perceptron = NonLinear(n_kernels, n_heads)

    def forward(self, pos, edge_types):
        """
        If n is the number of the atoms in a particular molecule:
        pos - tensor of the atom's positions with size [n, 3]
        edge_types - integer tensor of the atom's pairwise connection types with size [n, n]
        """
        x = torch.cdist(pos, pos)                     # [n, 3] -> [n, n]
        psi_3d = self.gaussian_kernels(x, edge_types) # [n, n] -> [n, n, n_kernels]
        phi_3d = self.perceptron(psi_3d)              # [n, n, n_kernels] -> [n, n, n_heads]
        phi_3d = phi_3d.permute(2, 0, 1)
        return phi_3d, psi_3d

class Dropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

class PositionalEncoding(nn.Module):
    def __init__(self, n_heads, max_degree, n_kernels, n_spatial, atom_feature_dim):
        super().__init__()
        self.atom_encoder_2d = AtomEncoding2D(max_degree=max_degree, atom_feature_dim=atom_feature_dim)
        self.atom_encoder_3d = AtomEncoding3D(n_kernels=n_kernels, atom_feature_dim=atom_feature_dim)
        self.bond_encoder_2d = BondEncoding2D(n_heads=n_heads, n_spatial=n_spatial)
        self.bond_encoder_3d = BondEncoding3D(n_heads=n_heads, n_kernels=n_kernels)

    def forward(self, data_atoms, data_bonds):
        """
        data_atoms.x - atoms features
        data_atoms.pos - atoms positions
        data_atoms.atoms - atoms types
        data_bonds.edge_index - bonds: [atoms from, atoms_to]
        data_bonds.edge_attr - [bond length, bonds types]
        """
        n = len(data_atoms.atoms)
        edge_types = torch.zeros(n, n).long()
        atoms_from, atoms_to = data_bonds.edge_index
        edge_types[atoms_from, atoms_to] = edge_types[atoms_to, atoms_from] = data_bonds.edge_attr[1]
        degrees = (edge_types != 0).sum(1)

        phi_3d, psi_3d = self.bond_encoder_3d(data_atoms.pos, edge_types)
        phi_spd, phi_edge = self.bond_encoder_2d(data_atoms.atoms, data_bonds.edge_index)
        phi_degree = self.atom_encoder_2d(data_atoms.atoms, degrees)
        phi_3d_sum = self.atom_encoder_3d(psi_3d)

        return {
            'atoms': (phi_degree, phi_3d_sum),   # [n, d]
            'bonds': (phi_3d, phi_spd, phi_edge) # [n_heads, n, n]
        }

class AttentionBlock(nn.Module):
    def __init__(self, atom_feature_dim, scaling):
        super().__init__()
        self.scaling = scaling
        self.Q = nn.Linear(atom_feature_dim, atom_feature_dim, bias=False)
        self.K = nn.Linear(atom_feature_dim, atom_feature_dim, bias=False)
        self.V = nn.Linear(atom_feature_dim, atom_feature_dim, bias=False)
        self.dropout_module = Dropout(0.1, module_name=self.__class__.__name__)

    def forward(self, x, phi_3d, phi_spd, phi_edge, delta_pos):
        """
        x - [n + 1, d]
        phi_3d, phi_spd, phi_edge - [n, n]
        """
        Q = self.Q(x) # [n + 1, d]
        K = self.K(x) # [n + 1, d]
        V = self.V(x) # [n + 1, d]
        attn = Q @ K.transpose(-1, -2) * self.scaling # [n + 1, n + 1] # / np.sqrt(Q.size(-1)))
        stride = x.shape[0] - phi_3d.shape[0]
        attn[stride:, stride:] += phi_spd + phi_edge + phi_3d # [n + 1, n + 1]
        attn = F.softmax(attn, dim=-1) # [n + 1]
        attn = self.dropout_module(attn)
        # attn = attn.unsqueeze(-1) * delta_pos.unsqueeze(1)
        attn = attn @ V # [n + 1, d]
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, atom_feature_dim, model_dim=1):
        super().__init__()
        scaling = (atom_feature_dim // n_heads) ** -0.5
        self.heads = nn.Sequential(*[
            AttentionBlock(atom_feature_dim, scaling) for i in range(n_heads)
        ])
        self.W = nn.Linear(n_heads * atom_feature_dim, model_dim, bias=False)

    def forward(self, x, phi_3d, phi_spd, phi_edge, delta_pos=None):
        """
        x - [n + 1, d]
        phi_3d, phi_spd, phi_edge - [n_heads, n, n]
        """
        # delta_pos - нормализованные радиус-вектора попарных расстояний между атомами [n, n, 3]
        attn = [
            head(x, phi_3d[h], phi_spd[h], phi_edge[h], delta_pos) # [n + 1, d]
            for h, head in enumerate(self.heads)
        ]
        attn = torch.cat(attn, dim=-1) # [n + 1, n_heads x d]
        attn = self.W(attn)            # [n + 1, d]
        return attn


class TransformerMLayer(nn.Module):
    def __init__(self, n_heads, atom_feature_dim, model_dim):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(n_heads, atom_feature_dim, model_dim)
        self.feedforward = NonLinear(model_dim, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, phi_3d, phi_spd, phi_edge):
        """
        x - [n + 1, d]
        phi_3d, phi_spd, phi_edge - [n_heads, n, d]
        """
        attn = self.multihead_attn(x, phi_3d, phi_spd, phi_edge)
        x = self.norm1(x + attn)
        z = self.feedforward(x)
        x = self.norm2(x + z)
        return x # [n + 1, d]

class TransformerMEncoder(nn.Module):
    def __init__(self, n_heads, max_degree, n_kernels, n_spatial, atom_feature_dim, n_encoder_layers):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, atom_feature_dim))
        self.positional_encoder = PositionalEncoding(
            n_heads=n_heads, max_degree=max_degree, n_kernels=n_kernels,
            n_spatial=n_spatial, atom_feature_dim=atom_feature_dim
        )
        self.transformer_m_layers = nn.Sequential(*[
            TransformerMLayer(
                n_heads=n_heads, atom_feature_dim=atom_feature_dim, model_dim=atom_feature_dim
            ) for _ in range(n_encoder_layers)
        ])

    def forward(self, data_atoms, data_bonds):
        phi = self.positional_encoder(data_atoms, data_bonds)
        phi_degree, phi_3d_sum = phi['atoms']
        phi_3d, phi_spd, phi_edge = phi['bonds']

        x = data_atoms.x + phi_degree + phi_3d_sum # [n, d]
        x = torch.cat([self.cls_token, x]) # [n + 1, d]

        for layer in self.transformer_m_layers:
            x = layer(x, phi_3d, phi_spd, phi_edge)
        return x[0] # cls_token



class TransformerM(nn.Module):
    def __init__(self, n_heads, max_degree, n_kernels, n_spatial, atom_feature_dim, n_encoder_layers):
        super().__init__()
        self.encoder = TransformerMEncoder(
            n_heads=n_heads, max_degree=max_degree, n_kernels=n_kernels,
            n_spatial=n_spatial, atom_feature_dim=atom_feature_dim,
            n_encoder_layers=n_encoder_layers
        )
        self.regressor = nn.Linear(atom_feature_dim, 1)

    def forward(self, data_atoms, data_bonds):
        out = self.encoder(data_atoms, data_bonds)
        out = self.regressor(out)
        return out


# CVAE

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=None):
        super(Embedding,self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embed_dim,
                                padding_idx=padding_idx)

    def forward(self,x):
        # x: bs*seq_len
        return self.emb(x) #x: bs*seq_len*emd_dim

class Encoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 cond_dim,
                 hidden_size,
                 num_layers
                 ):

        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cond_dim = cond_dim

        self.enc = nn.LSTM(
            input_size = emb_dim+cond_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = False,
        )

    def forward(self, x, c):
        # x: tensor of shape (bs * seq_length * emb_dim)
        # c: tensor of shape (bs * cond_dim)

        seq_len = x.shape[1]
        c = c.unsqueeze(1).repeat(1,seq_len,1) #bs*seq_len*cond_dim

        x_emb = torch.cat([x, c], dim=-1).float()
        #packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input=x_emb, lengths=l.to('cpu'), batch_first=True, enforce_sorted=False)
        #_, (h_enc, _) = self.lstm(packed_x_embed)

        out, (h_enc, c_enc) = self.enc(x_emb)
        return h_enc

class Parametrizator(nn.Module):
    def __init__(self, hidden_size, latent_dim, num_layers):
        super(Parametrizator, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_dim
        self.num_layers = num_layers

        self.mean = torch.nn.Linear(in_features = self.hidden_size * self.num_layers, out_features= self.latent_size)
        self.log_variance = torch.nn.Linear(in_features= self.hidden_size * self.num_layers, out_features= self.latent_size)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)

        z = mu + noise * std
        return z

    def forward(self, hid_state):
        enc_h = hid_state.view(-1, self.hidden_size*self.num_layers)
        mu = self.mean(enc_h)
        log_var = self.log_variance(enc_h)

        z = self.reparametrize(mu,log_var)
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 cond_dim,
                 latent_dim,
                 hidden_size,
                 num_layers,
                 ):

        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.c_init = nn.Linear(latent_dim, hidden_size)
        self.h_init = nn.Linear(latent_dim, hidden_size)

        self.dec = nn.LSTM(
            input_size = latent_dim+emb_dim+cond_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = False,
        )

    def forward(self, z, x, c, hidden=None):
        # z: tensor of shape (bs * latent_dim)
        # x: tensor of shape (bs * seq_length * emb_dim)
        # c: tensor of shape (bs * cond_dim)

        z_hid = z.unsqueeze(0).repeat(self.num_layers, 1, 1) #n_layers*bs*latent_dim

        c_0 = torch.tanh(self.c_init(z_hid))    #n_layers*bs*hid_size
        h_0 = torch.tanh(self.h_init(z_hid))    #n_layers*bs*hid_size

        seq_len = x.shape[1]

        z = z.unsqueeze(1).repeat(1, seq_len, 1) # bs*seq_len*latent_dim
        c = c.unsqueeze(1).repeat(1, seq_len, 1) # bs*seq_len*cond_dim

        z = torch.cat([z,x,c], dim=-1).float()   #bs*seq_len*(latent_dim+emb_dim+cond_dim)

        if hidden is None:
            out, hid = self.dec(z,(h_0,c_0))
        else:
            out, hid = self.dec(z,hidden)

        return out, hid

class Predictor(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128,vocab_size)

        self.predictor = nn.Sequential(self.fc1,self.fc2)

    def forward(self, x):
        return self.predictor(x)


class CVAE(nn.Module):
    def __init__(self, vocab_size,
                 emb_dim, cond_dim,
                 hidden_size, latent_dim,
                 num_layers, padding_idx=None):
        super(CVAE, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.emb_x = Embedding(vocab_size, emb_dim, padding_idx)

        self.encoder = Encoder(emb_dim,
                               cond_dim,
                               hidden_size,
                               num_layers)

        self.parametrize = Parametrizator(hidden_size, latent_dim, num_layers)
        self.decoder = Decoder(emb_dim, cond_dim, latent_dim, hidden_size, num_layers)
        self.predictor = Predictor(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, c):
        x_emb = self.emb_x(x) #bs*seq_len*emb_dim
        h = self.encoder(x_emb,c)
        z, mu, logvar = self.parametrize(h)
        out, _ = self.decoder(z,x_emb,c)
        preds = self.predictor(out)

        return self.softmax(preds), preds, mu, logvar

    def sample(self, z, x, c, seq_len):
        # z: tensor of shape (bs * latent_dim)
        # x: tensor of shape (bs * 1)
        # c: tensor of shape (bs * cond_dim)

        with torch.no_grad():
            hidden = (torch.zeros((self.num_layers, z.size(0), self.hidden_size)),
                      torch.zeros((self.num_layers, z.size(0), self.hidden_size)))

            outputs = [x.item()]

            for _ in range(1,seq_len):
                x = self.emb_x(x) #(bs * (1+i) * emb_dim)
                x, hidden = self.decoder(z, x, c, hidden)
                x = self.predictor(x) #(bs*seq_len*vocab_size)
                x = x.argmax(-1) #(bs*(1+i))
                outputs.append(x[:,-1].item())

            return outputs
