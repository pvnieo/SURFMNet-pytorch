# stdlib
from os import listdir
from os.path import isfile, join
from itertools import combinations
# 3p
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class FAUSTDataset(Dataset):
    """FAUST dataset"""
    def __init__(self, root, dim_basis=120):
        self.root = root
        self.dim_basis = dim_basis
        self.samples = [self.loader(join(root, f)) for f in listdir(root) if isfile(join(root, f))]
        self.combinations = list(combinations(range(len(self.samples)), 2))

    def loader(self, path):
        """
        load dict stored at path. Dict has keys:size:
            target_evals: 500 * 1
            target_evecs: num_vertices * 500
            target_shot: num_vertices * 352
        """
        mat = sio.loadmat(path)
        return (torch.Tensor(mat['target_shot']).float(),
                torch.Tensor(mat['target_evecs'])[:, :self.dim_basis].float(),
                torch.Tensor(mat['target_evals']).flatten()[:self.dim_basis].float())

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, index):
        idx1, idx2 = self.combinations[index]
        feat_x, evecs_x, evals_x = self.samples[idx1]
        feat_y, evecs_y, evals_y = self.samples[idx2]

        return [feat_x, evecs_x, evals_x, feat_y, evecs_y, evals_y]


class SURFMNetLoss(nn.Module):
    """
    Calculate the loss as presented in the SURFMNet paper.
    """
    def __init__(self, w_bij=1e3, w_orth=1e3, w_lap=1, w_pre=1e5, sub_pre=0.2):
        """Init SURFMNetLoss

        Keyword Arguments:
            w_bij {float} -- Bijectivity penalty weight (default: {1e3})
            w_orth {float} -- Orthogonality penalty weight (default: {1e3})
            w_lap {float} -- Laplacian commutativity penalty weight (default: {1})
            w_pre {float} -- Descriptor preservation via commutativity penalty weight (default: {1e5})
            sub_pre {float} -- Percentage of subsampled vertices used to compute
                               descriptor preservation via commutativity penalty (default: {0.2})
        """
        super().__init__()
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap
        self.w_pre = w_pre
        self.sub_pre = sub_pre

    def forward(self, C1, C2, feat_1, feat_2, evecs_1, evecs_2, evals_1, evals_2, device):
        """Compute soft error loss

        Arguments:
            C1 {torch.Tensor} -- matrix representation of functional correspondence.
                                Shape: batch_size x num-eigenvectors x num-eigenvectors.
            C2 {torch.Tensor} -- matrix representation of functional correspondence.
                                Shape: batch_size x num-eigenvectors x num-eigenvectors.
            feat_1 {Torch.Tensor} -- learned feature 1. Shape: batch-size x num-vertices x num-features
            feat_2 {Torch.Tensor} -- learned feature 2. Shape: batch-size x num-vertices x num-features
            evecs_1 {Torch.Tensor} -- eigen vectors decomposition of shape 1. Shape: batch-size x num-vertices x num-eigenvectors
            evecs_2 {Torch.Tensor} -- eigen vectors decomposition of shape 2. Shape: batch-size x num-vertices x num-eigenvectors
            evals_1 {Torch.Tensor} -- eigen values of shape 1. Shape: batch-size x num-eigenvectors
            evals_2 {Torch.Tensor} -- eigen values of shape 2. Shape: batch-size x num-eigenvectors
            device {Torch.device} -- device used (cpu or gpu)
        Returns:
            float -- total loss
        """
        criterion = nn.MSELoss(reduction="mean")
        eye = torch.eye(C1.size(1), C1.size(2)).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C1.size(0), dim=0).to(device)

        # Bijectivity penalty
        bijectivity_penalty = criterion(torch.bmm(C1, C2), eye_batch) + criterion(torch.bmm(C2, C1), eye_batch)
        bijectivity_penalty *= self.w_bij

        # Orthogonality penalty
        orthogonality_penalty = criterion(torch.bmm(C1.transpose(1, 2), C2), eye_batch)
        orthogonality_penalty += criterion(torch.bmm(C2.transpose(1, 2), C1), eye_batch)
        orthogonality_penalty *= self.w_orth

        # Laplacian commutativity penalty
        laplacian_penalty = criterion(torch.einsum('abc,ac->abc', C1, evals_1), torch.einsum('ab,abc->abc', evals_2, C1))
        laplacian_penalty += criterion(torch.einsum('abc,ac->abc', C2, evals_2), torch.einsum('ab,abc->abc', evals_1, C2))
        laplacian_penalty *= self.w_lap

        # Descriptor preservation via commutativity
        # see `Informative Descriptor Preservation via Commutativity for Shape Matching` for more information
        # http://www.lix.polytechnique.fr/~maks/papers/fundescEG17.pdf
        num_desc = int(feat_1.size(2) * self.sub_pre)
        descs = np.random.choice(feat_1.size(2), num_desc)
        feat_1 = feat_1[:, :, descs].transpose(1, 2).unsqueeze(2)
        feat_2 = feat_2[:, :, descs].transpose(1, 2).unsqueeze(2)
        M_1 = torch.einsum('abcd,ade->abcde', feat_1, evecs_1)
        M_1 = torch.einsum('afd,abcde->abcfe', evecs_1.transpose(1, 2), M_1)
        M_2 = torch.einsum('abcd,ade->abcde', feat_2, evecs_2)
        M_2 = torch.einsum('afd,abcde->abcfe', evecs_2.transpose(1, 2), M_2)
        C1_expand = torch.repeat_interleave(C1.unsqueeze(1).unsqueeze(1), repeats=num_desc, dim=1)
        C2_expand = torch.repeat_interleave(C2.unsqueeze(1).unsqueeze(1), repeats=num_desc, dim=1)
        source1, target1 = torch.einsum('abcde,abcef->abcdf', C1_expand, M_1), torch.einsum('abcef,abcfd->abced', M_2, C1_expand)
        source2, target2 = torch.einsum('abcde,abcef->abcdf', C2_expand, M_2), torch.einsum('abcef,abcfd->abced', M_1, C2_expand)
        preservation_penalty = criterion(source1, target1) + criterion(source2, target2)
        preservation_penalty *= self.w_pre

        return bijectivity_penalty + orthogonality_penalty + laplacian_penalty + preservation_penalty


if __name__ == "__main__":
    dataroot = "./data/Faust_original/MAT"
    dataset = FAUSTDataset(dataroot)
    print(len(dataset))
    print(len(dataset[0]), dataset[0][1].size())

    bs, n_points, n_feat, n_basis = 10, 1000, 352, 100
    C1, C2 = torch.rand(bs, n_basis, n_basis), torch.rand(bs, n_basis, n_basis)
    evals_1, evals_2 = torch.rand(bs, n_basis), torch.rand(bs, n_basis)
    feat_1, feat_2 = torch.rand(bs, n_points, n_feat), torch.rand(bs, n_points, n_feat)
    evecs_1, evecs_2 = torch.rand(bs, n_points, n_basis), torch.rand(bs, n_points, n_basis)
    criterion = SURFMNetLoss(1e3, 1e3, 1, 1e5, 0.2)
    print(criterion(C1, C2, feat_1, feat_2, evecs_1, evecs_2, evals_1, evals_2))
