# 3p
import numpy as np
import torch
import torch.nn as nn


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

    def forward(self, C1, C2, feat_1, feat_2, evecs_1, evecs_2, evecs_trans_1, evecs_trans_2, evals_1, evals_2, device):
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
            evecs_trans_1: {Torch.Tensor} -- inverse eigen vectors decomposition of shape 1. defined as evecs_x.t() @ mass_matrix.
                                             Shape: batch-size x num-eigenvectors x num-vertices
            evecs_trans_2: {Torch.Tensor} -- inverse eigen vectors decomposition of shape 2. defined as evecs_y.t() @ mass_matrix.
                                             Shape: batch-size x num-eigenvectors x num-vertices
            evals_1 {Torch.Tensor} -- eigen values of shape 1. Shape: batch-size x num-eigenvectors
            evals_2 {Torch.Tensor} -- eigen values of shape 2. Shape: batch-size x num-eigenvectors
            device {Torch.device} -- device used (cpu or gpu)
        Returns:
            float -- total loss
        """
        criterion = FrobeniusLoss()
        eye = torch.eye(C1.size(1), C1.size(2)).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C1.size(0), dim=0).to(device)

        # Bijectivity penalty
        bijectivity_penalty = criterion(torch.bmm(C1, C2), eye_batch) + criterion(torch.bmm(C2, C1), eye_batch)
        bijectivity_penalty *= self.w_bij

        # Orthogonality penalty
        orthogonality_penalty = criterion(torch.bmm(C1.transpose(1, 2), C1), eye_batch)
        orthogonality_penalty += criterion(torch.bmm(C2.transpose(1, 2), C2), eye_batch)
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
        M_1 = torch.einsum('afd,abcde->abcfe', evecs_trans_1, M_1)
        M_2 = torch.einsum('abcd,ade->abcde', feat_2, evecs_2)
        M_2 = torch.einsum('afd,abcde->abcfe', evecs_trans_2, M_2)
        C1_expand = torch.repeat_interleave(C1.unsqueeze(1).unsqueeze(1), repeats=num_desc, dim=1)
        C2_expand = torch.repeat_interleave(C2.unsqueeze(1).unsqueeze(1), repeats=num_desc, dim=1)
        source1, target1 = torch.einsum('abcde,abcef->abcdf', C1_expand, M_1), torch.einsum('abcef,abcfd->abced', M_2, C1_expand)
        source2, target2 = torch.einsum('abcde,abcef->abcdf', C2_expand, M_2), torch.einsum('abcef,abcfd->abced', M_1, C2_expand)
        preservation_penalty = criterion(source1, target1) + criterion(source2, target2)
        preservation_penalty *= self.w_pre

        return bijectivity_penalty + orthogonality_penalty + laplacian_penalty + preservation_penalty


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum((a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)
