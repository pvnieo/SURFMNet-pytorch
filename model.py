# 3p
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Implement one residual block as presented in FMNet paper."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-3, momentum=1e-3)
        self.fc2 = nn.Linear(out_dim, out_dim)

        if in_dim != out_dim:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.BatchNorm1d(out_dim)  # non implemented in original FMNet paper, suggested in resnet paper
            )
        else:
            self.projection = None

    def forward(self, x):
        x_res = F.relu(self.bn(self.fc1(x).transpose(1, 2)).transpose(1, 2))
        x_res = self.bn(self.fc2(x_res).transpose(1, 2)).transpose(1, 2)
        if self.projection:
            x = self.projection(x)
        x_res += x
        return F.relu(x_res)


class RefineNet(nn.Module):
    """Implement the refine net of FMNet. Take as input hand-crafted descriptors.
       Output learned descriptors well suited to the task of correspondence"""
    def __init__(self, n_residual_blocks=7, in_dim=352):
        super().__init__()
        model = []
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_dim, in_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """One pass in refine net.

        Arguments:
            x {torch.Tensor} -- input hand-crafted descriptor. Shape: batch-size x num-vertices x num-features

        Returns:
            torch.Tensor -- learned descriptor. Shape: batch-size x num-vertices x num-features
        """
        return self.model(x)


class FunctionalMapNet(nn.Module):
    """Compute the functional map matrix representation."""
    def __init__(self):
        super().__init__()

    def forward(self, feat_x, feat_y, evecs_x, evecs_y):
        """One pass in functional map net.

        Arguments:
            feat_x {Torch.Tensor} -- learned feature 1. Shape: batch-size x num-vertices x num-features
            feat_y {Torch.Tensor} -- learned feature 2. Shape: batch-size x num-vertices x num-features
            evecs_x {Torch.Tensor} -- eigen vectors decomposition of shape 1. Shape: batch-size x num-vertices x num-eigenvectors
            evecs_y {Torch.Tensor} -- eigen vectors decomposition of shape 2. Shape: batch-size x num-vertices x num-eigenvectors

        Returns:
            Torch.Tensor -- matrix representation of functional correspondence.
                            Shape: batch_size x num-eigenvectors x num-eigenvectors.
            Torch.Tensor -- matrix representation of functional correspondence.
                            Shape: batch_size x num-eigenvectors x num-eigenvectors.
        """
        # compute linear operator matrix representation C1 and C2
        F_hat = torch.bmm(evecs_x.transpose(1, 2), feat_x)
        G_hat = torch.bmm(evecs_y.transpose(1, 2), feat_y)
        F_hat, G_hat = F_hat.transpose(1, 2), G_hat.transpose(1, 2)

        Cs_1 = []
        for i in range(feat_x.size(0)):
            C = torch.inverse(F_hat[i].t() @ F_hat[i]) @ F_hat[i].t() @ G_hat[i]
            Cs_1.append(C.t().unsqueeze(0))
        C1 = torch.cat(Cs_1, dim=0)

        Cs_2 = []
        for i in range(feat_x.size(0)):
            C = torch.inverse(G_hat[i].t() @ G_hat[i]) @ G_hat[i].t() @ F_hat[i]
            Cs_2.append(C.t().unsqueeze(0))
        C2 = torch.cat(Cs_2, dim=0)

        return C1, C2


class SURFMNet(nn.Module):
    """Implement the SURFMNet network as described in the paper."""
    def __init__(self, n_residual_blocks=7, in_dim=352):
        """Initialize network.

        Keyword Arguments:
            n_residual_blocks {int} -- number of resnet blocks in FMNet (default: {7})
            in_dim {int} -- Input features dimension (default SHOT descriptor) (default: {352})
        """
        super().__init__()

        self.refine_net = RefineNet(n_residual_blocks, in_dim)
        self.funcmap_net = FunctionalMapNet()

    def forward(self, feat_x, feat_y, evecs_x, evecs_y):
        """One pass in FMNet.

        Arguments:
            feat_x {Torch.Tensor} -- hand crafted feature 1. Shape: batch-size x num-vertices x num-features
            feat_y {Torch.Tensor} -- hand crafted feature 2. Shape: batch-size x num-vertices x num-features
            evecs_x {Torch.Tensor} -- eigen vectors decomposition of shape 1. Shape: batch-size x num-vertices x num-eigenvectors
            evecs_y {Torch.Tensor} -- eigen vectors decomposition of shape 2. Shape: batch-size x num-vertices x num-eigenvectors

        Returns:
            Torch.Tensor -- soft correspondence matrix. Shape: batch_size x num_vertices x num_vertices.
            Torch.Tensor -- matrix representation of functional correspondence.
                            Shape: batch_size x num-eigenvectors x num-eigenvectors.
        """
        feat_x = self.refine_net(feat_x)
        feat_y = self.refine_net(feat_y)
        C1, C2 = self.funcmap_net(feat_x, feat_y, evecs_x, evecs_y)
        return C1, C2, feat_x, feat_y


if __name__ == "__main__":
    bs, n_points, n_feat, n_basis, n_basis2 = 10, 1000, 352, 100, 120
    feat_x = torch.rand(bs, n_points, n_feat)
    feat_y = torch.rand(bs, n_points, n_feat)
    evecs_x = torch.rand(bs, n_points, n_basis)
    evecs_y = torch.rand(bs, n_points, n_basis2)
    surfmnet = SURFMNet()
    C1, C2, feat_1, feat_2 = surfmnet(feat_x, feat_y, evecs_x, evecs_y)
    print(C1.size(), C2.size())
