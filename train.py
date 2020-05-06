# stdlib
import argparse
import os
# 3p
import numpy as np
import torch
# project
from model import SURFMNet
from utils import FAUSTDataset, SURFMNetLoss


def train_fmnet(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create dataset
    dataset = FAUSTDataset(args.dataroot, args.dim_basis)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    # create model
    surfmnet = SURFMNet(n_residual_blocks=args.num_blocks, in_dim=args.feat_dim).to(device)
    optimizer = torch.optim.Adam(surfmnet.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    criterion = SURFMNetLoss(args.wb, args.wo, args.wl, args.wd, args.sub_wd).to(device)

    # Training loop
    iterations = 0
    for epoch in range(1, args.n_epochs + 1):
        for i, data in enumerate(dataloader):
            feat_x, evecs_x, evals_x, feat_y, evecs_y, evals_y = data

            # sample vertices
            vertices = np.random.choice(feat_x.size(1), args.n_vertices)
            feat_x, feat_y = feat_x[:, vertices, :].to(device), feat_y[:, vertices, :].to(device)
            evecs_x, evecs_y = evecs_x[:, vertices, :].to(device), evecs_y[:, vertices, :].to(device)
            evals_x, evals_y = evals_x.to(device), evals_y.to(device)

            # do iteration
            optimizer.zero_grad()
            surfmnet.train()
            C1, C2, feat_1, feat_2 = surfmnet(feat_x, feat_y, evecs_x, evecs_y)
            loss = criterion(C1, C2, feat_1, feat_2, evecs_x, evecs_y, evals_x, evals_y, device)
            loss.backward()
            optimizer.step()

            # log and save model
            iterations += 1
            if iterations % args.log_interval == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, loss:{loss}")

        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(surfmnet.state_dict(), os.path.join(args.save_dir, 'epoch{}.pth'.format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lunch the training of SURFMNet model."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n-epochs", type=int, default=50, help="number of epochs of training")

    parser.add_argument("--feat-dim", type=int, default=352, help="Input feature dimension")
    parser.add_argument('--dim-basis', type=int, default=120,
                        help='number of eigenvectors used for representation. The first 500 are precomputed and stored in input')
    parser.add_argument("-nv", "--n-vertices", type=int, default=1500, help="Number of vertices used per shape")
    parser.add_argument("-nb", "--num-blocks", type=int, default=7, help="number of resnet blocks")
    parser.add_argument("--wb", type=float, default=1e3, help="Bijectivity penalty weight")
    parser.add_argument("--wo", type=float, default=1e3, help="Orthogonality penalty weight")
    parser.add_argument("--wl", type=float, default=1., help="Laplacian commutativity penalty weight")
    parser.add_argument("--wd", type=float, default=1e5, help="Descriptor preservation via commutativity penalty weight")
    parser.add_argument("--sub-wd", type=float, default=0.2,
                        help="Percentage of subsampled vertices used to compute descriptor preservation commutativity penalty")

    parser.add_argument('-d', '--dataroot', required=False, default="./data/faust/train", help='root directory of the dataset')
    parser.add_argument('--save-dir', required=False, default="./data/models/", help='root directory of the dataset')
    parser.add_argument("--n-cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--no-cuda', action='store_true', help='Disable GPU computation')
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="interval between model checkpoints")
    parser.add_argument("--log-interval", type=int, default=1, help="interval between logging train information")

    args = parser.parse_args()
    train_fmnet(args)
