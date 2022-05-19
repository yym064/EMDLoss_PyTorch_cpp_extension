import torch
import dist_emd

if __name__ == "__main__":

    a = torch.randn(5, 50, 3).cuda()
    b = torch.randn(5, 50, 3).cuda()

    print(f'input shape {a.size()}: ')
    print(f'target shape {b.size()}: ')
    cost = dist_emd.earth_mover_distance(a, b)
    print(f'cost shape {cost.size()}: ')

