import torch
import dist_emd

if __name__ == "__main__":

    # forward test
    a = torch.randn(5, 50, 3, requires_grad = True).cuda()
    b = torch.randn(5, 50, 3, requires_grad = True).cuda()

    print(f'input shape {a.size()}: ')
    print(f'target shape {b.size()}: ')
    cost = dist_emd.earth_mover_distance(a, b)
    print(f'cost shape {cost.size()}: ')

    # backward test
    model = torch.nn.Sequential(
        torch.nn.Linear(150, 150),
        torch.nn.ReLU(),
        torch.nn.Linear(150, 150)
    )

    model.cuda()

    a = a.view(5, -1)
    a_ = model(a)
    a_ = a_.view(5, 50, -1)

    cost = (dist_emd.earth_mover_distance(a_, b)).sum()
    cost.backward()
    print(f'backward is done')

