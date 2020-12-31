import torch
import emd_cuda


class EarthMoverDistanceFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        
        match = emd_cuda.forward_approx(xyz1, xyz2)
        cost  = emd_cuda.forward_cost(xyz1, xyz2, match)
        # match = ext.approx_match_forward(xyz1, xyz2)
        # cost = ext.match_cost_forward(xyz1, xyz2, match)
        
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.backward_cost(grad_cost, xyz1, xyz2, match)
        # grad_xyz1, grad_xyz2 = ext.match_cost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def earth_mover_distance(xyz1, xyz2):
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)

    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)

    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
    return cost
