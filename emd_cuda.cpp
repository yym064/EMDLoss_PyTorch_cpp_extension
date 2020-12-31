#include <torch/extension.h>
#include <vector>


at::Tensor approx_match_cuda_forward(const at::Tensor xyz1,
                                     const at::Tensor xyz2);
at::Tensor match_cost_cuda_forward(const at::Tensor xyz1, const at::Tensor xyz2,
                                   const at::Tensor match);
std::vector<at::Tensor> match_cost_cuda_backward(const at::Tensor grad_cost,
                                                 const at::Tensor xyz1,
                                                 const at::Tensor xyz2,
                                                 const at::Tensor match);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor
approx_match_forward(const at::Tensor xyz1, const at::Tensor xyz2)
{
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  return approx_match_cuda_forward(xyz1, xyz2);
}

at::Tensor
match_cost_forward(const at::Tensor xyz1, const at::Tensor xyz2,
                   const at::Tensor match)
{
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  return match_cost_cuda_forward(xyz1, xyz2, match);
}

std::vector<at::Tensor>
match_cost_backward(const at::Tensor grad_cost, const at::Tensor xyz1,
                    const at::Tensor xyz2, const at::Tensor match)
{
  CHECK_EQ(xyz1.size(0), xyz2.size(0));
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  return match_cost_cuda_backward(grad_cost, xyz1, xyz2, match);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_approx", &approx_match_forward, "Approx forward (CUDA)");
  m.def("forward_cost", &match_cost_forward, "Matchcost forward (CUDA)");
  m.def("backward_cost", &match_cost_backward, "Matchcost backward (CUDA)");
}