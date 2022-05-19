from torch.utils.cpp_extension import load
emd_cuda = load(
    'emd_cuda', ['emd_cuda.cpp', 'emd_kernel.cu'], verbose=True
)
help(emd_cuda)