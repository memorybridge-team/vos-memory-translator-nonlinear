"""First-run deterministic execution settings, configured before CUDA setup."""
import os

import torch

CUBLAS_WORKSPACE = ':4096:8'


def configure_determinism(device):
    device = torch.device(device)
    workspace = None
    if device.type == 'cuda':
        workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if torch.cuda.is_initialized() and workspace != CUBLAS_WORKSPACE:
            raise ValueError('configure CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization')
        if workspace is None:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = CUBLAS_WORKSPACE
            workspace = CUBLAS_WORKSPACE
        elif workspace != CUBLAS_WORKSPACE:
            raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 to match the first-run policy')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return execution_profile(device)


def execution_profile(device):
    """Expected checkpoint metadata; inspecting it never initializes CUDA."""
    return dict(schema_version='cmmt.deterministic_execution.v1',
                deterministic_algorithms=True, warn_only=False, broadcast_buffers=False,
                cudnn_benchmark=False, cudnn_deterministic=True,
                cublas_workspace_config=CUBLAS_WORKSPACE if torch.device(device).type == 'cuda' else None, torch_version=str(torch.__version__))
