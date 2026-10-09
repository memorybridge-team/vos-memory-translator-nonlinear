"""통신 증거·옵션 binding만 검사한다. 모델/tensor/실제 GPU 연산 없음."""
import pytest
from vos_memory_inspector.state_training.common import write_json, sha256
from vos_memory_inspector.state_training.remote_run import communication_environment


def test_communication_environment_rejects_unverified_options_and_rank_mapping(tmp_path):
    uuids = ['GPU-a891f407-b755-67eb-4727-a9da5d0c0730', 'GPU-56cadeba-6178-8504-350e-1d8d9c2a9a99']
    proof = tmp_path / 'pass.json'
    env = dict(NCCL_DEBUG='INFO', NCCL_P2P_DISABLE='1')
    value = dict(status='CUDA_COMMUNICATION_PASS', selected_env=env,
                 rank_results=[dict(rank=r, local_rank=r, current_device=r, all_reduce_sum=3.0,
                                    physical_uuid=uuid.removeprefix('GPU-')) for r, uuid in enumerate(uuids)])
    write_json(proof, value)
    cfg = dict(communication_env=env, communication_evidence=str(proof), communication_evidence_sha256=sha256(proof), gpu_uuids=uuids)
    assert communication_environment(cfg) == env
    with pytest.raises(ValueError, match='UNAPPROVED_NCCL_ENVIRONMENT'):
        communication_environment(dict(cfg, communication_env=dict(NCCL_IB_DISABLE='1')))
    with pytest.raises(ValueError, match='COMMUNICATION_RANK_UUID_MAPPING'):
        communication_environment(dict(cfg, gpu_uuids=list(reversed(uuids))))
    with pytest.raises(ValueError, match='COMMUNICATION_ENVIRONMENT_NOT_VERIFIED'):
        communication_environment(dict(cfg, communication_env=dict(NCCL_DEBUG='INFO')))
    proof.write_text('{}')
    with pytest.raises(ValueError, match='COMMUNICATION_EVIDENCE_CHANGED'):
        communication_environment(cfg)
