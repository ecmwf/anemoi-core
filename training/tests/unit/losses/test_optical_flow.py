import torch

from anemoi.training.losses.optical_flow import OpticalFlowConsistencyLoss


def test_optical_flow_loss_zero_for_static_teacher_match() -> None:
    loss = OpticalFlowConsistencyLoss(
        x_dim=4,
        y_dim=4,
        patch_size=3,
        search_radius=1,
        seed_spacing=1,
        max_seeds=16,
        rain_threshold=0.05,
        downsample=1,
        preprocess_sigma=0.0,
        mask_threshold=None,
        delta=0.1,
    )
    loss.add_scaler(3, torch.ones(16), name="node_weights")

    static = torch.zeros(1, 2, 1, 16, 1)
    static[..., 5, 0] = 1.0
    static[..., 6, 0] = 0.5
    pred = static[:, -1:, :, :, :].repeat(1, 2, 1, 1, 1).clone().requires_grad_(True)
    target = pred.detach().clone()

    out = loss(pred, target, input_context=static)
    torch.testing.assert_close(out, torch.tensor(0.0), atol=1e-6, rtol=0.0)

    out.backward()
    assert pred.grad is not None
