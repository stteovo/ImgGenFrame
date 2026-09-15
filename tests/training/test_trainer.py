"""FlowMatchingTrainer 测试：单步有限、checkpoint 往返、种子复现。"""

import torch

from zimage.models import S3DiTConfig
from zimage.pipeline import ZImagePipeline
from zimage.training.config import TrainingSpec
from zimage.training.dataset import SyntheticLatentDataset
from zimage.training.trainer import FlowMatchingTrainer


def _tiny_model():
    cfg = S3DiTConfig(hidden_size=128, depth=2, num_heads=1, ffn_dim=341)
    torch.manual_seed(0)
    return ZImagePipeline(cfg)


def _dataset():
    return SyntheticLatentDataset(
        num_images=8, latent_size=8, latent_channels=16, text_len=4, cap_feat_dim=2560, seed=0
    )


def test_train_step_finite():
    torch.manual_seed(0)
    model = _tiny_model()
    trainer = FlowMatchingTrainer(model, TrainingSpec(), device="cpu")
    x1, text = _dataset().sample_batch(2)
    loss = trainer.train_step(x1, text)
    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0


def test_checkpoint_save_load_resume():
    torch.manual_seed(0)
    model = _tiny_model()
    trainer = FlowMatchingTrainer(model, TrainingSpec(), device="cpu")
    ds = _dataset()
    x1, text = ds.sample_batch(2)

    # 训练 3 步后保存
    for _ in range(3):
        trainer.train_step(x1, text)
    trainer.save_checkpoint("/tmp/zk_test_ckpt.pt")
    params_before = [p.detach().clone() for p in model.parameters()]
    step_before = trainer.step

    # 新建 trainer 加载并续训
    torch.manual_seed(1234)  # 打乱 RNG，验证 checkpoint 恢复
    model2 = _tiny_model()
    trainer2 = FlowMatchingTrainer(model2, TrainingSpec(), device="cpu")
    trainer2.load_checkpoint("/tmp/zk_test_ckpt.pt")
    assert trainer2.step == step_before
    for p1, p2 in zip(params_before, model2.parameters()):
        assert torch.equal(p1, p2)

    # 续训一步正常且 loss 有限
    loss = trainer2.train_step(x1, text)
    assert torch.isfinite(torch.tensor(loss))


def test_seed_reproducibility_first_step():
    x1, text = _dataset().sample_batch(2)

    torch.manual_seed(0)
    m1 = _tiny_model()
    torch.manual_seed(999)
    t1 = FlowMatchingTrainer(m1, TrainingSpec(), device="cpu")
    l1 = t1.train_step(x1, text)

    torch.manual_seed(0)
    m2 = _tiny_model()
    torch.manual_seed(999)
    t2 = FlowMatchingTrainer(m2, TrainingSpec(), device="cpu")
    l2 = t2.train_step(x1, text)

    assert l1 == l2


def test_loss_decreases_on_single_image_overfit():
    # 单图过拟合：损失应显著下降（证明模型有记忆能力）
    torch.manual_seed(0)
    model = _tiny_model()
    spec = TrainingSpec(learning_rate=1e-3, batch_size=2, max_steps=200, grad_clip=1.0)
    trainer = FlowMatchingTrainer(model, spec, device="cpu")
    ds = SyntheticLatentDataset(num_images=1, latent_size=8, latent_channels=16, text_len=4, seed=0)

    losses = []
    for _ in range(200):
        x1, text = ds.sample_batch(2)
        losses.append(trainer.train_step(x1, text))

    assert losses[-1] < losses[0] * 0.5, f"loss 未显著下降：{losses[0]:.3f} → {losses[-1]:.3f}"
