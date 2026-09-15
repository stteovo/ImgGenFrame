# Flow Matching 学习笔记：从 DDPM 到 Z-Image

## 0. 一句话脉络

DDPM 用「加噪—去噪」的随机过程拟合数据分布；score matching 揭示去噪的实质是学数据分布对数密度的梯度；rectified flow / flow matching 放弃随机微分方程，改用一条**确定性直线路径**，让「噪声→数据」的传输成本最低、采样最快；Z-Image 采用 flow matching 作为训练目标，并加上 logit-normal 时间采样与动态 time shift 两处工程增强。

---

## 1. DDPM（Denoising Diffusion Probabilistic Models）

- 前向过程：逐步加噪 `x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε`，t=1..T，T 很大（1000+）。
- 反向过程：学一个网络 `ε_θ(x_t, t)` 预测每一步加的噪声，逐步去噪。
- 训练目标：`L = ||ε_θ(x_t, t) - ε||²`（变分下界的简化，权重为 1）。
- 采样：T 步马尔可夫链，慢（上千次前向）。
- 核心贡献：把「一步生成」拆成「多步小步去噪」，训练稳定、样本质量高。

## 2. Score Matching（分数匹配）

- 关键观察（Tweedie 公式 + score 网络）：去噪目标与数据分布的 score（`∇_{x} log p(x)`）等价，ε 预测 ≈ 负 score 的缩放。
- 因此扩散模型本质是在学数据分布的梯度场：score 指向高概率密度方向。
- Langevin 采样：`x ← x + η ∇log p(x) + noise`，用 score 爬山式采样。
- 统一视角：DDPM 是 score-based generative model 的一种离散化。

## 3. Rectified Flow / Flow Matching

问题：DDPM 的路径是随机的、弯弯绕绕的，采样步数多、路径冗余。

- **Flow（流）**：用常微分方程（ODE）`dx/dt = v(x, t)` 定义一个确定性的「传输映射」，把噪声分布推到数据分布。
- **线性插值路径**：`x_t = (1-t) x_0 + t x_1`，x_0~噪声、x_1~数据。这是噪声与数据之间的**直线**，传输成本最低。
- **速度目标**：`v = x_1 - x_0`（直线的切向量，恒定）。
- **训练（Flow Matching / Conditional Flow Matching）**：`L = E[ ||v_θ(x_t, t) - (x_1 - x_0)||² ]`。
  - 论文中常用「conditional」版本：给每个样本配一个 (x_0, x_1) 对，学条件速度场，再边缘化。
- **采样（Rectified Flow）**：从 x_0~噪声出发，ODE `x_{t+dt} = x_t + dt · v_θ(x_t, t)`，一步到位（Euler 即可），几十步就能出图。
- 相比 DDPM：路径直 → 步数少；确定性 → 可复现、可做蒸馏。

## 4. Z-Image 怎么用 Flow Matching

- **目标**（[PAPER §4.3]）：`x_t = t·x1 + (1−t)·x0`（x0=噪声、x1=原图），`v = x1 − x0`，MSE。预测 velocity（v-prediction），与我们的 `flow_matching.py` 完全一致。
- **logit-normal t 采样**：训练时 t 不是均匀采，而是 `z~N(0,1), t=sigmoid(z)`，把训练集中在**中间 t**（那里路径曲率/难度最大），两端稀疏。σ 论文未给 → 我们按 SD3 惯例取 σ=1.0（[ASSUMPTION]）。
- **动态 time shift**：多分辨率训练时，不同分辨率的 token 数不同 → 噪声水平（SNR）不同；Flux 式 shift `t_shifted = e^μ·t/(1+(e^μ−1)t)` 把时间表按序列长度重排，补偿分辨率差异。
- **采样**：Euler ODE + CFG（条件/无条件两次前向加权）；发布权重推理用固定 shift（base=6.0、Turbo=3.0），训练用动态 shift（两条路径显式区分）。

## 5. 本项目实现（`src/zimage/diffusion/`）

| 文件 | 内容 | 对齐来源 |
| --- | --- | --- |
| `flow_matching.py` | 线性插值 / velocity 目标 / MSE 损失 / flow pair 采样 | [PAPER §4.3] |
| `timestep_sampling.py` | uniform / logit-normal 采样、`calculate_shift`、`apply_time_shift` | [PAPER §4.3] + [OFFICIAL-CODE] diffusers |
| `scheduler.py` | Euler ODE 采样器（t=1−sigma，sigma 表 `linspace(1.0, 1/N, N)`） | [OFFICIAL-CODE] pipeline_z_image |

28 个 pytest 通过，含：
- 端点/中点闭式校验（t=0/0.5/1）、zero-noise、确定性 seed、batch 广播、dtype、BF16。
- logit-normal 经验 CDF 与理论 `Φ(logit(t))` 对比（统计分布验证）。
- `calculate_shift` / `apply_time_shift` 与 diffusers 逐值对齐。
- 恒定速度场下 Euler 精确复原数据的端到端测试。

## 6. 为什么不能简单替换

- 用 DDPM ε-prediction 替代 flow matching 会改变采样器语义（ε vs v 需要重参数化，v = x1−x0 与 ε 差一个尺度），且路径更弯、步数更多。
- 用均匀 t 采样替代 logit-normal 会把大量训练算力浪费在「已很简单的两端」，降低中间 t 的建模精度。
- 忽略动态 shift 会导致多分辨率训练时不同分辨率样本的 SNR 不一致，训练失衡。

## 7. 关键易错点

- x0/x1 方向与论文相反（v 预测变 ε 预测而采样器没同步改）——本项目明确约定 x0=噪声、x1=数据。
- 训练用动态 shift、推理用固定 shift 不区分——两条路径已显式分开并记录。
- logit-normal 的 σ 拍脑袋——已标 [ASSUMPTION] σ=1.0 待消融（open_questions A7）。
