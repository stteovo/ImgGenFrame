使用 flow-matching skill。

现在只实现 Flow Matching training primitives。

不要接入完整模型。

创建：

src/zimage/diffusion/

├── flow_matching.py
├── timestep_sampling.py
└── scheduler.py

实现：

1. linear interpolation
2. velocity target
3. timestep sampling
4. logit-normal timestep sampling
5. dynamic shifting
6. loss computation
7. inference ODE step

数学定义必须明确：

x_t = (1-t)x_0 + tx_1

v_target = x_1 - x_0

L = ||v_pred - v_target||²

要求：

tests/diffusion/

├── test_flow_matching.py
├── test_timestep_sampling.py
└── test_scheduler.py

测试：

- t=0
- t=1
- t=0.5
- zero noise
- deterministic seed
- batch broadcasting
- dtype
- BF16

另外写：

docs/learning/02_flow_matching.md

要求从：

DDPM
→ score matching
→ rectified flow
→ flow matching
→ Z-Image

解释它们之间的关系。

不要实现模型训练。