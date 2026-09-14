使用 rope-3d skill。

现在只实现 Z-Image 3D RoPE。

目标：

src/zimage/models/rope3d.py

要求：

1. 明确实现 temporal / height / width 三个 positional axes
2. 支持 variable H/W
3. 支持 text tokens
4. 支持 image tokens
5. 支持 batch 内不同 resolution
6. 支持 cached frequency
7. 尽量避免 Python loop

首先不要接入 S3-DiT。

先实现 unit tests：

tests/models/test_rope3d.py

至少测试：

- shape
- deterministic
- position sensitivity
- H/W change
- temporal position
- text/image token position
- head dimension compatibility

增加一个 visualization/debug script：

scripts/debug_rope3d.py

用于打印不同位置 token 的 frequency / embedding 差异。

完成后告诉我：

1. 3D RoPE 数学形式
2. 为什么 Z-Image 需要 3D 而不是普通 1D RoPE
3. text token 如何处理
4. image token 如何处理
5. 当前实现哪些地方来自论文
6. 哪些是工程实现