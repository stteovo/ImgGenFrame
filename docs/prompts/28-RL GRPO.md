使用 evaluation + distillation + RL skills。

设计 Z-Image post-training RL。

reward 至少拆成：

instruction following
+
aesthetic
+
text rendering
+
semantic correctness

不要只使用一个 reward。

先做 offline reward evaluation。

然后再考虑 GRPO。

要求：

记录 reward hacking。

尤其检查：

reward ↑
但 human quality ↓

的情况。