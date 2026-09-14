使用 reproduction-audit skill。

现在审计当前项目。

检查：

1. paper fidelity
2. official code fidelity
3. mathematical correctness
4. tensor shapes
5. dtype
6. numerical stability
7. reproducibility
8. experiment isolation
9. dataset versioning
10. evaluation consistency

把发现的问题分级：

P0:
incorrect implementation

P1:
major discrepancy

P2:
minor discrepancy

P3:
engineering improvement

不要修改代码。

先生成：

docs/reproduction/audit_<date>.md

然后等待我的确认。