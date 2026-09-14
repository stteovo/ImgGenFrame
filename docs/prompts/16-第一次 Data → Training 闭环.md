现在把 data pipeline 和 training pipeline 接起来。

完整 pipeline：

raw image
↓
profile
↓
quality filter
↓
caption
↓
concept extraction
↓
embedding
↓
kNN
↓
Louvain/community detection
↓
KG mapping
↓
sampling weight
↓
training dataset
↓
S3-DiT

要求：

dataset 必须 versioned。

例如：

dataset_v001

不能依赖“当前目录里的图片是什么”。

训练 config 必须写：

dataset_version: v001