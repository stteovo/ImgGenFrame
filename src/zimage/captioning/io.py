"""Caption 记录 IO：parquet 读写 + dataset versioning。

parquet 输出（[IMPLEMENTATION] 用 pandas/pyarrow）；嵌套字段 JSON 序列化。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import CaptionRecord


def records_to_dataframe(records: List[CaptionRecord]):
    import pandas as pd

    rows = []
    for r in records:
        d = r.to_dict()
        for k in ("objects", "attributes", "style", "lighting", "composition", "ocr", "tags", "quality", "watermark"):
            if k in d:
                d[k] = json.dumps(d[k], ensure_ascii=False)
        rows.append(d)
    return pd.DataFrame(rows)


def write_parquet(records: List[CaptionRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records_to_dataframe(records).to_parquet(path, index=False)


def read_parquet(path: str | Path) -> List[CaptionRecord]:
    import pandas as pd

    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        d = row.to_dict()
        for k in ("objects", "attributes", "style", "lighting", "composition", "ocr", "tags", "quality", "watermark"):
            if k in d and isinstance(d[k], str):
                d[k] = json.loads(d[k])
        records.append(CaptionRecord.from_dict(d))
    return records
