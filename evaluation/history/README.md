# 历史逐题评测记录

四份 JSONL 保留原实验的题目 ID、片段 ID、计分和复核分类。题目 ID 中的数字为 CRUD-RAG 相应组别的原始行号，片段 ID 由旧索引生成。

| 文件 | 题数 | 可复算的信息 |
|---|---:|---|
| chinese-1000.jsonl | 1,000 | 候选和最终来源覆盖、Recall@5、回退、完整答对、引用支持 |
| rerank-ab-322.jsonl | 322 | 同一顺序的冻结候选，两组最终选证、答对及改退题数 |
| fact-audit-43.jsonl | 43 | 四种事实复核分类及各组答对数 |
| answer-ab-32.jsonl | 32 | 固定证据下两套回答提示的判分 |

复算只需要 Python 标准库。

```sh
python evaluation/history/replay.py
```

本机的 `.vgrag-eval/historical-oa/` 是 Git 忽略目录，保存来自 `/tmp/oa-*` 的逐题日志、索引清单和配套统计脚本。来源路径和文件 SHA-256 在该目录的 `manifest.json` 中。可核对快照哈希，并把公开的精简记录逐题对回原文件。补跑重复 ID 以最后一条为准。事实复核采用当时的人工定稿分类。

```sh
python evaluation/history/replay.py --raw-dir .vgrag-eval/historical-oa
```

原千题日志保存了候选命中数量及最终片段 ID。若要检查每题最初的完整候选列表和次序，需要用历史索引重跑检索。322 题冻结候选的列表和次序可直接回查。43 题事实复核留有分项检查与旧评分理由，原 322 题重排 A/B 的两臂回答全文需要重新生成。32 题回答提示对照的原始回答保存在本机原始档案。评测设置与分母见 [EVALUATION.md](../../EVALUATION.md)。
