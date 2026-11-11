# vector-graph-rag-chinese-light

应用milvus的vector-graph-rag过程中发现milvus的评测主要针对英文文档，而本人实际使用过程中对于中文文档的抽取、建链和检索的需求偏多，且因为服务器资源问题，此前在集成工程的本地 macOS + Docker 环境中，使用 **Chroma + 中文 BGE-small**，以 2 路上传 12 份 Markdown（144 个片段），同时跑索引和 3 路问答，估算知识库新增内存开销约 **639 MiB**（包含原应用和数据库的增量，详见评测记录），作为轻量部署的参考，故此项目面向中文文档的轻量 Vector Graph RAG，正确率大概在百分之八十左右。通过 Chroma、本地中文 BGE-small 和硅基流动的 Qwen，完成原文向量召回、实体与关系召回、一跳扩展、原文重排和回答生成。

## 选型

| 组件 | 默认选型 | 考虑 |
|---|---|---|
| 向量存储 | Chroma 1.0.20 | 原文与图索引分集合保存，通过实体键扩展关系。 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | 原文、实体、关系共用本地模型，降低常驻内存。 |
| 对话模型 | Qwen/Qwen3.5-35B-A3B | 调用硅基流动 API，负责抽取、选证和回答。 |
| 本地文档清单 | SQLite | 原文、来源和位置独立持久化，索引完成后发布来源的新版本。 |

开发阶段在原集成工程跑过 1,000 道中文多文档题，完整答对率为 79.1%。该分数使用历史图索引与集成工程数据处理流程。独立包的验证与历史评测过程见 [EVALUATION.md](EVALUATION.md)。

## 快速开始

需要 Python 3.10 及以上版本和 Docker。首次加载嵌入模型会下载模型文件。

```sh
docker compose up -d
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

编辑 .env 填入密钥，再导出配置。

```sh
set -a
source .env
set +a
vgrag-zh ingest examples/policy.md
vgrag-zh ingest examples/roster.md
vgrag-zh ask '示例项目的重大变更由谁审批？'
```

命令返回 JSON，包含 answer、citations 和 status。引用给出来源标识、片段 ID、原文位置和文本，新导入的片段还提供标题路径、偏移和 token 数。相同来源再次导入时发布该来源的新片段。删除来源可运行以下命令。

```sh
vgrag-zh remove examples/roster.md
```

默认支持 TXT、Markdown 和 DOCX，文本文件可读取 UTF-8、GBK 等常见编码。PDF 解析使用可选依赖。

```sh
pip install -e '.[documents]'
vgrag-zh ingest handbook.pdf
```

PDF 按页读取文本层并保留坐标，扫描页可通过 PaddleOCR 或 Tesseract 入库。安装步骤和格式处理规则见 [文档切分方案](docs/文档切分方案.md)。

## Python API

```python
from vector_graph_rag_chinese_light import ChineseLightRAG

rag = ChineseLightRAG(data_dir=".vgrag")
rag.upsert_text("policy", "示例项目的重大变更必须由示例值守岗审批。")
rag.upsert_text("roster", "示例值守岗由示例成员担任。")

result = rag.query("示例项目的重大变更由谁审批？")
print(result.as_dict())

citations = rag.search("谁担任示例值守岗？")
```

source 是调用方指定的稳定标识。更新文档时沿用该标识，文档清单会替换它的旧片段。查询读取已发布清单，支持失败后重试导入。旧向量保留为缓存，磁盘占用会随更新增长。请将数据目录与 Chroma 持久化卷一起备份。

## 配置

配置从进程环境读取，CLI 和库都适用。默认数据目录为 .vgrag，CLI 可通过 --data-dir 指定。

| 环境变量 | 默认值或用途 |
|---|---|
| SILICONFLOW_API_KEY | 硅基流动模型密钥 |
| VGRAG_CHAT_MODEL | Qwen/Qwen3.5-35B-A3B |
| VGRAG_CHROMA_URL | http://127.0.0.1:8000 |
| VGRAG_EMBEDDING_CACHE | 可选的本地嵌入模型缓存目录 |
| VGRAG_RETRIEVAL_MODE | `single-stage`（默认）或 `two-stage`（实验） |

未配置模型密钥时，可以导入文档并返回直接向量检索的原文摘录。之后配置密钥并再次导入来源，可补建关系索引。模型请求失败时保留原文引用并返回状态，调用方可据此决定是否重试。

## 检索过程

1. 四种格式转为带标题路径的结构块，统一按 BGE tokenizer 切分，目标 384 token，按完整句子保留重叠。
2. 模型抽取最多 4 条带类型的事实关系。实体须出现在片段内，区分人员、岗位、组织和事项等类型。
3. BGE 对原文、实体及关系编码。关系记录关联来源片段。
4. 问题向量直接召回原文。问题实体和关系召回后，沿共享实体扩展一跳。
5. 按直接向量前 4 段、图召回片段、其余向量片段合并，去重后最多 24 段。
6. Qwen 选最多 8 段原文，再根据这些原文生成回答。

文档清单在一次 SQLite 写事务中发布。发布之前的 Chroma 写入失败时，清单继续指向原版本。查询以清单片段 ID 限定向量范围。来源删除后，后续查询使用更新的清单。

### 实验模式：关系和原文两阶段重排

```sh
vgrag-zh ask --retrieval-mode two-stage '示例项目的重大变更由谁审批？'
```

Python 调用可用 `ChineseLightRAG(retrieval_mode="two-stage")`。两种模式共用文档清单与 Chroma 索引，可直接在已有知识库上切换。两阶段模式有关系候选且已配置对话模型时，会额外调用一次同一模型，从向量召回和一跳扩展的关系中筛选最多 12 条，再回溯来源并进入原文选证。关系重排请求失败或返回无效结果时，继续使用未经关系重排的候选。

## 与 milvus-Vector Graph RAG 的链路差异

以下对照上游提交 47590ad4eab5838cdcfec94f8a53ebfe70ebe330 中的默认生成式重排，以及其 Jev 两阶段冻结实验。

| 对比项 | 本项目当前中文链路 | 上游默认链路 | 上游 Jev 两阶段冻结实验 | 本项目的考虑与影响 |
|---|---|---|---|---|
| 向量存储 | Chroma 保存原文集合和图集合，图集合包含实体与关系记录 | Milvus 分实体、关系和原文三个集合 | 沿用冻结图索引和直接原文召回结果 | 面向轻量部署。存储选型与重排顺序可独立调整，Chroma 也能支持先排关系、再排原文。 |
| 文本切片 | 按标题和段落切分，目标 384 token，按完整句子重叠，保留原文位置 | 应用加载器支持配置切片 | 原文重排使用完整来源段落 | 合并相邻短段并保留完整列表项，效果需用同批语料复测。 |
| 嵌入模型 | 中文 BGE-small 编码原文、实体和关系 | 使用配置的嵌入模型 | 历史关系候选用 Contriever，直接原文召回用英文 BGE-large | 中文文档为主要目标，三类文本共用一个本地模型以控制常驻内存。 |
| 抽取与校验 | Qwen 抽取带类型的中文实体与关系，校验名称在原文出现，区分岗位和人员 | LLM 从片段抽取三元组，建立实体、关系及原文关联 | 复用冻结关系候选 | 中文示例和类型约束针对动作混入实体、岗位与人员误连等开发中遇到的问题。 |
| 初始召回 | 问题向量直接找原文，同时结合问题实体和关系召回 | 问题实体参与实体与关系检索 | 关系候选来自冻结图检索，直接原文单独召回 | 保留直接命中的制度条款、数字、时间和例外条件，再与图带回的材料合并。 |
| 图扩展 | 沿带类型的共享实体扩展一跳，再回溯来源片段 | 沿实体与关系引用扩展，跳数可配置，默认一跳 | 对冻结关系候选评分后回溯来源 | 一跳限制候选规模；更长关系链的覆盖需要另做对照。 |
| 关系重排 | 图召回直接回溯原文 | LLM 先选择有用关系 | Jev 先给候选关系评分，保留前 64 条 | 当前把主要模型选证集中在原文阶段，图关系在召回时发挥作用。 |
| 原文候选合并 | 直接向量前 4 段、图回溯片段、其余直接向量片段，去重后最多 24 段 | 从选中关系取原文，数量不足时用直接向量召回补齐 | 轮流取关系来源段，与直接向量前 10 段交替合并，去重后最多 16 段 | 合并顺序影响哪些片段进入重排，也会影响互补证据能否保留。 |
| 原文重排 | Qwen 对原文候选选证，最多 8 段 | 默认单阶段先选关系，再按关联取原文 | Jev 对完整来源段落再评分，按排序评测前 5 或 10 段 | 原文重排可直接核对限定条件，但模型需要从原文重新识别关系链。 |
| 回答生成 | Qwen 读取问题、选中片段的标题和原文，返回回答及来源引用 | 从选中的关系与关联原文生成回答 | 冻结实验主要衡量证据召回 | 原文片段可直接用于引用，来源位置由独立文档清单保存。 |
| 证据链示例 | 两段分别写“变更由示例岗审批”和“示例成员担任示例岗”，原文重排决定两段是否保留 | 先判断两条关系是否有用，再回溯对应原文 | 先评关系，再评其来源段落 | 原文选证可能只留下审批岗位，丢掉担任者。历史千题有 161 道候选已找齐、最终来源未找齐的题。 |
| 评测结果口径 | 历史中文千题完整答对率 79.1%，Recall@5 为 85.43% | 上游有独立的默认方案历史评测 | 英文 MuSiQue 与 2Wiki 各 1,000 题，Jev 1.14 平均 Recall@5 为 86.61% | 语料、模型、证据匹配和答案指标不同。具体条件见本项目评测记录。 |

[上游架构](https://zilliztech.github.io/vector-graph-rag/architecture/how-it-works/) · [上游两阶段实验](https://github.com/zilliztech/vector-graph-rag/blob/47590ad4eab5838cdcfec94f8a53ebfe70ebe330/evaluation/jev/two-stage/README.md) · [本项目评测记录](EVALUATION.md)

## 本地评测

合成样例可以通过公共 API 评测。

```sh
python evaluation/run.py \
  --documents evaluation/documents.example.jsonl \
  --questions evaluation/questions.example.jsonl \
  --data-dir .vgrag/evaluation \
  --output .vgrag/evaluation-result.jsonl
```

提供 --judge-model Qwen/Qwen3.6-35B-A3B 时，额外对答案做语义评分。每行记录保留题目、回答、引用和评分。模型调用会产生费用。使用新的数据目录和输出路径可以保留独立实验条件。

[中文千题复现步骤](EVALUATION.md#在独立项目中复测中文千题)说明如何从固定版本的 CRUD-RAG 生成评测输入，并用同一索引分别运行单阶段和两阶段。脚本输出整体与双文档、三文档分组的 Recall@5、来源找齐题数、完整答对题数和平均及 P95 查询耗时。

本地数据、密钥和虚拟环境已配置 Git 忽略规则。来源与授权记录见 [NOTICE.md](NOTICE.md)。

## 开发检查

运行单元检查时使用 `python -m unittest discover -s tests -v`。安装 `pip install -e '.[documents,test]'` 可运行 DOCX 和 PDF 文件样例。中文 OCR 样例需要 Tesseract、`chi_sim` 和 `eng` 语言数据，以及系统中文字体。设置 `VGRAG_TEST_CHROMA_URL` 并启动 Chroma 后，同一命令还会运行导入、问答、更新和失败恢复的集成检查。

## 许可证

本项目采用 [MIT](LICENSE) 许可证。
