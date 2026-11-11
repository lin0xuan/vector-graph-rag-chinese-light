# 来源与授权记录

本项目从先前集成工程中的中文检索模块抽离，独立包新增本地索引清单、文档加载、Python API 和 CLI。

检索设计参考 Zilliz 的 Vector Graph RAG，包括实体与关系向量化、通过共享实体扩展、一跳关系回溯原文，以及关系与原文重排的对照。核对的仓库提交为 47590ad4eab5838cdcfec94f8a53ebfe70ebe330，上游采用 MIT 许可证。

- [Vector Graph RAG](https://github.com/zilliztech/vector-graph-rag)
- [参考版本](https://github.com/zilliztech/vector-graph-rag/tree/47590ad4eab5838cdcfec94f8a53ebfe70ebe330)
- [上游许可证](https://github.com/zilliztech/vector-graph-rag/blob/47590ad4eab5838cdcfec94f8a53ebfe70ebe330/LICENSE)

历史评测引用 CRUD-RAG、MuSiQue 和 2Wiki 数据及上游已公开指标。数据的取得、使用和再分发需遵循各自的许可。仓库只带合成示例。

本项目源码默认采用 MIT 许可证，见 LICENSE。参考上游的许可文本保存在 third_party/Vector-Graph-RAG-LICENSE。
