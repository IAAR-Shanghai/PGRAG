# Pseudo-Graph Retrieval-Augmented Generation (PG-RAG)

PG-RAG是一个具有任务适应性的检索增强生成框架。它融合了检索增强生成（RAG）的核心概念，并遵循了索引、检索、生成的范式。PG-RAG通过有效的知识抽取和知识梳理，成功地规避了传统RAG在索引构造时所面临的众多限制。其层次游走策略确保了检索器能够在不同难度任务下具备灵活的适应性。此外，PG-RAG为大型模型提供了高密度的上下文，有效缓解了传统RAG系统在响应生成阶段受到上下文长度限制的问题。

## JsonToKG

`JsonToKG`是PG-RAG的一个重要组件，负责将JSON格式的数据转换为知识图谱（KG）。它通过递归解析JSON数据，并将解析后的数据插入到Neo4j数据库中，从而构建和扩展知识图谱。

### 功能

- 解析JSON数据。
- 构建和扩展Neo4j知识图谱。

### 使用指南

```python
from json_to_kg import JsonToKG

# 初始化
json_kg = JsonToKG()

# 调用方法处理JSON数据并构建知识图谱
# 示例：json_kg.process_and_insert_data('path/to/your/json/data')
```

## TopicFusionManager

`TopicFusionManager`是PG-RAG的核心组件之一，专注于知识图谱中的主题融合。它包括关键词融合和相似性融合两种功能，以帮助在知识图谱中更有效地管理和组织主题。

### 功能

- **关键词融合**：通过共享关键词创建主题之间的连接，并生成超级主题节点。
- **相似性融合**：基于嵌入向量的相似度分析，合并语义上相近的主题节点。

### 使用指南

```python
from topic_fusion_manager import TopicFusionManager

# 初始化
manager = TopicFusionManager()

# 执行关键词融合
manager.keyword_fusion()

# 执行相似性融合
manager.similarity_fusion()
```
