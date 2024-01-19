# -*- coding: utf-8 -*-
import networkx as nx
import torch
from py2neo import Graph
from PGRAG.pipline.embedding import TextSimilarityCalculator

class TopicFusionManager:
    def __init__(self):
        """
        初始化，连接到Neo4j数据库。
        """
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "1234"))
        self.sim_calculator = TextSimilarityCalculator()

    def keyword_fusion(self):
        """
        执行关键词融合过程。
        创建主题之间的连接基于共享的关键词，并创建超主题节点。
        """
        # 提取所有 Topic 节点的关键词
        query = "MATCH (n:Topic) RETURN n.主题, n.关键词"
        topics = self.graph.run(query).data()

        # 创建主题节点到关键词的映射
        topic_keywords = {topic['n.主题']: set(topic['n.关键词'].split()) for topic in topics if topic['n.关键词']}

        # 使用 NetworkX 创建图
        G = nx.Graph()

        # 在共享关键词的主题节点之间添加边
        for topic1, keywords1 in topic_keywords.items():
            G.add_node(topic1, keywords=keywords1)
            for topic2, keywords2 in topic_keywords.items():
                if topic1 != topic2 and keywords1 & keywords2:
                    G.add_edge(topic1, topic2)

        # 为每个连通分量创建一个超主题节点并连接相关的主题节点
        connected_components = list(nx.connected_components(G))
        for component in connected_components:
            super_topic_keywords = set()
            for topic in component:
                super_topic_keywords.update(G.nodes[topic]['keywords'])
            super_topic_keywords = ' '.join(super_topic_keywords)

            # 创建超主题节点
            super_topic_query = f"MERGE (s:SuperTopic {{关键词: '{super_topic_keywords}'}}) RETURN s"
            super_topic = self.graph.run(super_topic_query).evaluate()

            # 将 Topic 节点连接到 SuperTopic
            for topic_title in component:
                relation_query = f"""
                MATCH (t:Topic {{主题: $topic_title}}), (s:SuperTopic {{关键词: $super_topic_keywords}})
                MERGE (t)-[:属于]->(s)
                """
                self.graph.run(relation_query, topic_title=topic_title, super_topic_keywords=super_topic_keywords)

        print("关键词融合完成")

    def similarity_fusion(self):
        """
        执行相似性融合过程。
        对每个单一主题节点，检查其与其他超主题节点下的主题节点的相似性，并进行适当的更新。
        """
        # 第一步：获取所有单一主题的超主题节点
        query_single = """
        MATCH (t:Topic)-[:属于]->(st:SuperTopic)
        WITH st, count(t) AS topics_count
        WHERE topics_count = 1
        RETURN st.关键词 AS super_topic_keywords
        """
        single_super_topics = self.graph.run(query_single).data()

        # 第二步：对每个单一主题节点，检查其与其他超主题节点下的主题节点的相似性
        for single_super_topic in single_super_topics:
            # 获取单一主题节点的嵌入和名称
            query_t = """
            MATCH (t:Topic)-[:属于]->(st:SuperTopic {关键词: $keywords})
            RETURN t.主题 AS topic
            """
            single_topic_name = self.graph.run(query_t, keywords=single_super_topic['super_topic_keywords']).evaluate()

            query_te = """
            MATCH (t:Topic)-[:属于]->(st:SuperTopic {关键词: $keywords})
            RETURN t.主题嵌入 AS embedding
            """
            single_topic_embedding = self.graph.run(query_te,
                                                    keywords=single_super_topic['super_topic_keywords']).evaluate()

            # 获取其他所有超主题节点下的主题节点
            query_others = """
            MATCH (t:Topic)-[:属于]->(st:SuperTopic)
            WHERE NOT st.关键词 = $keywords
            RETURN t.主题嵌入 AS embedding, t.主题 AS topic
            """
            other_topics = self.graph.run(query_others, keywords=single_super_topic['super_topic_keywords']).data()

            # 检查相似性
            for other_topic in other_topics:
                similarity = self.sim_calculator.calculate_similarity_from_embedding(eval(single_topic_embedding),
                                                                                     eval(other_topic['embedding']))

                if similarity > 0.85:
                    # 获取相似主题节点的超主题节点的关键词
                    query_other_super_topic_keywords = """
                    MATCH (t:Topic {主题: $topic})-[:属于]->(st:SuperTopic)
                    RETURN st.关键词 AS super_topic_keywords
                    """
                    other_super_topic_keywords = self.graph.run(query_other_super_topic_keywords,
                                                                topic=other_topic['topic']).evaluate()

                    # 如果相似，更新超主题节点
                    update_query = """
                    MATCH (t:Topic {主题: $single_topic}), (st:SuperTopic {关键词: $other_super_keywords})
                    MERGE (t)-[:属于]->(st)
                    RETURN st
                    """
                    self.graph.run(update_query, single_topic=single_topic_name,
                                   other_super_keywords=other_super_topic_keywords)

                    # 将single_topic的关键词添加到other_super_keywords的SuperTopic现有的关键词集合中
                    merge_keywords_query = """
                    MATCH (st1:SuperTopic {关键词: $single_keywords}), (st2:SuperTopic {关键词: $other_super_keywords})
                    WITH st1, st2
                    SET st2.关键词 = st2.关键词 + ' ' + st1.关键词
                    DETACH DELETE st1
                    """
                    self.graph.run(merge_keywords_query, single_keywords=single_super_topic['super_topic_keywords'],
                                   other_super_keywords=other_super_topic_keywords)

                    break  # 找到相似节点后停止查找

        print("相似性融合完成")
