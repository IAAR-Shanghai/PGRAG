# -*- coding: utf-8 -*-
from py2neo import Graph
from PGRAG.PG_GEN.embedding import TextSimilarityCalculator, tensor

sim_calculator=TextSimilarityCalculator()

class BottomUpSearch:
    def __init__(self):
        try:
            self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "1234"))
            print("成功连接到Neo4j数据库.")
        except Exception as e:
            print(f"连接到Neo4j时出错: {e}")
            raise

    def get_neighbor_node_ids(self, node_id):
        try:
            parent_id = self.get_parent_node_id(node_id, verbose=False)
            if parent_id == -1:
                return []

            query_neighbors = """
            MATCH (parent)-[:基础链接]->(neighbor)
            WHERE id(parent) = $parent_id AND NOT id(neighbor) = $node_id
            RETURN collect(id(neighbor)) AS neighbor_ids
            """
            results = self.graph.run(query_neighbors, parent_id=parent_id, node_id=node_id).data()
            return results[0]['neighbor_ids'] if results else []
        except Exception as e:
            print(f"检索节点 {node_id} 的邻居节点ID时出错: {e}")
            return []

    def get_parent_node_id(self, node_id, verbose=True):
        try:
            query = """
            MATCH (parent)-[:基础链接]->(n)
            WHERE id(n) = $node_id
            RETURN id(parent) AS parent_id
            """
            parent_id = self.graph.run(query, node_id=node_id).evaluate()
            if verbose:
                if parent_id is not None:
                    print(f"节点 {node_id} 的父节点ID: {parent_id}")
                else:
                    print(f"找不到节点 {node_id} 的父节点.")
            return parent_id if parent_id is not None else -1
        except Exception as e:
            if verbose:
                print(f"检索节点 {node_id} 的父节点ID时出错: {e}")
            return -1

    def get_node_embedding(self, node_id):
        try:
            query = """
            MATCH (n)
            WHERE id(n) = $node_id
            RETURN 
                CASE 
                    WHEN 'Topic' IN labels(n) THEN n.主题嵌入
                    WHEN 'SubTopic' IN labels(n) THEN n.路由嵌入
                    WHEN 'Content' IN labels(n) THEN n.内容嵌入
                END AS embedding
            """
            result = self.graph.run(query, node_id=node_id).evaluate()
            return eval(result) if result else None
        except Exception as e:
            print(f"检索节点 {node_id} 的嵌入信息时出错: {e}")
            return None

    def calculate_similarity(self, node_embedding, kpr_embedding):
        # 直接使用 sim_calculator 实例计算两个嵌入之间的相似度
        similarity = sim_calculator.calculate_similarity_from_embedding(node_embedding, eval(kpr_embedding))
        return similarity

    def depth_search_for_similarity(self, node_id, kpr_embedding, seed_similarity, visited_node_ids, bottom_up_candidate_paths, current_depth=0):
        if node_id in visited_node_ids:
            return

        node_embedding = self.get_node_embedding(node_id)
        if node_embedding is None:
            return

        similarity = self.calculate_similarity(node_embedding, kpr_embedding)
        if similarity - seed_similarity >= -0.2:
            visited_node_ids.add(node_id)
            bottom_up_candidate_paths.append((node_id, similarity))
            print(f"节点 {node_id} 符合条件，相似度为 {similarity}, 已添加到候选路径")

            if -0.2 > similarity - seed_similarity >= -0.5:
                children = self.get_child_node_ids(node_id)
                for child_id in children:
                    self.depth_search_for_similarity(child_id, kpr_embedding, seed_similarity, visited_node_ids, bottom_up_candidate_paths, current_depth+1)

    def bottom_up_search(self, seed_topic_paths_with_similarity, path_ids, kpr_embedding, similarity_threshold=0.5):
        bottom_up_candidate_paths = []
        visited_node_ids = set()

        for (path, seed_similarity), path_id in zip(seed_topic_paths_with_similarity, path_ids):
            print(f'----在种子路径：{path_id} 上进行游走----')
            for node_id in reversed(path_id):
                neighbors = self.get_neighbor_node_ids(node_id)
                for neighbor_id in neighbors:
                    self.depth_search_for_similarity(neighbor_id, kpr_embedding, seed_similarity, visited_node_ids, bottom_up_candidate_paths)

        return bottom_up_candidate_paths

    # get_child_node_ids 方法实现，用于获取节点的所有子节点ID
    def get_child_node_ids(self, node_id):
        try:
            query = """
            MATCH (n)-[:基础链接]->(child)
            WHERE id(n) = $node_id
            RETURN collect(id(child)) AS child_ids
            """
            results = self.graph.run(query, node_id=node_id).data()
            return results[0]['child_ids'] if results else []
        except Exception as e:
            print(f"检索节点 {node_id} 的子节点ID时出错: {e}")
            return []


