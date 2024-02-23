# -*- coding: utf-8 -*-
from py2neo import Graph
from PGRAG.PG_GEN.embedding import TextSimilarityCalculator,tensor


class GraphPathToJsonConverter:
    def __init__(self):
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "1234"))
        self.sim_calculator = TextSimilarityCalculator()

    def get_full_path_details(self, bottom_up_paths):
        paths_details = []
        for path_id, _ in bottom_up_paths:
            # 假设存在一个方法get_full_paths_for_node来获取包含特定节点的所有完整路径
            all_paths = self.get_full_paths_for_node(path_id)
            for path_nodes in all_paths:
                path_ids = [node.identity for node in path_nodes]
                path_attributes = [{key: node[key] for key in node.keys()} for node in path_nodes]
                path_embeddings = [self.get_node_embedding(node) for node in path_nodes]
                paths_details.append({
                    "path_ids": path_ids,
                    "path_attributes": path_attributes,
                    "path_embeddings": path_embeddings,
                })
        return paths_details

    def get_full_paths_for_node(self, node_id):
        # 这里应实现具体的Cypher查询来获取完整路径
        # 以下是示意性的查询逻辑
        query = """
        MATCH path=(root:Topic)-[:基础链接*]->(target:Content)
        WHERE any(node in nodes(path) WHERE id(node) = $node_id)
        RETURN nodes(path) AS nodes
        """
        results = self.graph.run(query, node_id=node_id).data()
        all_paths = []
        for result in results:
            nodes_in_path = result['nodes']
            all_paths.append(nodes_in_path)
        return all_paths

    def get_node_embedding(self, node):
        # 根据节点类型获取嵌入向量
        if 'Topic' in node.labels:
            embedding_str = node.get('主题嵌入', "[]")
        elif 'SubTopic' in node.labels:
            embedding_str = node.get('路由嵌入', "[]")
        elif 'Content' in node.labels:
            embedding_str = node.get('路径嵌入', "[]")
        else:
            return []
        return eval(embedding_str)

    def convert_paths_to_json(self, bottom_up_paths, qdse):
        final_json = {}  # 存储所有主题及其子路径的结构
        processed_path_ids = set()  # 存储已处理的path_id集合

        # 遍历给定的bottom_up_paths列表
        for path_id, _ in bottom_up_paths:
            # 如果path_id已经处理过，则跳过
            if path_id in processed_path_ids:
                continue

            all_paths = self.get_full_paths_for_node(path_id)  # 获取包含特定节点的所有完整路径
            processed_path_ids.add(path_id)  # 将path_id添加到已处理集合中

            for path_nodes in all_paths:
                max_similarity = 0  # 为当前路径初始化最大相似性为0
                topic_name = ""  # 初始化主题名称
                current_level = final_json  # 初始化当前层级指向final_json

                for node in path_nodes:
                    max_similarity = 0
                    try:
                        node_embedding = self.get_node_embedding(node)  # 获取节点嵌入
                    except Exception as e:
                        continue
                    # 计算当前节点嵌入与所有kpr_embedding的相似性的最大值
                    for kpr_embedding in qdse:
                        similarity = self.sim_calculator.calculate_similarity_from_embedding(node_embedding,
                                                                                             eval(kpr_embedding))
                        max_similarity = max(max_similarity, similarity)

                    # 假设有方法从节点获取名称和类型
                    if 'Topic' in node.labels:
                        node_key = node['主题'].strip('\'')
                    elif 'SubTopic' in node.labels:
                        node_key = node['路标'].strip('\'')
                    elif 'Content' in node.labels:
                        node_key = node['事实'].strip('\'')

                    # 检查当前层级是否已存在节点键
                    if node_key not in current_level:
                        # 如果节点是事实节点，直接添加相似性值
                        if 'Content' in node.labels:
                            current_level[node_key] = max_similarity
                        else:
                            # 对于主题和子主题节点，创建新的字典来保存子节点
                            current_level[node_key] = {} if 'Topic' in node.labels else {}
                            # 更新当前层级的引用，指向新添加的节点
                            current_level = current_level[node_key]
                    else:
                        # 如果当前层级已存在节点键，更新当前层级的引用，除非是事实节点
                        if 'Content' not in node.labels:
                            current_level = current_level[node_key]

        return final_json

    # def convert_paths_to_json(self, bottom_up_paths, kpr_embedding):
    #     final_json = {}  # 存储所有主题及其子路径的结构
    #
    #     # 遍历给定的bottom_up_paths列表
    #     for path_id, _ in bottom_up_paths:
    #         all_paths = self.get_full_paths_for_node(path_id)  # 获取包含特定节点的所有完整路径
    #         for path_nodes in all_paths:
    #             max_similarity = 0  # 为当前路径初始化最大相似性为0
    #             topic_name = ""  # 初始化主题名称
    #             current_level = final_json  # 初始化当前层级指向final_json
    #
    #             for node in path_nodes:
    #                 node_embedding = self.get_node_embedding(node)  # 获取节点嵌入
    #
    #                 # 计算当前节点嵌入与kpr_embedding的相似性
    #                 similarity = self.sim_calculator.calculate_similarity_from_embedding(node_embedding, eval(kpr_embedding))
    #                 # 更新最大相似性
    #                 max_similarity = max(max_similarity, similarity)
    #                 # 假设有方法从节点获取名称和类型
    #                 if 'Topic' in node.labels:
    #                     node_key = node['主题']
    #                 elif 'SubTopic' in node.labels:
    #                     node_key = node['路标']
    #                 elif 'Content' in node.labels:
    #                     node_key = node['事实']
    #
    #                 # 检查当前层级是否已存在节点键
    #                 if node_key not in current_level:
    #                     # 如果节点是事实节点，直接添加相似性值
    #                     if 'Content' in node.labels:
    #                         current_level[node_key] = similarity
    #                     else:
    #                         # 对于主题和子主题节点，创建新的字典来保存子节点
    #                         current_level[node_key] = {} if 'Topic' in node.labels else {}
    #                         # 更新当前层级的引用，指向新添加的节点
    #                         current_level = current_level[node_key]
    #                 else:
    #                     # 如果当前层级已存在节点键，更新当前层级的引用，除非是事实节点
    #                     if 'Content' not in node.labels:
    #                         current_level = current_level[node_key]
    #
    #     return final_json