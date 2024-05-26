# -*- coding: utf-8 -*-
import json
from py2neo import Graph
import os
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class Neo4jDataInserter:
    def __init__(self, graph_uri, graph_auth, emb_model_name, max_workers=20):
        self.graph = Graph(graph_uri, auth=graph_auth)
        self.emb_model = SentenceTransformer(emb_model_name)
        self.max_workers = max_workers
        print('初始化成功！')

    def recursive_json_iterator(self, json_data, path='', topic_paths=None):
        if topic_paths is None:
            topic_paths = []

        if isinstance(json_data, dict):
            for key, value in json_data.items():
                current_path = f"{path} '{key}'".lstrip()
                self.recursive_json_iterator(value, current_path, topic_paths)
        elif isinstance(json_data, list):
            topic_path = f"{path} '{' '.join(map(str, json_data))}'".lstrip()
            topic_paths.append(topic_path)
        else:
            topic_path = f"{path} '{json_data}'".lstrip()
            topic_paths.append(topic_path)
        return topic_paths

    def load_json_files(self, mindmap_json_dir, raw_doc_dir):
        file_names = [file for file in os.listdir(mindmap_json_dir) if file.endswith('.json')]
        all_json_contents = {}
        all_news = []
        print('总文件数:', len(file_names))

        for file_name in file_names:
            mindmap_json_file_path = os.path.join(mindmap_json_dir, file_name)
            base_name, _ = os.path.splitext(file_name)
            raw_data_path = os.path.join(raw_doc_dir, base_name + ".txt")
            with open(raw_data_path, 'r', encoding='utf-8') as file:
                news = file.read()
            all_news.append(news)
            with open(mindmap_json_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                try:
                    parsed_line = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f'JSON解析错误在文件{file_name}', e)
                all_json_contents[file_name] = parsed_line

        return all_json_contents, all_news

    def process_and_insert_single_data(self, raw_data, json_data):
        title = list(json_data.keys())[0]
        emb_t = self.emb_model.encode(title, normalize_embeddings=True)
        document_properties = {
            "主题": title,
            "主题嵌入": emb_t.tolist()
        }
        label = "Topic"
        primary_key = "主题"
        topic_paths = self.recursive_json_iterator(json_data.get(title, {}))

        for key, value in document_properties.items():
            if value:
                update_query = f"MERGE (d:{label} {{{primary_key}: $primary_key_value}}) SET d.{key} = $value RETURN d"
                updated_node = self.graph.run(update_query,
                                              parameters={"primary_key_value": document_properties[primary_key],
                                                          "value": value}).evaluate()

        topic_name = updated_node.get('主题')
        print(f'名为“{topic_name}”的主题节点插入成功！')
        for topic_path in topic_paths:
            split_parts = topic_path.strip().strip("'").split("' '")
            parts = [part.strip() for part in split_parts if part.strip()]
            print('-------------------------------')
            print('待插入主题路径：', parts)

            for j, sub_topic_type in enumerate(parts[:-1]):
                if j == 0:
                    create_TST_query = "MATCH (d:Topic {主题: $topic_name}) MERGE (d)-[r:基础链接]->(st:SubTopic {路标: $sub_topic_type}) RETURN d, st"
                    TST_result = self.graph.run(create_TST_query,
                                                parameters={"topic_name": topic_name,
                                                            "sub_topic_type": sub_topic_type}).data()
                    print('主题到子主题插入成功！结果显示：', TST_result)
                else:
                    match_query_parts = [f"-[r{k}:基础链接]->(pst{k}:SubTopic {{路标: $part{k}}})" for k, part in
                                         enumerate(parts[:j], 1)]
                    match_query = "MATCH (d:Topic {主题: $topic_name}) " + ''.join(match_query_parts)
                    merge_query = f" WITH pst{j} MERGE (pst{j})-[r:基础链接]->(st:SubTopic {{路标: $sub_topic_type}}) RETURN st"
                    create_STST_query = match_query + merge_query

                    params = {"topic_name": topic_name, "sub_topic_type": sub_topic_type}
                    for k, part in enumerate(parts[:j], 1):
                        params[f"part{k}"] = part

                    STST_result = self.graph.run(create_STST_query, parameters=params).data()
                    print('插入的主题路径：', STST_result)
                if j == len(parts) - 2:
                    emb_fp = self.emb_model.encode(parts[0] + ' '.join(parts[1:]), normalize_embeddings=True)
                    fact = parts[j + 1]
                    match_query_parts = [f"-[r{k}:基础链接]->(pst{k}:SubTopic {{路标: $part{k}}})" for k, part in
                                         enumerate(parts[:j + 1], 1)]
                    match_query = f"MATCH (d:Topic {{主题: $topic_name}}) " + ''.join(match_query_parts)
                    merge_query = f" MERGE (c:Content {{事实: $fact, 路径嵌入: $fp}}) WITH pst{j + 1}, c MERGE (pst{j + 1})-[r:基础链接]->(c) RETURN c"

                    create_STC_query = match_query + merge_query

                    params = {"topic_name": topic_name, "fact": fact, "fp": emb_fp.tolist()}

                    for k, part in enumerate(parts[:j + 1], 1):
                        params[f"part{k}"] = part

                    STC_result = self.graph.run(create_STC_query, parameters=params).data()
                    print("**完整的路径插入成功！结果显示：", STC_result)

    def chunked_data(self, data, size):
        """将列表分成指定大小的块。"""
        for i in range(0, len(data), size):
            yield data[i:i + size]

    def process_and_insert_data(self, raw_doc_dir, mindmap_json_dir, start_batch=0, batch_size=20):
        result, all_news = self.load_json_files(mindmap_json_dir, raw_doc_dir)
        all_batches = zip(self.chunked_data(all_news, batch_size), self.chunked_data(list(result.values()), batch_size))
        for i, (batch_news, batch_json_data) in enumerate(all_batches):
            if i < start_batch:  # 跳过已处理的批次
                continue
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(executor.map(self.process_and_insert_single_data, batch_news, batch_json_data))
            print(f"已完成第{i + 1}批处理，大小：{batch_size}")

    def update_single_subtopic_embedding(self, subtopic_path):
        """
        更新单个子主题节点的路由嵌入。
        """
        subtopic = subtopic_path['st']
        path_names = subtopic_path['path_names']

        path_str = ' '.join(path_names)
        emb_p = self.emb_model.encode(path_str, normalize_embeddings=True)
        print(path_str)
        query_update_embedding = """
        MATCH path = (t:Topic)-[:基础链接*]->(st:SubTopic {路标: $subtopic_label})
        WHERE [node IN nodes(path) | CASE WHEN node:Topic THEN node.主题 WHEN node:SubTopic THEN node.路标 END] = $path_names
        SET st.路由嵌入 = $embedding 
        RETURN count(st) as updated
        """
        result = self.graph.run(query_update_embedding, subtopic_label=subtopic['路标'], path_names=path_names,
                                embedding=emb_p.tolist()).evaluate()

        if result == 0:
            print(f"Failed to update routing embedding for path: {path_str}")

    def update_subtopic_embeddings(self):
        query_subtopics_paths = """
        MATCH path = (t:Topic)-[:基础链接*]->(st:SubTopic)
        RETURN st, [node IN nodes(path) | CASE WHEN node:Topic THEN node.主题 WHEN node:SubTopic THEN node.路标 END] AS path_names
        """
        subtopics_paths = self.graph.run(query_subtopics_paths).data()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self.update_single_subtopic_embedding, subtopics_paths)

    def execute(self, raw_doc_dir, mindmap_json_dir, start_batch=0, batch_size=20):
        self.process_and_insert_data(raw_doc_dir, mindmap_json_dir, start_batch, batch_size)
        self.update_subtopic_embeddings()

class TopicAndContentFusion:
    def __init__(self, graph_uri, graph_auth, emb_model_name, topic_threshold=0.92, content_threshold=0.98):
        self.graph = Graph(graph_uri, auth=graph_auth)
        self.emb_model = SentenceTransformer(emb_model_name)
        self.topic_threshold = topic_threshold
        self.content_threshold = content_threshold
        self.topic_clusters = []  
        self.content_clusters = []  
        self.processed_topic_nodes = set()  
        self.processed_content_nodes = set() 

    def get_topic_node_ids(self):
        query = """
            MATCH (n:Topic)
            RETURN ID(n) AS topicNodeID
        """
        return self.graph.run(query).data()

    def get_content_node_ids(self):
        query = """
            MATCH (n:Content)
            RETURN ID(n) AS contentNodeID
        """
        return self.graph.run(query).data()

    def cluster_nodes(self, node_type, threshold):
        if node_type == 'Topic':
            node_id_list = self.get_topic_node_ids()
            processed_nodes = self.processed_topic_nodes
            clusters = self.topic_clusters
            embedding_field = 'n.主题嵌入'
            embedding_index = 'topic-embeddings'
            super_node_label = 'SuperTopic'
        elif node_type == 'Content':
            node_id_list = self.get_content_node_ids()
            processed_nodes = self.processed_content_nodes
            clusters = self.content_clusters
            embedding_field = 'n.路径嵌入'
            embedding_index = 'fact-embeddings'
            super_node_label = 'SuperContent'
        else:
            raise ValueError("Unsupported node type. Use 'Topic' or 'Content'.")

        print(f"{node_type} node count: {len(node_id_list)}")
        print(node_id_list)

        for node_dict in tqdm(node_id_list, desc=f"Clustering {node_type}s"):
            node_id = node_dict[f'{node_type.lower()}NodeID']
            if node_id in processed_nodes:
                continue

            query = f"""
            MATCH (n:{node_type}) WHERE ID(n) = {node_id}
            WITH {embedding_field} AS embedding
            CALL db.index.vector.queryNodes('{embedding_index}', {min(len(node_id_list), 100)}, embedding) YIELD node, score
            WHERE score > {threshold}
            RETURN ID(node) AS similarNodeId
            """
            similar_nodes = self.graph.run(query).data()
            similar_node_ids = {item['similarNodeId'] for item in similar_nodes}
            
            new_cluster = list(similar_node_ids)
            clusters.append(new_cluster)
            processed_nodes.update(similar_node_ids)
            processed_nodes.add(node_id)

        print(f"Total clusters for {node_type}: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            print(f"{node_type} Cluster {i+1}: {cluster}")

            super_node_id = f"{super_node_label}_{i}"
            self.graph.run(f"MERGE (:{super_node_label} {{id: $id}})", id=super_node_id)

            matches = " ".join(f"MATCH (n{idx}) WHERE ID(n{idx}) = {node_id}" for idx, node_id in enumerate(cluster))
            creates = " ".join(f"CREATE (n{idx})-[:相似链接]->(st)" for idx in range(len(cluster)))

            query = f"""
            {matches}
            MATCH (st:{super_node_label} {{id: '{super_node_id}'}})
            {creates}
            RETURN NULL
            """
            self.graph.run(query)

    def fuse_topics_and_contents(self):
        self.cluster_nodes('Topic', self.topic_threshold)
        self.cluster_nodes('Content', self.content_threshold)
        print("Fusion of topics and contents completed.")

