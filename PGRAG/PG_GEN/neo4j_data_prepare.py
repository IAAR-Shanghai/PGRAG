# -*- coding: utf-8 -*-
import os
import json
from py2neo import Graph
from PGRAG.PG_GEN.embedding import TextSimilarityCalculator
from concurrent.futures import ThreadPoolExecutor

class JsonToKG:
    def __init__(self):
        """
        初始化连接到Neo4j数据库和文本相似度计算器。
        """
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "iaar1234"))
        self.sim_calculator = TextSimilarityCalculator()

    def recursive_json_iterator(self, json_data, path='', topic_paths=None):
        """
        递归解析JSON数据。
        输入: json_data - JSON格式的数据
        输出: topic_paths - 包含所有路径的列表
        """
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

    def load_json_files(self, raw_doc_directory_path):
        """
        从指定目录加载JSON文件。
        输入: raw_doc_directory_path - JSON文件所在的目录路径
        输出: all_json_contents - 包含所有JSON内容的字典
        """
        json_files = [file for file in os.listdir(raw_doc_directory_path) if file.endswith('.json')]
        all_json_contents = {}

        for json_file in json_files:
            file_path = os.path.join(raw_doc_directory_path, json_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                try:
                    # 尝试解析JSON数据
                    parsed_line = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f'JSON解析错误在文件{json_file}', e)
                # parsed_line = ast.literal_eval(file.readline())
                all_json_contents[json_file] = parsed_line

        return all_json_contents

    def process_and_insert_single_data(self, json_data):
        emb_t = self.sim_calculator.calculate_embedding(str(json_data))

        document_properties = {
            "主题": json_data.get("主题", None),
            "主题嵌入": str(emb_t),
            "关键词": json_data.get("元信息", {}).get("关键词", None)
        }

        label = "Topic"
        primary_key = "主题"
        topic_paths = self.recursive_json_iterator(json_data.get("详情", {}))

        for key, value in document_properties.items():
            if value:
                update_query = f"MERGE (d:{label} {{{primary_key}: $primary_key_value}}) SET d.{key} = $value RETURN d"
                updated_node = self.graph.run(update_query,
                                              parameters={"primary_key_value": document_properties[primary_key],
                                                          "value": value}).evaluate()

        topic_name = updated_node.get('主题')
        print(f'名为“{topic_name}”的主题节点插入成功！')
        # ...
        for topic_path in topic_paths:
            split_parts = topic_path.strip().strip("'").split("' '")
            parts = [part.strip() for part in split_parts if part.strip()]
            print('-------------------------------')
            print('待插入主题路径：', parts)

            # 插入子主题
            for j, sub_topic_type in enumerate(parts[:-1]):
                if j == 0:
                    # 新的第一个子主题
                    create_TST_query = "MATCH (d:Topic {主题: $topic_name}) MERGE (d)-[r:基础链接]->(st:SubTopic {路标: $sub_topic_type}) RETURN d, st"
                    TST_result = self.graph.run(create_TST_query,
                                                parameters={"topic_name": topic_name,
                                                            "sub_topic_type": sub_topic_type}).data()
                    print('主题到子主题插入成功！结果显示：', TST_result)
                else:
                    # 新的中间子主题
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
                    # 叶子主题
                    emb_fp = self.sim_calculator.calculate_embedding(parts[0] + ' '.join(parts[1:]))
                    fact = parts[j + 1]
                    match_query_parts = [f"-[r{k}:基础链接]->(pst{k}:SubTopic {{路标: $part{k}}})" for k, part in
                                         enumerate(parts[:j + 1], 1)]
                    match_query = f"MATCH (d:Topic {{主题: $topic_name}}) " + ''.join(match_query_parts)
                    merge_query = f" MERGE (c:Content {{事实: $fact, 路径嵌入: $fp}}) WITH pst{j + 1}, c MERGE (pst{j + 1})-[r:基础链接]->(c) RETURN c"

                    create_STC_query = match_query + merge_query

                    params = {"topic_name": topic_name, "fact": fact, "fp": str(emb_fp)}

                    for k, part in enumerate(parts[:j + 1], 1):
                        params[f"part{k}"] = part

                    STC_result = self.graph.run(create_STC_query, parameters=params).data()
                    print("**完整的路径插入成功！结果显示：", STC_result)

    def process_and_insert_data(self, raw_doc_directory_path):
        """
        处理并插入数据到Neo4j数据库。
        输入: raw_doc_directory_path - JSON文件所在的目录路径
        输出: 无（数据被插入到数据库）
        """
        result = self.load_json_files(raw_doc_directory_path)

        num_pool = int(min(len(result.keys()), 20))
        with ThreadPoolExecutor(max_workers=num_pool) as executor:
            # 使用 map 方法来并行执行函数
            res = executor.map(self.process_and_insert_single_data, list(result.values())[:3])

    def update_single_subtopic_embedding(self, subtopic_path):
        """
        更新单个子主题节点的路由嵌入。
        """
        subtopic = subtopic_path['st']
        path_names = subtopic_path['path_names']

        path_str = ' '.join(path_names)
        embedding = self.sim_calculator.calculate_embedding(path_str)
        print(path_str)
        query_update_embedding = """
        MATCH path = (t:Topic)-[:基础链接*]->(st:SubTopic {路标: $subtopic_label})
        WHERE [node IN nodes(path) | CASE WHEN node:Topic THEN node.主题 WHEN node:SubTopic THEN node.路标 END] = $path_names
        SET st.路由嵌入 = $embedding 
        RETURN count(st) as updated
        """
        result = self.graph.run(query_update_embedding, subtopic_label=subtopic['路标'], path_names=path_names, embedding=str(embedding)).evaluate()

        if result == 0:
            print(f"Failed to update routing embedding for path: {path_str}")
            
    def update_subtopic_embeddings(self):
        query_subtopics_paths = """
        MATCH path = (t:Topic)-[:基础链接*]->(st:SubTopic)
        WHERE not EXISTS(st.路由嵌入) 
        RETURN st, [node IN nodes(path) | CASE WHEN node:Topic THEN node.主题 WHEN node:SubTopic THEN node.路标 END] AS path_names
        """
        subtopics_paths = self.graph.run(query_subtopics_paths).data()
        # 使用 ThreadPoolExecutor 来限制线程数量
        with ThreadPoolExecutor(max_workers=14) as executor:  # 你可以根据需要调整 max_workers 的值
            executor.map(self.update_single_subtopic_embedding, subtopics_paths)