from py2neo import Graph
from PGRAG.PG_GEN.embedding import TextSimilarityCalculator, tensor


class SeedContextRecall:
    def __init__(self):
        # 连接到 Neo4j 数据库
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "1234"))
        self.sim_calculator = TextSimilarityCalculator()

    def seed_topic_recall_base_tn(self, ci_embedding, topM=2):
        query_content_per_topic = """
         MATCH (t:Topic)
         RETURN t.主题 AS topic, collect({content: t, embedding: t.主题嵌入}) AS contents
         """
        contents_per_topic = self.graph.run(query_content_per_topic).data()

        topic_similarity = []
        for topic_contents in contents_per_topic:
            topic = topic_contents['topic']
            contents = topic_contents['contents']
            best_content, best_similarity = None, -float('inf')
            for content in contents:
                similarity = self.sim_calculator.calculate_similarity_from_embedding(
                    eval(content['embedding']), eval(ci_embedding))
                if similarity > best_similarity:
                    best_content = content
                    best_similarity = similarity
            topic_similarity.append((topic, best_content, best_similarity))

        # 使用快速选择算法找到topM个最相似的项
        topM_seed_nodes = sorted(topic_similarity, key=lambda x: x[2], reverse=True)[:topM]

        # 构建字典返回
        seed_topics_similarity = {seed[0]: seed[2] for seed in topM_seed_nodes}
        # print('seed_topics_similarity:', seed_topics_similarity)
        return seed_topics_similarity

    # def seed_topic_recall_base_tn(self, ci_embedding, topM=2):
    #     # 查询每个主题下的内容节点及其嵌入
    #     query_content_per_topic = """
    #      MATCH (t:Topic)
    #      RETURN t.主题 AS topic, collect({content: t, embedding: t.主题嵌入}) AS contents
    #      """
    #     contents_per_topic = self.graph.run(query_content_per_topic).data_v1()
    #
    #     # 计算每个主题下内容节点与CI的相似度，并选择最相似的节点
    #     top_content_per_topic = []
    #     for topic_contents in contents_per_topic:
    #         topic = topic_contents['topic']
    #         contents = topic_contents['contents']
    #         best_topic = max(contents, key=lambda x: self.sim_calculator.calculate_similarity_from_embedding(
    #             eval(x['embedding']), eval(ci_embedding)))
    #         top_content_per_topic.append((topic, best_topic))

        # # 选择TopK个种子节点
        # top_content_per_topic.sort(
        #     key=lambda x: self.sim_calculator.calculate_similarity_from_embedding(eval(x[1]['embedding']), eval(ci_embedding)),
        #     reverse=True)
        # topM_seed_nodes = top_content_per_topic[:topM]
        #
        # # 提取种子节点的主题名
        # seed_topics = [seed[1]['content']['主题'] for seed in topM_seed_nodes]
        # # print('seed_topics:', seed_topics)
        # return seed_topics


    # def seed_topic_recall_base_tn(self, ci_embedding, topM=2):
    #     # ci_embedding = self.sim_calculator.calculate_embedding(CI)
    #     # 查询每个主题下的内容节点及其嵌入
    #     query_content_per_topic = """
    #      MATCH (t:Topic)
    #      RETURN t.主题 AS topic, collect(t) as contents
    #      """
    #     contents_per_topic = self.graph.run(query_content_per_topic).data_v1()
    #     # print(contents_per_topic)
    #     #
    #     # # 计算每个主题下内容节点与CI的相似度，并选择最相似的节点
    #     top_content_per_topic = []
    #     for topic_contents in contents_per_topic:
    #         topic = topic_contents['topic']
    #         contents = topic_contents['contents']
    #         # try:
    #         best_topic = max(contents, key=lambda x: self.sim_calculator.calculate_similarity_from_embedding(
    #             eval(x['主题嵌入']), eval(ci_embedding)))
    #         # print('best_topic',best_topic)
    #         # except Exception as e:
    #         #     print(e)
    #         top_content_per_topic.append((topic, best_topic))
    #     #
    #     # # 选择TopK个种子节点
    #     top_content_per_topic.sort(
    #         key=lambda x: self.sim_calculator.calculate_similarity_from_embedding(eval(x[1]['主题嵌入']),
    #                                                                               eval(ci_embedding)), reverse=True)
    #     topM_seed_nodes = top_content_per_topic[:topM]
    #     # 对于每个种子节点，查询从主题节点到内容节点的完整路径
    #     seed_topics = []
    #     for seed in topM_seed_nodes:
    #         topic_name = seed[1]['主题']
    #     #     query_path = """
    #     #      MATCH (t:Topic)
    #     #      RETURN t.主题 AS names
    #     #      """
    #     #     path_result = self.graph.run(query_path, fact=seed_fact).evaluate()
    #     #     if path_result:
    #     #         # path_str = ' '.join(filter(None, path_result))
    #         seed_topics.append(topic_name)
    #     print('seed_topics:',seed_topics)
    #     # return seed_topics

    def seed_topic_recall(self, ci_embedding, topM=2):
        # ci_embedding = self.sim_calculator.calculate_embedding(CI)
        # 查询每个主题下的内容节点及其嵌入
        query_content_per_topic = """
        MATCH (t:Topic)-[*]->(c:Content)
        RETURN t.主题 AS topic, collect(c) AS contents
        """
        contents_per_topic = self.graph.run(query_content_per_topic).data()

        # 计算每个主题下内容节点与CI的相似度，并选择最相似的节点
        top_content_per_topic = []
        for topic_contents in contents_per_topic:
            topic = topic_contents['topic']
            contents = topic_contents['contents']
            try:
                best_content = max(contents, key=lambda x: self.sim_calculator.calculate_similarity_from_embedding(eval(x['路径嵌入']), eval(ci_embedding)))
            except Exception as e:
                print(e)
            top_content_per_topic.append((topic, best_content))

        # 选择TopK个种子节点
        top_content_per_topic.sort(key=lambda x: self.sim_calculator.calculate_similarity_from_embedding(eval(x[1]['路径嵌入']), eval(ci_embedding)), reverse=True)
        topM_seed_nodes = top_content_per_topic[:topM]

        # 对于每个种子节点，查询从主题节点到内容节点的完整路径
        paths = []
        for seed in topM_seed_nodes:
            seed_fact = seed[1]['事实']
            query_path = """
            MATCH path = (t:Topic)-[*]->(c:Content {事实: $fact})
            RETURN [node IN nodes(path) | CASE WHEN node:Topic THEN node.主题 WHEN node:SubTopic THEN node.路标 WHEN node:Content THEN node.事实 ELSE null END] AS names
            """
            path_result = self.graph.run(query_path, fact=seed_fact).evaluate()
            if path_result:
                path_str = ' '.join(filter(None, path_result))
                paths.append(path_str)
        return paths

    def find_candidate_topics(self, topM_topics):
        # candidate_topics = set(path.split(' ')[0] for path in topM_topics)
        candidate_topics = topM_topics
        for topic in candidate_topics.copy():
            query_super_topic = """
            MATCH (t:Topic {主题: $topic})-[:属于]->(st:SuperTopic)<-[:属于]-(other:Topic)
            RETURN collect(other.主题) AS other_topics
            """
            result = self.graph.run(query_super_topic, topic=topic).evaluate()
            if result:
                for other_topic in result:
                    if other_topic not in candidate_topics:
                        candidate_topics.add(other_topic)
        # print(candidate_topics)
        return candidate_topics

    def find_topN_paths_per_candidate_topic(self, candidate_topics, kpr_embedding, topN=2):
        # kpr_embedding = self.sim_calculator.calculate_embedding(KPR.replace('\n', ' '))
        top_paths_with_similarity = []
        top_paths_ids = []

        for topic in candidate_topics:
            query_content_in_topic = """
            MATCH (t:Topic {主题: $topic})-[*]->(c:Content)
            RETURN c.事实 AS fact, c.路径嵌入 AS embedding
            """
            contents = self.graph.run(query_content_in_topic, topic=topic).data()
            scored_contents = []
            for content in contents:
                try:
                    embedding_content = eval(content['embedding'])
                    embedding_kpr = eval(kpr_embedding)
                    similarity_score = self.sim_calculator.calculate_similarity_from_embedding(embedding_content,
                                                                                               embedding_kpr)
                    scored_contents.append((content, similarity_score))
                except Exception as e:
                    print("Error occurred while processing embeddings:")
                    print("content['embedding'][:20]:", content['embedding'][:20])
                    print("kpr_embedding[:20]:", kpr_embedding[:20])
                    print("Error message:", str(e))

                # print('kpr_embedding:', kpr_embedding[:20])
                # print('content_embedding:', content['embedding'][:20])
                # break
            scored_contents.sort(key=lambda x: x[1], reverse=True)
            top_contents = scored_contents[:topN]

            for content, similarity in top_contents:
                seed_fact = content['fact']
                query_path = """
                MATCH path = (t:Topic {主题: $topic})-[*]->(c:Content {事实: $fact})
                RETURN path
                """
                path_result = self.graph.run(query_path, topic=topic, fact=seed_fact).evaluate()
                if path_result:
                    path_nodes = [node['主题'] if '主题' in node else node['路标'] if '路标' in node else node['事实'] for node in path_result.nodes]
                    top_paths_with_similarity.append((path_nodes, similarity))

                    path_ids = [node.identity for node in path_result.nodes]
                    top_paths_ids.append(path_ids)

        return top_paths_with_similarity, top_paths_ids
