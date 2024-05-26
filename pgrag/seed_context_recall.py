import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
def tensor(lst):
    return torch.tensor(lst)

class SeedContextRecall:
    def __init__(self, graph_uri, graph_auth, emb_model_name, 
                 eval_data_with_qe_and_qdse_file, seed_topic_file, candidate_topic_file,
                 recall_top_m, walk_top_m, num_threads, top_k):
        self.graph = Graph(graph_uri, auth=graph_auth)
        self.emb_model = SentenceTransformer(emb_model_name)
        self.eval_data_with_qe_and_qdse_file = eval_data_with_qe_and_qdse_file
        self.seed_topic_file = seed_topic_file
        self.candidate_topic_file = candidate_topic_file
        self.recall_top_m = recall_top_m
        self.walk_top_m = walk_top_m
        self.num_threads = num_threads
        self.top_k = top_k
        print('初始化成功！')

    def seed_topic_recall_base_tn(self, ci_embedding):
        query_seed_topic = """
            CALL db.index.vector.queryNodes('topic-embeddings', $M, $emb) 
            YIELD node AS similarTopic, score
            MATCH (similarTopic)
            RETURN ID(similarTopic) AS topic_id, score
        """
        similar_topics_with_score = self.graph.run(query_seed_topic, M=self.recall_top_m, emb=ci_embedding).data()
        seed_topics_with_score = {item['topic_id']: item['score'] for item in similar_topics_with_score}
        return seed_topics_with_score

    def topic_walking(self, seed_topic_ids, qe):
        candidate_topics_ids = []
        candidate_topics_names = []
        qe_emb = np.array(qe) 

        query_seed_topics = """
            MATCH (t:Topic) WHERE id(t) IN $seed_topic_ids
            RETURN collect(id(t)) AS seed_ids, collect(t.主题) AS seed_names
        """
        seed_topics = self.graph.run(query_seed_topics, seed_topic_ids=seed_topic_ids).data()
        if seed_topics:
            candidate_topics_ids.extend(seed_topics[0]['seed_ids'])
            candidate_topics_names.extend(seed_topics[0]['seed_names'])

        all_topic_ids = []
        all_topic_embs = []
        all_topic_names = []
        for seed_topic_id in seed_topic_ids:
            query_super_topic = """
                MATCH (t:Topic)-[:相似链接]->(st:SuperTopic)<-[:相似链接]-(other:Topic)
                WHERE id(t) = $seed_topic_id
                RETURN collect(id(other)) AS other_topic_ids, 
                    collect(other.主题嵌入) AS other_topic_embs,
                    collect(other.主题) AS other_topic_names
            """
            connected_topics = self.graph.run(query_super_topic, seed_topic_id=seed_topic_id).data()
            if connected_topics:
                other_topic_ids = connected_topics[0]['other_topic_ids']
                other_topic_embs = [np.array(emb) for emb in connected_topics[0]['other_topic_embs']]
                other_topic_names = connected_topics[0]['other_topic_names']
                all_topic_ids.extend(other_topic_ids)
                all_topic_embs.extend(other_topic_embs)
                all_topic_names.extend(other_topic_names)

        if all_topic_embs:
            all_topic_embs_matrix = np.vstack(all_topic_embs)  
            sims = qe_emb @ all_topic_embs_matrix.T 
            top_indices = np.argsort(-sims) 

            added_ids = set(candidate_topics_ids)  
            for index in top_indices:
                if len(added_ids) >= self.walk_top_m:
                    break
                topic_id = all_topic_ids[index]
                if topic_id not in added_ids:
                    added_ids.add(topic_id)
                    candidate_topics_ids.append(topic_id)
                    candidate_topics_names.append(all_topic_names[index])

        return candidate_topics_ids, candidate_topics_names

    def find_topN_paths_per_candidate_topic(self, candidate_topics, kpr_embedding):
        query_content_and_path = '''
            CALL db.index.vector.queryNodes('fact-embeddings', 10000, $embedding)
            YIELD node AS similarContent, score
            UNWIND $topics AS topic
            MATCH path = (t:Topic)-[*]->(similarContent)
            WHERE t.主题 = topic
            WITH path, score, t
            ORDER BY score DESC
            LIMIT $topN
            UNWIND nodes(path) AS p
            WITH p, 
                score, 
                CASE WHEN p:Topic THEN p.主题 
                    WHEN p:SubTopic THEN p.路标 
                    WHEN p:Content THEN p.事实 
                    ELSE null END AS pathAttribute
            WHERE pathAttribute IS NOT NULL
            RETURN collect(pathAttribute) AS pathAttributes, score
        '''
        results = self.graph.run(query_content_and_path, topics=candidate_topics, topN=self.top_k, embedding=kpr_embedding).data()
        return results

    def seed_topic_recall(self, question_info):
        question, question_embedding, qdse = question_info
        seed_topic_paths = {}
        print('查询问题：', question)
        topM_paths = self.seed_topic_recall_base_tn(question_embedding)
        seed_topic_paths = {
            'question': question,
            'qe': question_embedding,
            'topM_topic_paths': str(topM_paths),
            'qdse': qdse
        }
        with open(self.seed_topic_file, 'a', encoding='utf-8') as file:
            file.write(str(seed_topic_paths) + '\n')

    def execute(self):
        with open(self.eval_data_with_qe_and_qdse_file, 'r', encoding='utf-8') as file:
            eval_data = json.load(file)

        questions = []
        qes = []
        qdses = []
        for entry in eval_data:
            question_text = entry["question"]
            question_embedding = entry[question_text]["qe_bge-base-zh"]
            qdse = entry[question_text]['qdse_bge-base-zh']
            questions.append(question_text)
            qes.append(question_embedding)
            qdses.append(qdse)
        
        question_infos = [(question, qe, qdse) for question, qe, qdse in zip(questions, qes, qdses)]
        print(len(question_infos))

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(executor.map(self.seed_topic_recall, question_infos), total=len(question_infos)))

        with open(self.seed_topic_file, 'r', encoding='utf-8') as file:
            topM_seed_topic_lines = file.readlines()
        for topM_seed_topic_line in topM_seed_topic_lines:
            topM_seed_topics_info = eval(topM_seed_topic_line)
            question = topM_seed_topics_info['question']
            qe = topM_seed_topics_info['qe']
            topM_seed_topic_with_sim = eval(topM_seed_topics_info['topM_topic_paths'])
            qdse = topM_seed_topics_info['qdse']
            print(question)
            seed_topic_ids = list(topM_seed_topic_with_sim.keys())
            candidate_topic_ids, candidate_topic_names = self.topic_walking(seed_topic_ids, qe)
            print("候选主题:", candidate_topic_ids, candidate_topic_names)

            candidate_topics = {
                'question': question,
                'candidate_topic_ids': candidate_topic_ids,
                'candidate_topic_names': candidate_topic_names,
                'qdse': qdse,
                'qe': qe
            }
            with open(self.candidate_topic_file, 'a', encoding='utf-8') as file:
                file.write(str(candidate_topics) + '\n')


