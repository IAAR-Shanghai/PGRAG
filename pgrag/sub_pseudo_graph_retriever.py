import numpy as np
from tqdm import tqdm
from py2neo import Graph

class PG_RAG_Processor:
    def __init__(self, graph_uri, graph_auth, candidate_topic_file, 
                 matrix_templates_file, matrix_templates_with_sim_file, topK=14):
        self.graph_uri = graph_uri
        self.graph_auth = graph_auth
        self.candidate_topic_file = candidate_topic_file
        self.matrix_templates_file = matrix_templates_file
        self.matrix_templates_with_sim_file = matrix_templates_with_sim_file
        self.topK = topK
        self.graph = Graph(graph_uri, auth=graph_auth)
    
    def fetch_paths_and_embeddings(self, candidate_topics_ids):
        query = """
            MATCH path=(t:Topic)-[:基础链接*]->(f: Content)
            WHERE id(t) IN $topic_ids
            RETURN [node in nodes(path) | id(node)] AS node_ids, 
                [node in nodes(path) | coalesce(node.主题嵌入, node.路由嵌入, node.路径嵌入)] AS node_embs
        """
        paths_results = self.graph.run(query, topic_ids=candidate_topics_ids).data()
        path_ids = []
        path_embs = []
        for result in paths_results:
            path_ids.append(result['node_ids'])
            path_embs.append(result['node_embs'])
        return path_ids, path_embs


    def create_matrix_templates(self):
        with open(self.candidate_topic_file, 'r', encoding='utf-8') as file:
            candidate_topic_lines = file.readlines()
        
        for candidate_topic_line in candidate_topic_lines:
            candidate_topic_info = eval(candidate_topic_line)
            question = candidate_topic_info['question']
            candidate_topics_ids = candidate_topic_info['candidate_topic_ids']
            qdse = candidate_topic_info['qdse']
            path_ids, path_embs = self.fetch_paths_and_embeddings(candidate_topics_ids)
            
            max_len_ids = max(len(ids) for ids in path_ids)
            id_matrix = np.full((len(path_ids), max_len_ids), -1)
            emb_matrix = np.zeros((len(path_embs), max_len_ids, len(path_embs[0][0])))

            for i, ids in enumerate(path_ids):
                id_matrix[i, :len(ids)] = ids
            for i, embs in enumerate(path_embs):
                for j, emb in enumerate(embs):
                    emb_matrix[i, j, :] = emb
                    emb_matrix[i, -1, :] = emb[-1]

            matrix_templates = {
                'question': question,
                'ID Matrix': id_matrix.tolist(),
                'EMB Matrix': emb_matrix.tolist(),
                'qdse': qdse
            }

            with open(self.matrix_templates_file, 'a', encoding='utf-8') as file:
                file.write(str(matrix_templates) + '\n')

    def compute_similarity_matrices(self):
        with open(self.matrix_templates_file, 'r', encoding='utf-8') as file:
            matrix_templates_lines = file.readlines()

        for matrix_templates_line in tqdm(matrix_templates_lines, desc="Processing SM"):
            matrix_templates_info = eval(matrix_templates_line)
            question = matrix_templates_info['question']
            matrix_id_list = matrix_templates_info['ID Matrix']
            matrix_emb_list = matrix_templates_info['EMB Matrix']
            kps_emb_list = matrix_templates_info['qdse']
            
            matrix_emb = np.array(matrix_emb_list)
            kps_emb = np.array(kps_emb_list)

            num_kps = kps_emb.shape[0]
            if num_kps == 1024:
                num_kps = 1
            num_matrices, num_vectors_per_matrix, emb_dim = matrix_emb.shape
            flattened_matrix_emb = matrix_emb.reshape(-1, emb_dim)
            sims = kps_emb @ flattened_matrix_emb.T
            reshaped_sims = sims.reshape(num_kps, num_matrices, num_vectors_per_matrix)

            matrix_templates_with_sim = {
                'question': question,
                'SIM Matrix': reshaped_sims.tolist(),
                'ID Matrix': matrix_id_list,
                'qdse': kps_emb_list
            }

            with open(self.matrix_templates_with_sim_file, 'a', encoding='utf-8') as file:
                file.write(str(matrix_templates_with_sim) + '\n')

    def process_top_k_ids(self, contexts_ids_file, final_contexts_file):
        with open(self.matrix_templates_with_sim_file, 'r', encoding='utf-8') as file:
            matrix_templates_with_sim_lines = file.readlines()

        processor = MatrixProcessor()
        for matrix_templates_with_sim_line in tqdm(matrix_templates_with_sim_lines, desc="Processing max id"):
            matrix_templates_with_sim_info = eval(matrix_templates_with_sim_line)
            question = matrix_templates_with_sim_info['question']
            matrix_sim = np.array(matrix_templates_with_sim_info['SIM Matrix'])
            matrix_id = np.array(matrix_templates_with_sim_info['ID Matrix'])

            final_matrix = np.zeros_like(matrix_id, dtype=np.float64)
            top_values = []
            top_indices = []

            for matrix in matrix_sim:
                last_non_zeros = []

                for row_index, row in enumerate(matrix):
                    for col_index in range(len(row) - 1, -1, -1):
                        if row[col_index] != 0:
                            last_non_zeros.append((row[col_index], row_index, col_index))
                            break

                last_non_zeros_sorted = sorted(last_non_zeros, key=lambda x: x[0], reverse=True)[:self.topK]
                top_values.append([val[0] for val in last_non_zeros_sorted])
                top_indices.append([(val[1], val[2]) for val in last_non_zeros_sorted])

                control_matrices, pathway_matrices = processor.create_control_and_pathway_matrices(matrix, matrix_id, top_values, top_indices)
                temp_result_matrix = processor.color_matrices(control_matrices, pathway_matrices)
                final_matrix += temp_result_matrix
            top_k_ids = processor.find_top_k_ids(final_matrix, matrix_id, self.topK)
            top_k_contexts = self.convert_paths_to_json(top_k_ids)
            contexts_ids = {
                'question': question,
                'top_k_ids': top_k_ids
            }
            with open(contexts_ids_file, 'a', encoding='utf-8') as file:
                file.write(str(contexts_ids) + '\n')

            final_contexts = {
                'question': question,
                'top_k_contexts': top_k_contexts
            }
            with open(final_contexts_file, 'a', encoding='utf-8') as file:
                file.write(str(final_contexts) + '\n')

    def convert_paths_to_json(self, top_k_ids):
        query = '''
            MATCH (f)
            WHERE id(f) IN $id_list
            OPTIONAL MATCH path=(t:Topic)-[:基础链接*]->(f: Content)
            WITH COLLECT(nodes(path)) AS all_nodes
            RETURN DISTINCT all_nodes
        '''
        result = self.graph.run(query, id_list=top_k_ids).data()
        all_paths = list(result[0]['all_nodes'])
        # print('all_paths:', all_paths)
        final_json = {}
        for path_nodes in all_paths:
            current_level = final_json  # 初始化当前层级指向final_json
            for node in path_nodes:
                # 假设有方法从节点获取名称和类型
                if 'Topic' in node.labels:
                    node_key = node['主题'].strip('\'').strip('。')
                    node_embedding = node['主题嵌入']  # 获取节点嵌入
                elif 'SubTopic' in node.labels:
                    node_key = node['路标'].strip('\'')
                    node_embedding = node['路由嵌入']  # 获取节点嵌入
                elif 'Content' in node.labels:
                    node_key = node['事实'].strip('\'')
                    node_embedding = node['路径嵌入']  # 获取节点嵌入
                # 检查当前层级是否已存在节点键
                if node_key not in current_level:
                    # 如果节点是事实节点，直接添加
                    if 'Content' in node.labels:
                        current_level[node_key] = {} 
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

class MatrixProcessor:
    def create_control_and_pathway_matrices(self, matrix_sim, matrix_id, top_values, top_indices, support_threshold=0.03, continue_threshold=0.05):
        control_matrices = []
        pathway_matrices = []

        for value, (row_idx, col_idx) in zip(top_values[-1], top_indices[-1]):
            control_matrix = np.zeros_like(matrix_sim)
            pathway_matrix = np.zeros_like(matrix_id)

            f = col_idx
            while f >= 0:
                diff = abs(matrix_sim[row_idx, f] - value)
                if diff <= support_threshold:
                    control_matrix[row_idx, f] = 1 * matrix_sim[row_idx, f]
                elif support_threshold < diff <= continue_threshold:
                    control_matrix[row_idx, f] = 0.5 * matrix_sim[row_idx, f]
                else:
                    control_matrix[row_idx, f] = 0
                    break
                f -= 1

            col_f = f + 1
            sub_root = matrix_id[row_idx, col_f]

            for i in range(matrix_sim.shape[0]):
                if i == row_idx:
                    continue
                for j in range(col_f, matrix_sim.shape[1]):
                    diff = abs(matrix_sim[i, j] - value)
                    if diff > continue_threshold:
                        break
                    elif diff <= support_threshold:
                        control_matrix[i, j] = 1 * matrix_sim[i, j]
                    elif support_threshold < diff <= continue_threshold:
                        control_matrix[i, j] = 0.5 * matrix_sim[i, j]

            control_matrices.append(control_matrix)

            for j in range(col_f, matrix_id.shape[1]):
                if matrix_id[row_idx, j] == -1:
                    break
                pathway_matrix[row_idx, j] = 1

            for direction in [-1, 1]:
                i = row_idx
                while 0 <= i + direction < matrix_id.shape[0]:
                    i += direction
                    if matrix_id[i, col_f] != sub_root:
                        break
                    for j in range(col_f, matrix_id.shape[1]):
                        if matrix_id[i, j] == -1:
                            break
                        pathway_matrix[i, j] = 1

            pathway_matrices.append(pathway_matrix)

        return control_matrices, pathway_matrices

    def color_matrices(self, control_matrices, pathway_matrices):
        result_matrix = np.zeros_like(control_matrices[0])

        for control_matrix, pathway_matrix in zip(control_matrices, pathway_matrices):
            result_matrix += np.multiply(control_matrix, pathway_matrix)

        return result_matrix

    def find_top_k_ids(self, final_matrix, matrix_id, topK):
        row_sums = np.sum(final_matrix, axis=1)
        top_k_row_indices = np.argsort(row_sums)[-topK:][::-1]

        top_k_ids = []
        for row_idx in top_k_row_indices:
            last_id = -1
            for col_idx in range(matrix_id.shape[1]):
                id_val = matrix_id[row_idx, col_idx]
                if id_val == -1:
                    break
                last_id = id_val
            if last_id != -1:
                top_k_ids.append(int(last_id))

        return top_k_ids
