# -*- coding: utf-8 -*-
# 原始数据的文件名
import json
import os
from PGRAG.Context_Recall.IO import JsonFileHandler
from PGRAG.Context_Recall.bottom_up_search import BottomUpSearch
from PGRAG.Context_Recall.context_prepare import GraphPathToJsonConverter
from PGRAG.Context_Recall.topic_path_router import SeedContextRecall

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    graph_uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "1234"
    eval_data_file = 'data/eval/output_data.json'
    seed_topic_path_file = 'data/context_recall/seed_topic_path.json'
    candidate_topic_path_file = 'data/context_recall/candidate_topic_path_top3.json'
    qdse_file = 'data/eval/updated_data_with_qdse.json'
    qds_file = 'data/eval/kps_data.json'
    topN_candidate_topic_paths_file = 'data/context_recall/topN_candidate_topic_path_70.json'
    topN_topic_paths_file = 'data/context_recall/topN_topic_path_70.json'
    topK_context_token_len_file = 'data/context_recall/topK_context_token_len.json'
    seed_context_recall = SeedContextRecall()
    searcher = BottomUpSearch()
    converter = GraphPathToJsonConverter()
    # # # 创建JsonFileHandler实例，指定要操作的文件路径
    file_handler = JsonFileHandler('data/output/final_context_top14.json')
    # 指定输出文件名
    # #种子主题召回
    # 读取output_data.json文件
    with open(eval_data_file, 'r', encoding='utf-8') as file:
        eval_data = json.load(file)
    # 创建一个字典，用于存储问题和对应的问题嵌入
    # 遍历output_data中的每个条目
    i=0
    questions = []
    qes = []
    for i, entry in enumerate(eval_data):
        if (i + 3) % 3 == 0:  # 单文档问题
            question_text = entry["question"]
            print(question_text)
            question_embedding = entry[question_text]["qe_bge-base-zh"]
            questions.append(question_text)
            qes.append(question_embedding)
        i = i + i
        # 将问题和问题嵌入存储在字典中
    for i, question in enumerate(questions):
        print(i)
        print('查询问题：',question)
        topM_paths = seed_context_recall.seed_topic_recall_base_tn(qes[i], topM=9)##6s/条
        print('种子路径：', topM_paths)
        with open(seed_topic_path_file, 'a', encoding='utf-8') as file:
            file.write(str(topM_paths)+'\n')
        print('----------------------------种子主题召回召回acc1')

    # ##主题游走，召回候选主题
    with open(seed_topic_path_file, 'r', encoding='utf-8') as file:
        topM_seed_topic_lines = file.readlines()
        # print(topM_seed_topic_lines[:3])
        for topM_seed_topic_line in topM_seed_topic_lines:
            # print(type(topM_seed_topic_line))
            topM_seed_topic_line = topM_seed_topic_line.replace('\"','”').replace('\'','\"').replace('\\xa0',' ')
            topM_seed_topic_with_sim = json.loads(topM_seed_topic_line)
    #         '''原主题游走，待优化
    #         # print(list(topM_seed_topic_with_sim.keys())[:1])
    #         # topM_seed_topic = set(list(topM_seed_topic_with_sim.keys())[:1])
    #         # print((topM_seed_topic))
    #         # print(type(topM_seed_topic))
    #         # candidate_topics = seed_context_recall.find_candidate_topics(topM_seed_topic)
    #         '''
            candidate_topics = set(list(topM_seed_topic_with_sim.keys())[:3])
            print("候选主题:", candidate_topics)
            with open(candidate_topic_path_file, 'a', encoding='utf-8') as file:
                file.write(str(candidate_topics)+'\n')
    print('----------------------------种子主题召回acc2')

    ##证据召回
    with open(candidate_topic_path_file, 'r', encoding='utf-8') as file:
        topM_candidate_topic_lines = file.readlines()
    print(len(topM_candidate_topic_lines))
    with open(qdse_file, 'r', encoding='utf-8') as file:
        qdse = json.load(file)
    i = 0
    for topM_candidate_topic_line in topM_candidate_topic_lines:
        print(i)
        # print(topM_candidate_topic_line[:15])
        question = qdse[3*i]['question']#单文档
        print(question)
        topM_candidate_topic_line = topM_candidate_topic_line.replace('\"','”').replace('\'','\"').replace('\\xa0',' ')
        topM_candidate_topics = set(eval(topM_candidate_topic_line))
        print(topM_candidate_topics)
        topN_bottom_up_paths = []
        for kpr_embedding in qdse[3*i][question]['qdse']:#单文档
            # print('kpr_embedding:',kpr_embedding[:20])
        #     # 对每批候选主题和kpr_embedding找出TopN个内容节点及其路径
            topN_paths, top_paths_ids = seed_context_recall.find_topN_paths_per_candidate_topic(topM_candidate_topics, kpr_embedding, topN=2)
            # 过滤出相似性大于0.85的路径及其ID
            filtered_seed_topic_paths = []
            filtered_top_paths_ids = []
            for (path, similarity), path_id in zip(topN_paths, top_paths_ids):
                if similarity > 0.70:
                    filtered_seed_topic_paths.append((path, similarity))
                    filtered_top_paths_ids.append(path_id)
            for path, similarity in filtered_seed_topic_paths:
                print("种子路径:", path, "相似性:", similarity)
                #     # 打印对应的节点ID序列
            for path_id in filtered_top_paths_ids:
                print("对应路径节点ID序列:", path_id)
            bottom_up_paths = searcher.bottom_up_search(filtered_seed_topic_paths, filtered_top_paths_ids, kpr_embedding)
            topN_bottom_up_paths = topN_bottom_up_paths + bottom_up_paths
        print('至下而上游走结果:',topN_bottom_up_paths)
        print()
        with open(topN_topic_paths_file, 'a', encoding='utf-8') as file:
            file.write(str(topN_bottom_up_paths)+'\n')
            print('文件已写入！')
        i = i + 1
    print('----------------------------证据召回acc')

