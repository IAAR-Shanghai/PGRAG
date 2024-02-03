# -*- coding: utf-8 -*-

from FlagEmbedding import FlagModel
import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '../llms/model_cache'
class TextSimilarityCalculator:
    def __init__(self):
        # 初始化FlagModel
        self.model = FlagModel('BAAI/bge-large-zh-v1.5',
                               query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                               use_fp16=True)

    def calculate_embedding(self, texts):
        # 计算文本列表的嵌入向量
        embeddings = self.model.encode(texts)
        return embeddings

    def calculate_similarity(self, embeddings1, embeddings2):
        # 计算两组嵌入向量之间的相似度
        similarity = embeddings1 @ embeddings2.T
        return similarity

def tensor(lst):
    return torch.tensor(lst)
# # 使用示例
# sim_calculator = TextSimilarityCalculator()
# sentences_1 = ["样例数据-1", "样例数据-2"]
# sentences_2 = ["样例数据-3", "样例数据-4"]
#
# embeddings_1 = sim_calculator.calculate_embedding(sentences_1)
# embeddings_2 = sim_calculator.calculate_embedding(sentences_2)
# similarity = sim_calculator.calculate_similarity(embeddings_1, embeddings_2)
# print(similarity)


