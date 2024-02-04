# -*- coding: utf-8 -*-
import os
import torch
from transformers import BertTokenizer, BertModel
from FlagEmbedding import FlagModel
os.environ['TRANSFORMERS_CACHE'] = '../llms/BertModelCache'
class TextSimilarityCalculator:
    def __init__(self):
        # 初始化模型和分词器

        self.tokenizer = BertTokenizer.from_pretrained('BAAI/bge-base-zh')
        self.model = BertModel.from_pretrained('BAAI/bge-base-zh')

    def calculate_embedding(self, text):
        # 将文本转换为嵌入向量
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 取[CLS]标记的嵌入作为整个句子的嵌入
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding

    def calculate_similarity_from_embedding(self, embedding1, embedding2):
        # 计算两个嵌入向量之间的相似度
        cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return cos_sim.item()

def tensor(lst):
    return torch.tensor(lst)
# # # 使用示例
# sim_calculator = TextSimilarityCalculator()
# text1 = "示例文本1"
# text2 = "示例文本2"
#
# embedding1 = sim_calculator.calculate_embedding(text1)
# embedding2 = sim_calculator.calculate_embedding(text2)
# sim_calculator = TextSimilarityCalculator()
# similarity = sim_calculator.calculate_similarity_from_embedding(embedding1, embedding2)
#
# print("相似度:", similarity)
