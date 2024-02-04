# -*- coding: utf-8 -*-
import os
import threading

from PGRAG.PG_GEN.neo4j_data_prepare import JsonToKG
from PGRAG.PG_GEN.TopicFusion import TopicFusionManager


if __name__ == '__main__':
    dm = JsonToKG()
    raw_doc_directory_path = r'data/pg_gen/mindmap_str_processed_batch0/'
    #Step 1：单篇思维导图入库（需改为多线程或者多进程，同时处理多篇思维导图）
    dm.process_and_insert_data(raw_doc_directory_path, top_n=10)
    #Step 2：Step 1结束后，计算节点嵌入（现为多线程，同时对一批节点进行嵌入计算，但本地依旧很慢，需要在服务器上加速）
    dm.update_subtopic_embeddings()
    manager = TopicFusionManager()
    #Step 3：Step 2结束后，通过抽取的关键词对主题节点进行初步融合（速度可以接受，先不优化）
    manager.keyword_fusion()
    #5735/11914=0.48
    #Step 4：Step 3结束后，通过相似性对主题节点进行二次融合（有些慢，需改为并行）
    manager.similarity_fusion()
    #4422/11914=0.37
