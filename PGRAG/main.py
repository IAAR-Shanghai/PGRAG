# -*- coding: utf-8 -*-
from llms.remote import GPT_transit
import os
from PGRAG.pipline.neo4j_data_prepare import JsonToKG
from PGRAG.pipline.TopicFusion import TopicFusionManager

if __name__ == '__main__':
    dm = JsonToKG()
    raw_doc_directory_path = r'../data/bm/mindmap'
    dm.process_and_insert_data(raw_doc_directory_path)
    dm.update_subtopic_embeddings()
    manager = TopicFusionManager()
    manager.keyword_fusion()
    manager.similarity_fusion()
