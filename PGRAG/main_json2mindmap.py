# -*- coding: utf-8 -*-
import os
import threading

from PGRAG.PG_GEN.neo4j_data_prepare import JsonToKG
from PGRAG.PG_GEN.TopicFusion import TopicFusionManager


if __name__ == '__main__':
    dm = JsonToKG()
    raw_doc_directory_path = r'data/pg_gen/mindmap_str_processed_batch0/'
    # dm.process_and_insert_data(raw_doc_directory_path)
    # dm.update_subtopic_embeddings()
    manager = TopicFusionManager()
    # manager.keyword_fusion()
    #5735/11914=0.48
    manager.similarity_fusion()
