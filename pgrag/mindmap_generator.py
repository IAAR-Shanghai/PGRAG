# -*- coding: utf-8 -*-
import os
from llms.remote import GPT_transit
import concurrent.futures
from tqdm import tqdm

class MindmapGeneration:
    def __init__(self, model_name, num_threads, raw_news_files_dir, title_files_dir, fcis_files_dir, mindmaps_str_files_dir, mindmaps_json_files_dir):
        self.gpt = GPT_transit(model_name=model_name, report=True)
        self.num_threads = num_threads
        self.raw_news_files_dir = raw_news_files_dir
        self.title_files_dir = title_files_dir
        self.fcis_files_dir = fcis_files_dir
        self.mindmaps_str_files_dir = mindmaps_str_files_dir
        self.mindmaps_json_files_dir = mindmaps_json_files_dir

    def extract_mt(self):
        raw_news_files_to_process = [os.path.join(self.raw_news_files_dir, file) for file in os.listdir(self.raw_news_files_dir) if file.endswith('.txt')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(executor.map(self.gpt.process_et, raw_news_files_to_process, [self.gpt] * len(raw_news_files_to_process), [self.title_files_dir] * len(raw_news_files_to_process)), total=len(raw_news_files_to_process)))

    def extract_fcis(self):
        raw_news_files_to_process = [os.path.join(self.raw_news_files_dir, file) for file in os.listdir(self.raw_news_files_dir) if file.endswith('.txt')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(executor.map(self.gpt.process_efvi, raw_news_files_to_process, [self.gpt] * len(raw_news_files_to_process), [self.fcis_files_dir] * len(raw_news_files_to_process)), total=len(raw_news_files_to_process)))

    def generate_mindmaps_str(self):
        title_files_to_process = [os.path.join(self.title_files_dir, file) for file in os.listdir(self.title_files_dir) if file.endswith('.txt')]
        fcis_files_to_process = [os.path.join(self.fcis_files_dir, file) for file in os.listdir(self.fcis_files_dir) if file.endswith('.txt')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(executor.map(self.gpt.process_gm, title_files_to_process, fcis_files_to_process, [self.gpt] * len(fcis_files_to_process), [self.mindmaps_str_files_dir] * len(fcis_files_to_process)), total=len(fcis_files_to_process)))

    def generate_mindmaps_json(self):
        mindmap_str_files_to_process = [os.path.join(self.mindmaps_str_files_dir, file) for file in os.listdir(self.mindmaps_str_files_dir) if file.endswith('.txt')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(executor.map(self.gpt.mindmap_str_to_json, mindmap_str_files_to_process, [self.mindmaps_json_files_dir] * len(mindmap_str_files_to_process)), total=len(mindmap_str_files_to_process)))

    def execute(self):
        self.extract_mt()
        self.extract_fcis()
        self.generate_mindmaps_str()
        self.generate_mindmaps_json()

