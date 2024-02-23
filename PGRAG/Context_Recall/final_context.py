# -*- coding: utf-8 -*-
import copy
from operator import itemgetter
import tiktoken
import copy
from operator import itemgetter

class TopKJsonExtractor:
    def __init__(self, data):
        self.data = data

    def extract_topk(self, top_k):
        items_with_scores = self._collect_items_with_scores(self.data)
        topk_items_sorted = sorted(items_with_scores, key=itemgetter('score'), reverse=True)[:top_k]
        # print('topk_items_sorted:',topk_items_sorted)
        topk_structure = self._rebuild_topk_structure(topk_items_sorted)
        return topk_structure

    def _collect_items_with_scores(self, data, path=None, items_with_scores=None):
        if items_with_scores is None:
            items_with_scores = []
        if path is None:
            path = []

        for key, value in data.items():
            if isinstance(value, dict):
                self._collect_items_with_scores(value, path + [key], items_with_scores)
            else:
                items_with_scores.append({
                    'path': path + [key],
                    'score': value
                })

        return items_with_scores

    def _rebuild_topk_structure(self, topk_items_sorted):
        tree = {}
        for item in topk_items_sorted:
            current_level = tree
            for part in item['path']:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
        return tree



    def num_tokens_from_string(self, string, encoding_name):
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
