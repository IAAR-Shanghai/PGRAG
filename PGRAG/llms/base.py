# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import copy
import json
import os
import re
from abc import ABC, abstractmethod

from loguru import logger


class BaseLLM(ABC):
    def __init__(
            self, 
            model_name: str = None, 
            temperature: float = 1.0, 
            max_new_tokens: int = 4096,
            top_p: float = 0.9,
            top_k: int = 5,
            **more_params
        ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }

    @abstractmethod
    def request(self, query:str) -> str:
        return ''

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        try:
            response = self.request(query)
        except Exception as e:
            logger.warning(repr(e))
            response = ''
        return response

    def extract_mindmap_in_json(self, news_body) -> dict:
        template = self._read_prompt_template('extract_mindmap_in_json.txt')
        query = template.format(news_body=news_body)
        query.replace('{{', '{').replace('}}', '}')
        # print(query)
        respond = self.safe_request(query)
        print(respond)
        real_respond = respond.replace('```json', '').replace('```', '').strip()
        json_obj = json.loads(real_respond)
        return query, json_obj

    def query_deconstruction(self, question):
        template = self._read_prompt_template('query_deconstruction.txt')
        query = template.format(question=question)
        # query.replace('{{', '{').replace('}}', '}')
        print('query:',query)
        respond = self.safe_request(query)
        print('respond:',respond)
        # real_respond = respond.replace('```json', '').replace('```', '').strip()
        # json_obj = json.loads(real_respond)
        # return query

    def process_input_output_pair(self, input_data, output_data):
        # 保存目录名
        output_dir = "data/bm/mindmap/"
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        # 获取当前已有的文件数量
        file_count = len(os.listdir(output_dir))
        # 生成文件名，六位数字
        filename_output = os.path.join(output_dir, f"{file_count + 1}.json")
        # 写入到文件
        with open(filename_output, 'w') as output_file:
            output_file.write(str(output_data))
        print(f"结果已写入到{filename_output}")

    @staticmethod
    def _read_prompt_template(filename: str) -> str:
        path = os.path.join('prompts/', filename)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''
