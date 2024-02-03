# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import copy
import json
import os
import re
from abc import ABC, abstractmethod
import concurrent.futures
from tqdm import tqdm

from loguru import logger


class BaseLLM(ABC):
    def __init__(
            self, 
            model_name: str = None, 
            temperature: float = 1.0, 
            max_new_tokens: int = 2048,
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
        query = query.replace('{{', '{').replace('}}', '}')

        respond = self.safe_request(query)

        try:
            real_respond = respond.replace('```json', '').replace('```', '').strip()
            # json_obj = json.loads(real_respond)
            return query, real_respond
        except json.JSONDecodeError:
            raise ValueError("JSON字符串获取失败")

    def process_input_output_pair(self, line_no, output_data):
        output_dir = "data/eval/text2mindmap/"
        os.makedirs(output_dir, exist_ok=True)
        filename_output = os.path.join(output_dir, f"{line_no}.txt")

        with open(filename_output, 'w', encoding='UTF-8') as output_file:
            json.dump(output_data, output_file, ensure_ascii=False, indent=4)
        print(f"结果已写入到{filename_output}")

    def process_file(self, file_path, gpt_instance, error_log):
        line_no = os.path.basename(file_path).replace('.txt', '')
        try:
            with open(file_path, 'r', encoding='UTF-8') as file:
                news_body = file.read()
                _, output_data = gpt_instance.extract_mindmap_in_json(news_body)
                gpt_instance.process_input_output_pair(line_no, output_data)
        except Exception as e:
            with open(error_log, 'a') as log_file:
                log_file.write(f"{file_path}: {e}\n")

    def query_deconstruction(self, question):
        template = self._read_prompt_template('query_deconstruction.txt')
        query = template.format(question=question)
        # query.replace('{{', '{').replace('}}', '}')
        print('query:',query)
        respond = self.safe_request(query)
        print('respond:',respond)
        # real_respond = respond.replace('```json', '').replace('```', '').strip()
        # json_obj = json.loads(real_respond)
        return query

    def question_answer(self, context, question):
        # 读取模板
        template = self._read_prompt_template('quest_answer.txt')
        # 格式化查询
        query = template.format(context=context, question=question)
        # 发送查询并获取回答
        answers = self.safe_request(query)
        # 解析回答
        pattern = r'<response>\n(.*?)\n</response>'
        final_answers = re.findall(pattern, answers, re.DOTALL)
        # 写入文件
        with open('data_old/bm/final_answer/top10.txt', 'a', encoding='utf-8') as file:
            file.write(' '.join(final_answers).replace('\n',' ') + '\n')
        print('最终答案写入成功')
        return final_answers

    @staticmethod
    def _read_prompt_template(filename: str) -> str:
        path = os.path.join('prompts/', filename)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''
