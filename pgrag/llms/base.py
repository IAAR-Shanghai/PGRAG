import jieba
from text2vec import Similarity
import evaluate
import shutil
import os
import copy
import json
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
        self.post_init()
    
    def post_init(self):
        """Post initialization method for subclasses.
        Normally, this method should initialize the model and tokenizer.
        """
        ...

    def update_params(self, inplace: bool = True, **params):
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

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

    def safe_request_with_prompt(self, text, prompt_file) -> str:
        template = self._read_prompt_template(prompt_file)
        query = template.format(text=text)
        query = query.replace('{{', '{').replace('}}', '}')
        respond = self.safe_request(query)
        return respond

    def extract_fact_verification_items(self, news_body) -> str:
        prompt_file = 'extract_fact_verification_items.txt'
        respond = self.safe_request_with_prompt(news_body, prompt_file)
        return respond
    
    def extract_title(self, text) -> str:
        prompt_file = 'topic_extract.txt'
        respond = self.safe_request_with_prompt(text, prompt_file)
        return respond
    
    def gen_mindmap(self, title, text) -> str:
        prompt_file = 'gen_mindmap.txt'
        template = self._read_prompt_template(prompt_file)
        query = template.format(title=title, text=text)
        query = query.replace('{{', '{').replace('}}', '}')
        respond = self.safe_request(query)
        return respond
    
    def process_input_output_pair(self, line_no, output_data, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        filename_output = os.path.join(output_dir, f"{line_no}.txt")

        with open(filename_output, 'w', encoding='UTF-8') as output_file:
            output_file.write(output_data)


    def process_efvi(self, file_path, gpt_instance, output_dir):
        line_no = os.path.basename(file_path).replace('.txt', '')
        with open(file_path, 'r', encoding='UTF-8') as file:
            news_body = file.read()
        output_data = gpt_instance.extract_fact_verification_items(news_body)
        rougeL_score = self.rougeL_score(output_data, news_body)
        print(f"ROUGE-L:'{rougeL_score}")
        # 测试 BertScore 分数计算
        bert_score = self.bert_score(output_data, news_body)
        print(f"bert_score:'{bert_score}")
        if rougeL_score >= 0.15 and bert_score >= 0.85:
            gpt_instance.process_input_output_pair(line_no, output_data, output_dir)
        else:
            new_file_path = "data/raw_news/regen/"
            os.makedirs(new_file_path, exist_ok=True)
            shutil.move(file_path, new_file_path)

    def process_et(self, file_path, gpt_instance, output_dir):
        line_no = os.path.basename(file_path).replace('.txt', '')
        with open(file_path, 'r', encoding='UTF-8') as file:
            news_body = file.read()
        output_data = gpt_instance.extract_title(news_body)
        gpt_instance.process_input_output_pair(line_no, output_data, output_dir)


    def process_gm(self, title_files_dir, fcis_files_dir, gpt_instance, output_dir):
        line_no = os.path.basename(title_files_dir).replace('.txt', '')
        with open(title_files_dir, 'r', encoding='UTF-8') as file:
            title = file.read()
        ttv_path = os.path.join(fcis_files_dir, f"{line_no}.txt")
        with open(ttv_path, 'r', encoding='UTF-8') as file:
            text = file.read()
        output_data = gpt_instance.gen_mindmap(title, text)
        gpt_instance.process_input_output_pair(line_no, output_data, output_dir)

    def mindmap_str_to_json(self, mindmap_str_file_path, mindmap_json_dir):
        mindmap_json_file_path = os.path.join(mindmap_json_dir, os.path.basename(mindmap_str_file_path).replace('.txt', '.json'))
        with open(mindmap_str_file_path, 'r', encoding='utf-8') as file:
            mindmap_str = file.read()
            try:
                if '```json' in mindmap_str:
                    real_content = mindmap_str.replace('```json', '').replace('```', '').strip()
                    mindmap = json.loads(real_content)
                else:
                    # 否则直接使用原始响应字符串
                    mindmap = json.loads(mindmap_str)
                with open(mindmap_json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(mindmap, f, ensure_ascii=False, indent=4)
            except json.JSONDecodeError as e:
                print(f'JSON解析错误在文件{mindmap_str_file_path}', e)


    def query_deconstruction(self, question):
        template = self._read_prompt_template('query_deconstruction.txt')
        query = template.format(question=question)
        respond = self.safe_request(query)
        return respond


    @staticmethod
    def _read_prompt_template(filename: str) -> str:
        path = os.path.join('prompts/', filename)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''
        
    def rougeL_score(self, 
        continuation: str,
        reference: str
    ) -> float:
        f = lambda text: list(jieba.cut(text))
        # rouge = evaluate.load('/path/to/local/rouge')
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=[continuation], references=[[reference]], tokenizer=f, rouge_types=['rougeL'])
        score = results['rougeL']
        return score

    def bert_score(self, 
        continuation: str,
        reference: str
    ) -> float:
        from text2vec import Similarity
        sim = Similarity(model_name_or_path="/path/to/local/text2vec-base-chinese")
        score = sim.get_score(continuation, reference)
        return score
