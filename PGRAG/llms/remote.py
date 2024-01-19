# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import json
import re
import requests

from loguru import logger

from PGRAG.llms.base import BaseLLM
from PGRAG.configs import real_config as conf


class GPT_transit(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=4096, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    def request(self, query: str) -> str:
        url = conf.GPT_transit_url
        payload = json.dumps({
            "model": self.params['model_name'],
            "messages": [{"role": "user", "content": query}],
            "temperature": self.params['temperature'],
            'max_tokens': self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
        })
        headers = {
            'token': conf.GPT_transit_token,
            'User-Agent': 'llm_ruc_group_001',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '47.88.65.188:8001',
            'Connection': 'keep-alive'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()
        real_res = res["choices"][0]["message"]["content"]

        token_consumed = res['usage']['total_tokens']
        logger.info(f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res
