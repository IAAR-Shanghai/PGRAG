import json
import requests
from llms.base import BaseLLM
from configs import real_config as conf

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
            'Authorization': 'Bearer {}'.format(conf.GPT_transit_token),
            'Content-Type': 'application/json',
        }
        res = requests.request("POST", url, headers=headers, data=payload,timeout=300)
        print('res:', res.text)
        res = res.json()
        real_res = res["choices"][0]["message"]["content"]
        return real_res
