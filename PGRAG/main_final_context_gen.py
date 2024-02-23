# -*- coding: utf-8 -*-
# 主函数
from multiprocessing import Pool
from tqdm import tqdm
import json
from PGRAG.Context_Recall.IO import JsonFileHandler
from PGRAG.Context_Recall.context_prepare import GraphPathToJsonConverter
from PGRAG.Context_Recall.final_context import TopKJsonExtractor

qdse_file = 'data_v1/eval/updated_data_with_qdse.json'
topN_topic_paths_file = 'data_v1/context_recall/topN_topic_path_70.json'
converter = GraphPathToJsonConverter()
# # # 创建JsonFileHandler实例，指定要操作的文件路径
file_handler = JsonFileHandler('data/output/final_context_top14.json')
topK_context_token_len_file = 'data/context_recall/topK_context_token_len.json'
# 全局函数，避免在函数内部定义函数或使用 lambda 函数
def process_data(args):
    topN_topic_paths_line, qdse_item = args
    topN_topic_paths = eval(topN_topic_paths_line)
    question = qdse_item['question']
    qdse_i = qdse_item[question]['qdse']
    json_outputs = converter.convert_paths_to_json(topN_topic_paths, qdse_i)

    # 输出到文件
    file_handler.save_data_to_file({
        "question": question,
        question: {
            "context": json_outputs
        }
    })

    return {
        "question": question,
        question: {
            "context": json_outputs
        }
    }


def context_gen():
    # 上下文生成
    with open(topN_topic_paths_file, 'r', encoding='utf-8') as file:
        topN_topic_paths_lines = file.readlines()
    with open(qdse_file, 'r', encoding='utf-8') as file:
        qdse_list = json.load(file)
    # # 单文档问题
    # sub_qdse_list = [qdse_item for i, qdse_item in enumerate(qdse_list) if (i + 3) % 3 == 0]
    sub_qdse_list = qdse_list
    # 创建进程池
    with Pool(processes = 10) as pool:
        # 定义处理函数的参数列表
        args_list = [(topN_topic_paths_line, qdse_item)
                     for topN_topic_paths_line, qdse_item in zip(topN_topic_paths_lines[:3], sub_qdse_list[:3])]

        # 使用 tqdm 包装进程池的 map 函数以显示进度条
        with tqdm(total=len(args_list)) as pbar:
            results = []
            for result in pool.imap_unordered(process_data, args_list):
                results.append(result)
                pbar.update(1)

    # 输出新的数据列表
    print(results)

if __name__ == "__main__":
    context_gen()
    ###准备最终上下文
    i = 0
    topK = 14
    loaded_data = file_handler.load_data_from_file()
    file_handler2 = JsonFileHandler(f'data/output/top{topK}_context.json')
    token_size = []
    total_tokens = 0  # 用于累积总token数的变量
    print(len(loaded_data))
    new_data = []
    for query_context in loaded_data:
        i = i + 1
        print(i)
        print(query_context)
        print(type(query_context))
        question = query_context['question']
        context = query_context[question]['context']
        print('question:',question)
        print('context:',context)
        reranker = TopKJsonExtractor(context)
        topk_result = reranker.extract_topk(topK)
        print("TopK 结果:")
        print(topk_result)
        query_context[question]["topk_result"] = topk_result
        # new_data.append(query_context)
        print('result:',query_context)
        file_handler2.save_data_to_file(query_context)
        tokens = reranker.num_tokens_from_string(str(topk_result), "cl100k_base")
        print('token总长度：', tokens)
        token_size.append(tokens)
        total_tokens += tokens  # 累积总token数
    topK_context_token = {}
    average_token_length = total_tokens / len(loaded_data)  # 计算平均token长度
    print("平均token长度：", average_token_length)
    topK_context_token["topK"] = topK
    topK_context_token["average_token_length"] = average_token_length
    with open(topK_context_token_len_file, 'a', encoding='utf-8') as file:
        file.write(str(topK_context_token)+'\n')
        print('文件已写入！')
    print('----------------------------最终上下文准备acc')