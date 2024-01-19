import json
import random

# 文件路径
database_file_path = '../data/retrieve_database.txt'
json_file_path = '../data/QAGeneration_gpt-4-1106-preview_3docs.json'

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 从JSON文件中提取前20个结果对应的新闻片段
news_snippets = []
for result in json_data['results'][:20]:
    for key in ['news1', 'news2', 'news3']:
        snippet = result[key]
        # 随机截取20个字符的片段
        if len(snippet) > 8:
            start_index = random.randint(0, len(snippet) - 8)
            news_snippets.append(snippet[start_index:start_index + 8])
        else:
            news_snippets.append(snippet)

# 读取数据库文件并提取对应新闻
extracted_news_lines = []
used_lines = set()  # 用于存储已经匹配过的行，以避免重复匹配

with open(database_file_path, 'r', encoding='utf-8') as file:
    database_lines = file.readlines()

for snippet in news_snippets:
    found = False
    for attempt in range(18):  # 尝试最多三次
        for line in database_lines:
            if snippet in line and line not in used_lines:
                # 找到匹配的行，并标记为已使用
                extracted_news_lines.append(line.strip())
                used_lines.add(line)
                found = True
                break

        if found:
            break
        else:
            # 重新选择随机片段
            start_index = random.randint(0, len(snippet) - 8)
            snippet = snippet[start_index:start_index + 8]

    if not found:
        extracted_news_lines.append("未找到匹配的新闻")
        print("1")

# 存储提取的新闻到TXT文件
output_file_path = '../data/extracted_news.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for line in extracted_news_lines:
        file.write(line + '\n')

output_file_path
