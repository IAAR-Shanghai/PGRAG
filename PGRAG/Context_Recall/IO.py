# -*- coding: utf-8 -*-
import json

class JsonFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_data_to_file(self, data):
        """
        将数据作为JSON格式的新行保存到文件中。

        :param data: 要保存的数据。
        """
        with open(self.file_path, 'a', encoding='utf-8') as file:  # 使用追加模式'a'
            # 将数据转换为JSON字符串，并添加换行符以便作为新行写入
            json_str = json.dumps(data, ensure_ascii=False) + "\n"
            file.write(json_str)
        print(f"数据已作为新行添加到文件：{self.file_path}")

    def load_data_from_file(self):
        """
        从文件逐行加载JSON数据，每行一个JSON对象。

        :return: 从文件加载的数据列表。
        """
        data_list = []
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 将每行的JSON字符串转换回Python对象并添加到列表中
                data_list.append(json.loads(line))
        return data_list
