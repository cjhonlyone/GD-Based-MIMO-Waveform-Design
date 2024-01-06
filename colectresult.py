# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:33:54 2023

@author: Chat-GPT
"""

import pandas as pd
import os
import sys

def get_csv_files(path):
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv') and file.startswith('DL') :
                csv_files.append(os.path.join(root, file))
    return csv_files

path = '.'  # 指定文件夹路径
csv_files = get_csv_files(path)  # 获取所有csv文件路径

if csv_files == []:
    print('Error: nothing need to be concat!')
    sys.exit(1)

df_list = []
for file in csv_files:
    folder_name = os.path.basename(os.path.dirname(file))
    df = pd.read_csv(file)
    df.insert(0, 'Folder', folder_name)  # 在首列插入子文件夹名称
    df_list.append(df)

result = pd.concat(df_list, ignore_index=True)  # 合并所有DataFrame

result.to_csv('result.csv', index=False)
result.to_excel('result.xlsx', index=False)