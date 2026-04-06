#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复杂Jinja2模板引擎示例
使用外部模板文件和字典数据
"""

from jinja2 import Environment, FileSystemLoader
import os


def complex_example():
    
    # 设置模板环境
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))
    
    # 加载模板
    template = env.get_template('template')
    
    # 准备数据字典，包含特殊键
    data = {
        'name': '张三',
        'age': 30,
        'divider_before': '这个键前面会有一个分割线',
        'group': None,
        'city': '北京',
        'divider_after': '这个键后面会有一个分割线',
        'job': '程序员',
        'hobby': '编程'
    }
    
    # 渲染模板
    result = template.render(data=data)
    

    
    # 将结果保存到文件
    output_file = os.path.join(current_dir, 'output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    



if __name__ == "__main__":
   
    complex_example()
    
   