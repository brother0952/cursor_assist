#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复杂Jinja2模板引擎示例
使用外部模板文件和字典数据
"""

from jinja2 import Environment, FileSystemLoader
import os


def complex_example():
    """使用外部模板文件的复杂示例"""
    print("=== 复杂Jinja2示例：使用外部模板 ===")
    
    # 设置模板环境
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))
    
    # 加载模板
    template = env.get_template('template.html')
    
    # 准备数据字典，包含特殊键
    data = {
        'name': '张三',
        'age': 30,
        'divider_before': '这个键前面会有一个分割线',
        'special_item': '这是一个特殊项目',
        'city': '北京',
        'divider_after': '这个键后面会有一个分割线',
        'job': '程序员',
        'hobby': '编程'
    }
    
    # 渲染模板
    result = template.render(data=data)
    
    print("渲染结果:")
    print(result)
    
    # 将结果保存到文件
    output_file = os.path.join(current_dir, 'output.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\n结果已保存到 {output_file}")


def another_complex_example():
    """另一个复杂示例，展示不同数据的效果"""
    print("\n=== 另一个复杂示例 ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))
    template = env.get_template('template.html')
    
    # 使用不同的数据
    data2 = {
        'product': '笔记本电脑',
        'price': 5999,
        'divider_after': '这个后面有分割线',
        'spec': 'i7处理器，16GB内存',
        'divider_before': '这个前面有分割线',
        'features': '轻薄便携，长续航',
        'rating': 4.8
    }
    
    result = template.render(data=data2)
    print("渲染结果:")
    print(result)
    
    # 保存第二个输出
    output_file = os.path.join(current_dir, 'output2.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    print("复杂Jinja2模板引擎示例程序")
    print("=" * 50)
    
    complex_example()
    another_complex_example()
    
    print("\n" + "=" * 50)
    print("复杂Jinja2示例执行完成！")