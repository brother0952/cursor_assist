#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复杂Jinja2模板引擎示例 - 测试新功能
使用外部模板文件和字典数据，包含group功能
"""

from jinja2 import Environment, FileSystemLoader
import os


def complex_group_example():
    """使用外部模板文件的复杂示例，展示group功能"""
    print("=== 复杂Jinja2示例：使用group功能 ===")
    
    # 设置模板环境
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))
    
    # 加载模板
    template = env.get_template('template.html')
    
    # 准备数据字典，包含group标记
    # 当group键的值为None时，切换group状态（开/关）
    data = {
        'name': '张三',
        'age': 30,
        'divider_before': '这个前面会有一个普通分割线',
        'special_item': '这个应该被[[]]包围',
        'city': '北京',
        'group': None,  # 切换到group模式，此后内容应被{}包围
        'inside_group_1': '这个应该被{}包围',
        'inside_group_2': '这个也应该被{}包围',
        'divider_after': '这个后面会有一个普通分割线',
        'inside_group_3': '这个也应该被{}包围',
        'job': '程序员',
        'group': None,  # 再次切换group模式，回到普通模式
        'hobby': '编程',
        'final_value': '这个应该被[[]]包围'
    }
    
    result = template.render(data=data)
    
    print("渲染结果:")
    print(result)
    
    # 将结果保存到文件
    output_file = os.path.join(current_dir, 'complex_output.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\n结果已保存到 {output_file}")


def simple_example():
    """简单示例，不含group"""
    print("\n=== 简单Jinja2示例 ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))
    template = env.get_template('template.html')
    
    data = {
        'product': '笔记本电脑',
        'price': 5999,
        'spec': 'i7处理器，16GB内存',
        'rating': 4.8
        # 不包含group键，所以所有内容都会被[[]]包围
    }
    
    result = template.render(data=data)
    print("渲染结果:")
    print(result)
    
    # 保存简单示例结果
    output_file = os.path.join(current_dir, 'simple_output.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\n结果已保存到 {output_file}")


def another_group_example():
    """另一个group示例，从group开始"""
    print("\n=== 从group开始的示例 ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(current_dir))
    template = env.get_template('template.html')
    
    data = {
        'group': None,  # 立即开启group模式
        'first_in_group': '第一个在group中',
        'second_in_group': '第二个在group中',
        'divider_before': '这个前面会有group风格分割线{}',
        'after_divider': '这个也在group中',
        'group': None,  # 关闭group
        'outside_group': '这个在group外',
        'divider_after': '这个后面会有普通分割线',
        'after_divider_normally': '这个也在group外'
    }
    
    result = template.render(data=data)
    print("渲染结果:")
    print(result)
    
    # 保存另一个示例结果
    output_file = os.path.join(current_dir, 'another_output.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    print("复杂Jinja2模板引擎示例程序 - Group功能")
    print("=" * 60)
    
    complex_group_example()
    simple_example()
    another_group_example()
    
    print("\n" + "=" * 60)
    print("复杂Jinja2示例执行完成！")