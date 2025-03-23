import os
import re
import argparse
from pathlib import Path

def extract_number_from_optimized(filename):
    """从优化后的文件名中提取编号"""
    match = re.search(r'_(\d+)_\d+_photo_optimized', filename)
    return int(match.group(1)) if match else None

def extract_number_from_original(filename):
    """从原始DJI文件名中提取编号"""
    match = re.search(r'DJI_(\d+)', filename)
    return int(match.group(1)) if match else None

def process_photos(directory):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"错误：目录 '{directory}' 不存在")
        return
    
    # 获取目录下所有jpg文件
    photos = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
    
    if not photos:
        print(f"在目录 '{directory}' 中没有找到jpg图片")
        return
    
    # 分类存储原始图片和优化图片
    optimized_photos = {}
    original_photos = {}
    
    for photo in photos:
        if 'photo_optimized' in photo:
            number = extract_number_from_optimized(photo)
            if number:
                optimized_photos[number] = photo
        elif photo.startswith('DJI_'):
            number = extract_number_from_original(photo)
            if number:
                original_photos[number] = photo
    
    # 查找重复的图片对
    duplicates = []
    for number in optimized_photos:
        if number in original_photos:
            duplicates.append((original_photos[number], optimized_photos[number]))
    
    if not duplicates:
        print("没有找到重复的图片对")
        return
    
    # 处理重复的图片
    print("\n发现以下重复图片:")
    for original, optimized in duplicates:
        print(f"原始图片: {original}")
        print(f"优化图片: {optimized}\n")
    
    choice = input(f"是否删除所有[{len(duplicates)}]张原始图片? (y/n): ").lower()
    if choice == 'y':
        for original, _ in duplicates:
            try:
                os.remove(os.path.join(directory, original))
                print(f"已删除: {original}")
            except Exception as e:
                print(f"删除失败: {e}")
    else:
        print("保留所有原始图片")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='处理DJI图片重复文件')
    parser.add_argument('directory', nargs='?', default=os.getcwd(),
                      help='指定要处理的图片目录路径，默认为当前目录')
    
    args = parser.parse_args()
    directory = os.path.abspath(args.directory)
    
    print(f"正在处理目录: {directory}")
    process_photos(directory) 