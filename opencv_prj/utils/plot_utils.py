import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import platform

def setup_chinese_font():
    """设置matplotlib中文字体"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
        if Path(font_path).exists():
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
    elif system == 'Linux':
        # Linux系统
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 文泉驿微米黑
    elif system == 'Darwin':
        # macOS系统
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS自带的字体
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
def create_figure_with_chinese(title=None, xlabel=None, ylabel=None, figsize=(10, 6)):
    """创建支持中文的图表"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return fig, ax 