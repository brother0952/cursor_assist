import sys
import logging
import pandas as pd

def setup_logging():
    """设置日志记录，使用UTF-8编码"""
    try:
        # 确保使用UTF-8编码写入日志文件
        file_handler = logging.FileHandler('excel_data_reader.log', encoding='utf-8')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                file_handler,
                logging.StreamHandler()
            ]
        )
        return True
    except Exception as e:
        print(f"无法设置日志记录: {e}")
        return False

def check_dependencies():
    """检查必要的依赖库"""
    try:
        import pandas
        import openpyxl
        return True
    except ImportError as e:
        print(f"缺少必要的依赖库: {e}")
        print("请先安装依赖库: pip install pandas openpyxl")
        return False

def read_excel_data(file_path):
    """
    读取Excel文件数据并返回DataFrame
    :param file_path: Excel文件路径
    :return: 包含Excel数据的DataFrame
    """
    try:
        # 读取Excel文件，header=0表示第一行作为列名
        df = pd.read_excel(file_path, header=0)
        # 优化数据输出格式
        logging.info("成功读取Excel文件数据，前5行如下:")
        logging.info("\n" + df.head().to_string())  # 使用基本字符串格式输出
        logging.info(f"\n数据概览:\n{df.describe().to_string()}")  # 添加数据统计信息
        return df
    except Exception as e:
        logging.error(f"读取Excel文件出错: {e}")
        return None

def append_new_data(main_df, new_data_df):
    """
    将新数据追加到主DataFrame末尾
    :param main_df: 主DataFrame
    :param new_data_df: 要追加的新数据DataFrame
    :return: 合并后的DataFrame
    """
    if main_df is None:
        return new_data_df
    if new_data_df is None:
        return main_df
    # 确保列名一致才能追加
    if set(main_df.columns) == set(new_data_df.columns):
        return pd.concat([main_df, new_data_df], ignore_index=True)
    else:
        print("列名不匹配，无法追加数据")
        return main_df


if __name__ == "__main__":
    # 打开并读取txt文件内容
    with open('P020250709332852003794.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # 打印txt文件内容
    print(content)

    if not setup_logging():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
        
    try:
        import pandas as pd
        # 读取当前目录下的Excel文件
        excel_file = "P020250709332852003794.xlsx"
        data_df = read_excel_data(excel_file)
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        sys.exit(1)
    
    # 这里可以保存data_df到变量或文件，以便后续追加新数据
    # 示例: 如何追加新数据
    # new_data = read_excel_data("新月份数据.xlsx")
    # data_df = append_new_data(data_df, new_data)
