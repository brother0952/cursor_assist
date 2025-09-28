import pickle
import os

def save_dict_to_pickle(new_data, pickle_file="data.pickle"):
    """
    将字典保存到 pickle 文件，跳过已存在的键
    :param new_data: 要保存的新数据（字典）
    :param pickle_file: pickle 文件名
    """
    existing_data = {}

    # 1. 如果 pickle 文件存在，加载已有数据
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            existing_data = pickle.load(f)

    # 2. 合并数据（只保留新数据中不存在的键）
    updated_data = {**existing_data, **new_data}  # 新数据覆盖旧数据（如果键相同）
    # 或者只添加不存在的键：
    # updated_data = existing_data.copy()
    # for key, value in new_data.items():
    #     if key not in existing_data:
    #         updated_data[key] = value

    # 3. 保存更新后的字典
    with open(pickle_file, "wb") as f:
        pickle.dump(updated_data, f)

    #print(f"数据已保存到 {pickle_file}（跳过已存在的键）")


def dump_pickle(pickle_file:str="data.pickle"):
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            existing_data = pickle.load(f)
            # print(existing_data)
            return existing_data



def save_list_to_pickle(new_data_list, pickle_file="data.pickle"):
    """
    将字典列表保存到 pickle 文件，跳过已存在的字典
    :param new_data_list: 要保存的新数据（列表，元素是字典）
    :param pickle_file: pickle 文件名
    """
    existing_data_list = []

    # 1. 如果 pickle 文件存在，加载已有数据
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            existing_data_list = pickle.load(f)

    # 2. 只添加不重复的字典
    updated_data_list = existing_data_list.copy()
    for new_dict in new_data_list:
        if new_dict not in existing_data_list:  # 检查整个字典是否已存在
            updated_data_list.append(new_dict)

    # 3. 保存更新后的列表
    with open(pickle_file, "wb") as f:
        pickle.dump(updated_data_list, f)

    print(f"数据已保存到 {pickle_file}（跳过重复的字典）")


# 示例用法
if __name__ == "__main__":
    # # 新数据（假设要保存）
    # new_data = {
    #     "key1": "value1",
    #     "key2": "value2",
    #     "key3": "value3",
    # }

    # # 调用函数保存
    # save_dict_to_pickle(new_data)

    # 新数据（假设要保存）
    new_data_list = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
        {"name": "Alice", "age": 25},  # 重复项（会被跳过）
    ]

    # 调用函数保存
    # save_list_to_pickle(new_data_list)

    all_link = dump_pickle("link.pkl")
    # print(all_link)

    # try:
        # del all_link["202504"]

    cnt=0
    for k,v in enumerate(all_link):
        # print(v["date"])
        # if "202503" in v["date"]:
            # cnt+=1
            # print(v["date"],len(v["date"]),v["href"])
        # if v["href"].startswith(r"https://www.stats.gov.cn/sj/zxfb"):
            # print("ready to del")
        pass
    # print(len(all_link))
    # print(cnt)
    all_data = dump_pickle("data.pkl")
    # del all_data["202401"]
    # del all_data["202501"]
    # save_dict_to_pickle(all_data)
    # print(all_data["202401"])
    # print(all_data)
    save_list_to_pickle("link.pkl")