import requests
import os
import random
import json

def download_random_card():
    # 创建保存目录
    save_dir = 'hearthstone'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取卡牌数据的API
    api_url = "https://api.hearthstonejson.com/v1/latest/zhCN/cards.json"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # 获取所有卡牌数据
        print("正在获取卡牌列表...")
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            cards = json.loads(response.text)
            # 过滤掉没有id或name的卡牌
            valid_cards = [card for card in cards if 'id' in card and 'name' in card]
            
            if valid_cards:
                # 随机选择一张卡牌
                random_card = random.choice(valid_cards)
                card_id = random_card['id']
                card_name = random_card['name']
                
                # 构造图片URL
                img_url = f"https://art.hearthstonejson.com/v1/render/latest/zhCN/256x/{card_id}.png"
                
                print(f"开始下载卡牌：{card_name} (ID: {card_id})")
                img_response = requests.get(img_url, headers=headers)
                
                if img_response.status_code == 200:
                    file_path = os.path.join(save_dir, f"{card_name}.png")
                    with open(file_path, 'wb') as f:
                        f.write(img_response.content)
                    print(f"下载成功！保存在：{file_path}")
                else:
                    print(f"下载图片失败，状态码：{img_response.status_code}")
            else:
                print("没有找到有效的卡牌数据")
        else:
            print(f"获取卡牌列表失败，状态码：{response.status_code}")
            
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    download_random_card()