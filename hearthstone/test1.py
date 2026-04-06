import requests
import os

def download_single_card():
    # 创建保存目录
    save_dir = 'hearthstone'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 以萨满祭司的图腾为例
    card_id = "CS2_052"  # 这是治疗图腾的ID
    card_name = "治疗图腾"
    
    # 构造图片URL
    img_url = f"https://art.hearthstonejson.com/v1/render/latest/zhCN/256x/{card_id}.png"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        print(f"开始下载卡牌：{card_name}")
        response = requests.get(img_url, headers=headers)
        
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{card_name}.png")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"下载成功！保存在：{file_path}")
        else:
            print(f"下载失败，状态码：{response.status_code}")
            
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    download_single_card()