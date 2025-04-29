import json
import pandas as pd
import itertools

def analyze_config(config_path):
    # 读取配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 获取所有按钮的可能组合
    button_names = config['buttons']['names']
    all_combinations = []
    
    # 为每个LED创建一个亮度列
    led_columns = [f"{name} (%)" for name in config['leds']['names']]
    
    # 生成所有可能的按钮组合
    for i in range(len(button_names) + 1):
        combinations = list(itertools.combinations(range(len(button_names)), i))
        all_combinations.extend(combinations)
    
    # 创建结果数据
    rows = []
    for combo in all_combinations:
        # 初始化所有LED亮度为0
        led_brightness = [0] * config['leds']['count']
        
        # 检查每个规则
        for rule in config['rules']:
            # 检查规则是否匹配当前按钮组合
            if set(rule['buttons']).issubset(set(combo)):
                led_brightness[rule['action']['led']] = rule['action']['brightness']
        
        # 创建按钮状态描述
        button_state = ['OFF'] * len(button_names)
        for btn_idx in combo:
            button_state[btn_idx] = 'ON'
        
        # 组合所有数据
        row_data = button_state + led_brightness
        rows.append(row_data)
    
    # 创建DataFrame
    columns = button_names + led_columns
    df = pd.DataFrame(rows, columns=columns)
    
    return df

def main():
    config_path = 'config.json'
    df = analyze_config(config_path)
    
    # 设置pandas显示选项
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # 打印表格
    print("\nLED Control Configuration Analysis")
    print("=" * 80)
    print(df)
    
    # 保存到CSV文件
    csv_path = 'led_control_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nAnalysis has been saved to {csv_path}")

if __name__ == '__main__':
    main()