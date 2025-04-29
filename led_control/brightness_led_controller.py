import sys
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, 
                            QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

class BrightnessLEDController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_config()
        self.setup_ui()
        
    def load_config(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.button_states = [False] * self.config['buttons']['count']
        self.led_brightness = [0] * self.config['leds']['count']
        
    def setup_ui(self):
        self.setWindowTitle('LED Brightness Controller')
        self.setGeometry(100, 100, 1000, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 创建按钮面板
        button_panel = QWidget()
        button_layout = QGridLayout(button_panel)
        self.buttons = []
        for i in range(self.config['buttons']['count']):
            button = QPushButton(self.config['buttons']['names'][i])
            button.setCheckable(True)
            button.clicked.connect(lambda checked, idx=i: self.on_button_clicked(idx))
            row = i // 3
            col = i % 3
            button_layout.addWidget(button, row, col)
            self.buttons.append(button)
        
        # 创建LED显示面板
        led_panel = QWidget()
        led_layout = QGridLayout(led_panel)
        self.led_indicators = []
        self.brightness_bars = []
        
        for i in range(self.config['leds']['count']):
            led_widget = QWidget()
            led_v_layout = QVBoxLayout(led_widget)
            
            # LED 标签
            led_label = QLabel(self.config['leds']['names'][i])
            led_label.setAlignment(Qt.AlignCenter)
            
            # 亮度进度条
            brightness_bar = QProgressBar()
            brightness_bar.setMinimum(0)
            brightness_bar.setMaximum(100)
            brightness_bar.setValue(0)
            brightness_bar.setTextVisible(True)
            brightness_bar.setFormat('%v%')
            
            led_v_layout.addWidget(led_label)
            led_v_layout.addWidget(brightness_bar)
            
            row = i // 2
            col = i % 2
            led_layout.addWidget(led_widget, row, col)
            
            self.brightness_bars.append(brightness_bar)
            
        main_layout.addWidget(button_panel)
        main_layout.addWidget(led_panel)
        
    def on_button_clicked(self, button_index):
        self.button_states[button_index] = self.buttons[button_index].isChecked()
        self.apply_rules()
        
    def apply_rules(self):
        # 重置所有LED亮度
        self.led_brightness = [0] * self.config['leds']['count']
        
        # 应用规则
        for rule in self.config['rules']:
            # 检查规则中的按钮组合是否满足
            rule_matched = True
            for button_idx in rule['buttons']:
                if not self.button_states[button_idx]:
                    rule_matched = False
                    break
                    
            if rule_matched:
                led_idx = rule['action']['led']
                self.led_brightness[led_idx] = rule['action']['brightness']
                
        # 更新LED显示
        self.update_leds()
        
    def update_leds(self):
        for i, brightness in enumerate(self.led_brightness):
            self.brightness_bars[i].setValue(brightness)
            
            # 根据亮度设置进度条颜色
            color = self.get_brightness_color(brightness)
            self.brightness_bars[i].setStyleSheet(
                f"""
                QProgressBar {{
                    border: 2px solid grey;
                    border-radius: 5px;
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                }}
                """
            )
    
    def get_brightness_color(self, brightness):
        # 将亮度值转换为颜色，从暗黄到亮黄
        if brightness == 0:
            return '#404040'
        else:
            # 计算颜色强度
            intensity = int(255 * (0.3 + 0.7 * brightness / 100))
            return f'rgb({intensity}, {intensity}, 0)'

def main():
    app = QApplication(sys.argv)
    window = BrightnessLEDController()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()