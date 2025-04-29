import sys
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, 
                            QGridLayout, QVBoxLayout, QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt

class ConfigurableLEDController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_config()
        self.setup_ui()
        
    def load_config(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.button_states = [False] * self.config['buttons']['count']
        self.led_states = [False] * self.config['leds']['count']
        
    def setup_ui(self):
        self.setWindowTitle('Configurable LED Controller')
        self.setGeometry(100, 100, 800, 400)
        
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
        for i in range(self.config['leds']['count']):
            led = QLabel(self.config['leds']['names'][i])
            led.setAlignment(Qt.AlignCenter)
            led.setStyleSheet('background-color: gray; min-width: 100px; min-height: 50px')
            row = i // 2
            col = i % 2
            led_layout.addWidget(led, row, col)
            self.led_indicators.append(led)
            
        main_layout.addWidget(button_panel)
        main_layout.addWidget(led_panel)
        
    def on_button_clicked(self, button_index):
        self.button_states[button_index] = self.buttons[button_index].isChecked()
        self.apply_rules()
        
    def apply_rules(self):
        # 重置所有LED状态
        self.led_states = [False] * self.config['leds']['count']
        
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
                self.led_states[led_idx] = rule['action']['state']
                
        # 更新LED显示
        self.update_leds()
        
    def update_leds(self):
        for i, state in enumerate(self.led_states):
            color = 'yellow' if state else 'gray'
            self.led_indicators[i].setStyleSheet(
                f'background-color: {color}; min-width: 100px; min-height: 50px'
            )

def main():
    app = QApplication(sys.argv)
    window = ConfigurableLEDController()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()