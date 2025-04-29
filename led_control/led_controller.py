import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

class LEDController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LED Controller')
        self.setGeometry(100, 100, 400, 300)
        
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建网格布局
        layout = QGridLayout(central_widget)
        
        # 创建6个LED按钮
        self.led_buttons = []
        for i in range(6):
            button = QPushButton(f'LED {i+1}')
            button.setCheckable(True)  # 使按钮可切换
            button.clicked.connect(lambda checked, btn=button: self.toggle_led(btn))
            button.setMinimumSize(100, 50)
            
            # 将按钮添加到网格布局中（2行3列）
            row = i // 3
            col = i % 3
            layout.addWidget(button, row, col)
            self.led_buttons.append(button)
            
    def toggle_led(self, button):
        # 根据按钮状态改变颜色
        if button.isChecked():
            button.setStyleSheet('background-color: yellow')
            print(f"{button.text()} ON")
        else:
            button.setStyleSheet('')
            print(f"{button.text()} OFF")

def main():
    app = QApplication(sys.argv)
    window = LEDController()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()