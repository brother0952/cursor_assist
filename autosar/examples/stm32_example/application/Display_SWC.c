#include "Rte_Receiver.h"
#include "stm32f4xx_hal.h"

// LCD 显示相关函数（示例）
static void LCD_DisplaySpeed(uint8 speed) {
    // 实现 LCD 显示速度的代码
    char str[10];
    sprintf(str, "Speed:%d", speed);
    // LCD_WriteString(0, 0, str); // 假设的 LCD 函数
}

static void LCD_DisplayRPM(uint16 rpm) {
    // 实现 LCD 显示转速的代码
    char str[10];
    sprintf(str, "RPM:%d", rpm);
    // LCD_WriteString(0, 1, str); // 假设的 LCD 函数
}

void Display_Run(void) {
    VehicleData_Type vehicleData;
    
    if (Rte_Read_VehicleData(&vehicleData) == RTE_E_OK) {
        LCD_DisplaySpeed(vehicleData.speed);
        LCD_DisplayRPM(vehicleData.rpm);
        
        // 速度超过阈值时触发警告
        if (vehicleData.speed > 100) {
            HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, GPIO_PIN_SET); // 警告 LED
            Rte_Call_SpeedWarning();
        } else {
            HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, GPIO_PIN_RESET);
        }
    }
} 