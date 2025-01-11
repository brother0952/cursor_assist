#include "Rte_Sender.h"
#include "Mcal_Adc.h"

#define SPEED_ADC_CHANNEL 0
#define RPM_ADC_CHANNEL 1

// 将 ADC 值转换为速度
static uint8 ConvertToSpeed(uint16 adcValue) {
    // 示例转换公式：假设 ADC 12位分辨率
    return (uint8)((adcValue * 200) / 4096);
}

// 将 ADC 值转换为转速
static uint16 ConvertToRPM(uint16 adcValue) {
    // 示例转换公式：假设 ADC 12位分辨率
    return (uint16)((adcValue * 8000) / 4096);
}

void SpeedSensor_Run(void) {
    VehicleData_Type vehicleData;
    
    // 读取传感器数据
    uint16 speedAdc = Mcal_Adc_Read(SPEED_ADC_CHANNEL);
    uint16 rpmAdc = Mcal_Adc_Read(RPM_ADC_CHANNEL);
    
    // 转换数据
    vehicleData.speed = ConvertToSpeed(speedAdc);
    vehicleData.rpm = ConvertToRPM(rpmAdc);
    
    // 通过 RTE 发送数据
    Rte_Write_VehicleData(&vehicleData);
} 