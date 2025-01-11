#ifndef MCAL_ADC_H
#define MCAL_ADC_H

#include "Std_Types.h"

// ADC 初始化
void Mcal_Adc_Init(void);
// 读取 ADC 值
uint16 Mcal_Adc_Read(uint8 channel);

#endif 