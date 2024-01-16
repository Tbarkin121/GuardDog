<#ftl strip_whitespace = true>

<#if !MC??>
<#if SWIPdatas??>
<#list SWIPdatas as SWIP>
<#if SWIP.ipName == "MotorControl">
<#if SWIP.parameters??>
<#assign MC = SWIP.parameters>
<#break>
</#if>
</#if>
</#list>
</#if>
<#if !MC??>
<#stop "No MotorControl SW IP data found">
</#if>
<#if configs[0].peripheralParams.get("RCC")??>
<#assign RCC = configs[0].peripheralParams.get("RCC")>
</#if>
<#if !RCC??>
<#stop "No RCC found">
</#if>
</#if>

<#assign FOC = MC.M1_DRIVE_TYPE == "FOC" || MC.M2_DRIVE_TYPE == "FOC">
<#assign SIX_STEP = MC.M1_DRIVE_TYPE == "SIX_STEP" || MC.M2_DRIVE_TYPE == "SIX_STEP">

/**
  ******************************************************************************
  * @file    parameters_conversion_f30x.h
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file contains the definitions needed to convert MC SDK parameters
  *          so as to target the STM32F3 Family.
  *
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */
  
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __PARAMETERS_CONVERSION_F30X_H
#define __PARAMETERS_CONVERSION_F30X_H
<#if FOC>
#include "mc_stm_types.h"
<#if MC.PFC_ENABLED == true>
#include "pfc_parameters.h"
</#if>
</#if>
/************************* CPU & ADC PERIPHERAL CLOCK CONFIG ******************/
<#function _last_word text sep="_"><#return text?split(sep)?last></#function>
<#function TimerHandler timer><#return "${timer}_IRQHandler"></#function>
<#function Fx_TIM_Div TimFreq PWMFreq>
    <#list [1,2,3,4,5] as divider>
    <#-- (PWM_PERIOD_CYCLES*SQRT3FACTOR)/16384u must feet in 16 bits with SQRT3FACTOR=0xDDB4 -->
       <#if (TimFreq/(PWMFreq*divider)*56756)/16384 < 65535 >
            <#return divider >
        </#if>
    </#list>
    <#return 1 >
</#function>


<#assign SYSCLKFreq = RCC.get("SYSCLKFreq_VALUE")?number>
<#assign APB1TIM_Freq = RCC.get("APB1TimFreq_Value")?number>
<#assign TimerFreq = _last_word(MC.M1_PWM_TIMER_SELECTION)+"Freq_Value" >
<#assign ADV_TIM_CLKFreq = RCC.get(TimerFreq)?number>
<#assign TimerDivider = Fx_TIM_Div(ADV_TIM_CLKFreq,(MC.M1_PWM_FREQUENCY)?number) >
<#if MC.DRIVE_NUMBER != "1">
  <#assign TimerFreq2 = _last_word(MC.M2_PWM_TIMER_SELECTION)+"Freq_Value" >
  <#assign ADV_TIM_CLKFreq2 = RCC.get(TimerFreq2)?number>
  <#assign TimerDivider2 = Fx_TIM_Div(ADV_TIM_CLKFreq2,(MC.M2_PWM_FREQUENCY)?number) >
</#if>
<#-- For futur use 
<#assign ADC12outputFreq ${RCC.get("ADC12outputFreq_Value")}>
<#assign ADC34outputFreq ${RCC.get("ADC34outputFreq_Value")}>
-->
#define SYSCLK_FREQ      ${SYSCLKFreq}uL
#define TIM_CLOCK_DIVIDER  ${TimerDivider} 
#define ADV_TIM_CLK_MHz  ${(ADV_TIM_CLKFreq/(1000000*TimerDivider))?floor} /* Actual TIM clk including Timer clock divider*/
#define ADC_CLK_MHz     ${(SYSCLKFreq/(MC.ADC_CLOCK_WB_DIV?number*1000000))?floor}
#define HALL_TIM_CLK    ${SYSCLKFreq}uL
#define APB1TIM_FREQ  ${APB1TIM_Freq}uL
 <#if MC.DRIVE_NUMBER != "1">
#define TIM_CLOCK_DIVIDER2  ${TimerDivider2}
#define ADV_TIM_CLK_MHz2  ${(ADV_TIM_CLKFreq2/(1000000*TimerDivider2))?floor} /* Actual TIM clk including Timer clock divider*/
#define ADC_CLK_MHz2      ${(SYSCLKFreq/(MC.ADC_CLOCK_WB_DIV2?number*1000000))?floor}
#define HALL_TIM_CLK2     ${SYSCLKFreq}uL
 </#if>

/*************************  IRQ Handler Mapping  *********************/
<#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
#define TIMx_UP_M1_IRQHandler TIM1_UP_TIM16_IRQHandler
#define TIMx_BRK_M1_IRQHandler TIM1_BRK_TIM15_IRQHandler
#define R1_DMAx_M1_IRQHandler DMA1_Channel4_IRQHandler
</#if>
<#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M1_PWM_TIMER_SELECTION == 'TIM8'>
#define TIMx_UP_M1_IRQHandler TIM8_UP_IRQHandler
#define TIMx_BRK_M1_IRQHandler TIM8_BRK_IRQHandler
#define R1_DMAx_M1_IRQHandler DMA2_Channel2_IRQHandler
#define MC1USETIM8
</#if>
<#if FOC>
<#if MC.DRIVE_NUMBER != "1">
<#if MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M2_PWM_TIMER_SELECTION == 'TIM1'>
#define TIMx_UP_M2_IRQHandler TIM1_UP_TIM16_IRQHandler
#define TIMx_BRK_M2_IRQHandler TIM1_BRK_TIM15_IRQHandler
#define R1_DMAx_M2_IRQHandler DMA1_Channel4_IRQHandler
</#if>
<#if MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M2_PWM_TIMER_SELECTION == 'TIM8'>
#define TIMx_UP_M2_IRQHandler TIM8_UP_IRQHandler
#define TIMx_BRK_M2_IRQHandler TIM8_BRK_IRQHandler
#define R1_DMAx_M2_IRQHandler DMA2_Channel2_IRQHandler
</#if>
</#if>
</#if>
<#if SIX_STEP>
<#if MC.DRIVE_MODE == "VM">
#define VM
<#else> <#-- "CM" -->
#define CM
</#if>
<#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
#define PERIOD_COMM_Handler                     ${TimerHandler(_last_word(MC.LF_TIMER_SELECTION))}
<#elseif MC.M1_SPEED_SENSOR == "SENSORLESS_COMP">
<#elseif MC.M1_SPEED_SENSOR == "HALL_SENSORS">
<#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
#define	COMM_EVENT_M1_Handler                   TIM1_TRG_COM_TIM17_IRQHandler
</#if>
<#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M1_PWM_TIMER_SELECTION == 'TIM8'>
#define	COMM_EVENT_M1_Handler                   TIM8_TRG_COM_IRQHandler
</#if>
#define	HALL_TIM_M1_IRQHandler                  ${TimerHandler(_last_word(MC.M1_HALL_TIMER_SELECTION))}
</#if>
#define	ADC_ConvCplt_Handler                    ADC1_IRQHandler
</#if>

/*************************  ADC Physical characteristics  *********************/	
#define ADC_TRIG_CONV_LATENCY_CYCLES 3.5
#define ADC_SAR_CYCLES 13u

#define M1_VBUS_SW_FILTER_BW_FACTOR      6u
<#if MC.DRIVE_NUMBER != "1">
#define M2_VBUS_SW_FILTER_BW_FACTOR     6u
</#if>
<#if MC.ESC_ENABLE>
/* ESC parameters conversion */
/* 1060 us ON time is the minimum value to spin the motor*/
/* 1860 us ON time is the target for the maximum speed */
#define ESC_TON_MIN (APB1TIM_FREQ / 1000000 * 1060 )
#define ESC_TON_MAX (APB1TIM_FREQ / 1000000 * 1860 )
/* Arming : if 900 us <= TOn < 1060 us for 200 us, ESC is armed */
#define ESC_TON_ARMING ((APB1TIM_FREQ /1000000) * 900 )
</#if>

#define OPAMP1_InvertingInput_PC5         LL_OPAMP_INPUT_INVERT_IO0
#define OPAMP1_InvertingInput_PA3         LL_OPAMP_INPUT_INVERT_IO1 
#define OPAMP1_InvertingInput_PGA         LL_OPAMP_INPUT_INVERT_CONNECT_NO
#define OPAMP1_InvertingInput_FOLLOWER    LL_OPAMP_MODE_FOLLOWER
#define OPAMP2_InvertingInput_PC5         LL_OPAMP_INPUT_INVERT_IO0
#define OPAMP2_InvertingInput_PA5         LL_OPAMP_INPUT_INVERT_IO1 
#define OPAMP2_InvertingInput_PGA         LL_OPAMP_INPUT_INVERT_CONNECT_NO
#define OPAMP2_InvertingInput_FOLLOWER    LL_OPAMP_MODE_FOLLOWER
#define OPAMP3_InvertingInput_PB10        LL_OPAMP_INPUT_INVERT_IO0
#define OPAMP3_InvertingInput_PB2         LL_OPAMP_INPUT_INVERT_IO1 
#define OPAMP3_InvertingInput_PGA         LL_OPAMP_INPUT_INVERT_CONNECT_NO
#define OPAMP3_InvertingInput_FOLLOWER    LL_OPAMP_MODE_FOLLOWER
#define OPAMP4_InvertingInput_PB10        LL_OPAMP_INPUT_INVERT_IO0
#define OPAMP4_InvertingInput_PD8         LL_OPAMP_INPUT_INVERT_IO1 
#define OPAMP4_InvertingInput_PGA         LL_OPAMP_INPUT_INVERT_CONNECT_NO
#define OPAMP4_InvertingInput_FOLLOWER    LL_OPAMP_MODE_FOLLOWER

#define OPAMP1_NonInvertingInput_PA7      LL_OPAMP_INPUT_NONINVERT_IO1
#define OPAMP1_NonInvertingInput_PA5      LL_OPAMP_INPUT_NONINVERT_IO3
#define OPAMP1_NonInvertingInput_PA3      LL_OPAMP_INPUT_NONINVERT_IO2
#define OPAMP1_NonInvertingInput_PA1      LL_OPAMP_INPUT_NONINVERT_IO0
#define OPAMP2_NonInvertingInput_PD14     LL_OPAMP_INPUT_NONINVERT_IO1
#define OPAMP2_NonInvertingInput_PB14     LL_OPAMP_INPUT_NONINVERT_IO3
#define OPAMP2_NonInvertingInput_PB0      LL_OPAMP_INPUT_NONINVERT_IO2
#define OPAMP2_NonInvertingInput_PA7      LL_OPAMP_INPUT_NONINVERT_IO0
#define OPAMP3_NonInvertingInput_PB13     LL_OPAMP_INPUT_NONINVERT_IO1
#define OPAMP3_NonInvertingInput_PA5      LL_OPAMP_INPUT_NONINVERT_IO3
#define OPAMP3_NonInvertingInput_PA1      LL_OPAMP_INPUT_NONINVERT_IO2
#define OPAMP3_NonInvertingInput_PB0      LL_OPAMP_INPUT_NONINVERT_IO0
#define OPAMP4_NonInvertingInput_PD11     LL_OPAMP_INPUT_NONINVERT_IO1
#define OPAMP4_NonInvertingInput_PB11     LL_OPAMP_INPUT_NONINVERT_IO3
#define OPAMP4_NonInvertingInput_PA4      LL_OPAMP_INPUT_NONINVERT_IO2
#define OPAMP4_NonInvertingInput_PB13     LL_OPAMP_INPUT_NONINVERT_IO0

#define OPAMP1_PGAConnect_PC5             OPAMP_CSR_PGGAIN_3
#define OPAMP1_PGAConnect_PA3             (OPAMP_CSR_PGGAIN_3|OPAMP_CSR_PGGAIN_2)
#define OPAMP2_PGAConnect_PC5             OPAMP_CSR_PGGAIN_3
#define OPAMP2_PGAConnect_PA5             (OPAMP_CSR_PGGAIN_3|OPAMP_CSR_PGGAIN_2)
#define OPAMP3_PGAConnect_PB10            OPAMP_CSR_PGGAIN_3
#define OPAMP3_PGAConnect_PB2             (OPAMP_CSR_PGGAIN_3|OPAMP_CSR_PGGAIN_2)
#define OPAMP4_PGAConnect_PB10            OPAMP_CSR_PGGAIN_3
#define OPAMP4_PGAConnect_PD8             (OPAMP_CSR_PGGAIN_3|OPAMP_CSR_PGGAIN_2)

#define COMP1_InvertingInput_PA0          LL_COMP_INPUT_MINUS_IO1
#define COMP2_InvertingInput_PA2          LL_COMP_INPUT_MINUS_IO1
#define COMP3_InvertingInput_PD15         LL_COMP_INPUT_MINUS_IO1
#define COMP3_InvertingInput_PB12         LL_COMP_INPUT_MINUS_IO2
#define COMP4_InvertingInput_PE8          LL_COMP_INPUT_MINUS_IO1
#define COMP4_InvertingInput_PB2          LL_COMP_INPUT_MINUS_IO2
#define COMP5_InvertingInput_PD13         LL_COMP_INPUT_MINUS_IO1
#define COMP5_InvertingInput_PB10         LL_COMP_INPUT_MINUS_IO2
#define COMP6_InvertingInput_PD10         LL_COMP_INPUT_MINUS_IO1
#define COMP6_InvertingInput_PB15         LL_COMP_INPUT_MINUS_IO2
#define COMP7_InvertingInput_PC0          LL_COMP_INPUT_MINUS_IO1

#define COMPX_InvertingInput_DAC1        LL_COMP_INPUT_MINUS_DAC1_CH1
#define COMPX_InvertingInput_DAC2        LL_COMP_INPUT_MINUS_DAC1_CH2
#define COMPX_InvertingInput_VREF        LL_COMP_INPUT_MINUS_VREFINT
#define COMPX_InvertingInput_VREF_1_4    LL_COMP_INPUT_MINUS_1_4VREFINT
#define COMPX_InvertingInput_VREF_1_2    LL_COMP_INPUT_MINUS_1_2VREFINT
#define COMPX_InvertingInput_VREF_3_4    LL_COMP_INPUT_MINUS_3_4VREFINT

#define COMP1_NonInvertingInput_PA1    LL_COMP_INPUT_PLUS_IO1
#define COMP2_NonInvertingInput_PA3    LL_COMP_INPUT_PLUS_IO2
#define COMP2_NonInvertingInput_PA7    LL_COMP_INPUT_PLUS_IO1
#define COMP3_NonInvertingInput_PB14   LL_COMP_INPUT_PLUS_IO1
#define COMP3_NonInvertingInput_PD14   LL_COMP_INPUT_PLUS_IO2
#define COMP4_NonInvertingInput_PB0    LL_COMP_INPUT_PLUS_IO1
#define COMP4_NonInvertingInput_PE7    LL_COMP_INPUT_PLUS_IO2
#define COMP5_NonInvertingInput_PB13   LL_COMP_INPUT_PLUS_IO2
#define COMP5_NonInvertingInput_PD12   LL_COMP_INPUT_PLUS_IO1
#define COMP6_NonInvertingInput_PB11   LL_COMP_INPUT_PLUS_IO2
#define COMP6_NonInvertingInput_PD11   LL_COMP_INPUT_PLUS_IO1
#define COMP7_NonInvertingInput_PC1    LL_COMP_INPUT_PLUS_IO2
#define COMP7_NonInvertingInput_PA0    LL_COMP_INPUT_PLUS_IO1

/* USER CODE BEGIN Additional parameters */

/* USER CODE END Additional parameters */  

#endif /*__PARAMETERS_CONVERSION_F30X_H*/

/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
