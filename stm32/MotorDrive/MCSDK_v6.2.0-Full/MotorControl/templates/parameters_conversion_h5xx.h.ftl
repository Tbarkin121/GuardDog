<#ftl>
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
/**
  ******************************************************************************
  * @file    parameters_conversion_h5xx.h
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file implements the Parameter conversion on the base
  *          of stdlib H5xx for the first drive
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
#ifndef __PARAMETERS_CONVERSION_H5XX_H
#define __PARAMETERS_CONVERSION_H5XX_H

#include "pmsm_motor_parameters.h"
#include "power_stage_parameters.h"
#include "drive_parameters.h"
#include "mc_math.h"

/************************* CPU & ADC PERIPHERAL CLOCK CONFIG ******************/
<#function Fx_TIM_Div TimFreq PWMFreq>
    <#list [1,2,3,4,5] as divider>
       <#if (TimFreq/(PWMFreq *divider)) < 65535 >
            <#return divider >
        </#if>
    </#list>
    <#return 1 >
</#function>


<#assign SYSCLKFreq = RCC.get("SYSCLKFreq_VALUE")?number>
<#assign AUXTIMFreq = RCC.get("APB1TimFreq_Value")?number>
<#assign PWMTIMFreq = RCC.get("APB2TimFreq_Value")?number>
<#assign ADCFreq = RCC.get("ADCFreq_Value")?number>
<#assign TimerDivider = Fx_TIM_Div(SYSCLKFreq,(MC.M1_PWM_FREQUENCY)?number) >
<#assign TimerDivider2 = Fx_TIM_Div(SYSCLKFreq,(MC.M2_PWM_FREQUENCY)?number) >

#define SYSCLK_FREQ      ${SYSCLKFreq}uL
#define TIM_CLOCK_DIVIDER  ${TimerDivider}
<#-- Timer Auxiliary must be half of the PWM timer, because it is the case for high end H5-->
#define TIMAUX_CLOCK_DIVIDER (TIM_CLOCK_DIVIDER<#if AUXTIMFreq == PWMTIMFreq>*2</#if>)
#define ADV_TIM_CLK_MHz  ${(PWMTIMFreq/(1000000))?floor}/TIM_CLOCK_DIVIDER
#define ADC_CLK_MHz     ${(ADCFreq/(1000000))?floor}
#define HALL_TIM_CLK    ${AUXTIMFreq}uL

 <#if MC.DRIVE_NUMBER != "1">
#define TIM_CLOCK_DIVIDER2  ${TimerDivider2} //Not used, both motors configured to same freq
<#-- Timer Auxiliary must be half of the PWM timer, because it is the case for high end H5-->
#define TIMAUX_CLOCK_DIVIDER2 (TIM_CLOCK_DIVIDER2<#if AUXTIMFreq == PWMTIMFreq>*2</#if>)
#define ADV_TIM_CLK_MHz2  ${(PWMTIMFreq/(1000000))?floor}/TIM_CLOCK_DIVIDER2
#define ADC_CLK_MHz2      ${(ADCFreq/(4*1000000))?floor}
#define HALL_TIM_CLK2     ${AUXTIMFreq}uL
 </#if>

/*************************  IRQ Handler Mapping  *********************/
<#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
#define TIMx_UP_M1_IRQHandler TIM1_UP_IRQHandler
#define TIMx_BRK_M1_IRQHandler TIM1_BRK_IRQHandler
</#if>


/**********  AUXILIARY TIMER (SINGLE SHUNT) *************/
/* Defined here for legacy purposes */
#define R1_PWM_AUX_TIM                  TIM4

/*************************  ADC Physical characteristics  ************/			
#define ADC_TRIG_CONV_LATENCY_CYCLES 3.5
#define ADC_SAR_CYCLES 12.5

#define M1_VBUS_SW_FILTER_BW_FACTOR      10u
<#if MC.DRIVE_NUMBER != "1">
#define M2_VBUS_SW_FILTER_BW_FACTOR      6u
</#if>

#endif /*__PARAMETERS_CONVERSION_H5XX_H*/

/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
