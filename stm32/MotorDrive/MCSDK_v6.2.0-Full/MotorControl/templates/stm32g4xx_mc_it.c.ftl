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
  <#if MC??>
  <#else>
  <#stop "No MotorControl SW IP data found">
  </#if>
</#if>

<#assign FOC = MC.M1_DRIVE_TYPE == "FOC" || MC.M2_DRIVE_TYPE == "FOC">
<#assign SIX_STEP = MC.M1_DRIVE_TYPE == "SIX_STEP" || MC.M2_DRIVE_TYPE == "SIX_STEP">
<#assign M1_ENCODER = (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M2_ENCODER = (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M1_HALL_SENSOR = (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
<#assign M2_HALL_SENSOR = (MC.M2_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
<#if MC.M1_CS_OPAMP_PHASE_SHARED!="NONE"><#assign M1_PHASE_SHARED = MC.M1_CS_OPAMP_PHASE_SHARED><#else><#assign M1_PHASE_SHARED = MC.M1_CS_ADC_PHASE_SHARED></#if>
<#if MC.M2_CS_OPAMP_PHASE_SHARED!="NONE"><#assign M2_PHASE_SHARED = MC.M2_CS_OPAMP_PHASE_SHARED><#else><#assign M2_PHASE_SHARED = MC.M2_CS_ADC_PHASE_SHARED></#if>
<#if MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT'> 
  <#if M1_PHASE_SHARED!="V"><#assign M1_ADC = MC.M1_CS_ADC_V><#else><#assign M1_ADC = MC.M1_CS_ADC_U></#if>
<#else><#-- (M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT') || (M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  <#assign M1_ADC = MC.M1_CS_ADC_U>
</#if><#-- M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT' -->
<#if MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT'> 
  <#if M2_PHASE_SHARED!="V"><#assign M2_ADC = MC.M2_CS_ADC_V><#else><#assign M2_ADC = MC.M2_CS_ADC_U></#if>
<#else><#-- (M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT') || (M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  <#assign M2_ADC = MC.M2_CS_ADC_U>
</#if><#-- M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT' -->

<#if MC.M1_DRIVE_TYPE == "FOC">
  <#if MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT'> 
    <#assign M1_CS_TOPO ="R3_"+MC.M1_CS_ADC_NUM >
  <#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>  
    <#assign M1_CS_TOPO ="R1" >
  <#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'> 
    <#assign M1_CS_TOPO ="ICS" >
  </#if>
</#if>
<#if MC.M2_DRIVE_TYPE == "FOC">
  <#if MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT'> 
    <#assign M2_CS_TOPO ="R3_"+MC.M2_CS_ADC_NUM >
  <#elseif ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>  
    <#assign M2_CS_TOPO ="R1" >
  <#elseif MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'> 
    <#assign M2_CS_TOPO ="ICS" >
  </#if>
</#if>

/**
  ******************************************************************************
  * @file    stm32g4xx_mc_it.c 
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   Main Interrupt Service Routines.
  *          This file provides exceptions handler and peripherals interrupt 
  *          service routine related to Motor Control for the STM32G4 Family.
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
  * @ingroup STM32G4xx_IRQ_Handlers
  */

/* Includes ------------------------------------------------------------------*/
#include "mc_config.h"
<#-- Specific to FOC algorithm usage -->
#include "mc_type.h"
//cstat -MISRAC2012-Rule-3.1
#include "mc_tasks.h"
//cstat +MISRAC2012-Rule-3.1
#include "motorcontrol.h"
  <#if (MC.START_STOP_BTN == true) || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")>
#include "stm32g4xx_ll_exti.h"
  </#if>
#include "stm32g4xx_hal.h"
#include "stm32g4xx.h"
<#if MC.MCP_EN>
#include "mcp_config.h"  
</#if>

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/** @addtogroup MCSDK
  * @{
  */

/** @addtogroup STM32G4xx_IRQ_Handlers STM32G4xx IRQ Handlers
  * @{
  */
  
/* USER CODE BEGIN PRIVATE */
  
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/

#define SYSTICK_DIVIDER (SYS_TICK_FREQUENCY/1000U)

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/* USER CODE END PRIVATE */
<#-- Specific to 6_STEP algorithm usage -->
<#if SIX_STEP>
void PERIOD_COMM_IRQHandler(void);
void BEMF_READING_IRQHandler(void);
</#if><#-- SIX_STEP -->
<#-- Specific to FOC algorithm usage -->

<#if MC.MCP_OVER_STLNK_EN>
void DebugMon_Handler(void)
{
  /* USER CODE BEGIN DebugMonitor_IRQn 0 */

  /* USER CODE END DebugMonitor_IRQn 0 */

  STLNK_HWDataTransmittedIT(&STLNK);
  
  /* USER CODE BEGIN DebugMonitor_IRQn 1 */

  /* USER CODE END DebugMonitor_IRQn 1 */
}
</#if><#-- MC.MCP_OVER_STLNK_EN -->
<#if FOC>
/* Public prototypes of IRQ handlers called from assembly code ---------------*/
  <#if M1_ADC == "ADC1" || M1_ADC == "ADC2" || M2_ADC == "ADC1" || M2_ADC == "ADC2">
void ADC1_2_IRQHandler(void);
  </#if><#-- M1_ADC == "ADC1" || M1_ADC == "ADC2" || M2_ADC == "ADC1"
          || M2_ADC == "ADC2" -->
  <#if M1_ADC == "ADC3" || M2_ADC == "ADC3">
void ADC3_IRQHandler(void);
  </#if><#-- M1_ADC == "ADC3" || M2_ADC == "ADC3" -->
  <#if M1_ADC == "ADC4" || M2_ADC == "ADC4">
void ADC4_IRQHandler(void);
  </#if><#-- M1_ADC == "ADC4" || M2_ADC == "ADC4" -->
  <#if M1_ADC == "ADC5" || M2_ADC == "ADC5">
void ADC5_IRQHandler(void);
  </#if><#-- M1_ADC == "ADC5" || M2_ADC == "ADC5" -->
</#if><#-- Specific to FOC algorithm usage -->  
void TIMx_UP_M1_IRQHandler(void);
void TIMx_BRK_M1_IRQHandler(void);
<#if FOC>
	<#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
	|| ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) > 
    <#if _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM1" || _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM1">
void DMA1_Channel1_IRQHandler(void);
    </#if><#-- _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM1" || _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM1" -->
    <#if _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM8" || _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM8">
void DMA2_Channel1_IRQHandler(void);
    </#if><#-- _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM8" || _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM8" -->
  </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
  || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
 </#if><#-- Specific to FOC algorithm usage -->  

<#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
/**
  * @brief  This function handles BEMF sensing interrupt request.
  * @param[in] None
  */
void BEMF_READING_IRQHandler(void)
{
  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 0 */

  /* USER CODE END CURRENT_REGULATION_IRQn 0 */
  <#if MC.PHASE_U_BEMF_ADC == "ADC1" || MC.PHASE_V_BEMF_ADC == "ADC1" || MC.PHASE_W_BEMF_ADC == "ADC1">
  if(LL_ADC_IsActiveFlag_JEOC(ADC1) && LL_ADC_IsEnabledIT_JEOC(ADC1))
  {
  /* Clear Flags */
    LL_ADC_ClearFlag_JEOC(ADC1);
    BADC_IsZcDetected(&Bemf_ADC_M1, &PWM_Handle_M1._Super);
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- MC.PHASE_U_BEMF_ADC == "ADC1" || MC.PHASE_V_BEMF_ADC == "ADC1" || MC.PHASE_W_BEMF_ADC == "ADC1" -->
  <#if MC.PHASE_U_BEMF_ADC == "ADC2" || MC.PHASE_V_BEMF_ADC == "ADC2" || MC.PHASE_W_BEMF_ADC == "ADC2">
  if(LL_ADC_IsActiveFlag_JEOC(ADC2) && LL_ADC_IsEnabledIT_JEOC(ADC2))
  {
  /* Clear Flags */
    LL_ADC_ClearFlag_JEOC(ADC2);
    BADC_IsZcDetected(&Bemf_ADC_M1, &PWM_Handle_M1._Super);
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- MC.PHASE_U_BEMF_ADC == "ADC2" || MC.PHASE_V_BEMF_ADC == "ADC2" || MC.PHASE_W_BEMF_ADC == "ADC2" -->
  <#if MC.PHASE_U_BEMF_ADC == "ADC3" || MC.PHASE_V_BEMF_ADC == "ADC3" || MC.PHASE_W_BEMF_ADC == "ADC3">
  if(LL_ADC_IsActiveFlag_JEOC(ADC3) && LL_ADC_IsEnabledIT_JEOC(ADC3))
  {
  /* Clear Flags */
    LL_ADC_ClearFlag_JEOC(ADC3);
    BADC_IsZcDetected(&Bemf_ADC_M1, &PWM_Handle_M1._Super);
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- MC.PHASE_U_BEMF_ADC == "ADC3" || MC.PHASE_V_BEMF_ADC == "ADC3" || MC.PHASE_W_BEMF_ADC == "ADC3" -->
  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 1 */

  /* USER CODE END CURRENT_REGULATION_IRQn 1 */

  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 2 */

  /* USER CODE END CURRENT_REGULATION_IRQn 2 */
}

/**
  * @brief     LFtimer interrupt handler.
  * @param[in] None
  */
void PERIOD_COMM_IRQHandler(void)
{
  /* TIM Update event */

  if(LL_TIM_IsActiveFlag_UPDATE(Bemf_ADC_M1.pParams_str->LfTim) && LL_TIM_IsEnabledIT_UPDATE(Bemf_ADC_M1.pParams_str->LfTim))
  {
    LL_TIM_ClearFlag_UPDATE(Bemf_ADC_M1.pParams_str->LfTim);
    BADC_StepChangeEvent(&Bemf_ADC_M1, 0, &PWM_Handle_M1._Super);
    (void)TSK_HighFrequencyTask();
  }
  else
  {
    /* Nothing to do */
  }
}
</#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->

<#if (M1_ENCODER == true) || (M1_HALL_SENSOR == true)>
void SPD_TIM_M1_IRQHandler(void);
</#if><#-- (M1_ENCODER == true) || (M1_HALL_SENSOR == true) -->

<#if FOC>
  <#if MC.DRIVE_NUMBER != "1">
void TIMx_UP_M2_IRQHandler(void);
void TIMx_BRK_M2_IRQHandler(void);
    <#if (M2_ENCODER == true) || (M2_HALL_SENSOR == true)>
void SPD_TIM_M2_IRQHandler(void);
    </#if><#-- (M2_ENCODER == true) || (M2_HALL_SENSOR == true) -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- Specific to FOC algorithm usage -->
void HardFault_Handler(void);
void SysTick_Handler(void);
<#if MC.START_STOP_BTN == true>
void ${EXT_IRQHandler(_last_word(MC.START_STOP_GPIO_PIN)?number)}(void);
</#if><#-- MC.START_STOP_BTN == true -->

<#if FOC>
  <#if M1_ADC == "ADC1" || M1_ADC == "ADC2" || M2_ADC == "ADC1" || M2_ADC == "ADC2">
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief  This function handles ADC1/ADC2 interrupt request.
  * @param  None
  */
void ADC1_2_IRQHandler(void)
{
  /* USER CODE BEGIN ADC1_2_IRQn 0 */

  /* USER CODE END ADC1_2_IRQn 0 */
  
    <#if MC.DRIVE_NUMBER != "1">
      <#if (M1_ADC == "ADC1" && M2_ADC == "ADC2")
        || (M1_ADC == "ADC2" && M2_ADC == "ADC1")>
  /* Shared IRQ management - begin */
  if (LL_ADC_IsActiveFlag_JEOS(${M1_ADC}))
  {
      </#if><#-- (M1_ADC == "ADC1" && M2_ADC == "ADC2")
        || (M1_ADC == "ADC2" && M2_ADC == "ADC1") -->
    </#if><#-- MC.DRIVE_NUMBER > 1 --> 
    <#if M1_ADC == "ADC1" || M1_ADC == "ADC2">
    /* Clear Flags M1 */
    LL_ADC_ClearFlag_JEOS(${M1_ADC});
    </#if><#-- M1_ADC == "ADC1" || M1_ADC == "ADC2" -->

    <#if MC.DRIVE_NUMBER != "1">
      <#if (M1_ADC == "ADC1" && M2_ADC == "ADC2") || 
         (M1_ADC == "ADC2" && M2_ADC == "ADC1")>
  }
  else if (LL_ADC_IsActiveFlag_JEOS(${M2_ADC}))
  {
      </#if><#-- (M1_ADC == "ADC1" && M2_ADC == "ADC2") || 
         (M1_ADC == "ADC2" && M2_ADC == "ADC1") -->
      <#-- In case of same ADC for both motors, we must not clear the interrupt twice -->
      <#if M2_ADC == "ADC1" || M2_ADC == "ADC2">
        <#if M1_ADC != M2_ADC>
    /* Clear Flags M2 */
    LL_ADC_ClearFlag_JEOS(${M2_ADC});
        </#if><#-- M1_ADC != M2_ADC -->
      </#if><#-- M2_ADC == "ADC1" || M2_ADC == "ADC2" -->
      <#if (M1_ADC == "ADC1" && M2_ADC == "ADC2") || 
         (M1_ADC == "ADC2" && M2_ADC == "ADC1")>
  }
      </#if><#-- M1_ADC == "ADC1" && M2_ADC == "ADC2") || 
         (M1_ADC == "ADC2" && M2_ADC == "ADC1") -->
    </#if><#-- MC.DRIVE_NUMBER > 1 -->
  (void)TSK_HighFrequencyTask();
  


 /* USER CODE BEGIN HighFreq */

 /* USER CODE END HighFreq  */
 
 /* USER CODE BEGIN ADC1_2_IRQn 1 */

 /* USER CODE END ADC1_2_IRQn 1 */
}
  </#if><#-- M1_ADC == "ADC1" || M1_ADC == "ADC2" || M2_ADC == "ADC1"
          || M2_ADC == "ADC2" -->

  <#if M1_ADC == "ADC3" || M2_ADC == "ADC3">
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief  This function handles ADC3 interrupt request.
  * @param  None
  */
void ADC3_IRQHandler(void)
{
 /* USER CODE BEGIN ADC3_IRQn 0 */

 /* USER CODE END  ADC3_IRQn 0 */

  /* Clear Flags */
  LL_ADC_ClearFlag_JEOS(ADC3);
  /* Highfrequency task ADC3 */
  (void)TSK_HighFrequencyTask();


 /* USER CODE BEGIN ADC3_IRQn 1 */

 /* USER CODE END  ADC3_IRQn 1 */
}
  </#if><#-- M1_ADC == "ADC3" || M2_ADC == "ADC3" -->

  <#if M1_ADC == "ADC4" || M2_ADC == "ADC4">
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief  This function handles ADC4 interrupt request.
  * @param  None
  */
void ADC4_IRQHandler(void)
{
 /* USER CODE BEGIN ADC4_IRQn 0 */

 /* USER CODE END  ADC4_IRQn 0 */
 
  /* Clear Flags */
  LL_ADC_ClearFlag_JEOS(ADC4);
 
  /* Highfrequency task ADC4 */
  (void)TSK_HighFrequencyTask();

 /* USER CODE BEGIN ADC4_IRQn 1 */

 /* USER CODE END  ADC4_IRQn 1 */
}
  </#if><#-- M1_ADC == "ADC3" || M2_ADC == "ADC3" -->

  <#if M1_ADC == "ADC5" || M2_ADC == "ADC5">
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief  This function handles ADC5 interrupt request.
  * @param  None
  */
void ADC5_IRQHandler(void)
{
 /* USER CODE BEGIN ADC5_IRQn 0 */

 /* USER CODE END  ADC5_IRQn 0 */
 
  /* Clear Flags */
  LL_ADC_ClearFlag_JEOS(ADC5);

  /* Highfrequency task ADC5 */
  (void)TSK_HighFrequencyTask();

 /* USER CODE BEGIN ADC5_IRQn 1 */

 /* USER CODE END  ADC5_IRQn 1 */
}
  </#if><#-- M1_ADC == "ADC5" || M2_ADC == "ADC5" -->
</#if><#-- FOC -->

#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief  This function handles first motor TIMx Update interrupt request.
  * @param  None
  */
void TIMx_UP_M1_IRQHandler(void)
{
 /* USER CODE BEGIN TIMx_UP_M1_IRQn 0 */

 /* USER CODE END  TIMx_UP_M1_IRQn 0 */
 
  LL_TIM_ClearFlag_UPDATE(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
<#if FOC>
  (void)${M1_CS_TOPO}_TIMx_UP_IRQHandler(&PWM_Handle_M1);
  <#if MC.DRIVE_NUMBER != "1">
  TSK_DualDriveFIFOUpdate(M1);
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->
<#if SIX_STEP>
  (void)TSK_HighFrequencyTask();
</#if><#-- SIX_STEP -->
 /* USER CODE BEGIN TIMx_UP_M1_IRQn 1 */

 /* USER CODE END  TIMx_UP_M1_IRQn 1 */
}


void TIMx_BRK_M1_IRQHandler(void)
{
  /* USER CODE BEGIN TIMx_BRK_M1_IRQn 0 */

  /* USER CODE END TIMx_BRK_M1_IRQn 0 */
  if (0U == LL_TIM_IsActiveFlag_BRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)}))
  {
    /* Nothing to do */
  }
  else
  {
    LL_TIM_ClearFlag_BRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
<#if FOC>
    <#if (MC.M1_OCP_TOPOLOGY != "NONE") &&  (MC.M1_OCP_DESTINATION == "TIM_BKIN")>
    PWMC_OCP_Handler(&PWM_Handle_M1._Super);
    <#elseif (MC.M1_DP_TOPOLOGY != "NONE") &&  (MC.M1_DP_DESTINATION == "TIM_BKIN")>
    PWMC_DP_Handler(&PWM_Handle_M1._Super);
    <#else>
    PWMC_OVP_Handler(&PWM_Handle_M1._Super, ${_last_word(MC.M1_PWM_TIMER_SELECTION)});
    </#if>
</#if><#-- FOC -->

<#if SIX_STEP>
<#if  MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
    (void)SixPwm_BRK_IRQHandler(&PWM_Handle_M1);
  <#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
    (void)ThreePwm_BRK_IRQHandler(&PWM_Handle_M1);
  </#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
</#if><#-- SIX_STEP -->
  }
  
  if (0U == LL_TIM_IsActiveFlag_BRK2(${_last_word(MC.M1_PWM_TIMER_SELECTION)}))
  {
    /* Nothing to do */
  }
  else
  {
    LL_TIM_ClearFlag_BRK2(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
<#if FOC>
    <#if (MC.M1_OCP_TOPOLOGY != "NONE") &&  (MC.M1_OCP_DESTINATION == "TIM_BKIN2")>
    PWMC_OCP_Handler(&PWM_Handle_M1._Super);
    <#elseif (MC.M1_DP_TOPOLOGY != "NONE") &&  (MC.M1_DP_DESTINATION == "TIM_BKIN2")>
    PWMC_DP_Handler(&PWM_Handle_M1._Super);
    <#else>
    PWMC_OVP_Handler(&PWM_Handle_M1._Super, ${_last_word(MC.M1_PWM_TIMER_SELECTION)});
    </#if>
</#if><#-- FOC -->

<#if SIX_STEP>
<#if  MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
    (void)SixPwm_BRK_IRQHandler(&PWM_Handle_M1);
  <#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
    (void)ThreePwm_BRK_IRQHandler(&PWM_Handle_M1);
  </#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
</#if><#-- SIX_STEP -->
  }
  /* Systick is not executed due low priority so is necessary to call MC_Scheduler here */
  MC_Scheduler();

  /* USER CODE BEGIN TIMx_BRK_M1_IRQn 1 */

  /* USER CODE END TIMx_BRK_M1_IRQn 1 */
}
<#if FOC>
	<#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
	|| ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>  
    <#if _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM1" || _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM1">
      <#if _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM1">
        <#assign MOTOR = "M1">
      <#else>
        <#assign MOTOR = "M2">
      </#if>
      <#if (((MOTOR == "M1") && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')))
         || ((MOTOR == "M2") && ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))))> 
void DMA1_Channel1_IRQHandler (void)
{
  uint32_t tempReg1 = LL_DMA_IsActiveFlag_HT1(DMA1);
  uint32_t tempReg2 = LL_DMA_IsEnabledIT_HT(DMA1, LL_DMA_CHANNEL_1);
  if ((tempReg1 != 0U) && (tempReg2 != 0U))
  {
    (void)R1_DMAx_HT_IRQHandler(&PWM_Handle_${MOTOR});
    LL_DMA_ClearFlag_HT1(DMA1);
  }
  else
  {
    /* Nothing to do */
  }
  
  if (LL_DMA_IsActiveFlag_TC1(DMA1) != 0U)
  {
    LL_DMA_ClearFlag_TC1(DMA1);
    (void)R1_DMAx_TC_IRQHandler(&PWM_Handle_${MOTOR});
  }
  else
  {
    /* Nothing to do */
  }
  
    /* USER CODE BEGIN DMA1_Channel1_IRQHandler */

    /* USER CODE END DMA1_Channel1_IRQHandler */
}
      </#if>
    </#if><#-- _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM1" || _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM1" -->

    <#if _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM8" || _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM8">
      <#if _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM8">
        <#assign MOTOR = "M1">
      <#else>
        <#assign MOTOR = "M2">
      </#if>
      <#if (((MOTOR == "M1") && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')))
         || ((MOTOR == "M2") && ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))))> 
void DMA2_Channel1_IRQHandler (void)
{
  if (LL_DMA_IsActiveFlag_HT1(DMA2) && LL_DMA_IsEnabledIT_HT(DMA2, LL_DMA_CHANNEL_1))
  {
    (void)R1_DMAx_HT_IRQHandler(&PWM_Handle_${MOTOR});
    LL_DMA_ClearFlag_HT1(DMA2);
  }
  else
  {
    /* Nothing to do */
  }
  
  if (LL_DMA_IsActiveFlag_TC1(DMA2))
  {
    LL_DMA_ClearFlag_TC1(DMA2);
    (void)R1_DMAx_TC_IRQHandler(&PWM_Handle_${MOTOR});
  }
  else
  {
    /* Nothing to do */
  }

    /* USER CODE BEGIN DMA2_Channel1_IRQHandler */

    /* USER CODE END DMA2_Channel1_IRQHandler */
}
      </#if>
    </#if><#-- _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM8" || _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM8" -->
  </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
	|| ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->

  <#if MC.DRIVE_NUMBER != "1">
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief  This function handles second motor TIMx Update interrupt request.
  * @param  None
  */
void TIMx_UP_M2_IRQHandler(void)
{
 /* USER CODE BEGIN TIMx_UP_M2_IRQn 0 */

 /* USER CODE END  TIMx_UP_M2_IRQn 0 */

    LL_TIM_ClearFlag_UPDATE(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
    (void)${M2_CS_TOPO}_TIMx_UP_IRQHandler(&PWM_Handle_M2);

    TSK_DualDriveFIFOUpdate(M2);

 /* USER CODE BEGIN TIMx_UP_M2_IRQn 1 */

 /* USER CODE END  TIMx_UP_M2_IRQn 1 */
}

void TIMx_BRK_M2_IRQHandler(void)
{
  /* USER CODE BEGIN TIMx_BRK_M2_IRQn 0 */

  /* USER CODE END TIMx_BRK_M2_IRQn 0 */
  if (0U == LL_TIM_IsActiveFlag_BRK(${_last_word(MC.M2_PWM_TIMER_SELECTION)}))
  {
    /* Nothing to do */
  }
  else
  {
    LL_TIM_ClearFlag_BRK(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
<#if (MC.M2_OCP_TOPOLOGY != "NONE") &&  (MC.M2_OCP_DESTINATION == "TIM_BKIN")>
    PWMC_OCP_Handler(&PWM_Handle_M2._Super);
<#elseif (MC.M2_DP_TOPOLOGY != "NONE") &&  (MC.M2_DP_DESTINATION == "TIM_BKIN")>
    PWMC_DP_Handler(&PWM_Handle_M2._Super);
<#else>
    PWMC_OVP_Handler(&PWM_Handle_M2._Super, ${_last_word(MC.M2_PWM_TIMER_SELECTION)});
</#if>
<#if SIX_STEP>
<#if  MC.M2_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
    (void)SixPwm_BRK_IRQHandler(&PWM_Handle_M2);
  <#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
    (void)ThreePwm_BRK_IRQHandler(&PWM_Handle_M2);
  </#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
</#if><#-- SIX_STEP -->

  /* USER CODE BEGIN BRK */

  /* USER CODE END BRK */

  }
  if (0U == LL_TIM_IsActiveFlag_BRK2(${_last_word(MC.M2_PWM_TIMER_SELECTION)}))
  {
    /* Nothing to do */
  }
  else
  {
    LL_TIM_ClearFlag_BRK2(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
<#if (MC.M2_OCP_TOPOLOGY != "NONE") &&  (MC.M2_OCP_DESTINATION == "TIM_BKIN2")>
    PWMC_OCP_Handler(&PWM_Handle_M2._Super);
<#elseif (MC.M2_DP_TOPOLOGY != "NONE") &&  (MC.M2_DP_DESTINATION == "TIM_BKIN2")>
    PWMC_DP_Handler(&PWM_Handle_M2._Super);
<#else>
    PWMC_OVP_Handler(&PWM_Handle_M2._Super, ${_last_word(MC.M2_PWM_TIMER_SELECTION)});
</#if>
  /* USER CODE BEGIN BRK2 */

  /* USER CODE END BRK2 */
  }
  /* Systick is not executed due low priority so is necessary to call MC_Scheduler here */
  MC_Scheduler();
  /* USER CODE BEGIN TIMx_BRK_M2_IRQn 1 */

  /* USER CODE END TIMx_BRK_M2_IRQn 1 */
}

  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->
<#if (M1_ENCODER == true) || (M1_HALL_SENSOR == true)>
/**
  * @brief  This function handles TIMx global interrupt request for M1 Speed Sensor.
  * @param  None
  */
void SPD_TIM_M1_IRQHandler(void)
{
  /* USER CODE BEGIN SPD_TIM_M1_IRQn 0 */

  /* USER CODE END SPD_TIM_M1_IRQn 0 */

  <#if M1_HALL_SENSOR == true>
  /* HALL Timer Update IT always enabled, no need to check enable UPDATE state */
  if (0U == LL_TIM_IsActiveFlag_UPDATE(HALL_M1.TIMx))
  {
    /* Nothing to do */
  }
  else
  {
    LL_TIM_ClearFlag_UPDATE(HALL_M1.TIMx);
    (void)HALL_TIMx_UP_IRQHandler(&HALL_M1);
    /* USER CODE BEGIN M1 HALL_Update */

    /* USER CODE END M1 HALL_Update   */
  }

  /* HALL Timer CC1 IT always enabled, no need to check enable CC1 state */
  if (LL_TIM_IsActiveFlag_CC1 (HALL_M1.TIMx) != 0U) 
  {
    LL_TIM_ClearFlag_CC1(HALL_M1.TIMx);
    (void)HALL_TIMx_CC_IRQHandler(&HALL_M1);
    /* USER CODE BEGIN M1 HALL_CC1 */

    /* USER CODE END M1 HALL_CC1 */
  }
  else
  {
  /* Nothing to do */
  }
  <#else><#-- M1_HALL_SENSOR != true -->
 /* Encoder Timer UPDATE IT is dynamicaly enabled/disabled, checking enable state is required */
  if (LL_TIM_IsEnabledIT_UPDATE (ENCODER_M1.TIMx) != 0U)
  {
    if (LL_TIM_IsActiveFlag_UPDATE (ENCODER_M1.TIMx) != 0U)
    {
      LL_TIM_ClearFlag_UPDATE(ENCODER_M1.TIMx);
      (void)ENC_IRQHandler(&ENCODER_M1);
      /* USER CODE BEGIN M1 ENCODER_Update */

      /* USER CODE END M1 ENCODER_Update   */
    }
    else
    {
      /* No other IT to manage for encoder config */
    }
  }
  else
  {
    /* No other IT to manage for encoder config */
  }
  </#if><#-- M1_HALL_SENSOR == true -->
  /* USER CODE BEGIN SPD_TIM_M1_IRQn 1 */

  /* USER CODE END SPD_TIM_M1_IRQn 1 */
}
</#if><#-- (M1_ENCODER == true) || (M1_HALL_SENSOR == true) -->
  
<#if FOC>
  <#if MC.DRIVE_NUMBER != "1">
    <#if (M2_ENCODER == true) || (M2_HALL_SENSOR == true)>

/**
  * @brief  This function handles TIMx global interrupt request for M2 Speed Sensor.
  * @param  None
  */
void SPD_TIM_M2_IRQHandler(void)
{
  /* USER CODE BEGIN SPD_TIM_M2_IRQn 0 */

  /* USER CODE END SPD_TIM_M2_IRQn 0 */

      <#if (M2_HALL_SENSOR == true)>
  /* HALL Timer Update IT always enabled, no need to check enable UPDATE state */
  if (LL_TIM_IsActiveFlag_UPDATE(HALL_M2.TIMx) != 0)
  {
    LL_TIM_ClearFlag_UPDATE(HALL_M2.TIMx);
    (void)HALL_TIMx_UP_IRQHandler(&HALL_M2);
    /* USER CODE BEGIN M2 HALL_Update */

    /* USER CODE END M2 HALL_Update   */
  }
  else
  {
    /* Nothing to do */
  }
  /* HALL Timer CC1 IT always enabled, no need to check enable CC1 state */
  if (LL_TIM_IsActiveFlag_CC1 (HALL_M2.TIMx))
  {
    LL_TIM_ClearFlag_CC1(HALL_M2.TIMx);
    (void)HALL_TIMx_CC_IRQHandler(&HALL_M2);
    /* USER CODE BEGIN M2 HALL_CC1 */

    /* USER CODE END M2 HALL_CC1 */
  }
  else
  {
  /* Nothing to do */
  }
      <#else><#-- M2_HALL_SENSOR != true -->
  /* Encoder Timer UPDATE IT is dynamicaly enabled/disabled, checking enable state is required */
  if (LL_TIM_IsEnabledIT_UPDATE (ENCODER_M2.TIMx) && LL_TIM_IsActiveFlag_UPDATE (ENCODER_M2.TIMx))
  {
    LL_TIM_ClearFlag_UPDATE(ENCODER_M2.TIMx);
    ENC_IRQHandler(&ENCODER_M2);
    /* USER CODE BEGIN M2 ENCODER_Update */

    /* USER CODE END M2 ENCODER_Update   */
  }
  else
  {
  /* No other IT to manage for encoder config */
  }
      </#if><#-- M2_HALL_SENSOR == true -->
  /* USER CODE BEGIN SPD_TIM_M2_IRQn 1 */

  /* USER CODE END SPD_TIM_M2_IRQn 1 */
}
    </#if><#-- (M2_ENCODER == true) || (M2_HALL_SENSOR == true) -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->

<#-- ST MCWB monitoring usage management (used when MC.SERIAL_COMMUNICATION == true) -->
<#if MC.MCP_OVER_UART_A_EN>
/**
  * @brief This function handles DMA_RX_A channel DMACH_RX_A global interrupt.
  */
//cstat !MISRAC2012-Rule-8.4
void ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A}(void)
{
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A} 0 */
  
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A} 0 */

  
  /* Buffer is ready by the HW layer to be processed */
  if (0U == LL_DMA_IsActiveFlag_TC(DMA_RX_A, DMACH_RX_A))
  {
    /* Nothing to do */
  }
  else
  {
    LL_DMA_ClearFlag_TC (DMA_RX_A, DMACH_RX_A);
    ASPEP_HWDataReceivedIT (&aspepOverUartA);
  }
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A} 1 */
  
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A} 1 */

}

/* This section is present only when MCP over UART_A is used */
/**
  * @brief  This function handles USART interrupt request.
  * @param  None
  */
//cstat !MISRAC2012-Rule-8.4
void ${MC.MCP_IRQ_HANDLER_UART_A}(void)
{
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_UART_A} 0 */
  
  /* USER CODE END ${MC.MCP_IRQ_HANDLER_UART_A} 0 */
    
  if (0U == LL_USART_IsActiveFlag_TC(USARTA))
  {
    /* Nothing to do */
  }
  else
  {
    /* LL_GPIO_SetOutputPin(GPIOC , LL_GPIO_PIN_6) */
    /* Disable the DMA channel to prepare the next chunck of data */
    LL_DMA_DisableChannel(DMA_TX_A, DMACH_TX_A);
    LL_USART_ClearFlag_TC (USARTA);
    /* Data Sent by UART */
    /* Need to free the buffer, and to check pending transfer */
    ASPEP_HWDataTransmittedIT (&aspepOverUartA);
    /* LL_GPIO_ResetOutputPin(GPIOC , LL_GPIO_PIN_6) */
  }

  uint32_t flags;
  uint32_t oreFlag;
  uint32_t feFlag;
  uint32_t neFlag;
  uint32_t errorMask;
  uint32_t activeIdleFlag;
  uint32_t isEnabledIdelFlag;
  oreFlag = LL_USART_IsActiveFlag_ORE(USARTA);
  feFlag = LL_USART_IsActiveFlag_FE(USARTA);
  neFlag = LL_USART_IsActiveFlag_NE(USARTA);
  errorMask = LL_USART_IsEnabledIT_ERROR(USARTA);
  
  flags = ((oreFlag | feFlag | neFlag) & errorMask);
  if (0U == flags)
  {
    /* Nothing to do */
  }
  else
  { /* Stopping the debugger will generate an OverRun error */
    WRITE_REG(USARTA->ICR, USART_ICR_FECF|USART_ICR_ORECF|USART_ICR_NECF);
    /* We disable ERROR interrupt to avoid to trig one Overrun IT per additional byte recevied */
    LL_USART_DisableIT_ERROR (USARTA);
    LL_USART_EnableIT_IDLE (USARTA);
  }

  activeIdleFlag = LL_USART_IsActiveFlag_IDLE (USARTA);
  isEnabledIdelFlag = LL_USART_IsEnabledIT_IDLE (USARTA);
  flags = activeIdleFlag & isEnabledIdelFlag;
  if (0U == flags)
  {
    /* Nothing to do */
  }
  else
  { /* Stopping the debugger will generate an OverRun error */
    LL_USART_DisableIT_IDLE (USARTA);
    /* Once the complete unexpected data are received, we enable back the error IT */
    LL_USART_EnableIT_ERROR (USARTA);
    /* To be sure we fetch the potential pendig data */
    /* We disable the DMA request, Read the dummy data, endable back the DMA request */
    LL_USART_DisableDMAReq_RX (USARTA);
    (void)LL_USART_ReceiveData8(USARTA);
    LL_USART_EnableDMAReq_RX (USARTA);
    ASPEP_HWDMAReset (&aspepOverUartA);
  }

  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_UART_A} 1 */
 
  /* USER CODE END ${MC.MCP_IRQ_HANDLER_UART_A} 1 */
}
</#if><#-- MC.MCP_OVER_UART_A_EN -->

<#if MC.MCP_OVER_UART_B_EN>
/**
  * @brief This function handles DMA_RX_B channel DMACH_RX_B global interrupt.
  */
//cstat !MISRAC2012-Rule-8.4
void ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B}(void)
{
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B} 0 */
  
  /* USER CODE END ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B} 0 */
    
  /* Buffer is ready by the HW layer to be processed */
  if (LL_DMA_IsActiveFlag_TC(DMA_RX_B, DMACH_RX_B))
  {
    LL_DMA_ClearFlag_TC (DMA_RX_B, DMACH_RX_B);
    ASPEP_HWDataReceivedIT (&aspepOverUartB);
  }
  else
  {
    /* Nothing to do */
  }
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B} 1 */
  
  /* USER CODE END ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B} 1 */
}

/* This section is present only when MCP over UART_B is used */
/**
  * @brief  This function handles USART interrupt request.
  * @param  None
  */
void USARTB_IRQHandler(void)
{
  /* USER CODE BEGIN USARTB_IRQn 0 */
  
  /* USER CODE END USARTB_IRQn 0 */
  if (0U == LL_USART_IsActiveFlag_TC(USARTB))
  {
    /* Nothing to do */
  }
  else
  {
    /* Disable the DMA channel to prepare the next chunck of data */
    LL_DMA_DisableChannel(DMA_TX_B, DMACH_TX_B);
    LL_USART_ClearFlag_TC (USARTB);
    /* Data Sent by UART */
    /* Need to free the buffer, and to check pending transfer */
    ASPEP_HWDataTransmittedIT (&aspepOverUartB);
  }

  uint32_t flags;
  uint32_t oreFlag;
  uint32_t feFlag;
  uint32_t neFlag;
  uint32_t errorMask;
  uint32_t activeIdleFlag;
  uint32_t isEnabledIdelFlag;
  oreFlag = LL_USART_IsActiveFlag_ORE (USARTA);
  feFlag = LL_USART_IsActiveFlag_FE (USARTA);
  neFlag = LL_USART_IsActiveFlag_NE (USARTA);
  errorMask = LL_USART_IsEnabledIT_ERROR (USARTA);
  flags = ((oreFlag | feFlag | neFlag) & errorMask);
  if (0U == flags)
  {
    /* Nothing to do */
  }
  else
  { /* Stopping the debugger will generate an OverRun error */
    WRITE_REG(USARTB->ICR, USART_ICR_FECF|USART_ICR_ORECF|USART_ICR_NECF);
    /* We disable ERROR interrupt to avoid to trig one Overrun IT per additional byte recevied */
    LL_USART_DisableIT_ERROR (USARTB);
    LL_USART_EnableIT_IDLE (USARTB);
  }

  activeIdleFlag = LL_USART_IsActiveFlag_IDLE(USARTB);
  isEnabledIdelFlag = LL_USART_IsEnabledIT_IDLE(USARTB);
  flags = activeIdleFlag & isEnabledIdelFlag;
  if (0U == flags)
  {
    /* Nothing to do */
  }
  else
  { /* Stopping the debugger will generate an OverRun error */
    LL_USART_DisableIT_IDLE (USARTB);
    /* Once the complete unexpected data are received, we enable back the error IT */
    LL_USART_EnableIT_ERROR (USARTB);
    /* To be sure we fetch the potential pendig data */
    /* We disable the DMA request, Read the dummy data, endable back the DMA request */
    LL_USART_DisableDMAReq_RX (USARTB);
    (void)LL_USART_ReceiveData8(USARTB);
    LL_USART_EnableDMAReq_RX (USARTB);
    ASPEP_HWDMAReset (&aspepOverUartB);
  }
 
  /* USER CODE BEGIN USARTB_IRQn 1 */
 
  /* USER CODE END USARTB_IRQn 1 */
}
</#if><#-- MC.MCP_OVER_UART_B_EN -->

/**
  * @brief  This function handles Hard Fault exception.
  * @param  None
  */
void HardFault_Handler(void)
{
 /* USER CODE BEGIN HardFault_IRQn 0 */

 /* USER CODE END HardFault_IRQn 0 */
  TSK_HardwareFaultTask();
  
  /* Go to infinite loop when Hard Fault exception occurs */
  while (true)
  {

  }
 /* USER CODE BEGIN HardFault_IRQn 1 */

 /* USER CODE END HardFault_IRQn 1 */

}
<#if MC.RTOS == "NONE">

void SysTick_Handler(void)
{

#ifdef MC_HAL_IS_USED
static uint8_t SystickDividerCounter = SYSTICK_DIVIDER;
  /* USER CODE BEGIN SysTick_IRQn 0 */

  /* USER CODE END SysTick_IRQn 0 */
  if (SystickDividerCounter == SYSTICK_DIVIDER)
  {
    HAL_IncTick();
    HAL_SYSTICK_IRQHandler();
    SystickDividerCounter = 0;
  }
  SystickDividerCounter ++;
#endif /* MC_HAL_IS_USED */

  /* USER CODE BEGIN SysTick_IRQn 1 */
  /* USER CODE END SysTick_IRQn 1 */
    MC_RunMotorControlTasks();
  <#if MC.M1_POSITION_CTRL_ENABLING == true>
    TC_IncTick(&PosCtrlM1);
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true -->

  <#if MC.M2_POSITION_CTRL_ENABLING == true>
    TC_IncTick(&PosCtrlM2);
  </#if><#-- MC.M2_POSITION_CTRL_ENABLING == true -->

  /* USER CODE BEGIN SysTick_IRQn 2 */
  /* USER CODE END SysTick_IRQn 2 */
}
</#if><#--  MC.RTOS == "NONE" -->
  <#function EXT_IRQHandler line>
  <#local EXTI_IRQ =
        [ {"name": "EXTI0_IRQHandler", "line": 0}
        , {"name": "EXTI1_IRQHandler", "line": 1}
        , {"name": "EXTI2_IRQHandler", "line": 2}
        , {"name": "EXTI3_IRQHandler", "line": 3}
        , {"name": "EXTI4_IRQHandler", "line": 4}
        , {"name": "EXTI9_5_IRQHandler", "line": 9}
        , {"name": "EXTI15_10_IRQHandler", "line": 15}
        ] >
  <#list EXTI_IRQ as handler >
        <#if line <= (handler.line) >
           <#return  handler.name >
         </#if>
  </#list>
  <#return "EXTI15_10_IRQHandler" >
  </#function>

  <#function _last_word text sep="_"><#return text?split(sep)?last></#function>
  <#function _last_char text><#return text[text?length-1]></#function>

<#if MC.START_STOP_BTN == true || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#-- GUI, this section is present only if start/stop button and/or Position Control with Z channel is enabled -->
  
    <#assign EXT_IRQHandler_StartStopName = "" >
    <#assign EXT_IRQHandler_ENC_Z_M1_Name = "" >
    <#assign EXT_IRQHandler_ENC_Z_M2_Name = "" >
    <#assign Template_StartStop ="">
    <#assign Template_Encoder_Z_M1 ="">
    <#assign Template_Encoder_Z_M2 ="">

  <#if MC.START_STOP_BTN == true>
    <#assign EXT_IRQHandler_StartStopName = "${EXT_IRQHandler(_last_word(MC.START_STOP_GPIO_PIN)?number)}" >
     <#if _last_word(MC.START_STOP_GPIO_PIN)?number < 32>
       <#assign Template_StartStop = '/* USER CODE BEGIN START_STOP_BTN */
  if (0U == LL_EXTI_ReadFlag_0_31(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)}))
  {
    /* Nothing to do */
  }
  else
  {
    LL_EXTI_ClearFlag_0_31 (LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)});
    (void)UI_HandleStartStopButton_cb ();
  }'> 
    <#else><#-- _last_word(MC.START_STOP_GPIO_PIN)?number >= 32 -->
      <#assign Template_StartStop = '/* USER CODE BEGIN START_STOP_BTN */
  if (0U == LL_EXTI_ReadFlag_32_63(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)}))
  {
    /* Nothing to do */
  }
  else
  {
    LL_EXTI_ClearFlag_32_63 (LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)});
    (void)UI_HandleStartStopButton_cb ();
  }'> 
    </#if><#-- _last_word(MC.START_STOP_GPIO_PIN)?number < 32 -->
  </#if><#-- MC.START_STOP_BTN == true -->

  <#if MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z">
    <#assign EXT_IRQHandler_ENC_Z_M1_Name = "${EXT_IRQHandler(_last_word(MC.M1_ENC_Z_GPIO_PIN)?number)}" >
    <#if _last_word(MC.M1_ENC_Z_GPIO_PIN)?number < 32>
      <#assign Template_Encoder_Z_M1 = '/* USER CODE BEGIN ENCODER Z INDEX M1 */
  if (LL_EXTI_ReadFlag_0_31(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)}))
  {
    LL_EXTI_ClearFlag_0_31 (LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)});
    TC_EncoderReset(&PosCtrlM1);
  }'> 
    <#else><#-- _last_word(MC.M1_ENC_Z_GPIO_PIN)?number >= 32 -->
      <#assign Template_Encoder_Z_M1 = '/* USER CODE BEGIN ENCODER Z INDEX M1 */
  if (LL_EXTI_ReadFlag_32_63(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)}))
  {
    LL_EXTI_ClearFlag_32_63 (LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)});
    TC_EncoderReset(&PosCtrlM1);
  }'> 
    </#if><#-- _last_word(MC.M1_ENC_Z_GPIO_PIN)?number < 32 -->
  </#if><#-- MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z" -->

  <#if MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z">
    <#assign EXT_IRQHandler_ENC_Z_M2_Name = "${EXT_IRQHandler(_last_word(MC.M2_ENC_Z_GPIO_PIN)?number)}" >
    <#if _last_word(MC.M2_ENC_Z_GPIO_PIN)?number < 32>
      <#assign Template_Encoder_Z_M2 = '/* USER CODE BEGIN ENCODER Z INDEX M2 */
  if (LL_EXTI_ReadFlag_0_31(LL_EXTI_LINE_${_last_word(MC.M2_ENC_Z_GPIO_PIN)}))
  {
    LL_EXTI_ClearFlag_0_31 (LL_EXTI_LINE_${_last_word(MC.M2_ENC_Z_GPIO_PIN)});
    TC_EncoderReset(&PosCtrlM2);
  }'> 
    <#else><#-- _last_word(MC.M2_ENC_Z_GPIO_PIN)?number >= 32 -->
      <#assign Template_Encoder_Z_M2 = '/* USER CODE BEGIN ENCODER Z INDEX M2 */
  if (LL_EXTI_ReadFlag_32_63(LL_EXTI_LINE_${_last_word(MC.M2_ENC_Z_GPIO_PIN)}))
  {
    LL_EXTI_ClearFlag_32_63 (LL_EXTI_LINE_${_last_word(MC.M2_ENC_Z_GPIO_PIN)});
    TC_EncoderReset(&PosCtrlM2);
  }'> 
    </#if><#-- _last_word(MC.M2_ENC_Z_GPIO_PIN)?number < 32 -->
  </#if> <#-- MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z" -->

  <#if MC.START_STOP_BTN == true>
/**
  * @brief  This function handles Button IRQ on PIN P${ _last_char(MC.START_STOP_GPIO_PORT)}${_last_word(MC.START_STOP_GPIO_PIN)}.
    <#if (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") && "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M1_Name}">
  *                 and M1 Encoder Index IRQ on PIN P${ _last_char(MC.M1_ENC_Z_GPIO_PORT)}${_last_word(MC.M1_ENC_Z_GPIO_PIN)}.
    </#if><#-- MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z" && "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M1_Name}" -->
    <#if (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") && "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M2_Name}">
  *                 and M2 Encoder Index IRQ on PIN P${ _last_char(MC.M2_ENC_Z_GPIO_PORT)}${_last_word(MC.M2_ENC_Z_GPIO_PIN)}.
    </#if><#-- MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z" && "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M2_Name}" -->
  */
void ${EXT_IRQHandler_StartStopName} (void)
{
  ${Template_StartStop}

    <#if "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M1_Name}">
  ${Template_Encoder_Z_M1}
    </#if><#-- "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M1_Name}" -->
  
    <#if "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M2_Name}">
  ${Template_Encoder_Z_M2}
    </#if><#-- "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M2_Name}" -->
}
  </#if> <#-- MC.START_STOP_BTN == true -->

  <#if MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z">
    <#if "${EXT_IRQHandler_StartStopName}" != "${EXT_IRQHandler_ENC_Z_M1_Name}">
/**
  * @brief  This function handles M1 Encoder Index IRQ on PIN P${ _last_char(MC.M1_ENC_Z_GPIO_PORT)}${_last_word(MC.M1_ENC_Z_GPIO_PIN)}.
      <#if (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") && "${EXT_IRQHandler_ENC_Z_M1_Name}" == "${EXT_IRQHandler_ENC_Z_M2_Name}" >
  *                 and M2 Encoder Index IRQ on PIN P${ _last_char(MC.M2_ENC_Z_GPIO_PORT)}${_last_word(MC.M2_ENC_Z_GPIO_PIN)}.
      </#if><#-- MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z" && "${EXT_IRQHandler_ENC_Z_M1_Name}" == "${EXT_IRQHandler_ENC_Z_M2_Name}" -->
  */
void ${EXT_IRQHandler_ENC_Z_M1_Name} (void)
{
  ${Template_Encoder_Z_M1}
  
      <#if "${EXT_IRQHandler_ENC_Z_M1_Name}" == "${EXT_IRQHandler_ENC_Z_M2_Name}">
  ${Template_Encoder_Z_M2}
      </#if><#-- "${EXT_IRQHandler_ENC_Z_M1_Name}" == "${EXT_IRQHandler_ENC_Z_M2_Name}" -->

}
    </#if> <#-- "${EXT_IRQHandler_StartStopName}" != "${EXT_IRQHandler_ENC_Z_M1_Name}" -->
  </#if> <#-- MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z" -->

  <#if MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z">
    <#if "${EXT_IRQHandler_StartStopName}" != "${EXT_IRQHandler_ENC_Z_M2_Name}"
      && "${EXT_IRQHandler_ENC_Z_M1_Name}" != "${EXT_IRQHandler_ENC_Z_M2_Name}">
/**
  * @brief  This function handles M2 Encoder Index IRQ on PIN 
            P${ _last_char(MC.M2_ENC_Z_GPIO_PORT)}${_last_word(MC.M2_ENC_Z_GPIO_PIN)}.
  */
void ${EXT_IRQHandler_ENC_Z_M2_Name} (void)
{
  ${Template_Encoder_Z_M2}
  
}
    </#if><#-- ${EXT_IRQHandler_StartStopName}" != "${EXT_IRQHandler_ENC_Z_M2_Name}" 
           && "${EXT_IRQHandler_ENC_Z_M1_Name}" != "${EXT_IRQHandler_ENC_Z_M2_Name}" -->
  </#if><#--MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z" -->
</#if><#-- MC.START_STOP_BTN == true || MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z" || MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z" -->


/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

/**
  * @}
  */

/**
  * @}
  */
/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
