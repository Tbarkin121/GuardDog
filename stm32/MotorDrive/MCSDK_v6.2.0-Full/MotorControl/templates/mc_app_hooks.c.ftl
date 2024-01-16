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
<#assign CondIncludeMCConfig = (MC.M1_POTENTIOMETER_ENABLE == true) || (MC.M2_POTENTIOMETER_ENABLE == true) || (MC.ESC_ENABLE == true) >
<#assign CondBootHookUsed = (MC.M1_POTENTIOMETER_ENABLE == true) || (MC.M2_POTENTIOMETER_ENABLE == true) || (MC.ESC_ENABLE == true) >
<#assign CondPostMediumFrequencyHookM1Used = (MC.M1_POTENTIOMETER_ENABLE == true) || (MC.ESC_ENABLE == true) >
<#assign CondPostMediumFrequencyHookM2Used = (MC.M2_POTENTIOMETER_ENABLE == true) >
/**
  ******************************************************************************
  * @file    mc_app_hooks.c
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file implements default motor control app hooks.
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
  * @ingroup MCAppHooks
  */

/* Includes ------------------------------------------------------------------*/
#include "mc_type.h"
#include "mc_app_hooks.h"
<#if CondIncludeMCConfig>
#include "mc_config.h"
</#if><#-- CondIncludeMCConfig -->
<#if MC.M1_POTENTIOMETER_ENABLE == true || MC.M2_POTENTIOMETER_ENABLE == true>
#include "speed_potentiometer.h"
</#if><#-- MC.M1_POTENTIOMETER_ENABLE == true || MC.M2_POTENTIOMETER_ENABLE == true -->
<#if MC.ESC_ENABLE >
#include "esc.h"
</#if><#-- MC.ESC_ENABLE -->

/** @addtogroup MCSDK
  * @{
  */

/** @addtogroup MCTasks
  * @{
  */

/**
 * @defgroup MCAppHooks Motor Control Applicative hooks
 * @brief User defined functions that are called in the Motor Control tasks.
 *
 *
 * @{
 */

/**
 * @brief Hook function called right before the end of the MCboot function.
 * 
 * 
 * 
 */
__weak void MC_APP_BootHook(void)
{
<#if CondBootHookUsed>
  <#if MC.M1_POTENTIOMETER_ENABLE == true>
  /* RCM component initialization */
  (void)RCM_RegisterRegConv(&PotRegConv_M1);
  SPDPOT_Init(&SpeedPotentiometer_M1);
  </#if><#-- MC.M1_POTENTIOMETER_ENABLE == true -->
  <#if MC.M2_POTENTIOMETER_ENABLE == true>
  /* RCM component initialization */
  (void)RCM_RegisterRegConv(&PotRegConv_M2);
  SPDPOT_Init(&SpeedPotentiometer_M2);
  </#if><#-- MC.M2_POTENTIOMETER_ENABLE == true -->
  <#if MC.ESC_ENABLE == true>
  esc_boot(&ESC_M1);
  </#if><#-- MC.ESC_ENABLE == true -->
<#else><#-- CondBootHookUsed == false -->
  /* 
   * This function can be overloaded or the application can inject
   * code into it that will be executed at the end of MCboot().
   */

</#if><#-- CondBootHookUsed -->
/* USER CODE BEGIN BootHook */

/* USER CODE END BootHook */
}

/**
 * @brief Hook function called right after the Medium Frequency Task for Motor 1.
 * 
 * 
 * 
 */
__weak void MC_APP_PostMediumFrequencyHook_M1(void) 
{
<#if CondPostMediumFrequencyHookM1Used>
  <#if MC.M1_POTENTIOMETER_ENABLE == true>
  uint16_t rawValue = RCM_ExecRegularConv(&PotRegConv_M1);
  SPDPOT_Run(&SpeedPotentiometer_M1, rawValue);
  </#if><#-- MC.M1_POTENTIOMETER_ENABLE == true -->
  <#if MC.ESC_ENABLE == true>
  esc_pwm_control(&ESC_M1); 
  </#if><#-- MC.ESC_ENABLE == true -->
<#else><#-- CondPostMediumFrequencyHookM1Used == false -->
  /* 
   * This function can be overloaded or the application can inject
   * code into it that will be executed right after the Medium
   * Frequency Task of Motor 1
   */

</#if><#-- CondPostMediumFrequencyHookM1Used -->
/* USER SECTION BEGIN PostMediumFrequencyHookM1 */

/* USER SECTION END PostMediumFrequencyHookM1 */
}

<#if MC.DRIVE_NUMBER != "1">
/**
 * @brief Hook function called right after the Medium Frequency Task for Motor 2.
 * 
 * 
 * 
 */
__weak void MC_APP_PostMediumFrequencyHook_M2(void) 
{
<#if CondPostMediumFrequencyHookM2Used>
  <#if MC.M2_POTENTIOMETER_ENABLE == true>
  uint16_t rawValue = RCM_ExecRegularConv(&PotRegConv_M2);
  SPDPOT_Run(&SpeedPotentiometer_M2, rawValue);
  </#if><#-- MC.M2_POTENTIOMETER_ENABLE == true -->
<#else><#-- CondPostMediumFrequencyHookM2Used == false -->
  /* 
   * This function can be overloaded or the application can inject
   * code into it that will be executed right after the Medium
   * Frequency Task of Motor 2
   */

</#if><#-- CondPostMediumFrequencyHookM1Used -->
/* USER SECTION BEGIN PostMediumFrequencyHookM1 */

/* USER SECTION END PostMediumFrequencyHookM1 */
}
</#if><#-- MC.DRIVE_NUMBER != "1" -->

/** @} */

/** @} */

/** @} */

/************************ (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
