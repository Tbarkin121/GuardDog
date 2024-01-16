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
<#if MC.M1_PWM_TIMER_SELECTION == "PWM_TIM1">
  <#assign Stream   = "5"> 
</#if>
<#if MC.M2_PWM_TIMER_SELECTION == "PWM_TIM1">
  <#assign Stream2   = "5"> 
</#if>

<#assign M1_ENCODER = (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M2_ENCODER = (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M1_HALL_SENSOR = (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
<#assign M2_HALL_SENSOR = (MC.M2_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
<#assign FOC = MC.M1_DRIVE_TYPE == "FOC" || MC.M2_DRIVE_TYPE == "FOC">
<#if MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT'> 
  <#if MC.M1_CS_ADC_PHASE_SHARED!="V"><#assign M1_ADC = MC.M1_CS_ADC_V><#else><#assign M1_ADC = MC.M1_CS_ADC_U></#if>
<#else><#-- (M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT') || (M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  <#assign M1_ADC = MC.M1_CS_ADC_U>
</#if><#-- M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT' -->
/**
  ******************************************************************************
  * @file    stm32h5xx_mc_it.c 
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   Main Interrupt Service Routines.
  *          This file provides exceptions handler and peripherals interrupt 
  *          service routine related to Motor Control for the STM32H5 Family.
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
  * @ingroup STM32H5xx_IRQ_Handlers
  */ 

/* Includes ------------------------------------------------------------------*/
#include "mc_type.h"
#include "mc_config.h"
<#-- Specific to FOC algorithm usage -->
<#if MC.M1_DRIVE_TYPE == "FOC">
#include "mc_tasks.h"
#include "parameters_conversion.h"
#include "motorcontrol.h"
	<#if (MC.START_STOP_BTN == true) || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")>
#include "stm32h5xx_ll_exti.h"
	</#if>
#include "stm32h5xx_hal.h"
</#if>
#include "stm32h5xx.h"
<#-- Specific to 6_STEP algorithm usage -->
<#if MC.M1_DRIVE_TYPE == "SIX_STEP">
#include "stm32h5xx_it.h"
#include "6step_core.h"
	<#-- Only for the TERATERM usage -->
	<#if MC.SIX_STEP_COMMUNICATION_IF == "TERATERM_IF"><#-- TERATERM I/F usage -->
#include "6step_com.h"
	</#if>
</#if>
<#if MC.MCP_EN >
#include "mcp_config.h"
</#if>
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/** @addtogroup MCSDK
  * @{
  */

/** @addtogroup STM32H5xx_IRQ_Handlers STM32H5xx IRQ Handlers
  * @{
  */
  
/* USER CODE BEGIN PRIVATE */
  
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
<#-- Specific to FOC algorithm usage -->
<#if MC.M1_DRIVE_TYPE == "FOC">
#define SYSTICK_DIVIDER (SYS_TICK_FREQUENCY/1000)
</#if>
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/* USER CODE END PRIVATE */
<#-- Specific to 6_STEP algorithm usage -->
<#if MC.M1_DRIVE_TYPE == "SIX_STEP">
extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim4;
extern MC_Handle_t Motor_Device1;

void SysTick_Handler(void);
void TIM1_BRK_TIM9_IRQHandler(void);
void USART_IRQHandler(void);
</#if>

<#-- Specific to FOC algorithm usage -->
<#if MC.M1_DRIVE_TYPE == "FOC">
/* Public prototypes of IRQ handlers called from assembly code ---------------*/
 <#if M1_ADC == "ADC1">
void ADC1_IRQHandler(void);
  </#if><#-- M1_ADC == "ADC1" -->
  <#if M1_ADC == "ADC2">
void ADC2_IRQHandler(void);
  </#if><#-- M1_ADC == "ADC2" -->

void TIMx_UP_M1_IRQHandler(void);
void TIMx_BRK_M1_IRQHandler(void);
<#if MC.M1_DRIVE_TYPE == "FOC">
	<#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
  || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
void GPDMA1_Channel0_IRQHandler(void);
	</#if>
 </#if><#-- Specific to FOC algorithm usage -->	

	<#if (M1_ENCODER == true) || (M1_HALL_SENSOR == true)>
void SPD_TIM_M1_IRQHandler(void);
	</#if><#-- (M1_ENCODER == true) || (M1_HALL_SENSOR == true) -->
	<#if MC.DRIVE_NUMBER != "1">
void TIMx_UP_M2_IRQHandler(void);
void TIMx_BRK_M2_IRQHandler(void);
		<#if (M2_ENCODER == true) || (M2_HALL_SENSOR == true)>
void SPD_TIM_M2_IRQHandler(void);
		</#if><#-- (M2_ENCODER == true) || (M2_HALL_SENSOR == true) -->
	</#if><#-- MC.DRIVE_NUMBER > 1 -->

void HardFault_Handler(void);
void SysTick_Handler(void);
	<#if MC.START_STOP_BTN == true>
void ${EXT_IRQHandler(_last_word(MC.START_STOP_GPIO_PIN)?number)} (void);
	</#if>


/**
  * @brief  This function handles ADC1/ADC2 interrupt request.
  * @param  None
  */

  
 <#if M1_ADC == "ADC1">
void ADC1_IRQHandler(void)
  </#if><#-- M1_ADC == "ADC1" -->
  <#if M1_ADC == "ADC2">
void ADC2_IRQHandler(void)
  </#if><#-- M1_ADC == "ADC2" -->
{
  /* USER CODE BEGIN ADC_IRQn 0 */

  /* USER CODE END ADC_IRQn 0 */
  if(LL_ADC_IsActiveFlag_JEOS(${M1_ADC}))
  {
    // Clear Flags
  LL_ADC_ClearFlag_JEOS(${M1_ADC});

    TSK_HighFrequencyTask();          /*GUI, this section is present only if DAC is disabled*/
  }  
  /* USER CODE BEGIN ADC_IRQn 1 */

  /* USER CODE END ADC_IRQn 1 */
}

/**
  * @brief  This function handles first motor TIMx Update interrupt request.
  * @param  None
  */
void TIMx_UP_M1_IRQHandler(void)
{
  /* USER CODE BEGIN TIMx_UP_M1_IRQn 0 */

  /* USER CODE END TIMx_UP_M1_IRQn 0 */
  
  LL_TIM_ClearFlag_UPDATE(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
	<#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))>
		<#if MC.DRIVE_NUMBER == "1">
  R3_2_TIMx_UP_IRQHandler(&PWM_Handle_M1);
		<#else><#-- MC.DRIVE_NUMBER != 1 -->
  R3_2_TIMx_UP_IRQHandler(&PWM_Handle_M1);
		</#if><#-- MC.DRIVE_NUMBER == 1 -->
	<#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
  R3_1_TIMx_UP_IRQHandler(&PWM_Handle_M1);
	<#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
  ICS_TIMx_UP_IRQHandler(&PWM_Handle_M1);
	<#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>      
		<#if MC.DRIVE_NUMBER == "1">
  R1_TIMx_UP_IRQHandler(&PWM_Handle_M1);
		<#else><#-- MC.DRIVE_NUMBER != 1 -->
			<#if (MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1') || (MC.M1_PWM_TIMER_SELECTION == 'TIM1')>
  R1_TIM1_UP_IRQHandler(&PWM_Handle_M1);
			<#elseif (MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8') || (MC.M1_PWM_TIMER_SELECTION == 'TIM8')>
  R1_TIM8_UP_IRQHandler(&PWM_Handle_M1);
			</#if>
		</#if><#-- MC.DRIVE_NUMBER == 1 -->
	</#if>
	<#if MC.DRIVE_NUMBER != "1">
  TSK_DualDriveFIFOUpdate( M1 );
	</#if><#-- MC.DRIVE_NUMBER > 1 -->
  /* USER CODE BEGIN TIMx_UP_M1_IRQn 1 */

  /* USER CODE END TIMx_UP_M1_IRQn 1 */  
}

	<#if MC.DRIVE_NUMBER != "1">
/**
  * @brief  This function handles second motor TIMx Update interrupt request.
  * @param  None
  */
void TIMx_UP_M2_IRQHandler(void)
{
  /* USER CODE BEGIN TIMx_UP_M2_IRQn 0 */

  /* USER CODE END TIMx_UP_M2_IRQn 0 */ 
  LL_TIM_ClearFlag_UPDATE(PWM_Handle_M2.pParams_str->TIMx);
		<#if ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '2' ))>  
			<#if MC.DRIVE_NUMBER == "1">
  R3_2_TIMx_UP_IRQHandler(&PWM_Handle_M2);
			<#else><#-- MC.DRIVE_NUMBER != 1 -->
  R3_2_TIMx_UP_IRQHandler(&PWM_Handle_M2);
			</#if><#-- MC.DRIVE_NUMBER == 1 -->
		<#elseif MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
  ICS_TIMx_UP_IRQHandler(&PWM_Handle_M2);
		<#elseif ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
			<#if MC.DRIVE_NUMBER == "1">
			<#-- Any chance this can ever be valid? -->
				<#if (MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1') || (MC.M2_PWM_TIMER_SELECTION == 'TIM1')>
  R1_TIM1_UP_IRQHandler(&PWM_Handle_M2);
				<#elseif (MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM8') || (MC.M2_PWM_TIMER_SELECTION == 'TIM8')>
  R1_TIM8_UP_IRQHandler(&PWM_Handle_M2);
				</#if>
			<#else><#-- MC.DRIVE_NUMBER != 1 -->
				<#if (MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1') || (MC.M2_PWM_TIMER_SELECTION == 'TIM1')>
  R1_TIM1_UP_IRQHandler(&PWM_Handle_M2);
				<#elseif (MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM8') || (MC.M2_PWM_TIMER_SELECTION == 'TIM8')>
  R1_TIM8_UP_IRQHandler(&PWM_Handle_M2);
				</#if>
			</#if><#-- MC.DRIVE_NUMBER == 1 -->
		</#if>
  TSK_DualDriveFIFOUpdate( M2 );
  /* USER CODE BEGIN TIMx_UP_M2_IRQn 1 */

  /* USER CODE END TIMx_UP_M2_IRQn 1 */ 
}
	</#if><#-- MC.DRIVE_NUMBER > 1 -->

/**
  * @brief  This function handles first motor BRK interrupt.
  * @param  None
  */
void TIMx_BRK_M1_IRQHandler(void)
{
  /* USER CODE BEGIN TIMx_BRK_M1_IRQn 0 */

  /* USER CODE END TIMx_BRK_M1_IRQn 0 */ 
  if (LL_TIM_IsActiveFlag_BRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)}))
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
  }
  
  if (LL_TIM_IsActiveFlag_BRK2(${_last_word(MC.M1_PWM_TIMER_SELECTION)}))
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
  }
  /* Systick is not executed due low priority so is necessary to call MC_Scheduler here.*/
  MC_Scheduler();
  
  /* USER CODE BEGIN TIMx_BRK_M1_IRQn 1 */

  /* USER CODE END TIMx_BRK_M1_IRQn 1 */ 
}

<#if MC.M1_DRIVE_TYPE == "FOC">
  <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
    <#if _last_word(MC.M1_PWM_TIMER_SELECTION) == "TIM1" ||
			 _last_word(MC.M2_PWM_TIMER_SELECTION) == "TIM1" >
void GPDMA1_Channel0_IRQHandler(void)
{
  uint32_t tempReg1;
  uint32_t tempReg2;
  
  tempReg1 = LL_DMA_IsActiveFlag_HT(GPDMA1, LL_DMA_CHANNEL_0);
  tempReg2 = LL_DMA_IsEnabledIT_HT(GPDMA1, LL_DMA_CHANNEL_0);
    
  if ((tempReg1 != 0U) && (tempReg2 != 0U))
  {
    (void)R1_DMAx_HT_IRQHandler(&PWM_Handle_M1);  
    LL_DMA_ClearFlag_HT(GPDMA1, LL_DMA_CHANNEL_0);     
  } 

  if (LL_DMA_IsActiveFlag_TC(GPDMA1, LL_DMA_CHANNEL_0) != 0U)
  {
    LL_DMA_ClearFlag_TC(GPDMA1, LL_DMA_CHANNEL_0);
    (void)R1_DMAx_TC_IRQHandler(&PWM_Handle_M1);
  }

    /* USER CODE BEGIN DMA1_Channel1_IRQHandler */

    /* USER CODE END DMA1_Channel1_IRQHandler */
}
    </#if>
  </#if>
</#if>

	<#if (M1_ENCODER==true) || (M1_HALL_SENSOR == true)>
/**
  * @brief  This function handles TIMx global interrupt request for M1 Speed Sensor.
  * @param  None
  */
void SPD_TIM_M1_IRQHandler(void)
{
  /* USER CODE BEGIN SPD_TIM_M1_IRQn 0 */

  /* USER CODE END SPD_TIM_M1_IRQn 0 */ 
  
		<#if (M1_HALL_SENSOR == true)>
  /* HALL Timer Update IT always enabled, no need to check enable UPDATE state */
  if (LL_TIM_IsActiveFlag_UPDATE(HALL_M1.TIMx))
  {
    LL_TIM_ClearFlag_UPDATE(HALL_M1.TIMx);
    HALL_TIMx_UP_IRQHandler(&HALL_M1);
    /* USER CODE BEGIN M1 HALL_Update */

    /* USER CODE END M1 HALL_Update   */ 
  }
  else
  {
    /* Nothing to do */
  }
  /* HALL Timer CC1 IT always enabled, no need to check enable CC1 state */
  if (LL_TIM_IsActiveFlag_CC1 (HALL_M1.TIMx)) 
  {
    LL_TIM_ClearFlag_CC1(HALL_M1.TIMx);
    HALL_TIMx_CC_IRQHandler(&HALL_M1);
    /* USER CODE BEGIN M1 HALL_CC1 */

    /* USER CODE END M1 HALL_CC1 */ 
  }
  else
  {
  /* Nothing to do */
  }
		<#else>
 /* Encoder Timer UPDATE IT is dynamicaly enabled/disabled, checking enable state is required */
  if (LL_TIM_IsEnabledIT_UPDATE (ENCODER_M1.TIMx) && LL_TIM_IsActiveFlag_UPDATE (ENCODER_M1.TIMx))
  { 
    LL_TIM_ClearFlag_UPDATE(ENCODER_M1.TIMx);
    ENC_IRQHandler(&ENCODER_M1);
    /* USER CODE BEGIN M1 ENCODER_Update */

    /* USER CODE END M1 ENCODER_Update   */ 
  }
  else
  {
  /* No other IT to manage for encoder config */
  }
		</#if>
  /* USER CODE BEGIN SPD_TIM_M1_IRQn 1 */

  /* USER CODE END SPD_TIM_M1_IRQn 1 */ 
}
	</#if>

	<#if MC.DRIVE_NUMBER != "1">
/**
  * @brief  This function handles second motor BRK interrupt.
  * @param  None
  */
void TIMx_BRK_M2_IRQHandler(void)
{
  /* USER CODE BEGIN TIMx_BRK_M2_IRQn 0 */

  /* USER CODE END TIMx_BRK_M2_IRQn 0 */
  
  if (LL_TIM_IsActiveFlag_BRK(PWM_Handle_M2.pParams_str->TIMx))
  {
    LL_TIM_ClearFlag_BRK(PWM_Handle_M2.pParams_str->TIMx);
<#if FOC>
<#if (MC.M2_OCP_TOPOLOGY != "NONE") &&  (MC.M2_OCP_DESTINATION != "TIM_BKIN")>
    PWMC_OCP_Handler(&PWM_Handle_M2._Super);
    <#elseif (MC.M2_DP_TOPOLOGY != "NONE") &&  (MC.M2_DP_DESTINATION != "TIM_BKIN")>
    PWMC_DP_Handler(&PWM_Handle_M2._Super);
    <#else>
    PWMC_OVP_Handler(&PWM_Handle_M2._Super, ${_last_word(MC.M2_PWM_TIMER_SELECTION)});
    </#if>
</#if><#-- FOC -->
  /* USER CODE BEGIN BRK */

  /* USER CODE END BRK */

  }
   /* Systick is not executed due low priority so is necessary to call MC_Scheduler here.*/
  MC_Scheduler();
  /* USER CODE BEGIN TIMx_BRK_M2_IRQn 1 */

  /* USER CODE END TIMx_BRK_M2_IRQn 1 */
}

		<#if (M2_ENCODER==true) || (M2_HALL_SENSOR == true)>
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
  if (LL_TIM_IsActiveFlag_UPDATE(HALL_M2.TIMx))
  {
    LL_TIM_ClearFlag_UPDATE(HALL_M2.TIMx);
    HALL_TIMx_UP_IRQHandler(&HALL_M2);
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
    HALL_TIMx_CC_IRQHandler(&HALL_M2);
    /* USER CODE BEGIN M2 HALL_CC1 */

    /* USER CODE END M2 HALL_CC1 */ 
  }
  else
  {
  /* Nothing to do */
  }
			<#else>
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
			</#if>
  /* USER CODE BEGIN SPD_TIM_M2_IRQn 1 */

  /* USER CODE END SPD_TIM_M2_IRQn 1 */ 
}
		</#if>
	</#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- MC.M1_DRIVE_TYPE == "FOC" -->

<#-- ST MCWB monitoring usage management -->
<#if MC.MCP_OVER_UART_A_EN>
/**
  * @brief This function handles DMA_RX_A channel DMACH_RX_A global interrupt.
  */
void ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A}(void)
{
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A} 0 */
  
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A} 0 */
  
  /* Buffer is ready by the HW layer to be processed */ 
  if (LL_DMA_IsActiveFlag_TC (DMA_RX_A, DMACH_RX_A) ){
    LL_DMA_ClearFlag_TC (DMA_RX_A, DMACH_RX_A);
    ASPEP_HWDataReceivedIT (&aspepOverUartA);
  }
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A} 1 */
  
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_A} 1 */

}

/* This section is present only when serial communication is used */
/**
  * @brief  This function handles USART interrupt request.
  * @param  None
  */
void ${MC.MCP_IRQ_HANDLER_UART_A}(void)
{
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_UART_A} 0 */
  
  /* USER CODE END U${MC.MCP_IRQ_HANDLER_UART_A} 0 */
    
  if ( LL_USART_IsActiveFlag_TC (USARTA) )
  {
    /* Disable the DMA channel to prepare the next chunck of data*/
    LL_DMA_DisableChannel( DMA_TX_A, DMACH_TX_A );
    LL_USART_ClearFlag_TC (USARTA);
    /* Data Sent by UART*/
    /* Need to free the buffer, and to check pending transfer*/
    ASPEP_HWDataTransmittedIT (&aspepOverUartA);
  }
  if ( (LL_USART_IsActiveFlag_ORE (USARTA) || LL_USART_IsActiveFlag_FE (USARTA) || LL_USART_IsActiveFlag_NE (USARTA))
        && LL_USART_IsEnabledIT_ERROR (USARTA) )
  { /* Stopping the debugger will generate an OverRun error*/
    WRITE_REG(USARTA->ICR, USART_ICR_FECF|USART_ICR_ORECF|USART_ICR_NECF);
    /* We disable ERROR interrupt to avoid to trig one Overrun IT per additional byte recevied*/
    LL_USART_DisableIT_ERROR (USARTA);
    LL_USART_EnableIT_IDLE (USARTA);
  }
  if ( LL_USART_IsActiveFlag_IDLE (USARTA) && LL_USART_IsEnabledIT_IDLE (USARTA) )
  { /* Stopping the debugger will generate an OverRun error*/
    LL_USART_DisableIT_IDLE (USARTA);
    /* Once the complete unexpected data are received, we enable back the error IT*/
    LL_USART_EnableIT_ERROR (USARTA);
    /* To be sure we fetch the potential pendig data*/
    /* We disable the DMA request, Read the dummy data, endable back the DMA request */
    LL_USART_DisableDMAReq_RX (USARTA);
    LL_USART_ReceiveData8(USARTA);
    LL_USART_EnableDMAReq_RX (USARTA);
    ASPEP_HWDMAReset (&aspepOverUartA);
  }

  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_UART_A} 1 */
 
  /* USER CODE END ${MC.MCP_IRQ_HANDLER_UART_A} 1 */
}
</#if>

<#if MC.MCP_OVER_UART_B_EN>
/**
  * @brief This function handles DMA_RX_B channel DMACH_RX_B global interrupt.
  */
void ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B}(void)
{
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B} 0 */
  
  /* USER CODE END ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B} 0 */
    
  /* Buffer is ready by the HW layer to be processed */ 
  if (LL_DMA_IsActiveFlag_TC (DMA_RX_B, DMACH_RX_B) ){
    LL_DMA_ClearFlag_TC (DMA_RX_B, DMACH_RX_B);
    ASPEP_HWDataReceivedIT (&aspepOverUartB);
  }
  /* USER CODE BEGIN ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B} 1 */
  
  /* USER CODE END ${MC.MCP_IRQ_HANDLER_DMA_RX_UART_B} 1 */  
}

/* This section is present only when serial communication is used */
/**
  * @brief  This function handles USART interrupt request.
  * @param  None
  */
void USARTB_IRQHandler(void)
{
  /* USER CODE BEGIN USARTB_IRQn 0 */
  
  /* USER CODE END USARTB_IRQn 0 */
  if ( LL_USART_IsActiveFlag_TC (USARTB) )
  {
    /* Disable the DMA channel to prepare the next chunck of data*/
    LL_DMA_DisableChannel( DMA_TX_B, DMACH_TX_B );
    LL_USART_ClearFlag_TC (USARTB);
    /* Data Sent by UART*/
    /* Need to free the buffer, and to check pending transfer*/
    ASPEP_HWDataTransmittedIT (&aspepOverUartB);
  }
  if ( (LL_USART_IsActiveFlag_ORE (USARTB) || LL_USART_IsActiveFlag_FE (USARTB) || LL_USART_IsActiveFlag_NE (USARTB)) 
        && LL_USART_IsEnabledIT_ERROR (USARTB) )  
  { /* Stopping the debugger will generate an OverRun error*/
    LL_USART_ClearFlag_FE(USARTB);
    LL_USART_ClearFlag_ORE(USARTB);
    LL_USART_ClearFlag_NE(USARTB);
    /* We disable ERROR interrupt to avoid to trig one Overrun IT per additional byte recevied*/
    LL_USART_DisableIT_ERROR (USARTB);
    LL_USART_EnableIT_IDLE (USARTB);        
  }
  if ( LL_USART_IsActiveFlag_IDLE (USARTB) && LL_USART_IsEnabledIT_IDLE (USARTB) )
  { /* Stopping the debugger will generate an OverRun error*/
    LL_USART_DisableIT_IDLE (USARTB);
    /* Once the complete unexpected data are received, we enable back the error IT*/
    LL_USART_EnableIT_ERROR (USARTB);    
    /* To be sure we fetch the potential pendig data*/
    /* We disable the DMA request, Read the dummy data, endable back the DMA request */
    LL_USART_DisableDMAReq_RX (USARTB);
    LL_USART_ReceiveData8(USARTB);
    LL_USART_EnableDMAReq_RX (USARTB);
    LL_DMA_ClearFlag_TE (DMA_RX_B, DMACH_RX_B );    
    ASPEP_HWDMAReset (&aspepOverUartB);
  }  
 
  /* USER CODE BEGIN USARTB_IRQn 1 */
 
  /* USER CODE END USARTB_IRQn 1 */
}
</#if>

<#-- Specific to 6_STEP algorithm usage -->
<#if MC.M1_DRIVE_TYPE == "SIX_STEP">
	<#-- Only for the TERATERM usage -->
	<#if MC.SIX_STEP_COMMUNICATION_IF == "TERATERM_IF"><#-- TERATERM I/F usage -->
/* This section is present only when TeraTerm is used for 6-steps */
/**
  * @brief  This function handles USART interrupt request.
  * @param  None
  */
void USART_IRQHandler(void)
{
  /* USER CODE BEGIN USART2_IRQn 0 */

  /* USER CODE END USART2_IRQn 0 */
  HAL_UART_IRQHandler(UI_Params.pUART);
  /* USER CODE BEGIN USART2_IRQn 1 */
  MC_Com_ProcessInput();
  /* USER CODE END USART2_IRQn 1 */
}
	</#if>

/**
  * @brief This function handles System tick timer.
  * @param  None
  */
void SysTick_Handler(void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */

  /* USER CODE END SysTick_IRQn 0 */
  HAL_IncTick();
  /* USER CODE BEGIN SysTick_IRQn 1 */
  HAL_SYSTICK_IRQHandler();
  /* USER CODE END SysTick_IRQn 1 */
}

/**
  * @brief This function handles TIM1 break interrupt and TIM9 global interrupt.
  * @param  None
  */
void TIM1_BRK_TIM9_IRQHandler(void)
{
  /* USER CODE BEGIN TIM1_BRK_TIM9_IRQn 0 */
  Motor_Device1.status = MC_OVERCURRENT;
  MC_Core_LL_Error(&Motor_Device1);
  /* USER CODE END TIM1_BRK_TIM9_IRQn 0 */
  HAL_TIM_IRQHandler(&htim1);
  /* USER CODE BEGIN TIM1_BRK_TIM9_IRQn 1 */

  /* USER CODE END TIM1_BRK_TIM9_IRQn 1 */
}

	<#if MC.SIX_STEP_COMMUNICATION_IF == "PWM_IF"><#-- PWM I/F usage -->
/**
  * @brief This function handles TIM2 global interrupt.
  */
void TIM2_IRQHandler(void)
{
  /* USER CODE BEGIN TIM2_IRQn 0 */
		<#if MC.SIX_STEP_SENSING == "SENSORS_LESS"><#-- Sensorless usage -->
  /* PWM INTERFACE BEGIN 1 */
  MC_Core_LL_PwmInterfaceIrqHandler((uint32_t *) &htim2);
  /* PWM INTERFACE END 1 */
		</#if>
  /* USER CODE END TIM2_IRQn 0 */

		<#if MC.SIX_STEP_SENSING == "HALL_SENSORS"><#-- Hall Sensors usage -->
  HAL_TIM_IRQHandler(&htim2);
		</#if>

  /* USER CODE BEGIN TIM2_IRQn 1 */
  /* USER CODE END TIM2_IRQn 1 */
}

/**
  * @brief This function handles TIM4 global interrupt.
  */
void TIM4_IRQHandler(void)
{
  /* USER CODE BEGIN TIM4_IRQn 0 */
		<#if MC.SIX_STEP_SENSING == "HALL_SENSORS"><#-- Hall Sensors usage -->
  /* PWM INTERFACE BEGIN 1 */
  MC_Core_LL_PwmInterfaceIrqHandler((uint32_t *) &htim4);
  /* PWM INTERFACE END 1 */
		</#if>
  /* USER CODE END TIM4_IRQn 0 */

		<#if MC.SIX_STEP_SENSING == "SENSORS_LESS"><#-- Sensorless usage -->
  HAL_TIM_IRQHandler(&htim4);
		</#if>

  /* USER CODE BEGIN TIM4_IRQn 1 */

  /* USER CODE END TIM4_IRQn 1 */
}
	</#if>
</#if>

<#-- Specific to FOC algorithm usage -->
<#if MC.M1_DRIVE_TYPE == "FOC">
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
  while (1)
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

<#if  MC.M1_POSITION_CTRL_ENABLING == true >
    TC_IncTick(&PosCtrlM1);
</#if>

<#if  MC.M2_POSITION_CTRL_ENABLING == true >
    TC_IncTick(&PosCtrlM2);
</#if>	

  /* USER CODE BEGIN SysTick_IRQn 2 */
  /* USER CODE END SysTick_IRQn 2 */
}
	</#if>
	<#function EXT_IRQHandler line>
 	 <#return "EXTI"+line+"_IRQHandler" >
	</#function>

	<#function _last_word text sep="_"><#return text?split(sep)?last></#function>
	<#function _last_char text><#return text[text?length-1]></#function>

<#if MC.START_STOP_BTN == true || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") >
<#-- GUI, this section is present only if start/stop button and/or Position Control with Z channel is enabled -->
  
<#assign EXT_IRQHandler_StartStopName = "" >
<#assign EXT_IRQHandler_ENC_Z_M1_Name = "" >
<#assign EXT_IRQHandler_ENC_Z_M2_Name = "" >
<#assign Template_StartStop ="">
<#assign Template_Encoder_Z_M1 ="">
<#assign Template_Encoder_Z_M2 ="">

	<#if MC.START_STOP_BTN == true>
  <#assign EXT_IRQHandler_StartStopName = "${EXT_IRQHandler(_last_word(MC.START_STOP_GPIO_PIN)?number)}" >
  <#if _last_word(MC.START_STOP_GPIO_PIN)?number < 32 >
  <#assign Template_StartStop = '/* USER CODE BEGIN START_STOP_BTN */
  if ( LL_EXTI_ReadFallingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)}) ) 
  {                                                                                
    LL_EXTI_ClearFallingFlag_0_31 (LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)});  
    UI_HandleStartStopButton_cb ();                                               
  }'> 
		<#else>
  <#assign Template_StartStop = '/* USER CODE BEGIN START_STOP_BTN */
  if ( LL_EXTI_ReadFlag_32_63(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)}) )
  {
    LL_EXTI_ClearFlag_32_63 (LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)});
    UI_HandleStartStopButton_cb ();
  }'> 
		</#if>
</#if>

<#if MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z">
  <#assign EXT_IRQHandler_ENC_Z_M1_Name = "${EXT_IRQHandler(_last_word(MC.M1_ENC_Z_GPIO_PIN)?number)}" >
  <#if _last_word(MC.M1_ENC_Z_GPIO_PIN)?number < 32 >
  <#assign Template_Encoder_Z_M1 = '/* USER CODE BEGIN ENCODER Z INDEX M1 */
  if (LL_EXTI_ReadFlag_0_31(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)}))  
  {                                                                          
    LL_EXTI_ClearFlag_0_31 (LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)});  
    TC_EncoderReset(&PosCtrlM1);                                            
  }'> 
  <#else>
  <#assign Template_Encoder_Z_M1 = '/* USER CODE BEGIN ENCODER Z INDEX M1 */
  if (LL_EXTI_ReadFlag_32_63(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)})) 
  {                                                                          
    LL_EXTI_ClearFlag_32_63 (LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)});  
    TC_EncoderReset(&PosCtrlM1);                                            
  }'> 
  </#if> 	
</#if> 

<#if MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z">
  <#assign EXT_IRQHandler_ENC_Z_M2_Name = "${EXT_IRQHandler(_last_word(MC.M2_ENC_Z_GPIO_PIN)?number)}" >
  <#if _last_word(MC.M2_ENC_Z_GPIO_PIN)?number < 32 >
  <#assign Template_Encoder_Z_M2 = '/* USER CODE BEGIN ENCODER Z INDEX M2 */
  if (LL_EXTI_ReadFlag_0_31(LL_EXTI_LINE_${_last_word(MC.M2_ENC_Z_GPIO_PIN)}))  
  {                                                                           
    LL_EXTI_ClearFlag_0_31 (LL_EXTI_LINE_${_last_word(MC.M2_ENC_Z_GPIO_PIN)});  
    TC_EncoderReset(&PosCtrlM2);                                             
  }'> 
  <#else>
  <#assign Template_Encoder_Z_M2 = '/* USER CODE BEGIN ENCODER Z INDEX M2 */
  if (LL_EXTI_ReadFlag_32_63(LL_EXTI_LINE_${_last_word(MC.M2_ENC_Z_GPIO_PIN)}))  
  {                                                                            
    LL_EXTI_ClearFlag_32_63 (LL_EXTI_LINE_${_last_word(MC.M2_ENC_Z_GPIO_PIN)});  
    TC_EncoderReset(&PosCtrlM2);                                              
  }'> 
  </#if> 
</#if> 
  
<#if MC.START_STOP_BTN == true>
/**
  * @brief  This function handles Button IRQ on PIN P${ _last_char(MC.START_STOP_GPIO_PORT)}${_last_word(MC.START_STOP_GPIO_PIN)}.
<#if (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") && "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M1_Name}">
  *                 and M1 Encoder Index IRQ on PIN P${ _last_char(MC.M1_ENC_Z_GPIO_PORT)}${_last_word(MC.M1_ENC_Z_GPIO_PIN)}.
</#if>  
<#if (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") && "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M2_Name}">
  *                 and M2 Encoder Index IRQ on PIN P${ _last_char(MC.M2_ENC_Z_GPIO_PORT)}${_last_word(MC.M2_ENC_Z_GPIO_PIN)}.
</#if>  
  */
void ${EXT_IRQHandler_StartStopName} (void)
{
	${Template_StartStop}

	<#if "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M1_Name}" >
	${Template_Encoder_Z_M1}
	</#if>
	
	<#if "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M2_Name}" >
	${Template_Encoder_Z_M2}
	</#if>
}
</#if>

<#if MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z">
	<#if "${EXT_IRQHandler_StartStopName}" != "${EXT_IRQHandler_ENC_Z_M1_Name}" >
/**
  * @brief  This function handles M1 Encoder Index IRQ on PIN P${ _last_char(MC.M1_ENC_Z_GPIO_PORT)}${_last_word(MC.M1_ENC_Z_GPIO_PIN)}.
<#if (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") && "${EXT_IRQHandler_ENC_Z_M1_Name}" == "${EXT_IRQHandler_ENC_Z_M2_Name}" >
  *                 and M2 Encoder Index IRQ on PIN P${ _last_char(MC.M2_ENC_Z_GPIO_PORT)}${_last_word(MC.M2_ENC_Z_GPIO_PIN)}.
</#if>  
  */
void ${EXT_IRQHandler_ENC_Z_M1_Name} (void)
{
	${Template_Encoder_Z_M1}
	
	<#if "${EXT_IRQHandler_ENC_Z_M1_Name}" == "${EXT_IRQHandler_ENC_Z_M2_Name}" >
	${Template_Encoder_Z_M2}
	</#if>

}	
	</#if>
</#if> 

<#if MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z">
	<#if "${EXT_IRQHandler_StartStopName}" != "${EXT_IRQHandler_ENC_Z_M2_Name}" && "${EXT_IRQHandler_ENC_Z_M1_Name}" != "${EXT_IRQHandler_ENC_Z_M2_Name}">
/**
  * @brief  This function handles M2 Encoder Index IRQ on PIN P${ _last_char(MC.M2_ENC_Z_GPIO_PORT)}${_last_word(MC.M2_ENC_Z_GPIO_PIN)}.
  */	
void ${EXT_IRQHandler_ENC_Z_M2_Name} (void)
{
	${Template_Encoder_Z_M2}
	
}	
	</#if>
</#if> 

</#if>
</#if><#-- MC.M1_DRIVE_TYPE == "FOC" -->
/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

/**
  * @}
  */

/**
  * @}
  */
/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
