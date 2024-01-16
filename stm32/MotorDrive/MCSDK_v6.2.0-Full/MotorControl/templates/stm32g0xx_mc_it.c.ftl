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
<#if MC??>
<#else>
<#stop "No MotorControl SW IP data found">
</#if>

<#function _first_word text sep="_"><#return text?split(sep)?first></#function>
<#function _last_word text sep="_"><#return text?split(sep)?last></#function>
<#function _last_char text><#return text[text?length-1]></#function>

<#function EXT_IRQHandler line>
    <#local EXTI_IRQ =
        [ {"name": "EXTI0_1_IRQHandler", "line": 1} 
        , {"name": "EXTI2_3_IRQHandler", "line": 3} 
        , {"name": "EXTI4_15_IRQHandler", "line": 15}
        ] >
    <#list EXTI_IRQ as handler >
        <#if line <= (handler.line ) >
           <#return  handler.name >
         </#if>
    </#list>
     <#return "EXTI4_15_IRQHandler" >
</#function>

<#assign M1_ENCODER = (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M2_ENCODER = (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M1_HALL_SENSOR = (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
<#assign M2_HALL_SENSOR = (MC.M2_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
<#assign FOC = MC.M1_DRIVE_TYPE == "FOC" || MC.M2_DRIVE_TYPE == "FOC">
<#assign SIX_STEP = MC.M1_DRIVE_TYPE == "SIX_STEP" || MC.M2_DRIVE_TYPE == "SIX_STEP">

<#assign NVICParams = configs[0].peripheralNVICParams >
<#function IRQHandler_name component irq="IRQ">
  <#if NVICParams.containsKey(component) >
    <#assign IRQHandler = NVICParams.get(component)>
  <#else>  
    <#return "#error component ${component} not found at NVIC level (no IT activated) " >
  </#if>
  <#if IRQHandler.isEmpty() == true >
      <#return "IRQ ${irq} probably not activated at NVIC level" >
  </#if>
    <#list IRQHandler.keySet() as entry>
       <#if entry?contains(irq)>     
         <#return "void "+entry[0..entry.length()-2]+"Handler (void)">
       </#if>
    </#list>
  <#return "#error IRQ ${irq} not activated at NVIC level for component ${component}">
</#function>
<#-- 
Useful for debug
      <#list NVICParams.keySet() as nvicParam>
          ${nvicParam}
      </#list>    
-->
</#if>
/**
  ******************************************************************************
  * @file    stm32g0xx_mc_it.c 
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   Main Interrupt Service Routines.
  *          This file provides exceptions handler and peripherals interrupt 
  *          service routine related to Motor Control
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
  * @ingroup STM32G0xx_IRQ_Handlers
  */ 

/* Includes ------------------------------------------------------------------*/
#include "mc_type.h"
#include "mc_tasks.h"
#include "parameters_conversion.h"
#include "motorcontrol.h"
<#if (MC.START_STOP_BTN == true) || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z")>
#include "stm32g0xx_ll_exti.h"
</#if>
<#if MC.MCP_EN>
#include "mcp_config.h"  
</#if>
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */
/** @addtogroup MCSDK
  * @{
  */

/** @addtogroup STM32F0xx_IRQ_Handlers STM32F0xx IRQ Handlers
  * @{
  */

/* USER CODE BEGIN PRIVATE */
  
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define SYSTICK_DIVIDER (SYS_TICK_FREQUENCY/1000)

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/* USER CODE END PRIVATE */

<#if FOC>
${IRQHandler_name("DMA", "Channel1")};
<#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
${IRQHandler_name("DMA", "Ch4")};
</#if> <#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
</#if><#-- FOC -->
<#if SIX_STEP>
void BEMF_READING_IRQHandler(void);
void PERIOD_COMM_Handler(void);
</#if><#-- SIX_STEP -->
${IRQHandler_name(_last_word(MC.M1_PWM_TIMER_SELECTION), "UP")};
<#if M1_ENCODER == true >
<#assign SensorTimer=_last_word(MC.M1_ENC_TIMER_SELECTION)>
${IRQHandler_name(_last_word(MC.M1_ENC_TIMER_SELECTION))};
</#if>
<#if M1_HALL_SENSOR == true>
<#assign SensorTimer=_last_word(MC.M1_HALL_TIMER_SELECTION)>
${IRQHandler_name(_last_word(MC.M1_HALL_TIMER_SELECTION))};
</#if>
void HardFault_Handler(void);
void SysTick_Handler(void);
<#if MC.START_STOP_BTN == true>
void ${EXT_IRQHandler(_last_word(MC.START_STOP_GPIO_PIN)?number)} (void);
</#if>

<#if FOC>
/**
  * @brief  This function handles current regulation interrupt request.
  * @param  None
  */
${IRQHandler_name("DMA", "Channel1")}    
{
  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 0 */
    /* Debug High frequency task duration
     * LL_GPIO_SetOutputPin (GPIOB, LL_GPIO_PIN_3); 
     */
  /* USER CODE END CURRENT_REGULATION_IRQn 0 */
  
  /* Clear Flags */
  DMA1->IFCR = (LL_DMA_ISR_GIF1|LL_DMA_ISR_TCIF1|LL_DMA_ISR_HTIF1);
  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 1 */

  /* USER CODE END CURRENT_REGULATION_IRQn 1 */   
  
    TSK_HighFrequencyTask();

  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 2 */

  /* USER CODE END CURRENT_REGULATION_IRQn 2 */   
}
</#if><#-- FOC -->

/**
  * @brief  This function handles first motor TIMx Update, Break-in interrupt request.
  * @param  None
  */
${IRQHandler_name(_last_word(MC.M1_PWM_TIMER_SELECTION), "UP")}  
{
  /* USER CODE BEGIN TIMx_UP_BRK_M1_IRQn 0 */

  /* USER CODE END TIMx_UP_BRK_M1_IRQn 0 */   

  if(LL_TIM_IsActiveFlag_UPDATE(${_last_word(MC.M1_PWM_TIMER_SELECTION)}) && LL_TIM_IsEnabledIT_UPDATE(${_last_word(MC.M1_PWM_TIMER_SELECTION)}))
  {
    LL_TIM_ClearFlag_UPDATE(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
<#if FOC>
<#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>  

    R1_TIMx_UP_IRQHandler(&PWM_Handle_M1);


<#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && (MC.M1_CS_ADC_NUM == '1'))>
    R3_1_TIMx_UP_IRQHandler(&PWM_Handle_M1);
</#if> 
</#if><#-- FOC -->
<#if SIX_STEP>
    (void)TSK_HighFrequencyTask();
</#if><#-- SIX_STEP -->
    /* USER CODE BEGIN PWM_Update */

    /* USER CODE END PWM_Update */  
  }  
  if(LL_TIM_IsActiveFlag_BRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)}) && LL_TIM_IsEnabledIT_BRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)})) 
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
    SixPwm_BRK_IRQHandler(&PWM_Handle_M1);
<#else>
    ThreePwm_BRK_IRQHandler(&PWM_Handle_M1);
</#if>
</#if><#-- SIX_STEP -->
    /* USER CODE BEGIN Break */

    /* USER CODE END Break */ 
  }
  if (LL_TIM_IsActiveFlag_BRK2(${_last_word(MC.M1_PWM_TIMER_SELECTION)}) && LL_TIM_IsEnabledIT_BRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)})) 
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
    SixPwm_BRK_IRQHandler(&PWM_Handle_M1);
<#else>
    ThreePwm_BRK_IRQHandler(&PWM_Handle_M1);
</#if>
</#if><#-- SIX_STEP -->     
    /* USER CODE BEGIN Break */

    /* USER CODE END Break */ 
  }
  else 
  {
   /* No other interrupts are routed to this handler */
  }
  /* USER CODE BEGIN TIMx_UP_BRK_M1_IRQn 1 */

  /* USER CODE END TIMx_UP_BRK_M1_IRQn 1 */   
}

<#if FOC>
<#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>     
/**
  * @brief  This function handles first motor DMAx TC interrupt request. 
  *         Required only for R1 with rep rate > 1
  * @param  None
  */
${IRQHandler_name("DMA", "Ch4")}
{
  /* USER CODE BEGIN DMAx_R1_M1_IRQn 0 */

  /* USER CODE END DMAx_R1_M1_IRQn 0 */ 
  if (LL_DMA_IsActiveFlag_HT4(DMA1) && LL_DMA_IsEnabledIT_HT(DMA1, LL_DMA_CHANNEL_4))
  {
    R1_DMAx_HT_IRQHandler(&PWM_Handle_M1);
    LL_DMA_ClearFlag_HT4(DMA1);   
    /* USER CODE BEGIN DMAx_R1_M1_TC4 */

    /* USER CODE END DMAx_R1_M1_TC4 */     
  } 
  
  if (LL_DMA_IsActiveFlag_TC4(DMA1))
  {
    LL_DMA_ClearFlag_TC4(DMA1);   
    R1_DMAx_TC_IRQHandler(&PWM_Handle_M1);
    /* USER CODE BEGIN DMAx_R1_M1_TC4 */

    /* USER CODE END DMAx_R1_M1_TC4 */     
  }    
        
  /* USER CODE BEGIN DMAx_R1_M1_IRQn 1 */

  /* USER CODE END DMAx_R1_M1_IRQn 1 */ 
}
</#if>
</#if><#-- FOC -->

<#if  MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
/**
  * @brief  This function handles BEMF sensing interrupt request.
  * @param[in] None
  */
void BEMF_READING_IRQHandler(void)
{
  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 0 */

  /* USER CODE END CURRENT_REGULATION_IRQn 0 */

  if(LL_DMA_IsActiveFlag_TC1(DMA1) && LL_DMA_IsEnabledIT_TC(DMA1, LL_DMA_CHANNEL_1))
  {
  /* Clear Flags */
    LL_DMA_ClearFlag_TC1( DMA1 );
  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 1 */

  /* USER CODE END CURRENT_REGULATION_IRQn 1 */
    BADC_IsZcDetected( &Bemf_ADC_M1, &PWM_Handle_M1._Super );
  }
  /* USER CODE BEGIN CURRENT_REGULATION_IRQn 2 */

  /* USER CODE END CURRENT_REGULATION_IRQn 2 */
}

/**
  * @brief     LFtimer interrupt handler
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
}

</#if><#--  MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
<#if (M1_ENCODER == true) || (M1_HALL_SENSOR == true)>
/**
  * @brief  This function handles TIMx global interrupt request for M1 Speed Sensor.
  * @param  None
  */
${IRQHandler_name(SensorTimer)}  
{
  /* USER CODE BEGIN SPD_TIM_M1_IRQn 0 */

  /* USER CODE END SPD_TIM_M1_IRQn 0 */ 
  
<#if M1_HALL_SENSOR == true>
  /* HALL Timer Update IT always enabled, no need to check enable UPDATE state */
  if (LL_TIM_IsActiveFlag_UPDATE(HALL_M1.TIMx) != 0)
  {
    LL_TIM_ClearFlag_UPDATE(HALL_M1.TIMx);
    HALL_TIMx_UP_IRQHandler(&HALL_M1);
    /* USER CODE BEGIN HALL_Update */

    /* USER CODE END HALL_Update   */ 
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
    /* USER CODE BEGIN HALL_CC1 */

    /* USER CODE END HALL_CC1 */ 
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
    /* USER CODE BEGIN ENCODER_Update */

    /* USER CODE END ENCODER_Update   */ 
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
<#-- ST MCWB monitoring usage management (used when MC.MCP_OVER_UART_A_EN == true) -->
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
  
  /* USER CODE END ${MC.MCP_IRQ_HANDLER_UART_A} 0 */
    
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
  /* USER CODE BEGIN SysTick_IRQn 2 */
  /* USER CODE END SysTick_IRQn 2 */
}



<#if MC.START_STOP_BTN == true || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#-- GUI, this section is present only if start/stop button and/or Position Control with Z channel is enabled -->
  
  <#assign EXT_IRQHandler_StartStopName = "" >
  <#assign EXT_IRQHandler_ENC_Z_M1_Name = "" >
  <#assign Template_StartStop ="">
  <#assign Template_Encoder_Z_M1 ="">

  <#if MC.START_STOP_BTN == true>
    <#assign EXT_IRQHandler_StartStopName = "${EXT_IRQHandler(_last_word(MC.START_STOP_GPIO_PIN)?number)}" >
    <#if _last_word(MC.START_STOP_GPIO_PIN)?number < 32 >
      <#assign Template_StartStop = '/* USER CODE BEGIN START_STOP_BTN */
  if ( LL_EXTI_ReadRisingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)}) ||
       LL_EXTI_ReadFallingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)}) ) 
  {                                                                                
    LL_EXTI_ClearRisingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)});  
    LL_EXTI_ClearFallingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)});
    UI_HandleStartStopButton_cb ();                                               
  }'> 
    <#else>
      <#assign Template_StartStop = '/* USER CODE BEGIN START_STOP_BTN */
  if ( LL_EXTI_ReadRisingFlag_32_63(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)}) ||
       LL_EXTI_ReadFallingFlag_32_63(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)}) )   
  {                                                                                
    LL_EXTI_ClearRisingFlag_32_63(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)});  
    LL_EXTI_ClearFallingFlag_32_63(LL_EXTI_LINE_${_last_word(MC.START_STOP_GPIO_PIN)});
    UI_HandleStartStopButton_cb ();                                               
  }'> 
    </#if>
  </#if>

  <#if MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z">
    <#assign EXT_IRQHandler_ENC_Z_M1_Name = "${EXT_IRQHandler(_last_word(MC.M1_ENC_Z_GPIO_PIN)?number)}" >
    <#if _last_word(MC.M1_ENC_Z_GPIO_PIN)?number < 32 >
      <#assign Template_Encoder_Z_M1 = '/* USER CODE BEGIN ENCODER Z INDEX M1 */
  if ( LL_EXTI_ReadRisingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)}) ||
       LL_EXTI_ReadFallingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)}) ) 
  {                                                                                
    LL_EXTI_ClearRisingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)});  
    LL_EXTI_ClearFallingFlag_0_31(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)});
    TC_EncoderReset(&PosCtrlM1); 
  }'> 
  <#else>
  <#assign Template_StartStop = '/* USER CODE BEGIN ENCODER Z INDEX M1 */
  if ( LL_EXTI_ReadRisingFlag_32_63(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)}) ||
       LL_EXTI_ReadFallingFlag_32_63(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)}) )   
  {                                                                                
    LL_EXTI_ClearRisingFlag_32_63(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)});  
    LL_EXTI_ClearFallingFlag_32_63(LL_EXTI_LINE_${_last_word(MC.M1_ENC_Z_GPIO_PIN)});
    TC_EncoderReset(&PosCtrlM1); 
  }'> 
    </#if> 	
  </#if> <#-- MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z" -->
  
  <#if MC.START_STOP_BTN == true>
/**
  * @brief  This function handles Button IRQ on PIN P${ _last_char(MC.START_STOP_GPIO_PORT)}${_last_word(MC.START_STOP_GPIO_PIN)}.
    <#if (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") && "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M1_Name}">
  *                 and M1 Encoder Index IRQ on PIN P${ _last_char(MC.M1_ENC_Z_GPIO_PORT)}${_last_word(MC.M1_ENC_Z_GPIO_PIN)}.
    </#if>  
  */
void ${EXT_IRQHandler_StartStopName} (void)
{
	${Template_StartStop}

    <#if "${EXT_IRQHandler_StartStopName}" == "${EXT_IRQHandler_ENC_Z_M1_Name}" >
	${Template_Encoder_Z_M1}
    </#if>
	
}
  </#if> <#-- MC.START_STOP_BTN == true -->

  <#if MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z">
    <#if "${EXT_IRQHandler_StartStopName}" != "${EXT_IRQHandler_ENC_Z_M1_Name}" >
/**
  * @brief  This function handles M1 Encoder Index IRQ on PIN P${ _last_char(MC.M1_ENC_Z_GPIO_PORT)}${_last_word(MC.M1_ENC_Z_GPIO_PIN)}.
  */
void ${EXT_IRQHandler_ENC_Z_M1_Name} (void)
{
	${Template_Encoder_Z_M1}
	
}	
    </#if> <#-- "${EXT_IRQHandler_StartStopName}" != "${EXT_IRQHandler_ENC_Z_M1_Name}" -->
  </#if> <#-- MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z" --> 
</#if> <#-- MC.START_STOP_BTN == true || MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z" -->

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */



/**
  * @}
  */

/**
  * @}
  */

/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
