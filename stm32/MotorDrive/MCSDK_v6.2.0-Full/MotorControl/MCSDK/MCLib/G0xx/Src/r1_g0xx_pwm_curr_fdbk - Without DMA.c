/**
  ******************************************************************************
  * @file    r1_g0xx_pwm_curr_fdbk.c
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file provides firmware functions that implement current sensor
  *          class to be stantiated when the single shunt current sensing
  *          topology is used.
  * 
  *          It is specifically designed for STM32G0XX microcontrollers and
  *          implements the successive sampling of motor current using only one ADC.
  *           + MCU peripheral and handle initialization fucntion
  *           + three shunt current sensing
  *           + space vector modulation function
  *           + ADC sampling function
  *
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted, provided that the following conditions are met:
  *
  * 1. Redistribution of source code must retain the above copyright notice,
  *    this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  *    this list of conditions and the following disclaimer in the documentation
  *    and/or other materials provided with the distribution.
  * 3. Neither the name of STMicroelectronics nor the names of other
  *    contributors to this software may be used to endorse or promote products
  *    derived from this software without specific written permission.
  * 4. This software, including modifications and/or derivative works of this
  *    software, must execute solely and exclusively on microcontroller or
  *    microprocessor devices manufactured by or for STMicroelectronics.
  * 5. Redistribution and use of this software other than as permitted under
  *    this license is void and will automatically terminate your rights under
  *    this license.
  *
  * THIS SOFTWARE IS PROVIDED BY STMICROELECTRONICS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT
  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
  * PARTICULAR PURPOSE AND NON-INFRINGEMENT OF THIRD PARTY INTELLECTUAL PROPERTY
  * RIGHTS ARE DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW. IN NO EVENT
  * SHALL STMICROELECTRONICS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
  * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "pwm_curr_fdbk.h"
#include "r1_g0xx_pwm_curr_fdbk.h"
#include "mc_type.h"

/** @addtogroup MCSDK
  * @{
  */

/** @addtogroup pwm_curr_fdbk
  * @{
  */

/**
 * @defgroup R1_G0XX_pwm_curr_fdbk G0 R1 1 ADC PWM & Current Feedback
 * 
 * @brief STM32G0, 1-Shunt, 1 ADC, PWM & Current Feedback implementation
 *
 * This component is used in applications based on an STM32G0 MCU, using a one
 * shunt resistor current sensing topology and one ADC peripheral to acquire the current
 * values.
 * 
 * 
 * @{
 */

/* Private Defines -----------------------------------------------------------*/

#define DR_OFFSET 0x40u
#define ADC1_DR_Address     ADC1_BASE + DR_OFFSET

#define NB_CONVERSIONS 16u

#define REGULAR         ((uint8_t)0u)
#define BOUNDARY_1      ((uint8_t)1u)  /* Two small, one big */
#define BOUNDARY_2      ((uint8_t)2u)  /* Two big, one small */
#define BOUNDARY_3      ((uint8_t)3u)  /* Three equal        */

#define INVERT_NONE 0u
#define INVERT_A 1u
#define INVERT_B 2u
#define INVERT_C 3u

#define SAMP_NO 0u
#define SAMP_IA 1u
#define SAMP_IB 2u
#define SAMP_IC 3u
#define SAMP_NIA 4u
#define SAMP_NIB 5u
#define SAMP_NIC 6u
#define SAMP_OLDA 7u
#define SAMP_OLDB 8u
#define SAMP_OLDC 9u

#define CH1NORMAL           0x0060u
#define CH2NORMAL           0x6000u
#define CH3NORMAL           0x0060u
#define CH4NORMAL           0x7000u

#define CCMR1_PRELOAD_DISABLE_MASK 0xF7F7u
#define CCMR2_PRELOAD_DISABLE_MASK 0xFFF7u

#define CCMR1_PRELOAD_ENABLE_MASK 0x0808u
#define CCMR2_PRELOAD_ENABLE_MASK 0x0008u

/* DMA ENABLE mask */
#define CCR_ENABLE_Set          ((uint32_t)0x00000001u)
#define CCR_ENABLE_Reset        ((uint32_t)0xFFFFFFFEu)

#define CR2_JEXTSEL_Reset       ((uint32_t)0xFFFF8FFFu)
#define CR2_JEXTTRIG_Set        ((uint32_t)0x00008000u)
#define CR2_JEXTTRIG_Reset      ((uint32_t)0xFFFF7FFFu)

#define TIM_DMA_ENABLED_CC1 0x0200u
#define TIM_DMA_ENABLED_CC2 0x0400u
#define TIM_DMA_ENABLED_CC3 0x0800u

#define CR2_ADON_Set                ((uint32_t)0x00000001u)

/* ADC SMPx mask */
#define SMPR1_SMP_Set              ((uint32_t) (0x00000007u))
#define SMPR2_SMP_Set              ((uint32_t) (0x00000007u))
#define CR2_EXTTRIG_SWSTART_Set     ((u32)0x00500000)

#define ADC1_CR2_EXTTRIG_SWSTART_BB 0x42248158u

#define ADCx_IRQn     ADC1_COMP_IRQn
#define TIMx_UP_IRQn  TIM1_BRK_UP_TRG_COM_IRQn

/* Constant values -----------------------------------------------------------*/
static const uint8_t REGULAR_SAMP_CUR1[6] = {SAMP_NIC,SAMP_NIC,SAMP_NIA,SAMP_NIA,SAMP_NIB,SAMP_NIB};
static const uint8_t REGULAR_SAMP_CUR2[6] = {SAMP_IA,SAMP_IB,SAMP_IB,SAMP_IC,SAMP_IC,SAMP_IA};
static const uint8_t BOUNDR1_SAMP_CUR2[6] = {SAMP_IB,SAMP_IB,SAMP_IC,SAMP_IC,SAMP_IA,SAMP_IA};
static const uint8_t BOUNDR2_SAMP_CUR1[6] = {SAMP_IA,SAMP_IB,SAMP_IB,SAMP_IC,SAMP_IC,SAMP_IA};
static const uint8_t BOUNDR2_SAMP_CUR2[6] = {SAMP_IC,SAMP_IA,SAMP_IA,SAMP_IB,SAMP_IB,SAMP_IC};

/* Private function prototypes -----------------------------------------------*/
void R1G0XX_1ShuntMotorVarsInit(PWMC_Handle_t *pHdl);
void R1G0XX_1ShuntMotorVarsRestart(PWMC_Handle_t *pHdl);
void R1G0XX_TIMxInit(TIM_TypeDef* TIMx, PWMC_Handle_t *pHdl);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  It initializes TIM1, ADC, GPIO, DMA1 and NVIC for single shunt current
  *         reading configuration using STM32F0XX family.
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval none
  */
void R1G0XX_Init(PWMC_R1_G0_Handle_t *pHandle)
{
  uint16_t hTIM1_CR1;
  
  if ((uint32_t)pHandle == (uint32_t)&pHandle->_Super)
  {

    R1G0XX_1ShuntMotorVarsInit(&pHandle->_Super);
    
    LL_DBGMCU_APB2_GRP1_FreezePeriph(LL_DBGMCU_APB2_GRP1_TIM1_STOP);

    R1G0XX_TIMxInit(TIM1, &pHandle->_Super);


    /* DMA Event related to ADC conversion*/
    /* DMA channel configuration ----------------------------------------------*/
    LL_DMA_SetMemoryAddress(DMA1, LL_DMA_CHANNEL_1, (uint32_t)pHandle->hCurConv);
    LL_DMA_SetPeriphAddress(DMA1, LL_DMA_CHANNEL_1, (uint32_t)ADC1_DR_Address);
    LL_DMA_SetDataLength(DMA1, LL_DMA_CHANNEL_1, 2u);

    /* DMA1 channel 1 will be enabled after the CurrentReadingCalibration */


    /* Start calibration of ADC1 */
    LL_ADC_StartCalibration(ADC1);
    while(LL_ADC_IsCalibrationOnGoing(ADC1) == 1);
    (READ_BIT(ADC1->CR,ADC_CR_ADCAL) == RESET)?(LL_ADC_ReadReg(ADC1,DR)):(0);

    /* Enable ADC */
    ADC1->CFGR1 &= ~ADC_EXTERNALTRIGCONVEDGE_RISINGFALLING;
    LL_ADC_REG_SetDMATransfer(ADC1, LL_ADC_REG_DMA_TRANSFER_NONE);
    LL_ADC_Enable(ADC1);
    

    /* Wait ADC Ready */
    while (LL_ADC_IsActiveFlag_ADRDY(ADC1)==RESET)
    {}
    
    /* set the shadow variable bADCSMP2 to the current ADC SMP2 value */
    pHandle->bADCSMP2 = LL_ADC_GetSamplingTimeCommonChannels (ADC1, LL_ADC_SAMPLINGTIME_COMMON_2);


   
    R1G0XX_1ShuntMotorVarsRestart(&pHandle->_Super);

    LL_DMA_EnableIT_TC(DMA1, LL_DMA_CHANNEL_1);
    

    LL_TIM_EnableCounter(TIM1);
 
    hTIM1_CR1 = TIM1->CR1;
    hTIM1_CR1 |= TIM_CR1_CEN;
 
    pHandle->_Super.DTTest = 0u;
    pHandle->_Super.DTCompCnt = pHandle->_Super.hDTCompCnt;

  }
}

/**
  * @brief  It initializes TIMx for PWM generation,
  *          active vector insertion and adc triggering.
  * @param  TIMx Timer to be initialized
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval none
  */
__weak void R1G0XX_TIMxInit(TIM_TypeDef* TIMx, PWMC_Handle_t *pHdl)
{

  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;

  LL_TIM_CC_EnableChannel(TIMx, LL_TIM_CHANNEL_CH1);
  LL_TIM_CC_EnableChannel(TIMx, LL_TIM_CHANNEL_CH2);
  LL_TIM_CC_EnableChannel(TIMx, LL_TIM_CHANNEL_CH3);

  if ((pHandle->_Super.LowSideOutputs)== LS_PWM_TIMER)
  {
     LL_TIM_CC_EnableChannel(TIMx, LL_TIM_CHANNEL_CH1N);
     LL_TIM_CC_EnableChannel(TIMx, LL_TIM_CHANNEL_CH2N);
     LL_TIM_CC_EnableChannel(TIMx, LL_TIM_CHANNEL_CH3N);
  }

  LL_TIM_OC_EnablePreload(TIMx, LL_TIM_CHANNEL_CH1);
  LL_TIM_OC_EnablePreload(TIMx, LL_TIM_CHANNEL_CH2);
  LL_TIM_OC_EnablePreload(TIMx, LL_TIM_CHANNEL_CH3);
  LL_TIM_OC_EnablePreload(TIMx, LL_TIM_CHANNEL_CH4);
  LL_TIM_OC_EnablePreload(TIMx, LL_TIM_CHANNEL_CH5);
  LL_TIM_OC_EnablePreload(TIMx, LL_TIM_CHANNEL_CH6);

  LL_TIM_OC_SetDeadTime(TIMx, (pHandle->pParams_str->hDeadTime)/2u);
 

}

/**
  * @brief  It stores into handler the voltage present on the
  *         current feedback analog channel when no current is flowin into the
  *         motor
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval none
  */
__weak void R1G0XX_CurrentReadingCalibration(PWMC_Handle_t *pHdl)
{
  uint8_t bIndex = 0u;
  uint32_t wPhaseOffset = 0u;

  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;

  /* Set the CALIB flags to indicate the ADC calibartion phase*/
  pHandle->hFlags |= CALIB;
  
  /* ADC Channel and sampling time config for current reading */
  ADC1->CHSELR = 1 << pHandle->pParams_str->hIChannel;
  
  /* Disable DMA1 Channel1 */
  LL_DMA_DisableChannel(DMA1, LL_DMA_CHANNEL_1);
  
  /* ADC Channel used for current reading are read 
  in order to get zero currents ADC values*/   
  while (bIndex< NB_CONVERSIONS)
  {     
    /* Software start of conversion */
    LL_ADC_REG_StartConversion(ADC1);
    
    /* Wait until end of regular conversion */
    while (LL_ADC_IsActiveFlag_EOC(ADC1)==RESET)
    {}    
    
    wPhaseOffset += LL_ADC_REG_ReadConversionData12(ADC1);
    bIndex++;
  }
  
  pHandle->hPhaseOffset = (uint16_t)(wPhaseOffset/NB_CONVERSIONS);
  
  /* Reset the CALIB flags to indicate the end of ADC calibartion phase*/
  pHandle->hFlags &= (~CALIB);
  
}

/**
  * @brief  First initialization of class members
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval none
  */
__weak void R1G0XX_1ShuntMotorVarsInit(PWMC_Handle_t *pHdl)
{ 
  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;
  
  /* Init motor vars */
  pHandle->hPhaseOffset=0u;
  pHandle->bInverted_pwm_new=INVERT_NONE;
  pHandle->hFlags &= (~STBD3);
  pHandle->hFlags &= (~DSTEN);
  
  /* After reset value of dvDutyValues */
  pHandle->_Super.hCntPhA = pHandle->Half_PWMPeriod >> 1;
  pHandle->_Super.hCntPhB = pHandle->Half_PWMPeriod >> 1;
  pHandle->_Super.hCntPhC = pHandle->Half_PWMPeriod >> 1;
  
  /* Default value of DutyValues */
  pHandle->hCntSmp1 = (pHandle->Half_PWMPeriod >> 1) - pHandle->pParams_str->hTbefore;
  pHandle->hCntSmp2 = (pHandle->Half_PWMPeriod >> 1) + pHandle->pParams_str->hTafter;
  
  TIM1->CCR4 =  pHandle->hCntSmp1; /* First point */
  TIM1->CCR6 =  pHandle->hCntSmp2; /* Second point */ 
  /* Init of "regular" conversion registers */ 
  pHandle->bRegConvRequested = 0u;
  pHandle->bRegConvIndex = 0u;

}

/**
  * @brief  Initialization of class members after each motor start
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval none
  */
__weak void R1G0XX_1ShuntMotorVarsRestart(PWMC_Handle_t *pHdl)
{
  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;
  
  /* Default value of DutyValues */
  pHandle->hCntSmp1 = (pHandle->Half_PWMPeriod >> 1) - pHandle->pParams_str->hTbefore;
  pHandle->hCntSmp2 = (pHandle->Half_PWMPeriod >> 1) + pHandle->pParams_str->hTafter;
  
  pHandle->bInverted_pwm_new=INVERT_NONE;
  pHandle->hFlags &= (~STBD3); /*STBD3 cleared*/
  
  TIM1->CCR4 =  pHandle->hCntSmp1; /* First point */
  TIM1->CCR6 =  pHandle->hCntSmp2; /* Second point */ 

  /* After start value of dvDutyValues */
  pHandle->_Super.hCntPhA = pHandle->Half_PWMPeriod >> 1;
  pHandle->_Super.hCntPhB = pHandle->Half_PWMPeriod >> 1;
  pHandle->_Super.hCntPhC = pHandle->Half_PWMPeriod >> 1;
  
  /* Set the default previous value of Phase A,B,C current */
  pHandle->hCurrAOld=0;
  pHandle->hCurrBOld=0;


   }

/**
  * @brief  It computes and return latest converted motor phase currents motor
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval Curr_Components Ia and Ib current in Curr_Components format
  */
__weak void R1G0XX_GetPhaseCurrents(PWMC_Handle_t *pHdl,Curr_Components* pStator_Currents)
{  
  int32_t wAux;
  int16_t hCurrA = 0;
  int16_t hCurrB = 0;
  int16_t hCurrC = 0;
  uint8_t bCurrASamp = 0u;
  uint8_t bCurrBSamp = 0u;
  uint8_t bCurrCSamp = 0u;
  
  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;

  /* Disabling the External triggering for ADCx*/
  LL_ADC_REG_SetTriggerSource (ADC1, LL_ADC_REG_TRIG_SOFTWARE);
  
  /* Reset the update flag to indicate the start of FOC algorithm*/
  LL_TIM_ClearFlag_UPDATE(TIM1);
  
  /* First sampling point */
  wAux = (int32_t)(pHandle->hCurConv[0]) - (int32_t)(pHandle->hPhaseOffset);
  
  /* Check saturation */
  wAux = (wAux > S16_MIN) ? ((wAux < S16_MAX) ? wAux : S16_MAX) : S16_MIN;
  
  switch (pHandle->sampCur1)
  {
  case SAMP_IA:
    hCurrA = (int16_t)(wAux);
    bCurrASamp = 1u;
    break;
  case SAMP_IB:
    hCurrB = (int16_t)(wAux);
    bCurrBSamp = 1u;
    break;
  case SAMP_IC:
    hCurrC = (int16_t)(wAux);
    bCurrCSamp = 1u;
    break;
  case SAMP_NIA:
    wAux = -wAux;
    hCurrA = (int16_t)(wAux);
    bCurrASamp = 1u;
    break;
  case SAMP_NIB:
    wAux = -wAux;
    hCurrB = (int16_t)(wAux);
    bCurrBSamp = 1u;
    break;
  case SAMP_NIC:
    wAux = -wAux;
    hCurrC = (int16_t)(wAux);
    bCurrCSamp = 1u;
    break;
  case SAMP_OLDA:
    hCurrA = pHandle->hCurrAOld;
    bCurrASamp = 1u;
    break;
  case SAMP_OLDB:
    hCurrB = pHandle->hCurrBOld;
    bCurrBSamp = 1u;
    break;
  default:
    break;
  }
  
  /* Second sampling point */
  wAux = (int32_t)(pHandle->hCurConv[1]) - (int32_t)(pHandle->hPhaseOffset);
  
  wAux = (wAux > S16_MIN) ? ((wAux < S16_MAX) ? wAux : S16_MAX) : S16_MIN;

  
  switch (pHandle->sampCur2)
  {
  case SAMP_IA:
    hCurrA = (int16_t)(wAux);
    bCurrASamp = 1u;
    break;
  case SAMP_IB:
    hCurrB = (int16_t)(wAux);
    bCurrBSamp = 1u;
    break;
  case SAMP_IC:
    hCurrC = (int16_t)(wAux);
    bCurrCSamp = 1u;
    break;
  case SAMP_NIA:
    wAux = -wAux; 
    hCurrA = (int16_t)(wAux);
    bCurrASamp = 1u;
    break;
  case SAMP_NIB:
    wAux = -wAux; 
    hCurrB = (int16_t)(wAux);
    bCurrBSamp = 1u;
    break;
  case SAMP_NIC:
    wAux = -wAux; 
    hCurrC = (int16_t)(wAux);
    bCurrCSamp = 1u;
    break;
  default:
    break;
  }
    
  /* Computation of the third value */
  if (bCurrASamp == 0u)
  {
    wAux = -((int32_t)(hCurrB)) -((int32_t)(hCurrC));
    
    /* Check saturation */
	wAux = (wAux > S16_MIN) ? ((wAux < S16_MAX) ? wAux : S16_MAX) : S16_MIN;
    
    hCurrA = (int16_t)wAux; 
  }
  if (bCurrBSamp == 0u)
  {
    wAux = -((int32_t)(hCurrA)) -((int32_t)(hCurrC));
    
    /* Check saturation */
	wAux = (wAux > S16_MIN) ? ((wAux < S16_MAX) ? wAux : S16_MAX) : S16_MIN;
    
    hCurrB = (int16_t)wAux;
  }
  if (bCurrCSamp == 0u)
  {
    wAux = -((int32_t)(hCurrA)) -((int32_t)(hCurrB));
    
    /* Check saturation */
	wAux = (wAux > S16_MIN) ? ((wAux < S16_MAX) ? wAux : S16_MAX) : S16_MIN;
    
    hCurrC = (int16_t)wAux;
  }
  
  /* hCurrA, hCurrB, hCurrC values are the sampled values */
    
  pHandle->hCurrAOld = hCurrA;
  pHandle->hCurrBOld = hCurrB;

  pStator_Currents->qI_Component1 = hCurrA;
  pStator_Currents->qI_Component2 = hCurrB;
  
  pHandle->_Super.hIa = pStator_Currents->qI_Component1;
  pHandle->_Super.hIb = pStator_Currents->qI_Component2;
  pHandle->_Super.hIc = -pStator_Currents->qI_Component1 - pStator_Currents->qI_Component2;

  if (pHandle->bRegConvRequested != 0u)
  {
    /* Exec regular conversion @ bRegConvIndex */
    uint8_t bRegConvCh = pHandle->bRegConvCh[pHandle->bRegConvIndex];
    
    /* Set Sampling time and channel */
    ADC1->CHSELR = 1 <<  bRegConvCh;
    if (pHandle->bADCSMP2 != pHandle->bRegSmpTime[bRegConvCh] )
    {
      LL_ADC_SetSamplingTimeCommonChannels (ADC1, LL_ADC_SAMPLINGTIME_COMMON_2, pHandle->bRegSmpTime[bRegConvCh]);
      pHandle->bADCSMP2 = pHandle->bRegSmpTime[bRegConvCh];
    }
    
    /* Enable ADC1 EOC DMA */
    LL_ADC_REG_SetDMATransfer(ADC1, LL_ADC_REG_DMA_TRANSFER_NONE);
    
    /* Start ADC */
    LL_ADC_REG_StartConversion(ADC1);
    
    /* Flags the regular conversion ongoing */
    pHandle->hFlags |= REGCONVONGOING;
  }
}

/**
  * @brief  It turns on low sides switches. This function is intended to be
  *         used for charging boot capacitors of driving section. It has to be
  *         called each motor start-up when using high voltage drivers
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval none
  */
__weak void R1G0XX_TurnOnLowSides(PWMC_Handle_t *pHdl)
{
  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;

  pHandle->_Super.bTurnOnLowSidesAction = TRUE;

  TIM1->CCR1 = 0u;
  TIM1->CCR2 = 0u;
  TIM1->CCR3 = 0u;
  
  LL_TIM_ClearFlag_UPDATE(TIM1);
  while (LL_TIM_IsActiveFlag_UPDATE(TIM1) == RESET)
  {}
  
  /* Main PWM Output Enable */
  LL_TIM_EnableAllOutputs(TIM1);
  if ((pHandle->_Super.LowSideOutputs)== ES_GPIO)
  {
    LL_GPIO_SetOutputPin(pHandle->pParams_str->hCh1NPort, pHandle->pParams_str->hCh1NPin);
    LL_GPIO_SetOutputPin(pHandle->pParams_str->hCh2NPort, pHandle->pParams_str->hCh2NPin);
    LL_GPIO_SetOutputPin(pHandle->pParams_str->hCh3NPort, pHandle->pParams_str->hCh3NPin);
  }
  return; 
}

/**
  * @brief  It enables PWM generation on the proper Timer peripheral acting on
  *         MOE bit, enaables the single shunt distortion and reset the TIM status
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval none
  */
__weak void R1G0XX_SwitchOnPWM(PWMC_Handle_t *pHdl)
{
  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;
  
  pHandle->_Super.bTurnOnLowSidesAction = FALSE;
  /* enable break Interrupt */
  LL_TIM_ClearFlag_BRK(TIM1);
  LL_TIM_EnableIT_BRK(TIM1);


  LL_TIM_ClearFlag_UPDATE(TIM1);
  while (LL_TIM_IsActiveFlag_UPDATE(TIM1)==RESET)
  {}
  LL_TIM_ClearFlag_UPDATE(TIM1);
 /* Set all duty to 50% */
  /* Set ch4 ch6 for triggering */
  /* Clear Update Flag */

  LL_TIM_OC_SetCompareCH1(TIM1,(uint32_t)(pHandle->Half_PWMPeriod >> 1));
  LL_TIM_OC_SetCompareCH2(TIM1,(uint32_t)(pHandle->Half_PWMPeriod >> 1));
  LL_TIM_OC_SetCompareCH3(TIM1,(uint32_t)(pHandle->Half_PWMPeriod >> 1));
  LL_TIM_OC_SetCompareCH4(TIM1,(((uint32_t)(pHandle->Half_PWMPeriod >> 1)) + (uint32_t)pHandle->pParams_str->hTafter));
  LL_TIM_OC_SetCompareCH6(TIM1,(uint32_t)(pHandle->Half_PWMPeriod - 1u)); // CHO : temporary discard ch6

  while (LL_TIM_IsActiveFlag_UPDATE(TIM1)==RESET)
  {}
  /* dirty trick because the DMA is fired as soon as channel is enabled ...*/
  LL_DMA_EnableChannel(DMA1, LL_DMA_CHANNEL_1);
  LL_DMA_DisableChannel(DMA1, LL_DMA_CHANNEL_1);
  LL_DMA_SetDataLength(DMA1, LL_DMA_CHANNEL_1, 2u);
  LL_DMA_EnableChannel(DMA1, LL_DMA_CHANNEL_1);

  
  /* Main PWM Output Enable */  
  LL_TIM_EnableAllOutputs(TIM1);
  
  /* TIM output trigger 2 for ADC */
  LL_TIM_SetTriggerOutput2(TIM1, LL_TIM_TRGO2_OC4_RISING_OC6_RISING);



  /* Main PWM Output Enable */
  LL_TIM_ClearFlag_UPDATE(TIM1);
  LL_TIM_EnableIT_UPDATE(TIM1);

  if ((pHandle->_Super.LowSideOutputs)== ES_GPIO)
  {
    LL_GPIO_SetOutputPin(pHandle->pParams_str->hCh1NPort, pHandle->pParams_str->hCh1NPin);
    LL_GPIO_SetOutputPin(pHandle->pParams_str->hCh2NPort, pHandle->pParams_str->hCh2NPin);
    LL_GPIO_SetOutputPin(pHandle->pParams_str->hCh3NPort, pHandle->pParams_str->hCh3NPin);
  }
  

  
  /* Enabling distortion for single shunt */
  pHandle->hFlags |= DSTEN;
  return; 
}

/**
  * @brief  It disables PWM generation on the proper Timer peripheral acting on
  *         MOE bit, disables the single shunt distortion and reset the TIM status
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval none
  */
__weak void R1G0XX_SwitchOffPWM(PWMC_Handle_t *pHdl)
{
  uint16_t hAux;

  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;

  pHandle->_Super.bTurnOnLowSidesAction = FALSE;
  
  /* Main PWM Output Disable */
  LL_TIM_DisableAllOutputs(TIM1);
  if ((pHandle->_Super.LowSideOutputs)== ES_GPIO)
  {
    LL_GPIO_ResetOutputPin(pHandle->pParams_str->hCh1NPort, pHandle->pParams_str->hCh1NPin);
    LL_GPIO_ResetOutputPin(pHandle->pParams_str->hCh2NPort, pHandle->pParams_str->hCh2NPin);
    LL_GPIO_ResetOutputPin(pHandle->pParams_str->hCh3NPort, pHandle->pParams_str->hCh3NPin);
  }
  
  /* Disable UPDATE ISR */
  LL_TIM_DisableIT_UPDATE(TIM1);
  /* Disable break interrupt */
  LL_TIM_DisableIT_BRK(TIM1);
  
  /*Clear potential ADC Ongoing conversion*/
  if (LL_ADC_REG_IsConversionOngoing (ADC1))
  {
    LL_ADC_REG_StopConversion (ADC1);
    while ( LL_ADC_REG_IsConversionOngoing(ADC1))
    {
    }
  }
  LL_ADC_REG_SetTriggerSource (ADC1, LL_ADC_REG_TRIG_SOFTWARE);
    
  /* Disabling distortion for single */
  pHandle->hFlags &= (~DSTEN);

  while (LL_TIM_IsActiveFlag_UPDATE(TIM1)==RESET)
  {}

  
  /* Set all duty to 50% */
  hAux = pHandle->Half_PWMPeriod >> 1;
  TIM1->CCR1 = hAux;
  TIM1->CCR2 = hAux;
  TIM1->CCR3 = hAux;    
    
  return; 
}

/**
  * @brief  Implementation of the single shunt algorithm to setup the
  *         TIM1 register and DMA buffers values for the next PWM period.
  * @param  pHdl: handler of the current instance of the PWM component
  * @retval uint16_t It returns #MC_DURATION if the TIMx update occurs
  *          before the end of FOC algorithm else returns #MC_NO_ERROR
  */
__weak uint16_t R1G0XX_CalcDutyCycles(PWMC_Handle_t *pHdl)
{
  int16_t hDeltaDuty_0;
  int16_t hDeltaDuty_1;
  uint16_t hDutyV_0 = 0u;
  uint16_t hDutyV_1 = 0u;
  uint16_t hDutyV_2 = 0u;
  uint8_t bSector;
  uint8_t bStatorFluxPos;
  uint16_t hAux;

  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;
    
  bSector = (uint8_t)pHandle->_Super.hSector;
  
  if ((pHandle->hFlags & DSTEN) != 0u)
  { 
    switch (bSector)
    {
    case SECTOR_1:
      hDutyV_2 = pHandle->_Super.hCntPhA;
      hDutyV_1 = pHandle->_Super.hCntPhB;
      hDutyV_0 = pHandle->_Super.hCntPhC;
      break;
    case SECTOR_2:
      hDutyV_2 = pHandle->_Super.hCntPhB;
      hDutyV_1 = pHandle->_Super.hCntPhA;
      hDutyV_0 = pHandle->_Super.hCntPhC;
      break;
    case SECTOR_3:
      hDutyV_2 = pHandle->_Super.hCntPhB;
      hDutyV_1 = pHandle->_Super.hCntPhC;
      hDutyV_0 = pHandle->_Super.hCntPhA;
      break;
    case SECTOR_4:
      hDutyV_2 = pHandle->_Super.hCntPhC;
      hDutyV_1 = pHandle->_Super.hCntPhB;
      hDutyV_0 = pHandle->_Super.hCntPhA;
      break;
    case SECTOR_5:
      hDutyV_2 = pHandle->_Super.hCntPhC;
      hDutyV_1 = pHandle->_Super.hCntPhA;
      hDutyV_0 = pHandle->_Super.hCntPhB;
      break;
    case SECTOR_6:
      hDutyV_2 = pHandle->_Super.hCntPhA;
      hDutyV_1 = pHandle->_Super.hCntPhC;
      hDutyV_0 = pHandle->_Super.hCntPhB;
      break;
    default:
      break;
    }
    
    /* Compute delta duty */
    hDeltaDuty_0 = (int16_t)(hDutyV_1) - (int16_t)(hDutyV_0);
    hDeltaDuty_1 = (int16_t)(hDutyV_2) - (int16_t)(hDutyV_1);
    
    /* Check region */
    if ((uint16_t)hDeltaDuty_0<=pHandle->pParams_str->hTMin)
    {
      if ((uint16_t)hDeltaDuty_1<=pHandle->pParams_str->hTMin)
      {
        bStatorFluxPos = BOUNDARY_3;
      }
      else
      {
        bStatorFluxPos = BOUNDARY_2;
      }
    } 
    else 
    {
      if ((uint16_t)hDeltaDuty_1>pHandle->pParams_str->hTMin)
      {
        bStatorFluxPos = REGULAR;
      }
      else
      {
        bStatorFluxPos = BOUNDARY_1;
      }
    }
            
    if (bStatorFluxPos == REGULAR)
    {
      pHandle->bInverted_pwm_new = INVERT_NONE;
    }
    else if (bStatorFluxPos == BOUNDARY_1) /* Adjust the lower */
    {
      switch (bSector)
      {
      case SECTOR_5:
      case SECTOR_6:
        if (pHandle->_Super.hCntPhA - pHandle->pParams_str->hHTMin - hDutyV_0 > pHandle->pParams_str->hTMin)
        {
          pHandle->bInverted_pwm_new = INVERT_A;
          pHandle->_Super.hCntPhA -=pHandle->pParams_str->hHTMin;
          if (pHandle->_Super.hCntPhA < hDutyV_1)
          {
            hDutyV_1 = pHandle->_Super.hCntPhA;
          }
        }
        else
        {
          bStatorFluxPos = BOUNDARY_3;
          if ((pHandle->hFlags & STBD3) == 0u)
          {
            pHandle->bInverted_pwm_new = INVERT_A;
            pHandle->_Super.hCntPhA -=pHandle->pParams_str->hHTMin;
            pHandle->hFlags |= STBD3;
          } 
          else
          {
            pHandle->bInverted_pwm_new = INVERT_B;
            pHandle->_Super.hCntPhB -=pHandle->pParams_str->hHTMin;
            pHandle->hFlags &= (~STBD3);
          }
        }
        break;
      case SECTOR_2:
      case SECTOR_1:
        if (pHandle->_Super.hCntPhB - pHandle->pParams_str->hHTMin - hDutyV_0 > pHandle->pParams_str->hTMin)
        {
          pHandle->bInverted_pwm_new = INVERT_B;
          pHandle->_Super.hCntPhB -=pHandle->pParams_str->hHTMin;
          if (pHandle->_Super.hCntPhB < hDutyV_1)
          {
            hDutyV_1 = pHandle->_Super.hCntPhB;
          }
        }
        else
        {
          bStatorFluxPos = BOUNDARY_3;
          if ((pHandle->hFlags & STBD3) == 0u)
          {
            pHandle->bInverted_pwm_new = INVERT_A;
            pHandle->_Super.hCntPhA -=pHandle->pParams_str->hHTMin;
            pHandle->hFlags |= STBD3;
          } 
          else
          {
            pHandle->bInverted_pwm_new = INVERT_B;
            pHandle->_Super.hCntPhB -=pHandle->pParams_str->hHTMin;
            pHandle->hFlags &= (~STBD3);
          }
        }
        break;
      case SECTOR_4:
      case SECTOR_3:
        if (pHandle->_Super.hCntPhC - pHandle->pParams_str->hHTMin - hDutyV_0 > pHandle->pParams_str->hTMin)
        {
          pHandle->bInverted_pwm_new = INVERT_C;
          pHandle->_Super.hCntPhC -=pHandle->pParams_str->hHTMin;
          if (pHandle->_Super.hCntPhC < hDutyV_1)
          {
            hDutyV_1 = pHandle->_Super.hCntPhC;
          }
        }
        else
        {
          bStatorFluxPos = BOUNDARY_3;
          if ((pHandle->hFlags & STBD3) == 0u)
          {
            pHandle->bInverted_pwm_new = INVERT_A;
            pHandle->_Super.hCntPhA -=pHandle->pParams_str->hHTMin;
            pHandle->hFlags |= STBD3;
          } 
          else
          {
            pHandle->bInverted_pwm_new = INVERT_B;
            pHandle->_Super.hCntPhB -=pHandle->pParams_str->hHTMin;
            pHandle->hFlags &= (~STBD3);
          }
        }
        break;
      default:
        break;
      }
    }
    else if (bStatorFluxPos == BOUNDARY_2) /* Adjust the middler */
    {
      switch (bSector)
      {
      case SECTOR_4:
      case SECTOR_5: /* Invert B */
        pHandle->bInverted_pwm_new = INVERT_B;
        pHandle->_Super.hCntPhB -=pHandle->pParams_str->hHTMin;
        if (pHandle->_Super.hCntPhB > 0xEFFFu)
        {
          pHandle->_Super.hCntPhB = 0u;
        }
        break;
      case SECTOR_2:
      case SECTOR_3: /* Invert A */
        pHandle->bInverted_pwm_new = INVERT_A;
        pHandle->_Super.hCntPhA -=pHandle->pParams_str->hHTMin;
        if (pHandle->_Super.hCntPhA > 0xEFFFu)
        {
          pHandle->_Super.hCntPhA = 0u;
        }
        break;
      case SECTOR_6:
      case SECTOR_1: /* Invert C */
        pHandle->bInverted_pwm_new = INVERT_C;
        pHandle->_Super.hCntPhC -=pHandle->pParams_str->hHTMin;
        if (pHandle->_Super.hCntPhC > 0xEFFFu)
        {
          pHandle->_Super.hCntPhC = 0u;
        }
        break;
      default:
        break;
      }
    }
    else
    {
      if ((pHandle->hFlags & STBD3) == 0u)
      {
        pHandle->bInverted_pwm_new = INVERT_A;
        pHandle->_Super.hCntPhA -=pHandle->pParams_str->hHTMin;
        pHandle->hFlags |= STBD3;
      } 
      else
      {
        pHandle->bInverted_pwm_new = INVERT_B;
        pHandle->_Super.hCntPhB -=pHandle->pParams_str->hHTMin;
        pHandle->hFlags &= (~STBD3);
      }
    }
        
    if (bStatorFluxPos == REGULAR) /* Regular zone */
    {
      /* First point */
  /*    if ((hDutyV_1 - hDutyV_0 - pHandle->pParams_str->hDeadTime)> pHandle->pParams_str->hMaxTrTs)
      {
        pHandle->hCntSmp1 = hDutyV_0 + hDutyV_1 + pHandle->pParams_str->hDeadTime;
        pHandle->hCntSmp1 >>= 1;
      }
      else
      { */
        pHandle->hCntSmp1 = hDutyV_1 - pHandle->pParams_str->hTbefore;
     /* }*/
      /* Second point */
    /*  if ((hDutyV_2 - hDutyV_1 - pHandle->pParams_str->hDeadTime)> pHandle->pParams_str->hMaxTrTs)
      {
        pHandle->hCntSmp2 = hDutyV_1 + hDutyV_2 + pHandle->pParams_str->hDeadTime;
        pHandle->hCntSmp2 >>= 1;
      }
      else
      {*/
        pHandle->hCntSmp2 = hDutyV_2 - pHandle->pParams_str->hTbefore;
    /*  }*/
    }
    
    if (bStatorFluxPos == BOUNDARY_1) /* Two small, one big */
    {      
      /* First point */
    /*  if ((hDutyV_1 - hDutyV_0 - pHandle->pParams_str->hDeadTime)> pHandle->pParams_str->hMaxTrTs)
      {
        pHandle->hCntSmp1 = hDutyV_0 + hDutyV_1 + pHandle->pParams_str->hDeadTime;
        pHandle->hCntSmp1 >>= 1;
      }
      else 
      { */
        pHandle->hCntSmp1 = hDutyV_1 - pHandle->pParams_str->hTbefore;
     /* }*/
      /* Second point */
      pHandle->hCntSmp2 = pHandle->Half_PWMPeriod - pHandle->pParams_str->hHTMin + pHandle->pParams_str->hTSample;
    }
    
    if (bStatorFluxPos == BOUNDARY_2) /* Two big, one small */
    {
      /* First point */
   /*   if ((hDutyV_2 - hDutyV_1 - pHandle->pParams_str->hDeadTime)>= pHandle->pParams_str->hMaxTrTs)
      {
        pHandle->hCntSmp1 = hDutyV_1 + hDutyV_2 + pHandle->pParams_str->hDeadTime;
        pHandle->hCntSmp1 >>= 1;
      }
      else
      { */
        pHandle->hCntSmp1 = hDutyV_2 - pHandle->pParams_str->hTbefore;
    /*  }*/
      /* Second point */
      pHandle->hCntSmp2 = pHandle->Half_PWMPeriod - pHandle->pParams_str->hHTMin + pHandle->pParams_str->hTSample;
    }
    
    if (bStatorFluxPos == BOUNDARY_3)  
    {
      /* First point */
      pHandle->hCntSmp1 = hDutyV_0-pHandle->pParams_str->hTbefore; /* Dummy trigger */
      /* Second point */
      pHandle->hCntSmp2 = pHandle->Half_PWMPeriod - pHandle->pParams_str->hHTMin + pHandle->pParams_str->hTSample;
    }
  }
  else
  {
    pHandle->bInverted_pwm_new = INVERT_NONE;
    bStatorFluxPos = REGULAR;
  }

 
  /* Update Timer Ch 4,6 for ADC triggering and books the queue*/
/*  LL_TIM_OC_DisablePreload(TIM1, LL_TIM_CHANNEL_CH4);
  LL_TIM_OC_DisablePreload(TIM1, LL_TIM_CHANNEL_CH6);
  TIM1->CCR4 = 0x0u;
  TIM1->CCR6 = 0xFFFFu;
  LL_TIM_OC_EnablePreload(TIM1, LL_TIM_CHANNEL_CH4);
  LL_TIM_OC_EnablePreload(TIM1, LL_TIM_CHANNEL_CH6);  
*/
    /* Update ADC Trigger */
    TIM1->CCR4 =  pHandle->hCntSmp1; /* First point */
    TIM1->CCR6 =  pHandle->hCntSmp2; /* Second point */
    
 if (bStatorFluxPos == REGULAR)
  {
    LL_TIM_SetTriggerOutput2(TIM1, LL_TIM_TRGO2_OC4_RISING_OC6_RISING);
	// LL_TIM_SetCH5CombinedChannels(TIM1, LL_TIM_GROUPCH5_NONE);
    MODIFY_REG(TIM1->CCR5, (0x7 << 29), LL_TIM_GROUPCH5_NONE);
  }
 else {
    switch (pHandle->bInverted_pwm_new)
    {
      case INVERT_A:
        //LL_TIM_SetCH5CombinedChannels(TIM1, LL_TIM_GROUPCH5_OC1REFC);
        MODIFY_REG(TIM1->CCR5, (0x7 << 29), LL_TIM_GROUPCH5_OC1REFC);
        break;

      case INVERT_B:
        //LL_TIM_SetCH5CombinedChannels(TIM1, LL_TIM_GROUPCH5_OC2REFC);
        MODIFY_REG(TIM1->CCR5, (0x7 << 29), LL_TIM_GROUPCH5_OC2REFC);
        break;

      case INVERT_C:
	//LL_TIM_SetCH5CombinedChannels(TIM1, LL_TIM_GROUPCH5_OC3REFC);
        MODIFY_REG(TIM1->CCR5, (0x7 << 29), LL_TIM_GROUPCH5_OC3REFC);
        break;

      default:
        break;
    }    
    LL_TIM_SetTriggerOutput2(TIM1, LL_TIM_TRGO2_OC4_RISING_OC6_FALLING);
 }
 
   /* Update Timer Ch 1,2,3 (These value are required before update event) */
  TIM1->CCR1 = pHandle->_Super.hCntPhA;
  TIM1->CCR2 = pHandle->_Super.hCntPhB;
  TIM1->CCR3 = pHandle->_Super.hCntPhC;
   
  LL_GPIO_ResetOutputPin (GPIOB, LL_GPIO_PIN_3);

    /*check software error*/
  if (LL_TIM_IsActiveFlag_UPDATE(TIM1))
  {
    hAux = MC_DURATION;
    LL_GPIO_SetOutputPin (GPIOB, LL_GPIO_PIN_10);
    while (1)
    {
    }
    
  }
  else
  {
    hAux = MC_NO_ERROR;
  }
  if (pHandle->_Super.SWerror == 1u)
  {
    hAux = MC_DURATION;
    pHandle->_Super.SWerror = 0u;
  }  
  
  if ((pHandle->hFlags & REGCONVONGOING)==0u)
  {
  }
  else
  {
    pHandle->hRegConvValue[pHandle->bRegConvIndex] = (uint16_t)(ADC1->DR);
    
    /* ADC Channel and sampling time config for current reading */
    ADC1->CHSELR = 1u <<  pHandle->pParams_str->hIChannel;
    
    /* Enable ADC1 EOC DMA */
    LL_ADC_REG_SetDMATransfer(ADC1, LL_ADC_REG_DMA_TRANSFER_LIMITED);
    
    /* Clear regular conversion ongoing flag */
    pHandle->hFlags &= (uint16_t)~REGCONVONGOING;
    
    /* Prepare next conversion */
    pHandle->bRegConvIndex++;
    
    if (pHandle->bRegConvIndex >= pHandle->bRegConvRequested)
    {
      pHandle->bRegConvIndex = 0u;
    }
  }
      
  /* The following instruction can be executed after Update handler 
     before the get phase current (Second EOC) */
      
  /* Set the current sampled */
   if (bStatorFluxPos == REGULAR) /* Regual zone */
  {
    pHandle->sampCur1 = REGULAR_SAMP_CUR1[bSector];
    pHandle->sampCur2 = REGULAR_SAMP_CUR2[bSector];
  }
  
  if (bStatorFluxPos == BOUNDARY_1) /* Two small, one big */
  {
    pHandle->sampCur1 = REGULAR_SAMP_CUR1[bSector];
    pHandle->sampCur2 = BOUNDR1_SAMP_CUR2[bSector];
  }
  
  if (bStatorFluxPos == BOUNDARY_2) /* Two big, one small */
  {
    pHandle->sampCur1 = BOUNDR2_SAMP_CUR1[bSector];
    pHandle->sampCur2 = BOUNDR2_SAMP_CUR2[bSector];
  }
  
  if (bStatorFluxPos == BOUNDARY_3)  
  {
    if (pHandle->bInverted_pwm_new == INVERT_A)
    {
      pHandle->sampCur1 = SAMP_OLDB;
      pHandle->sampCur2 = SAMP_IA;
    }
    if (pHandle->bInverted_pwm_new == INVERT_B)
    {
      pHandle->sampCur1 = SAMP_OLDA;
      pHandle->sampCur2 = SAMP_IB;
    }
  }
    
  /* Limit for the Get Phase current (Second EOC Handler) */
      
  return (hAux);
}

/**
  * @brief  R1_G0XX implement MC IRQ function TIMER Update
  * @param  this related object
  * @retval void* It returns always #MC_NULL
  */
__weak void R1G0XX_TIMx_UP_IRQHandler(PWMC_Handle_t *pHdl)
{ 

  LL_ADC_REG_SetTriggerSource (ADC1, LL_ADC_REG_TRIG_EXT_TIM1_TRGO2);
  LL_ADC_REG_StartConversion (ADC1);

}

/**
  * @brief  Execute a regular conversion.
  *         The function is not re-entrant (can't executed twice at the same time)
  *         It returns 0xFFFF in case of conversion error.
  * @param  pHdl: handler of the current instance of the PWM component, ADC channel to be converted
  * @param  bChannel ADC channel used for the regular conversion
  * @retval uint16_t It returns converted value or oxFFFF for conversion error
  */
__weak uint16_t R1G0XX_ExecRegularConv(PWMC_Handle_t *pHdl, uint8_t bChannel)
{
  uint16_t hRetVal = 0xFFFFu;
  uint8_t i;
  bool bRegChFound = FALSE;

  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;
  
  if (bChannel < 18u)
  {
    /* Check if the channel has been already requested */
    for (i = 0u; i < pHandle->bRegConvRequested; i++)
    {
      if (pHandle->bRegConvCh[i] == bChannel)
      {
        hRetVal = pHandle->hRegConvValue[i];
        bRegChFound = TRUE;
        break;
      }
    }
    if (bRegChFound == FALSE)
    {
      if (pHandle->bRegConvRequested < MAX_REG_CONVERSIONS)
      {
        /* Add new channel to the list */
        pHandle->bRegConvCh[pHandle->bRegConvRequested] = bChannel;
        i = pHandle->bRegConvRequested;
        pHandle->bRegConvRequested++;
      }
    }
    if ((pHandle->hFlags & CALIB) == 0u)
    {
      if ((TIM1->DIER & LL_TIM_DIER_UIE)!= LL_TIM_DIER_UIE)
      {
        /* If the Timer update IT is not enabled, the PWM is switch Off */
        /* We can start the "regular" conversion immediately */
        /* Otherwise, the conversion is done within High frequency task 
        after the current sampling*/
        
        /* Set Sampling time and channel */
        ADC1->CHSELR = 1u <<  bChannel;
        if (pHandle->bADCSMP2 != pHandle->bRegSmpTime[bChannel] )
        {
          LL_ADC_SetSamplingTimeCommonChannels (ADC1, LL_ADC_SAMPLINGTIME_COMMON_2, pHandle->bRegSmpTime[bChannel]);
          pHandle->bADCSMP2 = pHandle->bRegSmpTime[bChannel];
        }
        
        
        /* Disable ADC1 EOC DMA */
        LL_ADC_REG_SetDMATransfer(ADC1, LL_ADC_REG_DMA_TRANSFER_NONE);
        
        /* Disabling the External triggering for ADCx*/
        LL_ADC_REG_SetTriggerSource (ADC1, LL_ADC_REG_TRIG_SOFTWARE );
        
        /* Clear EOC */
        LL_ADC_ClearFlag_EOC(ADC1);
        
        /* Start ADC */
        LL_ADC_REG_StartConversion(ADC1);
        
        /* Wait EOC */
        while (LL_ADC_IsActiveFlag_EOC(ADC1) == RESET)
        {
        }
        
        /* Read the "Regular" conversion (Not related to current sampling) */
        hRetVal = LL_ADC_REG_ReadConversionData12(ADC1);
        pHandle->hRegConvValue[i] = hRetVal;
        
        /* Enable ADC1 EOC DMA */
        LL_ADC_REG_SetDMATransfer(ADC1, LL_ADC_REG_DMA_TRANSFER_LIMITED);
      }
    }
  }
  
  return hRetVal;
}

/**
  * @brief  It sets the specified sampling time for the specified ADC channel
  *         on ADCx. It must be called once for each channel utilized by user
  * @param pHdl: handler of the current instance of the PWM component
  * @retval none
  */
__weak void R1G0XX_ADC_SetSamplingTime(PWMC_Handle_t *pHdl, ADConv_t ADConv_struct)
{ 
  PWMC_R1_G0_Handle_t *pHandle = (PWMC_R1_G0_Handle_t *)pHdl;
  
  if (ADConv_struct.Channel < 18u)
  {
    if (ADConv_struct.SamplTime < 8u)
    {
      pHandle->bRegSmpTime[ADConv_struct.Channel] = ADConv_struct.SamplTime;
      
      /* Select sampling time common 2 for all regular channels */
      /* sampling time common 1 is dedicated to current sampling time */
      MODIFY_REG (ADC1->SMPR, 1<<(ADConv_struct.Channel+ADC_SMPR_SMPSEL0_BITOFFSET_POS), 1<<(ADConv_struct.Channel+ADC_SMPR_SMPSEL0_BITOFFSET_POS));
    }
  }
}

/**
  * @}
  */
  
/**
  * @}
  */

/**
  * @}
  */
/************************ (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
