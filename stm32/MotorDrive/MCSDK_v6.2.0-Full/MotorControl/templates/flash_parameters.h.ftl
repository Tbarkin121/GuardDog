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
/**
  ******************************************************************************
  * @file    flash_parameters.h
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file includes the type definition of data aimed to be written by the application.
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
#ifndef FLASH_PARAMETERS_H
#define FLASH_PARAMETERS_H
#include "mc_type.h"
#include "mc_configuration_registers.h"

typedef struct
{
  float zestThresholdFreqHz; // MOTOR_ZEST_THRESHOLD_FREQ_HZ
  float zestInjectFreq; // MOTOR_ZEST_INJECT_FREQ
  float zestInjectD; // MOTOR_ZEST_INJECT_D
  float zestInjectQ; // MOTOR_ZEST_INJECT_Q
  float zestGainD; // MOTOR_ZEST_GAIN_D
  float zestGainQ; //MOTOR_ZEST_GAIN_Q
} zestFlashParams_t; 
  
typedef struct
{
  float pidSpdKp; //PID_SPD_KP
  float pidSpdKi; //PID_SPD_KI
} PIDSpeedFlashParams_t;  
      
typedef struct
{
  float voltage;
  float current;
  float frequency;
  float padding [1];
} scaleFlashParams_t; // 3 useful words + padding

typedef struct
{
  uint16_t        N;                   
  uint16_t        Nd;                  
  uint16_t        padding[2];
} polPulseFlashParams_t; // 2 useful words + 2 padding

typedef struct
{
  float limitOverVoltage;
  float limitRegenHigh;
  float limitRegenLow;
  float limitAccelHigh;
  float limitAccelLow;
  float limitUnderVoltage;
  float maxModulationIndex;
  float softOverCurrentTrip;
  float padding [1];
} boardFlashParams_t; // 8 useful words + padding

typedef struct
{
  MotorConfig_reg_t motor;
  zestFlashParams_t zest;
  PIDSpeedFlashParams_t PIDSpeed; 
  boardFlashParams_t board;
  float KSampleDelay;
  throttleParams_t throttle;
  scaleFlashParams_t scale;
  polPulseFlashParams_t polPulse;  
} FLASH_Params_t;

#endif // FLASH_PARAMETERS_H