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

<#function _last_word text sep="_"><#return text?split(sep)?last></#function>

<#-- Condition for STM32F302x8x MCU -->
<#assign CondMcu_STM32F302x8x = (McuName?? && McuName?matches("STM32F302.8.*"))>
<#-- Condition for STM32F072xxx MCU -->
<#assign CondMcu_STM32F072xxx = (McuName?? && McuName?matches("STM32F072.*"))>
<#-- Condition for STM32F030xxx MCU -->
<#assign CondMcu_STM32F030xxx = (McuName?? && McuName?matches("STM32F030.*"))>
<#-- Condition for STM32F0 Family -->
<#assign CondFamily_STM32F0 = (FamilyName?? && FamilyName=="STM32F0")>
<#-- Condition for STM32F3 Family -->
<#assign CondFamily_STM32F3 = (FamilyName?? && FamilyName == "STM32F3") >
<#-- Condition for STM32F4 Family -->
<#assign CondFamily_STM32F4 = (FamilyName?? && FamilyName == "STM32F4") >
<#-- Condition for STM32G4 Family -->
<#assign CondFamily_STM32G4 = (FamilyName?? && FamilyName == "STM32G4") >
<#-- Condition for STM32L4 Family -->
<#assign CondFamily_STM32L4 = (FamilyName?? && FamilyName == "STM32L4") >
<#-- Condition for STM32F7 Family -->
<#assign CondFamily_STM32F7 = (FamilyName?? && FamilyName == "STM32F7") >
<#-- Condition for STM32H5 Family -->
<#assign CondFamily_STM32H5 = (FamilyName?? && FamilyName == "STM32H5") >
<#-- Condition for STM32H7 Family -->
<#assign CondFamily_STM32H7 = (FamilyName?? && FamilyName == "STM32H7") >
<#-- Condition for STM32G0 Family -->
<#assign CondFamily_STM32G0 = (FamilyName?? && FamilyName=="STM32G0") >
<#-- Condition for STM32C0 Family -->
<#assign CondFamily_STM32C0 = (FamilyName?? && FamilyName=="STM32C0") >

<#assign FOC = MC.M1_DRIVE_TYPE == "FOC" || MC.M2_DRIVE_TYPE == "FOC">
<#assign SIX_STEP = MC.M1_DRIVE_TYPE == "SIX_STEP" || MC.M2_DRIVE_TYPE == "SIX_STEP">
<#assign M1_ENCODER = (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M2_ENCODER = (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>


<#if MC.M1_OVERMODULATION == true><#assign OVM ="_OVM"><#else><#assign OVM =""></#if>
<#if MC.M2_OVERMODULATION == true><#assign OVM2 ="_OVM"><#else><#assign OVM2 =""></#if>
<#macro setScandir Ph1 Ph2>
<#if Ph1?number < Ph2?number>
   LL_ADC_REG_SEQ_SCAN_DIR_FORWARD>>ADC_CFGR1_SCANDIR_Pos,
<#else>
   LL_ADC_REG_SEQ_SCAN_DIR_BACKWARD>>ADC_CFGR1_SCANDIR_Pos,
</#if>
<#return>
</#macro>
/**
  ******************************************************************************
  * @file    mc_config.c 
  * @author  Motor Control SDK Team,ST Microelectronics
  * @brief   Motor Control Subsystem components configuration and handler structures.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044,the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */
//cstat -MISRAC2012-Rule-21.1
#include "main.h" //cstat !MISRAC2012-Rule-21.1
//cstat +MISRAC2012-Rule-21.1
#include "mc_type.h"
#include "parameters_conversion.h"
#include "mc_parameters.h"
#include "mc_config.h"

/* USER CODE BEGIN Additional include */

/* USER CODE END Additional include */ 
<#if FOC>
  <#if MC.DRIVE_NUMBER == "1">
#define FREQ_RATIO 1                /* Dummy value for single drive */
#define FREQ_RELATION HIGHEST_FREQ  /* Dummy value for single drive */
  </#if><#-- MC.DRIVE_NUMBER == 1 -->

#include "pqd_motor_power_measurement.h"
</#if><#-- FOC -->

/* USER CODE BEGIN Additional define */

/* USER CODE END Additional define */ 

<#if FOC>
PQD_MotorPowMeas_Handle_t PQD_MotorPowMeasM1 =
{
  .ConvFact = PQD_CONVERSION_FACTOR
};

  <#if MC.DRIVE_NUMBER != "1">
PQD_MotorPowMeas_Handle_t PQD_MotorPowMeasM2=
{
  .ConvFact = PQD_CONVERSION_FACTOR2
};

  
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->


/**
  * @brief  PI / PID Speed loop parameters Motor 1.
  */
PID_Handle_t PIDSpeedHandle_M1 =
{
  .hDefKpGain          = (int16_t)PID_SPEED_KP_DEFAULT,
  .hDefKiGain          = (int16_t)PID_SPEED_KI_DEFAULT,
<#if FOC>
  .wUpperIntegralLimit = (int32_t)IQMAX * (int32_t)SP_KIDIV,
  .wLowerIntegralLimit = -(int32_t)IQMAX * (int32_t)SP_KIDIV,
  .hUpperOutputLimit   = (int16_t)IQMAX,
  .hLowerOutputLimit   = -(int16_t)IQMAX,
</#if><#-- FOC -->
<#if SIX_STEP>
  <#if MC.DRIVE_MODE == "VM">
  .wUpperIntegralLimit = (int32_t)PERIODMAX * (int32_t)SP_KIDIV,
  .wLowerIntegralLimit = 0,
  .hUpperOutputLimit   = (int16_t)PERIODMAX,
  .hLowerOutputLimit   = 0,
  <#else>
  .wUpperIntegralLimit = (int32_t)PERIODMAX_REF * (int32_t)SP_KIDIV,
  .wLowerIntegralLimit = 0,
  .hUpperOutputLimit   = (int16_t)PERIODMAX_REF,
  .hLowerOutputLimit   = 0,
  </#if><#-- MC.DRIVE_MODE == "VM" -->
</#if><#-- SIX_STEP -->
  .hKpDivisor          = (uint16_t)SP_KPDIV,
  .hKiDivisor          = (uint16_t)SP_KIDIV,
  .hKpDivisorPOW2      = (uint16_t)SP_KPDIV_LOG,
  .hKiDivisorPOW2      = (uint16_t)SP_KIDIV_LOG,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};




<#if FOC>
/**
  * @brief  PI / PID Iq loop parameters Motor 1.
  */
PID_Handle_t PIDIqHandle_M1 =
{
  .hDefKpGain          = (int16_t)PID_TORQUE_KP_DEFAULT,
  .hDefKiGain          = (int16_t)PID_TORQUE_KI_DEFAULT,
  .wUpperIntegralLimit = (int32_t)INT16_MAX * TF_KIDIV,
  .wLowerIntegralLimit = (int32_t)-INT16_MAX * TF_KIDIV,
  .hUpperOutputLimit   = INT16_MAX,
  .hLowerOutputLimit   = -INT16_MAX,
  .hKpDivisor          = (uint16_t)TF_KPDIV,
  .hKiDivisor          = (uint16_t)TF_KIDIV,
  .hKpDivisorPOW2      = (uint16_t)TF_KPDIV_LOG,
  .hKiDivisorPOW2      = (uint16_t)TF_KIDIV_LOG,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};

/**
  * @brief  PI / PID Id loop parameters Motor 1.
  */
PID_Handle_t PIDIdHandle_M1 =
{
  .hDefKpGain          = (int16_t)PID_FLUX_KP_DEFAULT,
  .hDefKiGain          = (int16_t)PID_FLUX_KI_DEFAULT,
  .wUpperIntegralLimit = (int32_t)INT16_MAX * TF_KIDIV,
  .wLowerIntegralLimit = (int32_t)-INT16_MAX * TF_KIDIV,
  .hUpperOutputLimit   = INT16_MAX,
  .hLowerOutputLimit   = -INT16_MAX,
  .hKpDivisor          = (uint16_t)TF_KPDIV,
  .hKiDivisor          = (uint16_t)TF_KIDIV,
  .hKpDivisorPOW2      = (uint16_t)TF_KPDIV_LOG,
  .hKiDivisorPOW2      = (uint16_t)TF_KIDIV_LOG,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};

  <#if MC.M1_FLUX_WEAKENING_ENABLING == true>
/**
  * @brief  FluxWeakeningCtrl component parameters Motor 1.
  */
FW_Handle_t FW_M1 =
{
  .hMaxModule             = MAX_MODULE,
  .hDefaultFW_V_Ref       = (int16_t)FW_VOLTAGE_REF,
  .hDemagCurrent          = ID_DEMAG,
  .wNominalSqCurr         = ((int32_t)NOMINAL_CURRENT*(int32_t)NOMINAL_CURRENT),
  .hVqdLowPassFilterBW    = M1_VQD_SW_FILTER_BW_FACTOR,
  .hVqdLowPassFilterBWLOG = M1_VQD_SW_FILTER_BW_FACTOR_LOG  
};

/**
  * @brief  PI Flux Weakening control parameters Motor 1.
  */
PID_Handle_t PIDFluxWeakeningHandle_M1 =
{
  .hDefKpGain          = (int16_t)FW_KP_GAIN,
  .hDefKiGain          = (int16_t)FW_KI_GAIN,
  .wUpperIntegralLimit = 0,
  .wLowerIntegralLimit = (int32_t)(-NOMINAL_CURRENT) * (int32_t)FW_KIDIV,
  .hUpperOutputLimit   = 0,
  .hLowerOutputLimit   = -INT16_MAX,
  .hKpDivisor          = (uint16_t)FW_KPDIV,
  .hKiDivisor          = (uint16_t)FW_KIDIV,
  .hKpDivisorPOW2      = (uint16_t)FW_KPDIV_LOG,
  .hKiDivisorPOW2      = (uint16_t)FW_KIDIV_LOG,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};
  </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING == true -->

  <#if MC.M2_FLUX_WEAKENING_ENABLING == true>
/**
  * @brief  FluxWeakeningCtrl component parameters Motor 2.
  */
FW_Handle_t FW_M2 =
{
  .hMaxModule             = MAX_MODULE2,
  .hDefaultFW_V_Ref       = (int16_t)FW_VOLTAGE_REF2,
  .hDemagCurrent          = ID_DEMAG2,
  .wNominalSqCurr         = ((int32_t)NOMINAL_CURRENT2*(int32_t)NOMINAL_CURRENT2),
  .hVqdLowPassFilterBW    = M2_VQD_SW_FILTER_BW_FACTOR,
  .hVqdLowPassFilterBWLOG = M2_VQD_SW_FILTER_BW_FACTOR_LOG
};

/**
  * @brief  PI Flux Weakening control parameters Motor 2.
  */
PID_Handle_t PIDFluxWeakeningHandle_M2 =
{
  .hDefKpGain          = (int16_t)FW_KP_GAIN2,
  .hDefKiGain          = (int16_t)FW_KI_GAIN2,
  .wUpperIntegralLimit = 0,
  .wLowerIntegralLimit = (int32_t)(-NOMINAL_CURRENT2) * (int32_t)FW_KIDIV2,
  .hUpperOutputLimit   = 0,
  .hLowerOutputLimit   = -INT16_MAX,
  .hKpDivisor          = (uint16_t)FW_KPDIV2,
  .hKiDivisor          = (uint16_t)FW_KIDIV2,
  .hKpDivisorPOW2      = (uint16_t)FW_KPDIV_LOG2,
  .hKiDivisorPOW2      = (uint16_t)FW_KIDIV_LOG2,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};
  </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING == true -->

  <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true>
/**
  * @brief  FeedForwardCtrl parameters Motor 1.
  */
FF_Handle_t FF_M1 =
{
  .hVqdLowPassFilterBW    = M1_VQD_SW_FILTER_BW_FACTOR,
  .wDefConstant_1D        = (int32_t)M1_CONSTANT1_D,
  .wDefConstant_1Q        = (int32_t)M1_CONSTANT1_Q,
  .wDefConstant_2         = (int32_t)M1_CONSTANT2_QD,
  .hVqdLowPassFilterBWLOG = M1_VQD_SW_FILTER_BW_FACTOR_LOG 
};
  </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true -->

  <#if MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true>
/**
  * @brief  FeedForwardCtrl parameters Motor 2.
  */
FF_Handle_t FF_M2 =
{
  .hVqdLowPassFilterBW    = M2_VQD_SW_FILTER_BW_FACTOR,
  .wDefConstant_1D        = (int32_t)M2_CONSTANT1_D,
  .wDefConstant_1Q        = (int32_t)M2_CONSTANT1_Q,
  .wDefConstant_2         = (int32_t)M2_CONSTANT2_QD,
  .hVqdLowPassFilterBWLOG = M2_VQD_SW_FILTER_BW_FACTOR_LOG
};
  </#if><#-- MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true -->

  <#if MC.M1_POSITION_CTRL_ENABLING == true>
PID_Handle_t PID_PosParamsM1 =
{
  .hDefKpGain          = (int16_t)PID_POSITION_KP_GAIN,
  .hDefKiGain          = (int16_t)PID_POSITION_KI_GAIN,
  .hDefKdGain          = (int16_t)PID_POSITION_KD_GAIN,
  .wUpperIntegralLimit = (int32_t)NOMINAL_CURRENT * (int32_t)PID_POSITION_KIDIV,
  .wLowerIntegralLimit = (int32_t)(-NOMINAL_CURRENT) * (int32_t)PID_POSITION_KIDIV,
  .hUpperOutputLimit   = (int16_t)NOMINAL_CURRENT,
  .hLowerOutputLimit   = -(int16_t)NOMINAL_CURRENT,
  .hKpDivisor          = (uint16_t)PID_POSITION_KPDIV,
  .hKiDivisor          = (uint16_t)PID_POSITION_KIDIV,
  .hKdDivisor          = (uint16_t)PID_POSITION_KDDIV,
  .hKpDivisorPOW2      = (uint16_t)PID_POSITION_KPDIV_LOG,
  .hKiDivisorPOW2      = (uint16_t)PID_POSITION_KIDIV_LOG,
  .hKdDivisorPOW2      = (uint16_t)PID_POSITION_KDDIV_LOG,
};

PosCtrl_Handle_t PosCtrlM1 =
{
  .SamplingTime  = 1.0f/MEDIUM_FREQUENCY_TASK_RATE,
  .SysTickPeriod = 1.0f/SYS_TICK_FREQUENCY,
    <#if MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z"> 
  .AlignmentCfg  = TC_ABSOLUTE_ALIGNMENT_SUPPORTED,
    <#else><#-- MC.M1_SPEED_SENSOR != "QUAD_ENCODER_Z" -->
  .AlignmentCfg  = TC_ABSOLUTE_ALIGNMENT_NOT_SUPPORTED,
    </#if><#-- MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z" -->
};
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true -->

  <#if MC.M2_POSITION_CTRL_ENABLING == true>
PID_Handle_t PID_PosParamsM2 =
{
  .hDefKpGain          = (int16_t)PID_POSITION_KP_GAIN2,
  .hDefKiGain          = (int16_t)PID_POSITION_KI_GAIN2,
  .hDefKdGain          = (int16_t)PID_POSITION_KD_GAIN2,
  .wUpperIntegralLimit = (int32_t)NOMINAL_CURRENT2 * (int32_t)PID_POSITION_KIDIV2,
  .wLowerIntegralLimit = (int32_t)(-NOMINAL_CURRENT2) * (int32_t)PID_POSITION_KIDIV2,
  .hUpperOutputLimit   = (int16_t)NOMINAL_CURRENT2,
  .hLowerOutputLimit   = -(int16_t)NOMINAL_CURRENT2,
  .hKpDivisor          = (uint16_t)PID_POSITION_KPDIV2,
  .hKiDivisor          = (uint16_t)PID_POSITION_KIDIV2,
  .hKdDivisor          = (uint16_t)PID_POSITION_KDDIV2,
  .hKpDivisorPOW2      = (uint16_t)PID_POSITION_KPDIV_LOG2,
  .hKiDivisorPOW2      = (uint16_t)PID_POSITION_KIDIV_LOG2,
  .hKdDivisorPOW2      = (uint16_t)PID_POSITION_KDDIV_LOG2,
};

PosCtrl_Handle_t PosCtrlM2 =
{
  .SamplingTime  = 1.0f/MEDIUM_FREQUENCY_TASK_RATE2,
  .SysTickPeriod = 1.0f/SYS_TICK_FREQUENCY,
    <#if MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z">
  .AlignmentCfg  = TC_ABSOLUTE_ALIGNMENT_SUPPORTED,
    <#else><#-- MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z" -->
  .AlignmentCfg  = TC_ABSOLUTE_ALIGNMENT_NOT_SUPPORTED,
    </#if><#-- MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z" -->
};
  </#if><#-- MC.M2_POSITION_CTRL_ENABLING == true -->
</#if><#-- FOC -->

/**
  * @brief  SpeednTorque Controller parameters Motor 1.
  */
SpeednTorqCtrl_Handle_t SpeednTorqCtrlM1 =
{
  .STCFrequencyHz             = MEDIUM_FREQUENCY_TASK_RATE,
  .MaxAppPositiveMecSpeedUnit = (uint16_t)(MAX_APPLICATION_SPEED_UNIT),
  .MinAppPositiveMecSpeedUnit = (uint16_t)(MIN_APPLICATION_SPEED_UNIT),
  .MaxAppNegativeMecSpeedUnit = (int16_t)(-MIN_APPLICATION_SPEED_UNIT),
  .MinAppNegativeMecSpeedUnit = (int16_t)(-MAX_APPLICATION_SPEED_UNIT),
<#if MC.M1_DRIVE_TYPE == "FOC">
  .MaxPositiveTorque          = (int16_t)NOMINAL_CURRENT,
  .MinNegativeTorque          = -(int16_t)NOMINAL_CURRENT,
</#if><#-- MC.M1_DRIVE_TYPE == "FOC" -->
<#if MC.M1_DRIVE_TYPE == "SIX_STEP">
  <#if MC.DRIVE_MODE == "VM">
  .MaxPositiveDutyCycle       = (uint16_t)PERIODMAX / 2U,
  <#else><#-- MC.DRIVE_MODE != "VM" -->
  .MaxPositiveDutyCycle       = (uint16_t)PERIODMAX_REF,
  </#if><#-- MC.DRIVE_MODE == "VM" -->
</#if><#-- MC.M1_DRIVE_TYPE == "SIX_STEP" -->
  .ModeDefault                = DEFAULT_CONTROL_MODE,
  .MecSpeedRefUnitDefault     = (int16_t)(DEFAULT_TARGET_SPEED_UNIT),
<#if MC.M1_DRIVE_TYPE == "FOC">
  .TorqueRefDefault           = (int16_t)DEFAULT_TORQUE_COMPONENT,
  .IdrefDefault               = (int16_t)DEFAULT_FLUX_COMPONENT,
</#if><#-- MC.M1_DRIVE_TYPE == "FOC" -->
<#if MC.M1_DRIVE_TYPE == "SIX_STEP">
  <#if MC.CURRENT_LIMITER_OFFSET>
  .DutyCycleRefDefault        = (uint16_t)PERIODMAX_REF,
  <#else><#-- != MC.CURRENT_LIMITER_OFFSET -->
  .DutyCycleRefDefault        = 0,
  </#if><#-- MC.CURRENT_LIMITER_OFFSET -->
</#if><#-- MC.M1_DRIVE_TYPE == "SIX_STEP" -->
};

<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M1_SPEED_SENSOR == "SENSORLESS_ADC")>
RevUpCtrl_Handle_t RevUpControlM1 =
{
  .hRUCFrequencyHz         = MEDIUM_FREQUENCY_TASK_RATE,
  .hStartingMecAngle       = (int16_t)((int32_t)(STARTING_ANGLE_DEG)* 65536/360),
  <#if MC.MOTOR_PROFILER == true>
  .bFirstAccelerationStage = ENABLE_SL_ALGO_FROM_PHASE,
  <#else><#-- MC.MOTOR_PROFILER == false -->
  .bFirstAccelerationStage = (ENABLE_SL_ALGO_FROM_PHASE-1u),
  </#if><#-- MC.MOTOR_PROFILER == true -->
  .hMinStartUpValidSpeed   = OBS_MINIMUM_SPEED_UNIT,
  <#if MC.M1_DRIVE_TYPE == "FOC">
  .hMinStartUpFlySpeed     = (int16_t)(OBS_MINIMUM_SPEED_UNIT/2),
  .OTFStartupEnabled       = ${MC.M1_OTF_STARTUP?string("true","false")},
  .OTFPhaseParams =
  {
    (uint16_t)500,
    0,
    (int16_t)PHASE5_FINAL_CURRENT,
    (void*)MC_NULL
  },
  </#if><#-- MC.M1_DRIVE_TYPE == "FOC" -->
  <#if MC.M1_DRIVE_TYPE == "SIX_STEP" &&  MC.DRIVE_MODE == "VM">
  .ParamsData  = 
  {
    {(uint16_t)PHASE1_DURATION,(int16_t)(PHASE1_FINAL_SPEED_UNIT),
    (uint16_t)PHASE1_VOLTAGE_DPP,&RevUpControlM1.ParamsData[1]},
    {(uint16_t)PHASE2_DURATION,(int16_t)(PHASE2_FINAL_SPEED_UNIT),
    (uint16_t)PHASE2_VOLTAGE_DPP,&RevUpControlM1.ParamsData[2]},
    {(uint16_t)PHASE3_DURATION,(int16_t)(PHASE3_FINAL_SPEED_UNIT),
    (uint16_t)PHASE3_VOLTAGE_DPP,&RevUpControlM1.ParamsData[3]},
    {(uint16_t)PHASE4_DURATION,(int16_t)(PHASE4_FINAL_SPEED_UNIT),
    (uint16_t)PHASE4_VOLTAGE_DPP,&RevUpControlM1.ParamsData[4]},
    {(uint16_t)PHASE5_DURATION,(int16_t)(PHASE5_FINAL_SPEED_UNIT),
    (uint16_t)PHASE5_VOLTAGE_DPP,(void*)MC_NULL},
  },
  <#elseif MC.M1_DRIVE_TYPE == "SIX_STEP" &&  MC.DRIVE_MODE == "CM">
  .ParamsData =
  {
    {(uint16_t)PHASE1_DURATION,(int16_t)(PHASE1_FINAL_SPEED_UNIT),
    PHASE1_FINAL_CURRENT_REF,&RevUpControlM1.ParamsData[1]},
    {(uint16_t)PHASE2_DURATION,(int16_t)(PHASE2_FINAL_SPEED_UNIT),
    PHASE2_FINAL_CURRENT_REF,&RevUpControlM1.ParamsData[2]},
    {(uint16_t)PHASE3_DURATION,(int16_t)(PHASE3_FINAL_SPEED_UNIT),
    PHASE3_FINAL_CURRENT_REF,&RevUpControlM1.ParamsData[3]},
    {(uint16_t)PHASE4_DURATION,(int16_t)(PHASE4_FINAL_SPEED_UNIT),
    PHASE4_FINAL_CURRENT_REF,&RevUpControlM1.ParamsData[4]},
    {(uint16_t)PHASE5_DURATION,(int16_t)(PHASE5_FINAL_SPEED_UNIT),
    PHASE5_FINAL_CURRENT_REF,(void*)MC_NULL},
  },
  <#else><#-- MC.M1_DRIVE_TYPE != "SIX_STEP" ||  MC.DRIVE_MODE != "VM" ||  MC.DRIVE_MODE != "CM" -->
  .ParamsData =
  {
    {(uint16_t)PHASE1_DURATION,(int16_t)(PHASE1_FINAL_SPEED_UNIT),(uint16_t)PHASE1_FINAL_CURRENT,&RevUpControlM1.ParamsData[1]},
    {(uint16_t)PHASE2_DURATION,(int16_t)(PHASE2_FINAL_SPEED_UNIT),(uint16_t)PHASE2_FINAL_CURRENT,&RevUpControlM1.ParamsData[2]},
    {(uint16_t)PHASE3_DURATION,(int16_t)(PHASE3_FINAL_SPEED_UNIT),(uint16_t)PHASE3_FINAL_CURRENT,&RevUpControlM1.ParamsData[3]},
    {(uint16_t)PHASE4_DURATION,(int16_t)(PHASE4_FINAL_SPEED_UNIT),(uint16_t)PHASE4_FINAL_CURRENT,&RevUpControlM1.ParamsData[4]},
    {(uint16_t)PHASE5_DURATION,(int16_t)(PHASE5_FINAL_SPEED_UNIT),(uint16_t)PHASE5_FINAL_CURRENT,(void*)MC_NULL},
  },
  </#if><#-- MC.M1_DRIVE_TYPE == "SIX_STEP" &&  MC.DRIVE_MODE == "VM" -->
};
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M1_SPEED_SENSOR == "SENSORLESS_ADC") -->
  
  
  
  
<#--------------------------------------------------------------------------------------------------------------------->
<#--                                                     Condition for FOC                                           -->
<#--------------------------------------------------------------------------------------------------------------------->
<#if FOC>

<#---------------------------------->
<#-- Condition for Single Shunt   -->
<#---------------------------------->
  <#if CondFamily_STM32H7 == false && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
PWMC_R1_Handle_t PWM_Handle_M1 =
{
  {
    .pFctGetPhaseCurrents       = &R1_GetPhaseCurrents,
    .pFctSetOffsetCalib         = &R1_SetOffsetCalib,
    .pFctGetOffsetCalib         = &R1_GetOffsetCalib,
    .pFctSwitchOffPwm           = &R1_SwitchOffPWM,
    .pFctSwitchOnPwm            = &R1_SwitchOnPWM,
    .pFctCurrReadingCalib       = &R1_CurrentReadingCalibration,
    .pFctTurnOnLowSides         = &R1_TurnOnLowSides,
    .pFctSetADCSampPointSectX   = &R1_CalcDutyCycles,
    .pFctOCPSetReferenceVoltage = MC_NULL,
    .pFctRLDetectionModeEnable  = MC_NULL,
    .pFctRLDetectionModeDisable = MC_NULL,
    .pFctRLDetectionModeSetDuty = MC_NULL,
    .pFctRLTurnOnLowSidesAndStart = MC_NULL,

    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING,
<#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M1_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M1_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M1_PWM_EN_U_Pin,
    .pwm_en_v_port     = M1_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M1_PWM_EN_V_Pin,
    .pwm_en_w_port     = M1_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M1_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M1_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M1_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M1_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->    
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES*SQRT3FACTOR) / 16384u,
    .Sector                     = 0,
    .CntPhA                     = 0,
    .CntPhB                     = 0,
    .CntPhC                     = 0,
    .SWerror                    = 0,
    .TurnOnLowSidesAction       = false,
    .OffCalibrWaitTimeCounter   = 0,
    .Motor                      = 0,
    .RLDetectionMode            = false,
    .SingleShuntTopology        = true,
    .Ia                         = 0,
    .Ib                         = 0,
    .Ic                         = 0,
    .LPFIqd_const               = LPF_FILT_CONST,
    .DTTest                     = 0,
    .DTCompCnt                  = DTCOMPCNT,
    .PWMperiod                  = PWM_PERIOD_CYCLES,
    .Ton                        = TON,
    .Toff                       = TOFF,
    .OverCurrentFlag            = false,
    .OverVoltageFlag            = false,
    .BrakeActionLock            = false,
    .driverProtectionFlag       = false,
  },
  <#if CondFamily_STM32F0 == true || CondFamily_STM32F4 == true>
  .DmaBuffCCR = {0,0,0,0,0,0},
<#else>
  .DmaBuffCCR                   = {0,0,0,0,0,0,0,0},
</#if>
  .PhaseOffset                  = 0,
<#if CondFamily_STM32F4 == true>
  .DmaBuffCCR_ADCTrig = {0,0,0},
</#if>
<#if CondFamily_STM32G0 == true>
  .CurConv = {0,0},
  .ADCRegularLocked = false,
</#if>
  .Half_PWMPeriod               = PWM_PERIOD_CYCLES / 2u,
  .CntSmp1                      = 0,
  .CntSmp2                      = 0,
  .sampCur1                     = 0,
  .sampCur2                     = 0,
  .CurrAOld                     = 0,
  .CurrBOld                     = 0,
  .Index                        = 0,
  .iflag                        = 0,
  .TCCnt                        = 0U,
  .UpdateFlagBuffer             = 0,
  .TCDoneFlag                   = true,

  .pParams_str                  = &R1_ParamsM1,
};

  </#if><#-- CondFamily_STM32H7 == false && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->

  <#if CondFamily_STM32H7 == false && ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
/**
  * @brief  PWM parameters Motor 2 for one shunt.
  */
PWMC_R1_Handle_t PWM_Handle_M2 =
{
  {
    .pFctGetPhaseCurrents       = &R1_GetPhaseCurrents,
    .pFctSetOffsetCalib         = &R1_SetOffsetCalib,
    .pFctGetOffsetCalib         = &R1_GetOffsetCalib,
    .pFctSwitchOffPwm           = &R1_SwitchOffPWM,
    .pFctSwitchOnPwm            = &R1_SwitchOnPWM,
    .pFctCurrReadingCalib       = &R1_CurrentReadingCalibration,
    .pFctTurnOnLowSides         = &R1_TurnOnLowSides,
    .pFctSetADCSampPointSectX   = &R1_CalcDutyCycles,
    .pFctOCPSetReferenceVoltage = MC_NULL,
    .pFctRLDetectionModeEnable  = MC_NULL,
    .pFctRLDetectionModeDisable = MC_NULL,
    .pFctRLDetectionModeSetDuty = MC_NULL,
    .pFctRLTurnOnLowSidesAndStart = MC_NULL,
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->        
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES2*SQRT3FACTOR) / 16384u,
    .Sector                     = 0,
    .CntPhA                     = 0,
    .CntPhB                     = 0,
    .CntPhC                     = 0,
    .SWerror                    = 0,
    .TurnOnLowSidesAction       = false,
    .OffCalibrWaitTimeCounter   = 0,
    .Motor                      = M2,
    .RLDetectionMode            = false,
    .SingleShuntTopology        = true,
    .Ia                         = 0,
    .Ib                         = 0,
    .Ic                         = 0,
    .LPFIqd_const               = LPF_FILT_CONST,
    .DTTest                     = 0,
    .DTCompCnt                  = DTCOMPCNT2,
    .PWMperiod                  = PWM_PERIOD_CYCLES2,
    .Ton                        = TON,
    .Toff                       = TOFF,
    .OverCurrentFlag            = false,
    .OverVoltageFlag            = false,
    .BrakeActionLock            = false,
    .driverProtectionFlag       = false,    
  },
  
  .DmaBuffCCR                   = {0,0,0,0,0,0},
<#if CondFamily_STM32F4 == true>
  .DmaBuffCCR_ADCTrig = {0,0,0},
</#if>
<#if CondFamily_STM32G0 == true>
  .CurConv = {0,0},
  .ADCRegularLocked = false,
</#if>
  .CntSmp1                      = 0,
  .CntSmp2                      = 0,
  .sampCur1                     = 0,
  .sampCur2                     = 0,
  .CurrAOld                     = 0,
  .CurrBOld                     = 0,
  .PhaseOffset                  = 0,
  .Index                        = 0,
  .Half_PWMPeriod               = PWM_PERIOD_CYCLES2 / 2u,
  .iflag                        = 0,
  .UpdateFlagBuffer             = 0,
  .pParams_str                  = &R1_ParamsM2,
};
  </#if><#-- CondFamily_STM32H7 == false && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->

<#---------------------------------->
<#-- Condition for three shunt    -->
<#---------------------------------->
  <#if (CondFamily_STM32H7 == false ) && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
/**
  * @brief  PWM parameters Motor 1 for one ADC.
  */
PWMC_R3_1_Handle_t PWM_Handle_M1 =
{
  {
    <#if CondFamily_STM32F4>
    .pFctGetPhaseCurrents       = &R3_1_GetPhaseCurrents,
    .pFctSetADCSampPointSectX   = &R3_1_SetADCSampPointSectX,
    <#else><#-- CondFamily_STM32F4 == false -->
    .pFctGetPhaseCurrents       = &R3_1_GetPhaseCurrents${OVM},
    .pFctSetADCSampPointSectX   = &R3_1_SetADCSampPointSectX${OVM},
    </#if><#-- CondFamily_STM32F4 -->
    .pFctSetOffsetCalib         = &R3_1_SetOffsetCalib,
    .pFctGetOffsetCalib         = &R3_1_GetOffsetCalib,
    .pFctSwitchOffPwm           = &R3_1_SwitchOffPWM,
    .pFctSwitchOnPwm            = &R3_1_SwitchOnPWM,          
	  <#if CondFamily_STM32F3  || CondFamily_STM32H5 || CondFamily_STM32G4 >            
    .pFctCurrReadingCalib       = &R3_1_CurrentReadingPolarization,
    <#else><#-- (CondFamily_STM32F3 || CondFamily_STM32H5 || CondFamily_STM32G4  ) -->
    .pFctCurrReadingCalib       = &R3_1_CurrentReadingCalibration,
    </#if><#-- CondFamily_STM32F3 || CondFamily_STM32H5 || CondFamily_STM32G4 -->
    .pFctTurnOnLowSides         = &R3_1_TurnOnLowSides,
    .pFctOCPSetReferenceVoltage = MC_NULL,
  
	  <#if CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0>
    .pFctRLDetectionModeEnable  = MC_NULL,
    .pFctRLDetectionModeDisable = MC_NULL,
    .pFctRLDetectionModeSetDuty = MC_NULL,
    .pFctRLTurnOnLowSidesAndStart = MC_NULL,
    <#else><#-- CondFamily_STM32F0 = false && CondFamily_STM32G0 == false && CondFamily_STM32C0 == false -->
    .pFctRLDetectionModeEnable  = &R3_1_RLDetectionModeEnable,
    .pFctRLDetectionModeDisable = &R3_1_RLDetectionModeDisable,
    .pFctRLDetectionModeSetDuty = &R3_1_RLDetectionModeSetDuty,
    .pFctRLTurnOnLowSidesAndStart = &R3_1_RLTurnOnLowSidesAndStart,
    </#if><#-- CondFamily_STM32F0 || CondFamily_STM32F7 || CondFamily_STM32G0 || CondFamily_STM32H5 || CondFamily_STM32G4 || CondFamily_STM32C0-->
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING,
<#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M1_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M1_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M1_PWM_EN_U_Pin,
    .pwm_en_v_port     = M1_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M1_PWM_EN_V_Pin,
    .pwm_en_w_port     = M1_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M1_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M1_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M1_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M1_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->    
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES*SQRT3FACTOR)/16384u,
    .Sector                     = 0,
    .CntPhA                     = 0,
    .CntPhB                     = 0,
    .CntPhC                     = 0,
    .SWerror                    = 0,
    .TurnOnLowSidesAction       = false,
    .OffCalibrWaitTimeCounter   = 0,
    <#if (CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0)>
    .Motor                      = 0,
    <#else><#-- CondFamily_STM32F0 == false && CondFamily_STM32G0 == false -->
    .Motor                      = M1,
    </#if><#-- CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0 -->
    .RLDetectionMode            = false,
    .SingleShuntTopology        = false,
    .Ia                         = 0,
    .Ib                         = 0,
    .Ic                         = 0,
    .LPFIqd_const               = LPF_FILT_CONST,
    .DTTest                     = 0,
    .DTCompCnt                  = DTCOMPCNT,
    .PWMperiod                  = PWM_PERIOD_CYCLES,
    .Ton                        = TON,
    .Toff                       = TOFF,
    .OverCurrentFlag            = false,
    .OverVoltageFlag            = false,
    .BrakeActionLock            = false,
    .driverProtectionFlag       = false,
  },
  .PhaseAOffset                 = 0,
  .PhaseBOffset                 = 0,
  .PhaseCOffset                 = 0,
  .Half_PWMPeriod               = PWM_PERIOD_CYCLES / 2u,
    <#if CondFamily_STM32F4 || CondFamily_STM32F7>
  .PolarizationCounter          = 0,
  .ADC_ExternalTriggerInjected  = 0,
  .ADCTriggerEdge               = 0,
    <#elseif CondFamily_STM32F0>
  .PolarizationCounter          = 0,
  .ADC1_DMA_converted           = {0,0},
    <#elseif CondFamily_STM32L4>
  .PolarizationCounter          = 0,
  .ADC_ExternalTriggerInjected  = 0,
    <#elseif CondFamily_STM32G0>
  .PolarizationCounter          = 0,
  .ADC1_DMA_converted[0]        = 0,
  .ADC1_DMA_converted[1]        = 0,
	  <#elseif CondFamily_STM32G4>
	    .ADC_ExternalPolarityInjected = 0,
		.PolarizationCounter = 0,
		.PolarizationSector = 0,
		.ADCRegularLocked = false,
    </#if><#-- CondFamily_STM32F4 || CondFamily_STM32F7 -->
 
    <#if CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0>
  .pParams_str                  = &R3_1_Params
    <#else><#-- CondFamily_STM32F0 = false && CondFamily_STM32G0 == false && CondFamily_STM32C0 == false -->
  .pParams_str                  = &R3_1_ParamsM1
    </#if><#-- CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0 -->
};
  </#if><#-- (CondFamily_STM32H7 == false ) && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->

  <#if (MC.DRIVE_NUMBER != "1") && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '1' ))>
/**
  * @brief  PWM parameters Motor 2 for one ADC.
  */
PWMC_R3_1_Handle_t PWM_Handle_M2 =
{
  {
    <#if CondFamily_STM32F4>
    .pFctGetPhaseCurrents       = &R3_1_GetPhaseCurrents,
    .pFctSetADCSampPointSectX   = &R3_1_SetADCSampPointSectX,
    <#else><#-- CondFamily_STM32F4 == false -->
    .pFctGetPhaseCurrents       = &R3_1_GetPhaseCurrents${OVM},
    .pFctSetADCSampPointSectX   = &R3_1_SetADCSampPointSectX${OVM},
    </#if><#-- CondFamily_STM32F4 -->
    .pFctSetOffsetCalib         = &R3_1_SetOffsetCalib,
    .pFctGetOffsetCalib         = &R3_1_GetOffsetCalib,
    .pFctSwitchOffPwm           = &R3_1_SwitchOffPWM,
    .pFctSwitchOnPwm            = &R3_1_SwitchOnPWM,          
	  <#if CondFamily_STM32F3 || CondFamily_STM32H5 || CondFamily_STM32G4 >            
    .pFctCurrReadingCalib       = &R3_1_CurrentReadingPolarization,
    <#else><#-- (CondFamily_STM32F3 || CondFamily_STM32H5 || CondFamily_STM32G4  ) -->
    .pFctCurrReadingCalib       = &R3_1_CurrentReadingCalibration,
    </#if><#-- CondFamily_STM32F3 || CondFamily_STM32H5 || CondFamily_STM32G4 -->
    .pFctTurnOnLowSides         = &R3_1_TurnOnLowSides,
    .pFctOCPSetReferenceVoltage = MC_NULL,
  
	  <#if CondFamily_STM32F7 || CondFamily_STM32G4>
    .pFctRLDetectionModeEnable  = MC_NULL,
    .pFctRLDetectionModeDisable = MC_NULL,
    .pFctRLDetectionModeSetDuty = MC_NULL,
    .pFctRLTurnOnLowSidesAndStart = MC_NULL,
    <#else><#-- CondFamily_STM32F7 = false && CondFamily_STM32H5==false && CondFamily_STM32G4 == false-->
    .pFctRLDetectionModeEnable  = &R3_1_RLDetectionModeEnable,
    .pFctRLDetectionModeDisable = &R3_1_RLDetectionModeDisable,
    .pFctRLDetectionModeSetDuty = &R3_1_RLDetectionModeSetDuty,
    .pFctRLTurnOnLowSidesAndStart = &R3_1_RLTurnOnLowSidesAndStart,
    </#if><#-- CondFamily_STM32F7 || CondFamily_STM32H5 || CondFamily_STM32G4 -->
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES2*SQRT3FACTOR)/16384u,
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->    

    .Sector                     = 0,
    .CntPhA                     = 0,
    .CntPhB                     = 0,
    .CntPhC                     = 0,
    .SWerror                    = 0,
    .TurnOnLowSidesAction       = false,
    .OffCalibrWaitTimeCounter   = 0,
    .Motor                      = M2,
    .RLDetectionMode            = false,
    .Ia                         = 0,
    .Ib                         = 0,
    .Ic                         = 0,
    .LPFIqd_const               = LPF_FILT_CONST,
    .DTTest                     = 0,
    .DTCompCnt                  = DTCOMPCNT2,
    .PWMperiod                  = PWM_PERIOD_CYCLES2,
    .Ton                        = TON2,
    .Toff                       = TOFF2,
    .OverCurrentFlag            = false,
    .OverVoltageFlag            = false,
    .BrakeActionLock            = false,
    .driverProtectionFlag       = false,
  },
  .PhaseAOffset                 = 0,
  .PhaseBOffset                 = 0,
  .PhaseCOffset                 = 0,
  .Half_PWMPeriod               = PWM_PERIOD_CYCLES2 / 2u,
    <#if CondFamily_STM32F4 || CondFamily_STM32F7>
  .PolarizationCounter          = 0,
  .ADC_ExternalTriggerInjected  = 0,
  .ADCTriggerEdge               = 0,
    <#elseif CondFamily_STM32L4>
  .PolarizationCounter          = 0,
  .ADC_ExternalTriggerInjected  = 0,
	  <#elseif CondFamily_STM32G4>
	    .ADC_ExternalPolarityInjected = 0,
		.PolarizationCounter = 0,
		.PolarizationSector = 0,
		.ADCRegularLocked = false,
    </#if><#-- CondFamily_STM32F4 || CondFamily_STM32F7 -->

  
  .pParams_str                  = &R3_1_ParamsM2
};
  </#if><#-- (MC.DRIVE_NUMBER != "1") && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '1' )) -->

<#-------------------------------------------------------------------->
<#-- Condition for 2 independent ADCs and/or 2 samplings on one ADC -->
<#-------------------------------------------------------------------->
  <#if ((CondFamily_STM32F4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32F3 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
     || (CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
     || (CondFamily_STM32G4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))))>


PWMC_R3_2_Handle_t PWM_Handle_M1 =
{
  ._Super = 
  {
    <#if CondFamily_STM32H7>
    .pFctGetPhaseCurrents       = &R3_2_GetPhaseCurrents,
    <#else><#-- CondFamily_STM32H7 == false -->
    .pFctGetPhaseCurrents       = &R3_2_GetPhaseCurrents${OVM},
    .pFctSetADCSampPointSectX   = &R3_2_SetADCSampPointSectX${OVM},
    .pFctSetOffsetCalib         = &R3_2_SetOffsetCalib,
    .pFctGetOffsetCalib         = &R3_2_GetOffsetCalib,
    </#if><#-- CondFamily_STM32H7 -->
 
    .pFctSwitchOffPwm           = &R3_2_SwitchOffPWM,
    .pFctSwitchOnPwm            = &R3_2_SwitchOnPWM,
               
	  <#if CondFamily_STM32F3 || CondFamily_STM32H7 || CondFamily_STM32G4 || CondFamily_STM32H5>
    .pFctCurrReadingCalib       = &R3_2_CurrentReadingPolarization,
    <#else><#-- CondFamily_STM32F3 == false && CondFamily_STM32H7 == false &&  CondFamily_STM32G4 == false && CondFamily_STM32H5 == false -->
    .pFctCurrReadingCalib       = &R3_2_CurrentReadingCalibration,
    </#if><#-- CondFamily_STM32F3 || CondFamily_STM32H7 ||  CondFamily_STM32G4 || CondFamily_STM32H5 -->
    .pFctTurnOnLowSides         = &R3_2_TurnOnLowSides,
    .pFctOCPSetReferenceVoltage = MC_NULL,
    .pFctRLDetectionModeEnable  = &R3_2_RLDetectionModeEnable,
    .pFctRLDetectionModeDisable = &R3_2_RLDetectionModeDisable,
    .pFctRLDetectionModeSetDuty = &R3_2_RLDetectionModeSetDuty,
    .pFctRLTurnOnLowSidesAndStart = &R3_2_RLTurnOnLowSidesAndStart,
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES*SQRT3FACTOR)/16384u,
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING,
<#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M1_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M1_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M1_PWM_EN_U_Pin,
    .pwm_en_v_port     = M1_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M1_PWM_EN_V_Pin,
    .pwm_en_w_port     = M1_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M1_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M1_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M1_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M1_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->        
    <#if CondFamily_STM32G4>
    .Sector                     = 0,
    .lowDuty                    = (uint16_t)0,
    .midDuty                    = (uint16_t)0,
    .highDuty                   = (uint16_t)0,
    </#if><#-- CondFamily_STM32G4 -->
    .CntPhA                     = 0,
    .CntPhB                     = 0,
    .CntPhC                     = 0,
    .SWerror                    = 0,
    .TurnOnLowSidesAction       = false,
    .OffCalibrWaitTimeCounter   = 0,
    <#if CondFamily_STM32F7>
    .Motor                      = 0,
    <#else><#-- CondFamily_STM32F7 == false -->
    .Motor                      = M1,
    </#if><#-- CondFamily_STM32F7 -->
    .RLDetectionMode            = false,
    .SingleShuntTopology        = false,
    .Ia                         = 0,
    .Ib                         = 0,
    .Ic                         = 0,
    .LPFIqd_const               = LPF_FILT_CONST,
    .DTTest                     = 0,
    .DTCompCnt                  = DTCOMPCNT,
    .PWMperiod                  = PWM_PERIOD_CYCLES,
    .Ton                        = TON,
    .Toff                       = TOFF,
    .OverCurrentFlag            = false,
    .OverVoltageFlag            = false,
    .BrakeActionLock            = false,
    .driverProtectionFlag       = false,
  },

  .Half_PWMPeriod               = PWM_PERIOD_CYCLES/2u,
  .PhaseAOffset                 = 0,
  .PhaseBOffset                 = 0,
  .PhaseCOffset                 = 0,

    <#if CondFamily_STM32L4>
  .PolarizationCounter          = 0,
  .ADC_ExternalTriggerInjected  = 0,
    </#if><#-- CondFamily_STM32L4 -->

    <#if CondFamily_STM32F7>
  .PolarizationCounter          = 0,
  .ADCTriggerEdge               = 0,
    </#if><#-- CondFamily_STM32F7 -->

    <#if CondFamily_STM32G4>
  .ADC_ExternalPolarityInjected = (uint8_t)0,
  .PolarizationCounter          = (uint8_t)0,
  .PolarizationSector           = (uint8_t)0,
  .ADCRegularLocked             = false,
    </#if><#-- CondFamily_STM32G4 -->
  .pParams_str                  = &R3_2_ParamsM1
};
  </#if><#-- Condition for 2 independent ADCs or 2 samplings on one ADC 
       ((CondFamily_STM32F4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32F3 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
     || (CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
     || (CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
     || (CondFamily_STM32G4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))) -->
  
  
  <#if MC.DRIVE_NUMBER != "1">
<#----------------------------------------------------------------------------->
<#-- Condition for F3xx/F4xx/G4xx series for Motor 2 when MC.DRIVE_NUMBER > 1-->
<#----------------------------------------------------------------------------->

<#------------------------------------------------------->
<#-- Condition for MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
<#------------------------------------------------------->
    <#if MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
      <#if (CondFamily_STM32F4 || CondFamily_STM32F3 || CondFamily_STM32G4 )>
/**
  * @brief  Current sensor parameters Dual Drive Motor 2
  */
PWMC_ICS_Handle_t PWM_Handle_M2 = {
  {
    .pFctGetPhaseCurrents       = &ICS_GetPhaseCurrents,
    .pFctSetOffsetCalib         = &ICS_SetOffsetCalib,
    .pFctGetOffsetCalib         = &ICS_GetOffsetCalib,
    .pFctSwitchOffPwm           = &ICS_SwitchOffPWM,
    .pFctSwitchOnPwm            = &ICS_SwitchOnPWM,
        <#if CondFamily_STM32G4 >
    .pFctCurrReadingCalib       = &ICS_CurrentReadingPolarization,
        <#else>
    .pFctCurrReadingCalib       = &ICS_CurrentReadingCalibration,
        </#if><#-- CondFamily_STM32G4 -->
    .pFctTurnOnLowSides         = &ICS_TurnOnLowSides,
    .pFctSetADCSampPointSectX   = &ICS_WriteTIMRegisters,
    .pFctOCPSetReferenceVoltage = MC_NULL,
    .pFctRLDetectionModeEnable  = MC_NULL,
    .pFctRLDetectionModeDisable = MC_NULL,
    .pFctRLDetectionModeSetDuty = MC_NULL,
    .pFctRLTurnOnLowSidesAndStart = MC_NULL,
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->         
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES2 * SQRT3FACTOR) / 16384u,
    .Sector                     = 0,
    .CntPhA                     = 0,
    .CntPhB                     = 0,
    .CntPhC                     = 0,
    .SWerror                    = 0,
    .TurnOnLowSidesAction       = false,
    .OffCalibrWaitTimeCounter   = 0,
    .Motor                      = M2,
    .RLDetectionMode            = false,
    .SingleShuntTopology        = false,
    .Ia                         = 0,
    .Ib                         = 0,
    .Ic                         = 0,
    .LPFIqd_const               = LPF_FILT_CONST,
    .DTTest                     = 0,
    .DTCompCnt                  = DTCOMPCNT2,
    .PWMperiod                  = PWM_PERIOD_CYCLES2,
    .Ton                        = TON2,
    .Toff                       = TOFF2,
    .OverCurrentFlag            = false,
    .OverVoltageFlag            = false,
    .BrakeActionLock            = false,
    .driverProtectionFlag       = false,    
  },
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->   
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->      
    .OverCurrentFlag            = false,
    .OverVoltageFlag            = false,
    .BrakeActionLock            = false,
    .driverProtectionFlag       = false,
        <#if CondFamily_STM32F3>
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->       
  .PhaseAOffset                 = 0,
  .PhaseBOffset                 = 0,
  .Half_PWMPeriod               = PWM_PERIOD_CYCLES2 / 2u,
  .PolarizationCounter          = 0,
          <#if MC.M2_PWM_TIMER_SELECTION == "PWM_TIM1">
  .ADCConfig1                   = (MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_U} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM1_CH4,
  .ADCConfig2                   = (MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_V} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM1_CH4,
          <#elseif MC.M2_PWM_TIMER_SELECTION == "PWM_TIM8">
            <#if (((MC.M2_CS_ADC_U == "ADC1") || (MC.M2_CS_ADC_U == "ADC2")) 
          && ((MC.M2_CS_ADC_V == "ADC1") || (MC.M2_CS_ADC_V == "ADC2")))>
  .ADCConfig1                   = (MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_U} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4_ADC12,
  .ADCConfig2                   = (MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_V} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4_ADC12,
            <#elseif (((MC.M2_CS_ADC_U == "ADC3") || (MC.M2_CS_ADC_U == "ADC4")) 
          && ((MC.M2_CS_ADC_V == "ADC3") || (MC.M2_CS_ADC_V == "ADC4")))>
  .ADCConfig1                   = (MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_U} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4__ADC34,
  .ADCConfig2                   = (MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_V} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4__ADC34,
              <#else>
  #error "Not supported ICS configuration"
              </#if><#-- (((MC.M2_CS_ADC_U == "ADC1") || (MC.M2_CS_ADC_U == "ADC2"))... -->
            </#if><#-- MC.M2_PWM_TIMER_SELECTION == "PWM_TIM1" -->
          </#if><#-- CondFamily_STM32F3 -->
        <#if CondFamily_STM32F3 || CondFamily_STM32G4>  
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->    
  	    </#if><#-- CondFamily_STM32F3 || CondFamily_STM32G4-->  
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->        
  .pParams_str                  = &ICS_ParamsM2   
};
      </#if><#-- ((CondFamily_STM32F4 || (CondFamily_STM32F3 &&  || (CondFamily_STM32G4 )) -->
    </#if><#-- (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  
<#------------------------------------------------------------------------------->
<#-- Condition for PID Speed loop parameters Motor 2 when MC.DRIVE_NUMBER > 1 -->
<#------------------------------------------------------------------------------->

/**
  * @brief  PI / PID Speed loop parameters Motor 2.
  */
PID_Handle_t PIDSpeedHandle_M2 =
{
  .hDefKpGain          = (int16_t)PID_SPEED_KP_DEFAULT2,
  .hDefKiGain          = (int16_t)PID_SPEED_KI_DEFAULT2,
  .wUpperIntegralLimit = (int32_t)IQMAX2 * (int32_t)SP_KIDIV2,
  .wLowerIntegralLimit = -(int32_t)IQMAX2 * (int32_t)SP_KIDIV2,
  .hUpperOutputLimit   = (int16_t)IQMAX2,
  .hLowerOutputLimit   = -(int16_t)IQMAX2,
  .hKpDivisor          = (uint16_t)SP_KPDIV2,
  .hKiDivisor          = (uint16_t)SP_KIDIV2,
  .hKpDivisorPOW2      = (uint16_t)SP_KPDIV_LOG2,
  .hKiDivisorPOW2      = (uint16_t)SP_KIDIV_LOG2,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};

/**
  * @brief  PI / PID Iq loop parameters Motor 2.
  */
PID_Handle_t PIDIqHandle_M2 =
{
  .hDefKpGain          = (int16_t)PID_TORQUE_KP_DEFAULT2,
  .hDefKiGain          = (int16_t)PID_TORQUE_KI_DEFAULT2,
  .wUpperIntegralLimit = (int32_t)INT16_MAX * TF_KIDIV2,
  .wLowerIntegralLimit = (int32_t)-INT16_MAX * TF_KIDIV2,
  .hUpperOutputLimit   = INT16_MAX,
  .hLowerOutputLimit   = -INT16_MAX,
  .hKpDivisor          = (uint16_t)TF_KPDIV2,
  .hKiDivisor          = (uint16_t)TF_KIDIV2,
  .hKpDivisorPOW2      = (uint16_t)TF_KPDIV_LOG2,
  .hKiDivisorPOW2      = (uint16_t)TF_KIDIV_LOG2,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};

/**
  * @brief  PI / PID Id loop parameters Motor 2.
  */
PID_Handle_t PIDIdHandle_M2 =
{
  .hDefKpGain          = (int16_t)PID_FLUX_KP_DEFAULT2,
  .hDefKiGain          = (int16_t)PID_FLUX_KI_DEFAULT2,
  .wUpperIntegralLimit = (int32_t)INT16_MAX * TF_KIDIV2,
  .wLowerIntegralLimit = (int32_t)-INT16_MAX * TF_KIDIV2,
  .hUpperOutputLimit   = INT16_MAX,
  .hLowerOutputLimit   = -INT16_MAX,
  .hKpDivisor          = (uint16_t)TF_KPDIV2,
  .hKiDivisor          = (uint16_t)TF_KIDIV2,
  .hKpDivisorPOW2      = (uint16_t)TF_KPDIV_LOG2,
  .hKiDivisorPOW2      = (uint16_t)TF_KIDIV_LOG2,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};

<#-------------------------------------------------------------------------------------->
<#-- Condition for SpeedNPosition sensor parameters Motor 2 when MC.DRIVE_NUMBER > 1 -->
<#-------------------------------------------------------------------------------------->
    <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || M2_ENCODER || (MC.M2_SPEED_SENSOR == "SENSORLESS_ADC")>
/**
  * @brief  SpeedNPosition sensor parameters Motor 2.
  */
VirtualSpeedSensor_Handle_t VirtualSpeedSensorM2 =
{
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM2,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15 * MAX_APPLICATION_SPEED_UNIT2),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT2),
    .bMaximumSpeedErrorsNumber = M2_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED2,
    .DPPConvFactor             = DPP_CONV_FACTOR2,
  },

  .hSpeedSamplingFreqHz        = MEDIUM_FREQUENCY_TASK_RATE2,
  .hTransitionSteps            = (int16_t)(TF_REGULATION_RATE2 * TRANSITION_DURATION2 / 1000.0),
};

    </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || M2_ENCODER || (MC.M2_SPEED_SENSOR == "SENSORLESS_ADC") -->
    <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL")>
/**
  * @brief  SpeedNPosition sensor parameters Motor 2 - State Observer + PLL.
  */
STO_PLL_Handle_t STO_PLL_M2 =
{
   ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM2,
    .SpeedUnit                 = SPEED_UNIT,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15 * MAX_APPLICATION_SPEED_UNIT2),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT2),
    .bMaximumSpeedErrorsNumber = M2_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED2,
    .DPPConvFactor             = DPP_CONV_FACTOR2,
  },

  .hC1                         = C12,
  .hC2                         = C22,
  .hC3                         = C32,
  .hC4                         = C42,
  .hC5                         = C52,
  .hF1                         = F12,
  .hF2                         = F22,

  .PIRegulator =
  {
    .hDefKpGain                = PLL_KP_GAIN2,
    .hDefKiGain                = PLL_KI_GAIN2,
    .hDefKdGain                = 0x0000U,
    .hKpDivisor                = PLL_KPDIV2,
    .hKiDivisor                = PLL_KIDIV2,
    .hKdDivisor                = 0x0000U,
    .wUpperIntegralLimit       = INT32_MAX,
    .wLowerIntegralLimit       = -INT32_MAX,
    .hUpperOutputLimit         = INT16_MAX,
    .hLowerOutputLimit         = -INT16_MAX,
    .hKpDivisorPOW2            = PLL_KPDIV_LOG2,
    .hKiDivisorPOW2            = PLL_KIDIV_LOG2,
    .hKdDivisorPOW2            = 0x0000U,
  },

  .SpeedBufferSizeUnit         = STO_FIFO_DEPTH_UNIT2,
  .SpeedBufferSizeDpp          = STO_FIFO_DEPTH_DPP2,
  .VariancePercentage          = PERCENTAGE_FACTOR2,
  .SpeedValidationBand_H       = SPEED_BAND_UPPER_LIMIT2,
  .SpeedValidationBand_L       = SPEED_BAND_LOWER_LIMIT2,
  .MinStartUpValidSpeed        = OBS_MINIMUM_SPEED_UNIT2,
  .StartUpConsistThreshold     = NB_CONSECUTIVE_TESTS2,
  .Reliability_hysteresys      = M2_SS_MEAS_ERRORS_BEFORE_FAULTS,
  .BemfConsistencyCheck        = M2_BEMF_CONSISTENCY_TOL,
  .BemfConsistencyGain         = M2_BEMF_CONSISTENCY_GAIN,
  .MaxAppPositiveMecSpeedUnit  = (uint16_t)(MAX_APPLICATION_SPEED_UNIT2 * 1.15),
  .F1LOG                       = F1_LOG2,
  .F2LOG                       = F2_LOG2,
  .SpeedBufferSizeDppLOG       = STO_FIFO_DEPTH_DPP_LOG2,
  .hForcedDirection            = 0x0000U
};
 
    </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL")-->
    <#if (MC.M2_SPEED_SENSOR == "STO_PLL")>
STO_Handle_t STO_M2 = 
{
  ._Super                        = (SpeednPosFdbk_Handle_t*)&STO_PLL_M2,//cstat !MISRAC2012-Rule-11.3
  .pFctForceConvergency1         = &STO_PLL_ForceConvergency1,
  .pFctForceConvergency2         = &STO_PLL_ForceConvergency2,
  .pFctStoOtfResetPLL            = &STO_OTF_ResetPLL,
  .pFctSTO_SpeedReliabilityCheck = &STO_PLL_IsVarianceTight
};

    </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") -->
/**
  * @brief  SpeednTorque Controller parameters Motor 2.
  */
SpeednTorqCtrl_Handle_t SpeednTorqCtrlM2 =
{
  .STCFrequencyHz             = MEDIUM_FREQUENCY_TASK_RATE2,
  .MaxAppPositiveMecSpeedUnit = (uint16_t)(MAX_APPLICATION_SPEED_UNIT2),
  .MinAppPositiveMecSpeedUnit = (uint16_t)(MIN_APPLICATION_SPEED_UNIT2),
  .MaxAppNegativeMecSpeedUnit = (int16_t)(-MIN_APPLICATION_SPEED_UNIT2),
  .MinAppNegativeMecSpeedUnit = (int16_t)(-MAX_APPLICATION_SPEED_UNIT2),
  .MaxPositiveTorque          = (int16_t)NOMINAL_CURRENT2,
  .MinNegativeTorque          = -(int16_t)NOMINAL_CURRENT2,
  .ModeDefault                = DEFAULT_CONTROL_MODE2,
  .MecSpeedRefUnitDefault     = (int16_t)(DEFAULT_TARGET_SPEED_UNIT2),
  .TorqueRefDefault           = (int16_t)DEFAULT_TORQUE_COMPONENT2,
  .IdrefDefault               = (int16_t)DEFAULT_FLUX_COMPONENT2,
};
    <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")>
RevUpCtrl_Handle_t RevUpControlM2 =
{
  .hRUCFrequencyHz         = MEDIUM_FREQUENCY_TASK_RATE2,
  .hStartingMecAngle       = (int16_t)((int32_t)(STARTING_ANGLE_DEG2)* 65536/360),
  .bFirstAccelerationStage = (ENABLE_SL_ALGO_FROM_PHASE2-1u),
  .hMinStartUpValidSpeed   = OBS_MINIMUM_SPEED_UNIT2,
  .hMinStartUpFlySpeed     = (int16_t)((OBS_MINIMUM_SPEED_UNIT2)/2),
  .OTFStartupEnabled       = ${MC.M2_OTF_STARTUP?string("true","false")},

  .OTFPhaseParams = 
  {
    (uint16_t)500,
    0,
    (int16_t)PHASE5_FINAL_CURRENT2,
    (void*)MC_NULL
  },

  .ParamsData = 
  {
    {(uint16_t)PHASE1_DURATION2,(int16_t)(PHASE1_FINAL_SPEED_UNIT2),(uint16_t)PHASE1_FINAL_CURRENT2,&RevUpControlM2.ParamsData[1]},
    {(uint16_t)PHASE2_DURATION2,(int16_t)(PHASE2_FINAL_SPEED_UNIT2),(uint16_t)PHASE2_FINAL_CURRENT2,&RevUpControlM2.ParamsData[2]},
    {(uint16_t)PHASE3_DURATION2,(int16_t)(PHASE3_FINAL_SPEED_UNIT2),(uint16_t)PHASE3_FINAL_CURRENT2,&RevUpControlM2.ParamsData[3]},
    {(uint16_t)PHASE4_DURATION2,(int16_t)(PHASE4_FINAL_SPEED_UNIT2),(uint16_t)PHASE4_FINAL_CURRENT2,&RevUpControlM2.ParamsData[4]},
    {(uint16_t)PHASE5_DURATION2,(int16_t)(PHASE5_FINAL_SPEED_UNIT2),(uint16_t)PHASE5_FINAL_CURRENT2,(void*)MC_NULL},
  },
};
    </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->
  </#if><#-- DUAL_DRIVE -->
</#if><#-- FOC -->





<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || M1_ENCODER || (MC.M1_SPEED_SENSOR == "SENSORLESS_ADC")>
/**
  * @brief  SpeedNPosition sensor parameters Motor 1 - Base Class.
  */
VirtualSpeedSensor_Handle_t VirtualSpeedSensorM1 =
{
  
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15*MAX_APPLICATION_SPEED_UNIT),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT),
    .bMaximumSpeedErrorsNumber = M1_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED,
    .DPPConvFactor             = DPP_CONV_FACTOR,
  },
  
  .hSpeedSamplingFreqHz        = MEDIUM_FREQUENCY_TASK_RATE,
  .hTransitionSteps            = (int16_t)((TF_REGULATION_RATE * TRANSITION_DURATION) / 1000.0),
};
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || M1_ENCODER || (MC.M1_SPEED_SENSOR == "SENSORLESS_ADC") -->

<#if FOC>
  <#if (MC.M1_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL")>
/**
  * @brief  SpeedNPosition sensor parameters Motor 1 - State Observer + PLL.
  */
STO_PLL_Handle_t STO_PLL_M1 =
{
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM,
    .SpeedUnit                 = SPEED_UNIT,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15 * MAX_APPLICATION_SPEED_UNIT),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT),
    .bMaximumSpeedErrorsNumber = M1_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED,
    .DPPConvFactor             = DPP_CONV_FACTOR,
  },

  .hC1                         = C1,
  .hC2                         = C2,
  .hC3                         = C3,
  .hC4                         = C4,
  .hC5                         = C5,
  .hF1                         = F1,
  .hF2                         = F2,

  .PIRegulator =
  {
    .hDefKpGain                = PLL_KP_GAIN,
    .hDefKiGain                = PLL_KI_GAIN,
    .hDefKdGain                = 0x0000U,
    .hKpDivisor                = PLL_KPDIV,
    .hKiDivisor                = PLL_KIDIV,
    .hKdDivisor                = 0x0000U,
    .wUpperIntegralLimit       = INT32_MAX,
    .wLowerIntegralLimit       = -INT32_MAX,
    .hUpperOutputLimit         = INT16_MAX,
    .hLowerOutputLimit         = -INT16_MAX,
    .hKpDivisorPOW2            = PLL_KPDIV_LOG,
    .hKiDivisorPOW2            = PLL_KIDIV_LOG,
    .hKdDivisorPOW2            = 0x0000U,
  },

  .SpeedBufferSizeUnit         = STO_FIFO_DEPTH_UNIT,
  .SpeedBufferSizeDpp          = STO_FIFO_DEPTH_DPP,
  .VariancePercentage          = PERCENTAGE_FACTOR,
  .SpeedValidationBand_H       = SPEED_BAND_UPPER_LIMIT,
  .SpeedValidationBand_L       = SPEED_BAND_LOWER_LIMIT,
  .MinStartUpValidSpeed        = OBS_MINIMUM_SPEED_UNIT,
  .StartUpConsistThreshold     = NB_CONSECUTIVE_TESTS,
  .Reliability_hysteresys      = M1_SS_MEAS_ERRORS_BEFORE_FAULTS,
  .BemfConsistencyCheck        = M1_BEMF_CONSISTENCY_TOL,
  .BemfConsistencyGain         = M1_BEMF_CONSISTENCY_GAIN,
  .MaxAppPositiveMecSpeedUnit  = (uint16_t)(MAX_APPLICATION_SPEED_UNIT * 1.15),
  .F1LOG                       = F1_LOG,
  .F2LOG                       = F2_LOG,
  .SpeedBufferSizeDppLOG       = STO_FIFO_DEPTH_DPP_LOG,
  .hForcedDirection            = 0x0000U
};

  </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") -->
  <#if (MC.M1_SPEED_SENSOR == "STO_PLL")>
STO_Handle_t STO_M1 = 
{
  ._Super                        = (SpeednPosFdbk_Handle_t *)&STO_PLL_M1, //cstat !MISRAC2012-Rule-11.3
  .pFctForceConvergency1         = &STO_PLL_ForceConvergency1,
  .pFctForceConvergency2         = &STO_PLL_ForceConvergency2,
  .pFctStoOtfResetPLL            = &STO_OTF_ResetPLL,
  .pFctSTO_SpeedReliabilityCheck = &STO_PLL_IsVarianceTight
};

  </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") -->

  <#if (MC.M2_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
/**
  * @brief  SpeedNPosition sensor parameters Motor 2 - State Observer + CORDIC.
  */
STO_CR_Handle_t STO_CR_M2 =
{ 
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM2,
    .SpeedUnit                 = SPEED_UNIT,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15 * MAX_APPLICATION_SPEED_UNIT2),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT2),
    .bMaximumSpeedErrorsNumber = M2_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED2,
    .DPPConvFactor             = DPP_CONV_FACTOR2,
  },

  .hC1                         = CORD_C12,
  .hC2                         = CORD_C22,
  .hC3                         = CORD_C32,
  .hC4                         = CORD_C42,
  .hC5                         = CORD_C52,
  .hF1                         = CORD_F12,
  .hF2                         = CORD_F22,
  .SpeedBufferSizeUnit         = CORD_FIFO_DEPTH_UNIT2,
  .SpeedBufferSizedpp          = CORD_FIFO_DEPTH_DPP2,
  .VariancePercentage          = CORD_PERCENTAGE_FACTOR2,
  .SpeedValidationBand_H       = SPEED_BAND_UPPER_LIMIT2,
  .SpeedValidationBand_L       = SPEED_BAND_LOWER_LIMIT2,
  .MinStartUpValidSpeed        = OBS_MINIMUM_SPEED_UNIT2,
  .StartUpConsistThreshold     = NB_CONSECUTIVE_TESTS2,
  .Reliability_hysteresys      = M2_SS_MEAS_ERRORS_BEFORE_FAULTS,
  .MaxInstantElAcceleration    = CORD_MAX_ACCEL_DPPP2,
  .BemfConsistencyCheck        = M2_CORD_BEMF_CONSISTENCY_TOL,
  .BemfConsistencyGain         = M2_CORD_BEMF_CONSISTENCY_GAIN,
  .MaxAppPositiveMecSpeedUnit  = (uint16_t)(MAX_APPLICATION_SPEED_UNIT2 * 1.15),
  .F1LOG                       = CORD_F1_LOG2 ,
  .F2LOG                       = CORD_F2_LOG2 ,
  .SpeedBufferSizedppLOG       = CORD_FIFO_DEPTH_DPP_LOG2
};
  </#if><#-- (MC.M2_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->

  <#if (MC.M2_SPEED_SENSOR == "STO_CORDIC")>
STO_Handle_t STO_M2 = 
{
  ._Super                        = (SpeednPosFdbk_Handle_t *)&STO_CR_M2, //cstat !MISRAC2012-Rule-11.3
  .pFctForceConvergency1         = &STO_CR_ForceConvergency1,
  .pFctForceConvergency2         = &STO_CR_ForceConvergency2,
  .pFctStoOtfResetPLL            = MC_NULL,
  .pFctSTO_SpeedReliabilityCheck = &STO_CR_IsSpeedReliable
};
  </#if><#-- (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->
  
  <#if (MC.M1_SPEED_SENSOR == "STO_CORDIC") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
/**
  * @brief  SpeedNPosition sensor parameters Motor 1 - State Observer + CORDIC.
  */
STO_CR_Handle_t STO_CR_M1 =
{ 
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM,
    .SpeedUnit                 = SPEED_UNIT,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15 * MAX_APPLICATION_SPEED_UNIT),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT),
    .bMaximumSpeedErrorsNumber = M1_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED,
    .DPPConvFactor             = DPP_CONV_FACTOR,
  },

  .hC1                         = CORD_C1,
  .hC2                         = CORD_C2,
  .hC3                         = CORD_C3,
  .hC4                         = CORD_C4,
  .hC5                         = CORD_C5,
  .hF1                         = CORD_F1,
  .hF2                         = CORD_F2,
  .SpeedBufferSizeUnit         = CORD_FIFO_DEPTH_UNIT,
  .SpeedBufferSizedpp          = CORD_FIFO_DEPTH_DPP,
  .VariancePercentage          = CORD_PERCENTAGE_FACTOR,
  .SpeedValidationBand_H       = SPEED_BAND_UPPER_LIMIT,
  .SpeedValidationBand_L       = SPEED_BAND_LOWER_LIMIT,
  .MinStartUpValidSpeed        = OBS_MINIMUM_SPEED_UNIT,
  .StartUpConsistThreshold     = NB_CONSECUTIVE_TESTS,
  .Reliability_hysteresys      = M1_SS_MEAS_ERRORS_BEFORE_FAULTS,
  .MaxInstantElAcceleration    = CORD_MAX_ACCEL_DPPP,
  .BemfConsistencyCheck        = M1_CORD_BEMF_CONSISTENCY_TOL,
  .BemfConsistencyGain         = M1_CORD_BEMF_CONSISTENCY_GAIN,
  .MaxAppPositiveMecSpeedUnit  = (uint16_t)(MAX_APPLICATION_SPEED_UNIT * 1.15),
  .F1LOG                       = CORD_F1_LOG,
  .F2LOG                       = CORD_F2_LOG,
  .SpeedBufferSizedppLOG       = CORD_FIFO_DEPTH_DPP_LOG
};
  </#if><#-- (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->

  <#if (MC.M1_SPEED_SENSOR == "STO_CORDIC")>
STO_Handle_t STO_M1 = 
{
  ._Super                        = (SpeednPosFdbk_Handle_t *)&STO_CR_M1, //cstat !MISRAC2012-Rule-11.3
  .pFctForceConvergency1         = &STO_CR_ForceConvergency1,
  .pFctForceConvergency2         = &STO_CR_ForceConvergency2,
  .pFctStoOtfResetPLL            = MC_NULL,
  .pFctSTO_SpeedReliabilityCheck = &STO_CR_IsSpeedReliable
};
  </#if><#-- (MC.M1_SPEED_SENSOR == "STO_CORDIC") -->

  <#if M1_ENCODER>
/**
  * @brief  SpeedNPosition sensor parameters Motor 1 - Encoder.
  */
ENCODER_Handle_t ENCODER_M1 =
{
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15 * MAX_APPLICATION_SPEED_UNIT),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT),
    .bMaximumSpeedErrorsNumber = M1_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED,
    .DPPConvFactor             = DPP_CONV_FACTOR,
  },

  .PulseNumber                 = M1_ENCODER_PPR * 4,
  .SpeedSamplingFreqHz         = MEDIUM_FREQUENCY_TASK_RATE,
  .SpeedBufferSize             = ENC_AVERAGING_FIFO_DEPTH,
  .TIMx                        = ${_last_word(MC.M1_ENC_TIMER_SELECTION)},
  .ICx_Filter                  = M1_ENC_IC_FILTER_LL,
};

/**
  * @brief  Encoder Alignment Controller parameters Motor 1.
  */
EncAlign_Handle_t EncAlignCtrlM1 =
{
  .hEACFrequencyHz = MEDIUM_FREQUENCY_TASK_RATE,
  .hFinalTorque    = FINAL_I_ALIGNMENT,
  .hElAngle        = ALIGNMENT_ANGLE_S16,
  .hDurationms     = M1_ALIGNMENT_DURATION,
  .bElToMecRatio   = POLE_PAIR_NUM,
};

  </#if><#-- M1_ENCODER -->
  <#if M2_ENCODER>
/**
  * @brief  SpeedNPosition sensor parameters Motor 2 - Encoder.
  */
ENCODER_Handle_t ENCODER_M2 =
{
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM2,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15 * MAX_APPLICATION_SPEED_UNIT2),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT2),
    .bMaximumSpeedErrorsNumber = M2_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED2,
    .DPPConvFactor             = DPP_CONV_FACTOR,
  },

  .PulseNumber                 = M2_ENCODER_PPR*4,
  .SpeedSamplingFreqHz         = MEDIUM_FREQUENCY_TASK_RATE2,
  .SpeedBufferSize             = ENC_AVERAGING_FIFO_DEPTH2,
  .TIMx                        = ${_last_word(MC.M2_ENC_TIMER_SELECTION)},
  .ICx_Filter                  = M2_ENC_IC_FILTER_LL,
};

/**
  * @brief  Encoder Alignment Controller parameters Motor 2.
  */
EncAlign_Handle_t EncAlignCtrlM2 =
{
  .hEACFrequencyHz = MEDIUM_FREQUENCY_TASK_RATE2,
  .hFinalTorque    = FINAL_I_ALIGNMENT2,
  .hElAngle        = ALIGNMENT_ANGLE_S162,
  .hDurationms     = M2_ALIGNMENT_DURATION,
  .bElToMecRatio   = POLE_PAIR_NUM2,
};

  </#if><#-- M2_ENCODER -->
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
OpenLoop_Handle_t OpenLoop_ParamsM1 =
{
  .hDefaultVoltage = OPEN_LOOP_VOLTAGE_d,
  .VFMode          = OPEN_LOOP_VF,
  .hVFOffset       = OPEN_LOOP_OFF,
  .hVFSlope        = OPEN_LOOP_K
};

  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
  <#if (MC.DRIVE_NUMBER != "1" && MC.M2_DBG_OPEN_LOOP_ENABLE == true)>
OpenLoop_Handle_t OpenLoop_ParamsM2 =
{
  .hDefaultVoltage = OPEN_LOOP_VOLTAGE_d2,
  .VFMode          = OPEN_LOOP_VF2,
  .hVFOffset       = OPEN_LOOP_OFF2,
  .hVFSlope        = OPEN_LOOP_K2
};

  </#if><#-- (MC.DRIVE_NUMBER > 1 &&  MC.M2_DBG_OPEN_LOOP_ENABLE == true) -->
  <#if ((CondFamily_STM32F3 && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '2' )))
    || (CondFamily_STM32F4 && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '2' )))
    || (CondFamily_STM32G4 && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '2' ))))>
/**
  * @brief  PWM parameters Motor 2 for three shunts two ADCs. 
  */
PWMC_R3_2_Handle_t PWM_Handle_M2=
{
  {
    .pFctGetPhaseCurrents       = &R3_2_GetPhaseCurrents${OVM2},
    .pFctSetADCSampPointSectX   = &R3_2_SetADCSampPointSectX${OVM2},
    .pFctSetOffsetCalib         = &R3_2_SetOffsetCalib,
    .pFctGetOffsetCalib         = &R3_2_GetOffsetCalib,
    .pFctSwitchOffPwm           = &R3_2_SwitchOffPWM,
    .pFctSwitchOnPwm            = &R3_2_SwitchOnPWM,
    <#if CondFamily_STM32F4>
    .pFctCurrReadingCalib       = &R3_2_CurrentReadingCalibration,
    <#else><#-- CondFamily_STM32F3 || CondFamily_STM32G4 -->
    .pFctCurrReadingCalib       = &R3_2_CurrentReadingPolarization,
     </#if><#-- if CondFamily_STM32F4 -->
    .pFctTurnOnLowSides         = &R3_2_TurnOnLowSides,
    .pFctOCPSetReferenceVoltage = MC_NULL,
    .pFctRLDetectionModeEnable  = &R3_2_RLDetectionModeEnable,
    .pFctRLDetectionModeDisable = &R3_2_RLDetectionModeDisable,
    .pFctRLDetectionModeSetDuty = &R3_2_RLDetectionModeSetDuty,
    .pFctRLTurnOnLowSidesAndStart = &R3_2_RLTurnOnLowSidesAndStart,
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING2,
<#if MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M2_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M2_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_U_Pin,
    .pwm_en_v_port     = M2_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_V_Pin,
    .pwm_en_w_port     = M2_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M2_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M2_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M2_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M2_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->        
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES2 * SQRT3FACTOR) / 16384u,
    .Sector                     = 0,
    .CntPhA                     = 0,
    .CntPhB                     = 0,
    .CntPhC                     = 0,
    .SWerror                    = 0,
    .TurnOnLowSidesAction       = false,
    .OffCalibrWaitTimeCounter   = 0,
    .Motor                      = M2,
    .RLDetectionMode            = false,
    .SingleShuntTopology        = false,
    .Ia                         = 0,
    .Ib                         = 0,
    .Ic                         = 0,
    .LPFIqd_const               = LPF_FILT_CONST,
    .DTTest                     = 0,
    .DTCompCnt                  = DTCOMPCNT2,
    .PWMperiod                  = PWM_PERIOD_CYCLES2,
    .Ton                        = TON2,
    .Toff                       = TOFF2
  },
  
  .PhaseAOffset                 = 0,
  .PhaseBOffset                 = 0,
  .PhaseCOffset                 = 0,
  .Half_PWMPeriod               = PWM_PERIOD_CYCLES2 / 2u,
  .pParams_str                  = &R3_2_ParamsM2,
};

</#if><#-- ((CondFamily_STM32F3 && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '2' )))
    || (CondFamily_STM32F4 && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '2' )))


    || (CondFamily_STM32G4 && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '2' )))) -->






  <#if (CondFamily_STM32F0 == false || CondFamily_STM32G0 == false || CondFamily_STM32C0 == false
     || CondFamily_STM32H7 == false) && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
/**
  * @brief  Current sensor parameters Dual Drive Motor 1 - ICS.
  */
PWMC_ICS_Handle_t PWM_Handle_M1 =
{
  { 
    .pFctGetPhaseCurrents       = &ICS_GetPhaseCurrents,
    .pFctSwitchOffPwm           = &ICS_SwitchOffPWM,
    .pFctSwitchOnPwm            = &ICS_SwitchOnPWM,
    <#if CondFamily_STM32G4>
    .pFctCurrReadingCalib       = &ICS_CurrentReadingPolarization,
    <#else><#-- CondFamily_STM32G4 == false -->
    .pFctCurrReadingCalib       = &ICS_CurrentReadingCalibration,
    </#if><#-- CondFamily_STM32G4 -->
    .pFctTurnOnLowSides         = &ICS_TurnOnLowSides,
    .pFctSetADCSampPointSectX   = &ICS_WriteTIMRegisters,
    .pFctSetOffsetCalib         = &ICS_SetOffsetCalib,
    .pFctGetOffsetCalib         = &ICS_GetOffsetCalib,
    .pFctOCPSetReferenceVoltage = MC_NULL,
    .pFctRLDetectionModeEnable  = MC_NULL,
    .pFctRLDetectionModeDisable = MC_NULL,
    .pFctRLDetectionModeSetDuty = MC_NULL,
    .pFctRLTurnOnLowSidesAndStart = MC_NULL,
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING,
<#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO">
  <#if MC.M1_SHARED_SIGNAL_ENABLE == false>
    .pwm_en_u_port     = M1_PWM_EN_U_GPIO_Port,
    .pwm_en_u_pin      = M1_PWM_EN_U_Pin,
    .pwm_en_v_port     = M1_PWM_EN_V_GPIO_Port,
    .pwm_en_v_pin      = M1_PWM_EN_V_Pin,
    .pwm_en_w_port     = M1_PWM_EN_W_GPIO_Port,
    .pwm_en_w_pin      = M1_PWM_EN_W_Pin,
  <#else>
    .pwm_en_u_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_u_pin      = M1_PWM_EN_UVW_Pin,
    .pwm_en_v_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_v_pin      = M1_PWM_EN_UVW_Pin,
    .pwm_en_w_port     = M1_PWM_EN_UVW_GPIO_Port,
    .pwm_en_w_pin      = M1_PWM_EN_UVW_Pin,
  </#if>  
<#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "ES_GPIO" -->
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
</#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "ES_GPIO" -->        
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES * SQRT3FACTOR) / 16384u,
    .Sector                     = 0,
    .CntPhA                     = 0,
    .CntPhB                     = 0,
    .CntPhC                     = 0,
    .SWerror                    = 0,
    .TurnOnLowSidesAction       = false,
    .OffCalibrWaitTimeCounter   = 0,
    .Motor                      = M1,
    .RLDetectionMode            = false,
    .SingleShuntTopology        = false,
    .Ia                         = 0,
    .Ib                         = 0,
    .Ic                         = 0,
    .LPFIqd_const               = LPF_FILT_CONST,
    .DTTest                     = 0,
    .DTCompCnt                  = DTCOMPCNT,
    .PWMperiod                  = PWM_PERIOD_CYCLES,
    .Ton                        = TON,
    .Toff                       = TOFF,
    .OverCurrentFlag            = false,
    .OverVoltageFlag            = false,
    .BrakeActionLock            = false,
    .driverProtectionFlag       = false,    
  },
  
  .PhaseAOffset                 = 0,
  .PhaseBOffset                 = 0,
  .Half_PWMPeriod               = PWM_PERIOD_CYCLES / 2u,
  .PolarizationCounter          = 0,

    <#if CondFamily_STM32L4>
      <#if MC.M1_PWM_TIMER_SELECTION == "PWM_TIM1">
  .ADCConfig1                   = (MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                 | LL_ADC_INJ_TRIG_EXT_TIM1_CH4,
  .ADCConfig2                   = (MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                 | LL_ADC_INJ_TRIG_EXT_TIM1_CH4,
      <#elseif MC.M1_PWM_TIMER_SELECTION == "PWM_TIM8">
  .ADCConfig1                   = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4,
  .ADCConfig2                   = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4,
      </#if><#-- MC.M1_PWM_TIMER_SELECTION == "PWM_TIM1" -->
    <#elseif CondFamily_STM32F3 >
      <#if MC.M1_PWM_TIMER_SELECTION == "PWM_TIM1">
  .ADCConfig1                   = (MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM1_CH4,
  .ADCConfig2                   = (MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM1_CH4,
      <#elseif MC.M1_PWM_TIMER_SELECTION == "PWM_TIM8">
        <#if (((MC.M1_CS_ADC_U == "ADC1") || (MC.M1_CS_ADC_U == "ADC2")) 
        && ((MC.M1_CS_ADC_V == "ADC1") || (MC.M1_CS_ADC_V == "ADC2")))>
  .ADCConfig1                   = (MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4_ADC12,
  .ADCConfig2                   = (MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4_ADC12,
        <#elseif (((MC.M1_CS_ADC_U == "ADC3") || (MC.M1_CS_ADC_U == "ADC4")) 
        && ((MC.M1_CS_ADC_V == "ADC3") || (MC.M1_CS_ADC_V == "ADC4")))>
  .ADCConfig1                   = (MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4__ADC34,
  .ADCConfig2                   = (MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V} << 8) | LL_ADC_INJ_TRIG_EXT_RISING
                                | LL_ADC_INJ_TRIG_EXT_TIM8_CH4__ADC34,
        <#else>
          #error "Not supported ICS configuration"
        </#if><#-- (((MC.M1_CS_ADC_U == "ADC1") || (MC.M1_CS_ADC_U == "ADC2"))... -->
      </#if><#-- MC.M1_PWM_TIMER_SELECTION == "PWM_TIM1" -->
    </#if><#-- CondFamily_STM32L4 -->
    <#if (CondFamily_STM32F7 == false) && (CondFamily_STM32F4 == false)>
    </#if><#-- (CondFamily_STM32F7 == false) && (CondFamily_STM32F4 == false) -->
  .pParams_str = &ICS_ParamsM1                               
};
  </#if><#-- (CondFamily_STM32F0 == false || CondFamily_STM32G0 == false || CondFamily_STM32C0 == false
           || CondFamily_STM32H7 == false) && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->




  <#if MC.DRIVE_NUMBER != "1">
    <#if (MC.M2_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
/**
  * @brief  SpeedNPosition sensor parameters Motor 2 - HALL.
  */
HALL_Handle_t HALL_M2 =
{
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM2,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15*MAX_APPLICATION_SPEED_UNIT2),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT2),
    .bMaximumSpeedErrorsNumber = M2_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED2,
    .DPPConvFactor             = DPP_CONV_FACTOR2,
  },

  .SensorPlacement             = HALL_SENSORS_PLACEMENT2,
  .PhaseShift                  = (int16_t)((HALL_PHASE_SHIFT2 * 65536) / 360),
  .SpeedSamplingFreqHz         = MEDIUM_FREQUENCY_TASK_RATE2,
  .SpeedBufferSize             = HALL_AVERAGING_FIFO_DEPTH2,
  .TIMClockFreq                = HALL_TIM_CLK2,
  .TIMx                        = ${_last_word(MC.M2_HALL_TIMER_SELECTION)},
  .ICx_Filter                  = M2_HALL_IC_FILTER_LL,
  .PWMFreqScaling              = PWM_FREQ_SCALING2,
  .HallMtpa                    = HALL_MTPA2,
  .H1Port                      = M2_HALL_H1_GPIO_Port,
  .H1Pin                       = M2_HALL_H1_Pin,
  .H2Port                      = M2_HALL_H2_GPIO_Port,
  .H2Pin                       = M2_HALL_H2_Pin,
  .H3Port                      = M2_HALL_H3_GPIO_Port,
  .H3Pin                       = M2_HALL_H3_Pin,
};
    </#if><#-- (MC.M2_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->

<#if SIX_STEP>
  <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
Bemf_ADC_Handle_t Bemf_ADC_M1 =
{
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15*MAX_APPLICATION_SPEED_UNIT),
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT),
    .bMaximumSpeedErrorsNumber = M1_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED,
    .DPPConvFactor             = DPP_CONV_FACTOR,
  },
  .Pwm_OFF = {
    .AdcThresholdDown = BEMF_THRESHOLD_DOWN,
    .AdcThresholdUp = BEMF_THRESHOLD_UP,
  <#if  MC.BEMF_OVERSAMPLING == false || CondFamily_STM32G4>
    .SamplingPoint = BEMF_ADC_TRIG_TIME,
  <#else>
    .SamplingPoint = BEMF_ADC_TRIG_TIME / BEMF_SAMPLING_RATE,
  </#if>  
  },
    <#if  MC.DRIVE_MODE == "VM">
  .Pwm_ON = {
    .AdcThresholdDown = BEMF_THRESHOLD_DOWN_ON,
    .AdcThresholdUp = BEMF_THRESHOLD_UP_ON,
  <#if  MC.BEMF_OVERSAMPLING == false || CondFamily_STM32G4>
    .SamplingPoint = BEMF_ADC_TRIG_TIME_ON,
    <#else>
    .SamplingPoint = BEMF_ADC_TRIG_TIME_ON / BEMF_SAMPLING_RATE,
    </#if>  
  },

  .OnSensingEnThres            = BEMF_PWM_ON_ENABLE_THRES,
  .OnSensingDisThres           = BEMF_PWM_ON_DISABLE_THRES,
    <#else><#-- MC.DRIVE_MODE != "VM" -->
  .OnSensingEnThres            = PWM_PERIOD_CYCLES,
  .OnSensingDisThres           = PWM_PERIOD_CYCLES,
    </#if><#-- MC.DRIVE_MODE == "VM" -->
  .Zc2CommDelay                = ZCD_TO_COMM,
  .pParams_str                 = &Bemf_ADC_ParamsM1,
  .TIMClockFreq                = SYSCLK_FREQ,
  .PWMFreqScaling              = PWM_FREQ_SCALING,
  .SpeedSamplingFreqHz         = MEDIUM_FREQUENCY_TASK_RATE,
  .SpeedBufferSize             = BEMF_AVERAGING_FIFO_DEPTH,
  .LowFreqTimerPsc             = LF_TIMER_PSC,
  .MinStartUpValidSpeed        = OBS_MINIMUM_SPEED_UNIT,
  .StartUpConsistThreshold     = NB_CONSECUTIVE_TESTS,

  .DemagParams =
  {
    .DemagMinimumSpeedUnit     = DEMAG_MINIMUM_SPEED,
    .RevUpDemagSpeedConv       = DEMAG_REVUP_CONV_FACTOR,
    .RunDemagSpeedConv         = DEMAG_RUN_CONV_FACTOR,
    .DemagMinimumThreshold     = MIN_DEMAG_TIME,
  },

  .DriveMode                   = DEFAULT_DRIVE_MODE,
  <#if  MC.BEMF_OVERSAMPLING && !CondFamily_STM32G4>
  .OverSamplingRate = BEMF_SAMPLING_RATE,
  .SamplingGuard = PWM_PERIOD_CYCLES / 10,
  </#if>  
};
  </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->

  <#if MC.DRIVE_MODE == "CM">
CurrentRef_Handle_t CurrentRef_M1 =
{
  .pParams_str = &CurrentRef_ParamsM1,
  .PWMperiod   = PWM_PERIOD_CYCLES_REF,
    <#if MC.CURRENT_LIMITER_OFFSET>
  .StartCntPh  = (uint16_t)PWM_PERIOD_CYCLES_REF,
    <#else><#-- MC.CURRENT_LIMITER_OFFSET == false -->
  .StartCntPh  = 0,
    </#if><#-- MC.CURRENT_LIMITER_OFFSET -->
};
  </#if><#-- MC.DRIVE_MODE == "CM" -->

  <#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
PWMC_SixPwm_Handle_t PWM_Handle_M1 =
{
  {
    .pFctSwitchOffPwm          = &SixPwm_SwitchOffPWM,
    .pFctSwitchOnPwm           = &SixPwm_SwitchOnPWM,
    <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
    .pFctSetADCTriggerChannel  = &SixPwm_SetADCTriggerChannel,
    </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
    .pFctTurnOnLowSides        = &SixPwm_TurnOnLowSides,
    .pGetFastDemagFlag         = &SixPwm_FastDemagFlag,
    .pGetQuasiSynchFlag        = &SixPwm_QuasiSynchFlag,
    <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" && MC.DRIVE_MODE == "CM">
    .StartCntPh                = (uint16_t) (0.9*BEMF_ADC_TRIG_TIME),
	<#else><#-- MC.DRIVE_MODE == "CM" -->
    .StartCntPh                = PWM_PERIOD_CYCLES,
    </#if><#-- MC.DRIVE_MODE == "CM" -->
    .SWerror                   = 0,
    .TurnOnLowSidesAction      = false,
    .Motor                     = 0,
    .PWMperiod                 = PWM_PERIOD_CYCLES,
    <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
    .DemagCounterThreshold     = MIN_DEMAG_TIME,
    <#else><#-- MC.M1_SPEED_SENSOR != "SENSORLESS_ADC" -->
    .DemagCounterThreshold     = 65535,
    </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
    .OverCurrentFlag  = 0,
    .OverVoltageFlag  = 0,
    .BrakeActionLock  = 0,
    .driverProtectionFlag = false,
  },
<#if  MC.QUASI_SYNC>
  .QuasiSynchDecay  = true,
<#else>
  .QuasiSynchDecay  = false,
</#if>
<#if  MC.BEMF_OVERSAMPLING>
  .Oversampling  = true,
<#else>
  .Oversampling  = false,
</#if>
<#if  MC.FAST_DEMAG>
  .FastDemag  = true,
<#else>
  .FastDemag  = false,
</#if>
  .pParams_str = &SixPwm_ParamsM1,
};
  <#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
PWMC_ThreePwm_Handle_t PWM_Handle_M1 =
{
  {
    .pFctSwitchOffPwm          = &ThreePwm_SwitchOffPWM,
    .pFctSwitchOnPwm           = &ThreePwm_SwitchOnPWM,
    <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
    .pFctSetADCTriggerChannel  = &ThreePwm_SetADCTriggerChannel,
    </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
    .pFctTurnOnLowSides        = &ThreePwm_TurnOnLowSides,
    .pGetFastDemagFlag         = &ThreePwm_FastDemagFlag,
    .pGetQuasiSynchFlag        = MC_NULL,
    <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" && MC.DRIVE_MODE == "CM">
    .StartCntPh                = (uint16_t) (0.9*BEMF_ADC_TRIG_TIME),
	<#else><#-- MC.DRIVE_MODE == "CM" -->
    .StartCntPh                = PWM_PERIOD_CYCLES,
    </#if><#-- MC.DRIVE_MODE == "CM" -->
    .SWerror                   = 0,
    .TurnOnLowSidesAction      = false,
    .Motor                     = 0,
    .PWMperiod                 = PWM_PERIOD_CYCLES,
    <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
    .DemagCounterThreshold     = MIN_DEMAG_TIME,
    <#else><#-- MC.M1_SPEED_SENSOR != "SENSORLESS_ADC" -->
    .DemagCounterThreshold     = 65535,
    </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
    .OverCurrentFlag  = 0,
    .OverVoltageFlag  = 0,
    .BrakeActionLock  = 0,
    .driverProtectionFlag = false,
  },
<#if  MC.FAST_DEMAG>
  .FastDemag  = true,
<#else>
  .FastDemag  = false,
</#if>
<#if  MC.BEMF_OVERSAMPLING>
  .Oversampling  = true,
<#else>
  .Oversampling  = false,
</#if>
  .pParams_str = &ThreePwm_ParamsM1,
};
  </#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
</#if><#-- SIX_STEP -->

<#if (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
/**
  * @brief  SpeedNPosition sensor parameters Motor 1 - HALL.
  */
HALL_Handle_t HALL_M1 =
{
  ._Super =
  {
    .bElToMecRatio             = POLE_PAIR_NUM,
    .hMaxReliableMecSpeedUnit  = (uint16_t)(1.15 * MAX_APPLICATION_SPEED_UNIT),
<#if FOC>
    .hMinReliableMecSpeedUnit  = (uint16_t)(MIN_APPLICATION_SPEED_UNIT),
<#else>
    .hMinReliableMecSpeedUnit  = 0,
</#if><#-- FOC -->
    .bMaximumSpeedErrorsNumber = M1_SS_MEAS_ERRORS_BEFORE_FAULTS,
    .hMaxReliableMecAccelUnitP = 65535,
    .hMeasurementFrequency     = TF_REGULATION_RATE_SCALED,
    .DPPConvFactor             = DPP_CONV_FACTOR,
  },

  .SensorPlacement             = HALL_SENSORS_PLACEMENT,
  .PhaseShift                  = (int16_t)(HALL_PHASE_SHIFT * 65536 / 360),
  .SpeedSamplingFreqHz         = MEDIUM_FREQUENCY_TASK_RATE,
  .SpeedBufferSize             = HALL_AVERAGING_FIFO_DEPTH,
  .TIMClockFreq                = HALL_TIM_CLK,
  .TIMx                        = ${_last_word(MC.M1_HALL_TIMER_SELECTION)},
  .ICx_Filter                  = M1_HALL_IC_FILTER_LL,
  .PWMFreqScaling              = PWM_FREQ_SCALING,
  .HallMtpa                    = HALL_MTPA,
  .H1Port                      = M1_HALL_H1_GPIO_Port,
  .H1Pin                       = M1_HALL_H1_Pin,
  .H2Port                      = M1_HALL_H2_GPIO_Port,
  .H2Pin                       = M1_HALL_H2_Pin,
  .H3Port                      = M1_HALL_H3_GPIO_Port,
  .H3Pin                       = M1_HALL_H3_Pin,
};
</#if><#-- (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") -->
    
    
    
    
<#if FOC>

  <#if MC.M1_ICL_ENABLED == true>
ICL_Handle_t ICL_M1 =
{
  .ICLstate               = ICL_ACTIVE,
  .hICLTicksCounter       = M1_ICL_CAPS_CHARGING_DELAY_TICKS,
  .hICLSwitchDelayTicks   = M1_ICL_RELAY_SWITCHING_DELAY_TICKS,
  .hICLChargingDelayTicks = M1_ICL_CAPS_CHARGING_DELAY_TICKS,
  .hICLVoltageThreshold   = M1_ICL_VOLTAGE_THRESHOLD_U16VOLT,
};
  </#if><#-- MC.M1_ICL_ENABLED == true -->

  <#if MC.M2_ICL_ENABLED == true && MC.DRIVE_NUMBER != "1">
ICL_Handle_t ICL_M2 =
{
  .ICLstate               = ICL_ACTIVE,
  .hICLTicksCounter       = M1_ICL_CAPS_CHARGING_DELAY_TICKS,
  .hICLSwitchDelayTicks   = M2_ICL_RELAY_SWITCHING_DELAY_TICKS,
  .hICLChargingDelayTicks = M2_ICL_CAPS_CHARGING_DELAY_TICKS,
  .hICLVoltageThreshold   = M1_ICL_VOLTAGE_THRESHOLD_U16VOLT,
};
  </#if><#-- MC.M2_ICL_ENABLED == true && MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->




<#if (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true)>
/**
  * temperature sensor parameters Motor 1.
  */
RegConv_t TempRegConv_M1 =
{
    .regADC                = ${MC.M1_TEMP_FDBK_ADC},
    .channel               = MC_${MC.M1_TEMP_FDBK_CHANNEL},
  .samplingTime          = M1_TEMP_SAMPLING_TIME,
};  
  
NTC_Handle_t TempSensor_M1 =
{
  .bSensorType             = REAL_SENSOR,

  .hLowPassFilterBW        = M1_TEMP_SW_FILTER_BW_FACTOR,
  .hOverTempThreshold      = (uint16_t)(OV_TEMPERATURE_THRESHOLD_d),
  .hOverTempDeactThreshold = (uint16_t)(OV_TEMPERATURE_THRESHOLD_d - OV_TEMPERATURE_HYSTERESIS_d),
  .hSensitivity            = (int16_t)(ADC_REFERENCE_VOLTAGE/dV_dT),
  .wV0                     = (uint16_t)((V0_V * 65536) / ADC_REFERENCE_VOLTAGE),
  .hT0                     = T0_C,
};
<#else><#-- (MC.M1_TEMPERATURE_READING == false  || MC.M1_OV_TEMPERATURE_PROT_ENABLING == false) -->
/**
  * Virtual temperature sensor parameters Motor 1.
  */
NTC_Handle_t TempSensor_M1 =
{
  .bSensorType     = VIRTUAL_SENSOR,
  .hExpectedTemp_d = 555,
  .hExpectedTemp_C = M1_VIRTUAL_HEAT_SINK_TEMPERATURE_VALUE,
};
</#if><#-- (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true) -->

<#if MC.DRIVE_NUMBER != "1">
  <#if (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true)>
/**
  * temperature sensor parameters Motor 2.
  */
RegConv_t TempRegConv_M2 =
{
    .regADC                = ${MC.M2_TEMP_FDBK_ADC},
    .channel               = MC_${MC.M2_TEMP_FDBK_CHANNEL},
  .samplingTime          = M2_TEMP_SAMPLING_TIME,
};
  
NTC_Handle_t TempSensor_M2 =
{
  .bSensorType             = REAL_SENSOR,
  .hLowPassFilterBW        = M2_TEMP_SW_FILTER_BW_FACTOR,
  .hOverTempThreshold      = (uint16_t)(OV_TEMPERATURE_THRESHOLD_d2),
  .hOverTempDeactThreshold = (uint16_t)(OV_TEMPERATURE_THRESHOLD_d2 - OV_TEMPERATURE_HYSTERESIS_d2),
  .hSensitivity            = (int16_t)(ADC_REFERENCE_VOLTAGE/dV_dT2),
  .wV0                     = (uint16_t)((V0_V2 * 65536) / ADC_REFERENCE_VOLTAGE),
  .hT0                     = T0_C2,
};
  <#else><#-- (MC.M2_TEMPERATURE_READING == false  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == false) -->
/**
  * Virtual temperature sensor parameters Motor 2.
  */
NTC_Handle_t TempSensor_M2 =
{
  .bSensorType = VIRTUAL_SENSOR,
  .hExpectedTemp_d = 555,
  .hExpectedTemp_C = M2_VIRTUAL_HEAT_SINK_TEMPERATURE_VALUE,
};
  </#if><#-- (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true) -->
</#if><#-- (MC.DRIVE_NUMBER > 1 -->

<#if MC.M1_BUS_VOLTAGE_READING == true>
/* Bus voltage sensor value filter buffer */
static uint16_t RealBusVoltageSensorFilterBufferM1[M1_VBUS_SW_FILTER_BW_FACTOR];

/**
  * Bus voltage sensor parameters Motor 1.
  */
RegConv_t VbusRegConv_M1 =
{
    .regADC                   = ${MC.M1_VBUS_ADC},
    .channel                  = MC_${MC.M1_VBUS_CHANNEL},
    .samplingTime             = M1_VBUS_SAMPLING_TIME,
};
    
RDivider_Handle_t BusVoltageSensor_M1 =
{
  ._Super =
  {
    .SensorType               = REAL_SENSOR,
    .ConversionFactor         = (uint16_t)(ADC_REFERENCE_VOLTAGE / VBUS_PARTITIONING_FACTOR),
  },
  
  .LowPassFilterBW            =  M1_VBUS_SW_FILTER_BW_FACTOR,
  .OverVoltageThreshold       = OVERVOLTAGE_THRESHOLD_d,
  <#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
  .OverVoltageThresholdLow    = OVERVOLTAGE_THRESHOLD_LOW_d,
  <#else><#-- MC.M1_ON_OVER_VOLTAGE != "TURN_ON_R_BRAKE" -->
  .OverVoltageThresholdLow    = OVERVOLTAGE_THRESHOLD_d,
  </#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->
  .OverVoltageHysteresisUpDir = true,
  .UnderVoltageThreshold      =  UNDERVOLTAGE_THRESHOLD_d,
  .aBuffer                    = RealBusVoltageSensorFilterBufferM1,
};
<#else><#-- MC.M1_BUS_VOLTAGE_READING == false -->
/**
  * Virtual bus voltage sensor parameters Motor 1.
  */
VirtualBusVoltageSensor_Handle_t BusVoltageSensor_M1 =
{
  ._Super =
  {
    .SensorType       = VIRTUAL_SENSOR,
    .ConversionFactor = 500,
  },
  .ExpectedVbus_d     = 1 + ((NOMINAL_BUS_VOLTAGE_V * 65536) / 500),
};
</#if><#-- MC.M1_BUS_VOLTAGE_READING == true -->

<#if MC.DRIVE_NUMBER != "1">
  <#if MC.M2_BUS_VOLTAGE_READING == true>
/* Bus voltage sensor value filter buffer */
uint16_t RealBusVoltageSensorFilterBufferM2[M2_VBUS_SW_FILTER_BW_FACTOR];

/**
  * Bus voltage sensor parameters Motor 2.
  */
RegConv_t VbusRegConv_M2 =
{
  .regADC                   = ${MC.M2_VBUS_ADC},
  .channel                  = MC_${MC.M2_VBUS_CHANNEL},
  .samplingTime             = M2_VBUS_SAMPLING_TIME,
};

RDivider_Handle_t BusVoltageSensor_M2 =
{
  ._Super =
  {
    .SensorType               = REAL_SENSOR,
    .ConversionFactor         = (uint16_t)(ADC_REFERENCE_VOLTAGE / VBUS_PARTITIONING_FACTOR2),

  },

  .LowPassFilterBW            = M2_VBUS_SW_FILTER_BW_FACTOR,
  .OverVoltageThreshold       = OVERVOLTAGE_THRESHOLD_d2,
      <#if MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
  .OverVoltageThresholdLow    = OVERVOLTAGE_THRESHOLD_LOW_d2,
      <#else><#-- MC.M2_ON_OVER_VOLTAGE != "TURN_ON_R_BRAKE" -->
  .OverVoltageThresholdLow    = OVERVOLTAGE_THRESHOLD_d2,
      </#if><#-- MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->
  .OverVoltageHysteresisUpDir = true,
  .UnderVoltageThreshold      = UNDERVOLTAGE_THRESHOLD_d2,
  .aBuffer                    = RealBusVoltageSensorFilterBufferM2,
};
  <#else><#-- MC.M2_BUS_VOLTAGE_READING == false --> 
/**
  * Virtual bus voltage sensor parameters Motor 2.
  */
VirtualBusVoltageSensor_Handle_t BusVoltageSensor_M2 =
{
  ._Super =
  {
    .SensorType       = VIRTUAL_SENSOR,
    .ConversionFactor = 500,
  },

  .ExpectedVbus_d     = 1 + ((NOMINAL_BUS_VOLTAGE_V2 * 65536) / 500),
};
  </#if><#-- MC.M2_BUS_VOLTAGE_READING == true -->
</#if><#-- MC.DRIVE_NUMBER > 1 -->

<#if FOC>
/** RAMP for Motor1
  *
  */
RampExtMngr_Handle_t RampExtMngrHFParamsM1 =
{
  .FrequencyHz = TF_REGULATION_RATE 
};

/**
  * @brief  CircleLimitation Component parameters Motor 1 - Base Component.
  */
CircleLimitation_Handle_t CircleLimitationM1 =
{
  .MaxModule = MAX_MODULE,
  <#if MC.M1_FLUX_WEAKENING_ENABLING  == true>
  .MaxVd     = (uint16_t)((MAX_MODULE * FW_VOLTAGE_REF) / 1000),
  <#else><#-- MC.M1_FLUX_WEAKENING_ENABLING  == false -->
  .MaxVd     = (uint16_t)((MAX_MODULE * 950) / 1000),
  </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING  == true -->
};
  <#if MC.DRIVE_NUMBER != "1">
/** RAMP for Motor2
  *
  */
RampExtMngr_Handle_t RampExtMngrHFParamsM2 =
{
  .FrequencyHz = TF_REGULATION_RATE2 
};

/**
  * @brief  CircleLimitation Component parameters Motor 2 - Base Component.
  */
CircleLimitation_Handle_t CircleLimitationM2 =
{
  .MaxModule = MAX_MODULE2,
    <#if MC.M2_FLUX_WEAKENING_ENABLING  == true>
  .MaxVd     = (uint16_t)((MAX_MODULE2 * FW_VOLTAGE_REF2) / 1000),
    <#else><#-- MC.M2_FLUX_WEAKENING_ENABLING  == false -->
  .MaxVd     = (uint16_t)((MAX_MODULE2 * 950) / 1000),
    </#if><#-- MC.M2_FLUX_WEAKENING_ENABLING  == true -->
};
  </#if>

  <#if MC.M1_MTPA_ENABLING == true>
MTPA_Handle_t MTPARegM1 =
{
  .SegDiv   = (int16_t)SEGDIV,
  .AngCoeff = M1_ANGC,
  .Offset   = OFST,
};
  </#if><#--  MC.M1_MTPA_ENABLING == true -->

  <#if MC.DRIVE_NUMBER != "1" &&  MC.M2_MTPA_ENABLING == true>
MTPA_Handle_t MTPARegM2 =
{
  .SegDiv   = (int16_t)SEGDIV2,
  .AngCoeff = M2_ANGC,
  .Offset   = OFST2,
};
  </#if><#-- MC.DRIVE_NUMBER > 1 &&  MC.M2_MTPA_ENABLING == true -->

  <#if MC.DRIVE_NUMBER != "1">
    <#if MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
DOUT_handle_t R_BrakeParamsM2 =
{
  .OutputState      = INACTIVE,
  .hDOutputPort     = M2_DISSIPATIVE_BRK_GPIO_Port,
  .hDOutputPin      = M2_DISSIPATIVE_BRK_Pin,
  .bDOutputPolarity = ${MC.M2_DISSIPATIVE_BRAKE_POLARITY}  
};
    </#if><#-- MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->

    <#if MC.M2_HW_OV_CURRENT_PROT_BYPASS == true && MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES">
DOUT_handle_t DOUT_OCPDisablingParamsM2 =
{
  .OutputState      = INACTIVE,
  .hDOutputPort     = M2_OCP_DISABLE_GPIO_Port,
  .hDOutputPin      = M2_OCP_DISABLE_Pin,
  .bDOutputPolarity = ${MC.OVERCURR_PROTECTION_HW_DISABLING2}   
};
    </#if><#-- MC.M2_HW_OV_CURRENT_PROT_BYPASS == true && MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" -->

    <#if MC.M2_ICL_ENABLED == true>
DOUT_handle_t ICLDOUTParamsM2 =
{
  .OutputState      = INACTIVE,
  .hDOutputPort     = M2_ICL_SHUT_OUT_GPIO_Port,
  .hDOutputPin      = M2_ICL_SHUT_OUT_Pin,
  .bDOutputPolarity = ${MC.M2_ICL_DIGITAL_OUTPUT_POLARITY}  
};
    </#if><#-- MC.M2_ICL_ENABLED == true -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->

  <#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
DOUT_handle_t R_BrakeParamsM1 =
{
  .OutputState      = INACTIVE,
  .hDOutputPort     = M1_DISSIPATIVE_BRK_GPIO_Port,
  .hDOutputPin      = M1_DISSIPATIVE_BRK_Pin,
  .bDOutputPolarity = ${MC.M1_DISSIPATIVE_BRAKE_POLARITY} 
};
  </#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->

  <#if MC.M1_HW_OV_CURRENT_PROT_BYPASS == true && MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES">
DOUT_handle_t DOUT_OCPDisablingParamsM1 =
{
  .OutputState      = INACTIVE,
  .hDOutputPort     = M1_OCP_DISABLE_GPIO_Port,
  .hDOutputPin      = M1_OCP_DISABLE_Pin,
  .bDOutputPolarity = ${MC.OVERCURR_PROTECTION_HW_DISABLING}
};
  </#if><#-- MC.M1_HW_OV_CURRENT_PROT_BYPASS == true && MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" -->

  <#if MC.M1_ICL_ENABLED == true>
DOUT_handle_t ICLDOUTParamsM1 =
{
  .OutputState      = INACTIVE,
  .hDOutputPort     = M1_ICL_SHUT_OUT_GPIO_Port,
  .hDOutputPin      = M1_ICL_SHUT_OUT_Pin,
  .bDOutputPolarity = ${MC.M1_ICL_DIGITAL_OUTPUT_POLARITY}    
};
  </#if><#-- MC.M1_ICL_ENABLED == true -->

  <#if MC.PFC_ENABLED == true>
PFC_Handle_t PFC = 
{
  .pParams                  = & PFC_Params,
  .pMPW1                    = &PQD_MotorPowMeasM1,
    <#if MC.DRIVE_NUMBER != "1">
  .pMPW2                    = &PQD_MotorPowMeasM2,
    <#else><#-- MC.DRIVE_NUMBER == 1 -->
  .pMPW2                    = MC_NULL,
    </#if><#-- MC.DRIVE_NUMBER > 1 -->
  .pVBS                     = (BusVoltageSensor_Handle_t *)&BusVoltageSensor_M1,
  .pBusVoltage              = MC_NULL,
  .cPICurr = {
    .hDefKpGain             = (int16_t)PFC_PID_CURR_KP_DEFAULT,
    .hDefKiGain             = (int16_t)PFC_PID_CURR_KI_DEFAULT,
    .hKpGain                = (int16_t)PFC_PID_CURR_KP_DEFAULT,
    .hKiGain                = (int16_t)PFC_PID_CURR_KI_DEFAULT,
    .wIntegralTerm          = 0,
    .wUpperIntegralLimit    = (int32_t)INT16_MAX * PFC_PID_CURR_KI_DIV,
    .wLowerIntegralLimit    = (int32_t)(-INT16_MAX) * PFC_PID_CURR_KI_DIV,
    .hUpperOutputLimit      = INT16_MAX,
    .hLowerOutputLimit      = 0,
    .hKpDivisor             = (uint16_t)PFC_PID_CURR_KP_DIV,
    .hKiDivisor             = (uint16_t)PFC_PID_CURR_KI_DIV,
    .hKpDivisorPOW2         = (uint16_t)LOG2(PFC_PID_CURR_KP_DIV),
    .hKiDivisorPOW2         = (uint16_t)LOG2(PFC_PID_CURR_KI_DIV),
    .hDefKdGain             = 0,
    .hKdGain                = 0,
    .hKdDivisor             = 0,
    .hKdDivisorPOW2         = 0,
    .wPrevProcessVarError   = 0
  },

  .cPIVolt = {
    .hDefKpGain             = (int16_t)PFC_PID_VOLT_KP_DEFAULT,
      .hDefKiGain           = (int16_t)PFC_PID_VOLT_KI_DEFAULT,
      .hKpGain              = (int16_t)PFC_PID_VOLT_KP_DEFAULT,
      .hKiGain              = (int16_t)PFC_PID_VOLT_KI_DEFAULT,
      .wIntegralTerm        = 0,
      .wUpperIntegralLimit  = (int32_t)(PFC_NOMINALCURRENTS16A / 2) * PFC_PID_VOLT_KI_DIV,
      .wLowerIntegralLimit  = (int32_t)(-PFC_NOMINALCURRENTS16A / 2) * PFC_PID_VOLT_KI_DIV,
      .hUpperOutputLimit    = (int16_t)(PFC_NOMINALCURRENTS16A / 2),
      .hLowerOutputLimit    = (int16_t)(-PFC_NOMINALCURRENTS16A / 2),
      .hKpDivisor           = (uint16_t)PFC_PID_VOLT_KP_DIV,
      .hKiDivisor           = (uint16_t)PFC_PID_VOLT_KI_DIV,
      .hKpDivisorPOW2       = (uint16_t)LOG2(PFC_PID_VOLT_KP_DIV),
      .hKiDivisorPOW2       = (uint16_t)LOG2(PFC_PID_VOLT_KI_DIV),
      .hDefKdGain           = 0,
      .hKdGain              = 0,
      .hKdDivisor           = 0,
      .hKdDivisorPOW2       = 0,
      .wPrevProcessVarError = 0
  },

  .bPFCen                   = false,
  .hTargetVoltageReference  = PFC_VOLTAGEREFERENCE,
  .hVoltageReference        = 0,
  .hStartUpDuration         = (int16_t)((PFC_STARTUPDURATION  *0.001) * (PFC_VOLTCTRLFREQUENCY * 1.0)),
  .hMainsSync               = 0,
  .hMainsPeriod             = 0,
  .bMainsPk                 = false,
};
  </#if><#-- MC.PFC_ENABLED == true -->

  <#if MC.MOTOR_PROFILER == true>
RampExtMngr_Handle_t RampExtMngrParamsSCC =
{
  .FrequencyHz = TF_REGULATION_RATE 
};

SCC_Handle_t SCC = 
{
  .pSCC_Params_str = &SCC_Params
};

RampExtMngr_Handle_t RampExtMngrParamsOTT =
{
  .FrequencyHz = MEDIUM_FREQUENCY_TASK_RATE 
};

OTT_Handle_t OTT = 
{
 
  .pOTT_Params_str = &OTT_Params
};
    <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
HT_Handle_t HT;
    </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") -->
  </#if><#-- MOTOR_PROFILER -->

  <#if MC.USE_STGAP1S>
GAP_Handle_t STGAP_M1 =
{
  .DeviceNum    = M1_STAGAP1AS_NUM,
  .DeviceParams =
  {
                  STGAP1AS_BRAKE,
                  STGAP1AS_UH,
                  STGAP1AS_UL,
                  STGAP1AS_VH,
                  STGAP1AS_VL,
                  STGAP1AS_WH,
                  STGAP1AS_WL,
   },
   .SPIx        = SPI2,
   .NCSPort     = M1_STGAP1S_SPI_NCS_GPIO_Port,
   .NCSPin      = M1_STGAP1S_SPI_NCS_Pin,
   .NSDPort     = M1_STGAP1S_SPI_NSD_GPIO_Port,
   .NSDPin      = M1_STGAP1S_SPI_NSD_Pin,
};
  </#if><#-- MC.USE_STGAP1S -->
</#if><#-- FOC -->

<#if MC.STSPIN32G4 == true >
/**
 * @brief Handler of STSPIN32G4 driver.
 */
STSPIN32G4_HandleTypeDef HdlSTSPING4;
</#if><#-- MC.STSPIN32G4 == true -->

<#if MC.M1_POTENTIOMETER_ENABLE == true>
/** @brief */
uint16_t SpeedPotentiometer_ValuesBuffer_M1[ 1 << POTENTIOMETER_LPF_BANDWIDTH_POW2_M1 ];

/*
 * Speed Potentiometer instance for motor 1
 */
RegConv_t PotRegConv_M1 =
{
  .regADC              = ${MC.POTENTIOMETER_ADC},
  .channel             = MC_${MC.POTENTIOMETER_CHANNEL},
  .samplingTime        = POTENTIOMETER_ADC_SAMPLING_TIME_M1,
};
  
SpeedPotentiometer_Handle_t SpeedPotentiometer_M1 =
{
  .Pot =
  {
    .LPFilterBandwidthPOW2 = POTENTIOMETER_LPF_BANDWIDTH_POW2_M1,
    .PotMeasArray          = (uint16_t *)SpeedPotentiometer_ValuesBuffer_M1
  },

  .pMCI                    = & Mci[0],
  .RampSlope               = (uint32_t)(POTENTIOMETER_RAMP_SLOPE_M1),
  .ConversionFactor        = (uint16_t)((65536.0) / ((POTENTIOMETER_MAX_SPEED_M1) - (POTENTIOMETER_MIN_SPEED_M1))),
  .SpeedAdjustmentRange    = (uint16_t)(POTENTIOMETER_SPEED_ADJUSTMENT_RANGE_M1),
  .MinimumSpeed            = (uint16_t)(POTENTIOMETER_MIN_SPEED_M1),
};
</#if><#-- MC.M1_POTENTIOMETER_ENABLE == true -->

<#if MC.DRIVE_NUMBER != "1">
<#if MC.M2_POTENTIOMETER_ENABLE == true>
/** @brief */
uint16_t SpeedPotentiometer_ValuesBuffer_M2[ 1 << POTENTIOMETER_LPF_BANDWIDTH_POW2_M2 ];

/*
 * Speed Potentiometer instance for motor 2
 */
RegConv_t PotRegConv_M2 =
{
  .regADC              = ${MC.POTENTIOMETER_ADC2},
  .channel             = MC_${MC.POTENTIOMETER_CHANNEL2},
  .samplingTime        = POTENTIOMETER_ADC_SAMPLING_TIME_M2,
};
  
SpeedPotentiometer_Handle_t SpeedPotentiometer_M2 =
{
  .Pot =
  {
    .LPFilterBandwidthPOW2 = POTENTIOMETER_LPF_BANDWIDTH_POW2_M2,
    .PotMeasArray          = (uint16_t *)SpeedPotentiometer_ValuesBuffer_M2
  },

  .pMCI                    = & Mci[1],
  .RampSlope               = (uint32_t)(POTENTIOMETER_RAMP_SLOPE_M2),
  .ConversionFactor        = (uint16_t)((65536.0) / ((POTENTIOMETER_MAX_SPEED_M2) - (POTENTIOMETER_MIN_SPEED_M2))),
  .SpeedAdjustmentRange    = (uint16_t)(POTENTIOMETER_SPEED_ADJUSTMENT_RANGE_M2),
  .MinimumSpeed            = (uint16_t)(POTENTIOMETER_MIN_SPEED_M2),
};
</#if><#-- MC.M2_POTENTIOMETER_ENABLE == true -->
</#if><#-- MC.DRIVE_NUMBER != "1" -->

MCI_Handle_t Mci[NBR_OF_MOTORS];
<#if MC.DRIVE_NUMBER != "1">
SpeednTorqCtrl_Handle_t *pSTC[NBR_OF_MOTORS]    = {&SpeednTorqCtrlM1 ,&SpeednTorqCtrlM2};
PID_Handle_t *pPIDIq[NBR_OF_MOTORS]             = {&PIDIqHandle_M1 ,&PIDIqHandle_M2};
PID_Handle_t *pPIDId[NBR_OF_MOTORS]             = {&PIDIdHandle_M1 ,&PIDIdHandle_M2};
NTC_Handle_t *pTemperatureSensor[NBR_OF_MOTORS] = {&TempSensor_M1 ,&TempSensor_M2};
PQD_MotorPowMeas_Handle_t *pMPM[NBR_OF_MOTORS]  = {&PQD_MotorPowMeasM1,&PQD_MotorPowMeasM2};   
  <#if MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true>
PosCtrl_Handle_t *pPosCtrl[NBR_OF_MOTORS]       = {<#if MC.M1_POSITION_CTRL_ENABLING >&PosCtrlM1<#else>MC_NULL</#if>,
                                                   <#if MC.M2_POSITION_CTRL_ENABLING>&PosCtrlM2<#else>MC_NULL</#if>};
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == false && MC.M2_POSITION_CTRL_ENABLING == false -->
  <#if MC.M1_FLUX_WEAKENING_ENABLING == true || MC.M2_FLUX_WEAKENING_ENABLING == true>
FW_Handle_t *pFW[NBR_OF_MOTORS]                 = {<#if MC.M1_FLUX_WEAKENING_ENABLING>&FW_M1<#else>MC_NULL</#if>,
                                                   <#if MC.M2_FLUX_WEAKENING_ENABLING>&FW_M2<#else>MC_NULL </#if>};
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true -->
  <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true>
FF_Handle_t *pFF[NBR_OF_MOTORS]                 = {<#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING>&FF_M1<#else>MC_NULL</#if>,
                                                <#if MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING>&FF_M2<#else>MC_NULL </#if>};
  </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
<#else><#-- MC.DRIVE_NUMBER == 1 -->
SpeednTorqCtrl_Handle_t *pSTC[NBR_OF_MOTORS]    = {&SpeednTorqCtrlM1};
NTC_Handle_t *pTemperatureSensor[NBR_OF_MOTORS] = {&TempSensor_M1};
  <#if FOC>
PID_Handle_t *pPIDIq[NBR_OF_MOTORS]             = {&PIDIqHandle_M1};
PID_Handle_t *pPIDId[NBR_OF_MOTORS]             = {&PIDIdHandle_M1};
PQD_MotorPowMeas_Handle_t *pMPM[NBR_OF_MOTORS]  = {&PQD_MotorPowMeasM1};   
    <#if MC.M1_POSITION_CTRL_ENABLING == true>
PosCtrl_Handle_t *pPosCtrl[NBR_OF_MOTORS]       = {&PosCtrlM1};
    </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true -->
    <#if MC.M1_FLUX_WEAKENING_ENABLING == true >
FW_Handle_t *pFW[NBR_OF_MOTORS]                 = {&FW_M1};
    </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING ==t rue -->
    <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true>
FF_Handle_t *pFF[NBR_OF_MOTORS]                 = {&FF_M1};
    </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
  </#if><#-- FOC -->
</#if><#-- MC.DRIVE_NUMBER > 1 -->
<#if MC.ESC_ENABLE == true>
ESC_Handle_t ESC_M1 =
{
  .pESC_params = &ESC_ParamsM1,
} ;
</#if><#-- MC.ESC_ENABLE == true -->
/* USER CODE BEGIN Additional configuration */
/* USER CODE END Additional configuration */ 

/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
