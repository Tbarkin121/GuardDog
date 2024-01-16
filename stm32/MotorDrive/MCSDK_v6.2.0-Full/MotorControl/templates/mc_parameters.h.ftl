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
</#if>

<#assign FOC = MC.M1_DRIVE_TYPE == "FOC" || MC.M2_DRIVE_TYPE == "FOC">
<#assign SIX_STEP = MC.M1_DRIVE_TYPE == "SIX_STEP" || MC.M2_DRIVE_TYPE == "SIX_STEP">
<#assign ACIM = MC.M1_DRIVE_TYPE == "ACIM" || MC.M2_DRIVE_TYPE == "ACIM">
<#-- Condition for STM32F302x8x MCU -->
<#assign CondMcu_STM32F302x8x = (McuName?? && McuName?matches("STM32F302.8.*"))>
<#-- Condition for STM32F072xxx MCU -->
<#assign CondMcu_STM32F072xxx = (McuName?? && McuName?matches("STM32F072.*"))>
<#-- Condition for STM32F446xCx or STM32F446xEx -->
<#assign CondMcu_STM32F446xCEx = (McuName?? && McuName?matches("STM32F446.(C|E).*"))>
<#-- Condition for STM32F0 Family -->
<#assign CondFamily_STM32F0 = (FamilyName?? && FamilyName=="STM32F0")>
<#-- Condition for STM32F3 Family -->
<#assign CondFamily_STM32F3 = (FamilyName?? && FamilyName == "STM32F3")>
<#-- Condition for STM32F4 Family -->
<#assign CondFamily_STM32F4 = (FamilyName?? && FamilyName == "STM32F4") >
<#-- Condition for STM32G4 Family -->
<#assign CondFamily_STM32G4 = (FamilyName?? && FamilyName == "STM32G4") >
<#-- Condition for STM32L4 Family -->
<#assign CondFamily_STM32L4 = (FamilyName?? && FamilyName == "STM32L4") >
<#-- Condition for STM32F7 Family -->
<#assign CondFamily_STM32F7 = (FamilyName?? && FamilyName == "STM32F7") >
<#-- Condition for STM32H7 Family -->
<#assign CondFamily_STM32H7 = (FamilyName?? && FamilyName == "STM32H7") >
<#-- Condition for STM32H5 Family -->
<#assign CondFamily_STM32H5 = (FamilyName?? && FamilyName == "STM32H5") >
<#-- Condition for STM32G0 Family -->
<#assign CondFamily_STM32G0 = (FamilyName?? && FamilyName == "STM32G0") >
<#-- Condition for STM32C0 Family -->
<#assign CondFamily_STM32C0 = (FamilyName?? && FamilyName == "STM32C0") >

/**
  ******************************************************************************
  * @file    mc_parameters.h
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file provides declarations of HW parameters specific to the 
  *          configuration of the subsystem.
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
#ifndef MC_PARAMETERS_H
#define MC_PARAMETERS_H

#include "mc_interface.h"  
<#if MC.ESC_ENABLE == true>
#include "esc.h"
</#if><#-- MC.ESC_ENABLE == true -->
<#if FOC>
  <#if CondFamily_STM32F4>
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_1_f4xx_pwm_curr_fdbk.h"
    </#if><#--  (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
#include "r3_2_f4xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_f4xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  </#if><#-- CondFamily_STM32F4 -->

  <#if CondFamily_STM32F0>
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_f0xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32F0 -->

  <#if CondFamily_STM32L4><#-- CondFamily_STM32L4 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
#include "ics_l4xx_pwm_curr_fdbk.h"
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) >
#include "r3_2_l4xx_pwm_curr_fdbk.h"
    </#if><#--((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))  -->
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_1_l4xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32L4 --->

  <#if CondFamily_STM32F7><#-- CondFamily_STM32F7 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"  
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
#include "ics_f7xx_pwm_curr_fdbk.h"
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) >
#include "r3_2_f7xx_pwm_curr_fdbk.h"
    </#if><#--((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))  -->
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_1_f7xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32F7 --->

  <#if CondFamily_STM32H7><#-- CondFamily_STM32H7 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_h7xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
#include "ics_h7xx_pwm_curr_fdbk.h"
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) >
#include "r3_2_h7xx_pwm_curr_fdbk.h"
    </#if><#--((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))  -->
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_1_h7xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32H7 --->

  <#if CondFamily_STM32H5 > <#-- CondFamily_STM32H5 -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) >
#include "r1_ps_pwm_curr_fdbk.h"  
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if  ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) >
#include "r3_2_h5xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) >
#include "r3_1_h5xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  </#if> <#-- CondFamily_STM32H5 -->


  <#if CondFamily_STM32F3><#-- CondFamily_STM32F3 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"    
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_f30x_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
#include "r3_2_f30x_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')
    || (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1'))>
#include "r3_1_f30x_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32F3 --->

  <#if CondFamily_STM32G4><#-- CondFamily_STM32G4 --->
   <#if MC.M1_SPEED_SENSOR == "HSO" || MC.M1_SPEED_SENSOR == "ZEST">
    <#if MC.M1_SPEED_SENSOR == "ZEST">
#include "zest.h"
      </#if>
#include "flash_parameters.h"   
#include "r3_g4xx_pwm_curr_fdbk.h" 
    <#else>
      <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
        || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) >
#include "r1_ps_pwm_curr_fdbk.h" 
      </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
              || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
      <#if (((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))  
         || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')))>
#include "r3_2_g4xx_pwm_curr_fdbk.h"
      </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
              || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
      <#if (((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1'))  
         || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1'))) >
#include "r3_1_g4xx_pwm_curr_fdbk.h"
      </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')) 
              || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')) -->
      <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')  >
#include "ics_g4xx_pwm_curr_fdbk.h"
      </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
    </#if>
  </#if> <#-- CondFamily_STM32G4 --->

  <#if (MC.MOTOR_PROFILER == true) && (MC.M1_SPEED_SENSOR != "HSO" && MC.M1_SPEED_SENSOR != "ZEST")>
#include "mp_self_com_ctrl.h"
#include "mp_one_touch_tuning.h"
  </#if><#-- MC.MOTOR_PROFILER == true -->

  <#if CondFamily_STM32G0>
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_g0xx_pwm_curr_fdbk.h"
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32G0 --->

  <#if CondFamily_STM32C0>
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_c0xx_pwm_curr_fdbk.h"
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#error C0 single shunt not supported yet
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  </#if><#-- CondFamily_STM32C0 --->

/* USER CODE BEGIN Additional include */

/* USER CODE END Additional include */  
  <#if CondFamily_STM32F4>
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
extern R3_1_Params_t R3_1_ParamsM1;
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
    <#if (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')>
extern R3_1_Params_t R3_1_ParamsM2;
    </#if><#-- (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) >
extern const R3_2_Params_t R3_2_ParamsM1;
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))  -->
    <#if ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
extern const R3_2_Params_t R3_2_ParamsM2;
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1;
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM2;
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
extern const ICS_Params_t ICS_ParamsM1;
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
    <#if (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
extern const ICS_Params_t ICS_ParamsM2;
    </#if><#-- (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  </#if><#-- CondFamily_STM32F4 --->

  <#if CondFamily_STM32F0>
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
extern const R3_1_Params_t R3_1_Params;
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1;
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32F0 --->

  <#if CondFamily_STM32F3><#-- CondFamily_STM32F3 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1;
    <#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
extern const ICS_Params_t ICS_ParamsM1;
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
      <#if MC.M1_USE_INTERNAL_OPAMP>
extern const R3_2_OPAMPParams_t R3_2_OPAMPParamsM1;
      </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->
extern const R3_2_Params_t R3_2_ParamsM1;
    <#elseif (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
extern const R3_1_Params_t R3_1_ParamsM1;
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM2;
    <#elseif (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')> 
extern const ICS_Params_t ICS_ParamsM2;
    <#elseif ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
      <#if MC.M2_USE_INTERNAL_OPAMP>
extern const R3_2_OPAMPParams_t R3_2_OPAMPParamsM2;
      </#if><#-- MC.M2_USE_INTERNAL_OPAMP -->
extern const R3_2_Params_t R3_2_ParamsM2;
    <#elseif (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')> 
extern const R3_1_Params_t R3_1_ParamsM2;
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32F3 --->

  <#if CondFamily_STM32L4 ><#-- CondFamily_STM32L4 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1;
    <#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' >
extern const ICS_Params_t ICS_ParamsM1;
    <#elseif  (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
extern const R3_1_Params_t R3_1_ParamsM1;
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) >
extern const R3_2_Params_t R3_2_ParamsM1;
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if> <#-- CondFamily_STM32L4 --->

  <#if CondFamily_STM32H5 > <#-- CondFamily_STM32H5 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1;    
    <#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' >
extern const ICS_Params_t ICS_ParamsM1;
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
extern const R3_1_Params_t R3_1_ParamsM1;
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))>
extern const R3_2_Params_t R3_2_ParamsM1;
    </#if> <#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if> <#-- CondFamily_STM32H5 --->

  <#if CondFamily_STM32F7><#-- CondFamily_STM32F7 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1;
    <#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
extern const ICS_Params_t ICS_ParamsM1;
    <#elseif (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
extern const R3_1_Params_t R3_1_ParamsM1;
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) >
extern const R3_2_Params_t R3_2_ParamsM1;
    </#if><#--  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32F7 --->

  <#if CondFamily_STM32H7><#-- CondFamily_STM32H7 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1;
    <#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
extern const ICS_Params_t ICS_ParamsM1;
    <#elseif (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
extern const R3_1_Params_t R3_1_ParamsM1;
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) >
extern const R3_2_Params_t R3_2_ParamsM1;
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32H7 --->

  <#if CondFamily_STM32G4><#-- CondFamily_STM32G4 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1; 
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
      <#if MC.M1_SPEED_SENSOR != "HSO" && MC.M1_SPEED_SENSOR != "ZEST" >    
extern const R3_2_Params_t R3_2_ParamsM1;
      </#if> <#-- MC.M1_SPEED_SENSOR != "HSO" && MC.M1_SPEED_SENSOR != "ZEST" --->
    <#elseif  ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1'))>
extern const R3_1_Params_t R3_1_ParamsM1;
    <#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
extern const ICS_Params_t ICS_ParamsM1;
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM2; 
    <#elseif ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
extern R3_2_Params_t R3_2_ParamsM2; 
    <#elseif (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')>
extern const R3_1_Params_t R3_1_ParamsM2;
    <#elseif (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
extern const ICS_Params_t ICS_ParamsM2;
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32G4 --->

  <#if CondFamily_STM32G0><#-- CondFamily_STM32G0 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern const R1_Params_t R1_ParamsM1;
    <#elseif (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
extern const R3_1_Params_t R3_1_Params; 
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32G0 --->
  
  <#if CondFamily_STM32C0><#-- CondFamily_STM32C0 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#error C0 single shunt not supported yet
    <#elseif ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
extern const R3_1_Params_t R3_1_Params; 
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32C0 --->
  <#if MC.PFC_ENABLED == true>
#include "pfc.h"
extern const PFC_Parameters_t PFC_Params;
  </#if><#-- MC.PFC_ENABLED == true -->

  <#if (MC.MOTOR_PROFILER == true) && (MC.M1_SPEED_SENSOR != "HSO" && MC.M1_SPEED_SENSOR != "ZEST")>
extern SCC_Params_t SCC_Params;
extern OTT_Params_t OTT_Params;
  </#if><#--MC.MOTOR_PROFILER == true -->
</#if><#-- FOC -->

<#if SIX_STEP>
  <#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
#include "pwmc_6pwm.h"
  <#else><#--  MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
#include "pwmc_3pwm.h"
  </#if><#--  MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
  <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
	<#if CondFamily_STM32F0>
	  <#if MC.BEMF_OVERSAMPLING>
#include "f0xx_bemf_ADC_OS_fdbk.h"
     <#else>	  
#include "f0xx_bemf_ADC_fdbk.h"
      </#if>
    </#if>
	<#if CondFamily_STM32G0>
	  <#if MC.BEMF_OVERSAMPLING>
#include "g0xx_bemf_ADC_OS_fdbk.h"
     <#else>	  
#include "g0xx_bemf_ADC_fdbk.h"
      </#if>
    </#if>
    <#if CondFamily_STM32C0>
      <#if  MC.BEMF_OVERSAMPLING> 
#include "c0xx_bemf_ADC_OS_fdbk.h"
      <#else>
#include "c0xx_bemf_ADC_fdbk.h"
      </#if>
    </#if>
	<#if CondFamily_STM32G4>	  
#include "g4xx_bemf_ADC_fdbk.h"
    </#if><#-- CondFamily_STM32G4 -->
  </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
  <#if MC.DRIVE_MODE == "CM">
#include "current_ref_ctrl.h"
  </#if><#-- MC.DRIVE_MODE == "CM" -->

/* USER CODE BEGIN Additional include */

/* USER CODE END Additional include */

  <#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
extern const SixPwm_Params_t SixPwm_ParamsM1;
  <#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
extern const ThreePwm_Params_t ThreePwm_ParamsM1;
  </#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->

  <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
extern const Bemf_ADC_Params_t Bemf_ADC_ParamsM1;
  </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
  <#if MC.DRIVE_MODE == "CM">
extern const CurrentRef_Params_t CurrentRef_ParamsM1;
  </#if><#-- MC.DRIVE_MODE == "CM" -->
</#if><#-- SIX_STEP -->

  <#if ACIM>
#include "r3_2_g4xx_pwm_curr_fdbk.h"
  </#if><#-- ACIM -->

<#if MC.ESC_ENABLE == true>
extern const ESC_Params_t ESC_ParamsM1;
</#if><#-- MC.ESC_ENABLE == true -->

<#if MC.M1_SPEED_SENSOR == "HSO" || MC.M1_SPEED_SENSOR == "ZEST">
  <#if MC.M1_SPEED_SENSOR == "ZEST">
extern ZEST_Params ZeST_params_M1;
extern const zestFlashParams_t *zestParams;
  </#if>
extern R3_Params_t R3_ParamsM1; 
extern const FLASH_Params_t flashParams;
extern const MotorConfig_reg_t *motorParams;
extern const boardFlashParams_t *boardParams;
extern const scaleFlashParams_t *scaleParams;
extern const throttleParams_t *throttleParams;
extern const float *KSampleDelayParams;
extern const PIDSpeedFlashParams_t *PIDSpeedParams;
</#if>


<#if FOC || ACIM >
  <#if MC.M1_SPEED_SENSOR != "HSO">
extern ScaleParams_t scaleParams_M1;
  </#if> <#-- MC.M1_SPEED_SENSOR != "HSO" -->
  <#if MC.DRIVE_NUMBER != "1">
    <#if MC.M2_SPEED_SENSOR != "HSO">
extern ScaleParams_t scaleParams_M2;
    </#if> <#-- MC.M2_SPEED_SENSOR != "HSO" -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->


<#if ACIM>
extern R3_2_Params_t R3_2_ParamsM1;
</#if><#-- ACIM -->
/* USER CODE BEGIN Additional extern */

/* USER CODE END Additional extern */  

#endif /* MC_PARAMETERS_H */
/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
