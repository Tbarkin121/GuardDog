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
<#assign M1_HALL_SENSOR = (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") >
<#assign M2_HALL_SENSOR = (MC.M2_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") >


/**
  ******************************************************************************
  * @file    mc_config.h 
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   Motor Control Subsystem components configuration and handler 
  *          structures declarations.
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
  
#ifndef MC_CONFIG_H
#define MC_CONFIG_H

<#-- Specific to FOC algorithm usage -->
#include "pid_regulator.h"
<#if FOC>
#include "speed_torq_ctrl.h"
</#if><#-- FOC -->
<#if SIX_STEP>
#include "speed_ctrl.h"
</#if><#-- SIX_STEP -->
#include "virtual_speed_sensor.h"
#include "ntc_temperature_sensor.h"
<#if FOC>
#include "revup_ctrl.h"
#include "pwm_curr_fdbk.h"
</#if><#-- FOC -->
<#if SIX_STEP>
#include "revup_ctrl_sixstep.h"
#include "pwm_common_sixstep.h"
  <#if MC.DRIVE_MODE == "CM">
#include "current_ref_ctrl.h"
  </#if><#-- MC.DRIVE_MODE == "CM" -->
</#if><#-- SIX_STEP -->
#include "mc_interface.h"
#include "mc_configuration_registers.h"
<#if (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true) 
  || (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true)
  || MC.M1_BUS_VOLTAGE_READING == true || MC.M2_BUS_VOLTAGE_READING == true
  ||  MC.M1_POTENTIOMETER_ENABLE == true || MC.M2_POTENTIOMETER_ENABLE >
#include "regular_conversion_manager.h"
</#if><#-- (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true) 
  || (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true)
  || MC.M1_BUS_VOLTAGE_READING == true || MC.M21_BUS_VOLTAGE_READING == true
  ||  MC.M1_POTENTIOMETER_ENABLE == true || MC.M2_POTENTIOMETER_ENABLE -->
<#if MC.M1_BUS_VOLTAGE_READING == true || MC.M2_BUS_VOLTAGE_READING == true>
#include "r_divider_bus_voltage_sensor.h"
</#if><#-- MC.M1_BUS_VOLTAGE_READING == true || MC.M2_BUS_VOLTAGE_READING == true -->
<#if MC.M1_BUS_VOLTAGE_READING == false || MC.M2_BUS_VOLTAGE_READING == false>
#include "virtual_bus_voltage_sensor.h"
</#if><#-- MC.M1_BUS_VOLTAGE_READING == false || MC.M2_BUS_VOLTAGE_READING == false -->
<#if FOC>
  <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true>
#include "feed_forward_ctrl.h"
  </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
  <#if MC.M1_FLUX_WEAKENING_ENABLING == true || MC.M2_FLUX_WEAKENING_ENABLING == true>
#include "flux_weakening_ctrl.h"
  </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING == true || MC.M2_FLUX_WEAKENING_ENABLING == true -->
  <#if MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true>
#include "trajectory_ctrl.h"
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true -->
#include "pqd_motor_power_measurement.h"
  <#if MC.USE_STGAP1S>
#include "gap_gate_driver_ctrl.h"
  </#if><#-- MC.USE_STGAP1S -->
</#if><#-- FOC -->

<#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" || (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true
  && MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES") || (MC.DRIVE_NUMBER != "1"
  && ((MC.M2_HW_OV_CURRENT_PROT_BYPASS == true && MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES")
  || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE"))>
#include "digital_output.h"
</#if><#-- Resistive break or OC protection bypass on OV -->
<#if MC.STSPIN32G4 == true >
#include "stspin32g4.h"
</#if><#-- MC.STSPIN32G4 == true -->
<#if MC.MOTOR_PROFILER == true>
#include "mp_one_touch_tuning.h"
#include "mp_self_com_ctrl.h"
  <#if  MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
#include "mp_hall_tuning.h"
  </#if><#-- MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR" -->
</#if><#-- MC.MOTOR_PROFILER == true -->

<#-- Specific to FOC algorithm usage -->
<#if FOC>
  <#-- Specific to F3 family usage -->
  <#if CondFamily_STM32F3 && (((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))
  || ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '1' )))>
#include "r3_1_f30x_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F3 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  <#if CondFamily_STM32F3 
  && (((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
  || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')))>
#include "r1_ps_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F3 && (((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || -->
  <#if CondFamily_STM32F3 && (((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')))>
#include "r3_2_f30x_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F3 && (((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))) -->
  <#if CondFamily_STM32F3 && ((MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'))>
#include "ics_f30x_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F3 && ((MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')) -->
  <#-- Specific to F4 family usage -->
  <#if CondFamily_STM32F4>
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    ||((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_1_f4xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
#include "r3_2_f4xx_pwm_curr_fdbk.h"    
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_f4xx_pwm_curr_fdbk.h"    
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  </#if><#-- CondFamily_STM32F4 -->
  <#-- Specific to G0 family usage -->
  <#if CondFamily_STM32G0>
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_g0xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_g0xx_pwm_curr_fdbk.h"
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
  </#if><#-- CondFamily_STM32G0 -->
  <#-- Specific to C0 family usage -->
  <#if CondFamily_STM32C0>
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#error C0 single shunt not supported yet
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_c0xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>

    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
  </#if><#-- CondFamily_STM32C0 -->
  <#-- Specific to L4 family usage -->
  <#if CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_1_l4xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  <#if CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))>
#include "r3_2_l4xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) -->
  <#if CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  <#if CondFamily_STM32L4 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_l4xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32L4 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  <#-- Specific to F7 family usage -->
  <#if CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_1_f7xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  <#if CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))>
#include "r3_2_f7xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) -->
  <#if CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  <#if CondFamily_STM32F7 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_f7xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F7 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  <#-- Specific to H7 family usage -->
  <#if CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_1_h7xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  <#if CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))>
#include "r3_2_h7xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) -->
  <#if CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_h7xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  <#if CondFamily_STM32H7 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_h7xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32H7 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  <#-- Specific to F0 family usage -->
  <#if CondFamily_STM32F0 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"      
  </#if><#-- CondFamily_STM32F0 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  <#if CondFamily_STM32F0 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_f0xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32F0 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->

  <#-- Specific to H5 family usage -->
  <#if CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))> 
#include "r1_ps_pwm_curr_fdbk.h"      
  </#if><#-- CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  <#if CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_1_h5xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  <#if CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))> 
#include "r3_2_h5xx_pwm_curr_fdbk.h"
  </#if><#-- CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )) -->
  
  <#-- Specific to G4 family usage -->
  <#if CondFamily_STM32G4>
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
#include "r3_2_g4xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
	<#if  ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')) || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1'))  >
#include "r3_1_g4xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_g4xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  </#if><#-- CondFamily_STM32G4 -->
</#if><#-- FOC -->

<#if SIX_STEP>
  <#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
#include "pwmc_6pwm.h"
  <#else><#-- !SIX_STEP -->
#include "pwmc_3pwm.h"
  </#if><#-- SIX_STEP -->
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
</#if><#-- SIX_STEP -->

<#if FOC>
  <#-- MTPA feature usage -->
  <#if MC.M1_MTPA_ENABLING == true || MC.M2_MTPA_ENABLING == true>
#include "max_torque_per_ampere.h"
  </#if><#-- MC.M1_MTPA_ENABLING == true || MC.M2_MTPA_ENABLING == true -->
  <#-- ICL feature usage -->
  <#if MC.M1_ICL_ENABLED == true || MC.M2_ICL_ENABLED == true>
#include "inrush_current_limiter.h"
  </#if><#-- MC.M1_ICL_ENABLED == true || MC.M2_ICL_ENABLED == true -->
  <#-- Open Loop feature usage -->
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true>
#include "open_loop.h"
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
</#if><#-- FOC -->

<#if FOC>
  <#-- Position sensors feature usage -->
  <#if M1_ENCODER || M2_ENCODER>
#include "encoder_speed_pos_fdbk.h"
#include "enc_align_ctrl.h"
  </#if><#-- M1_ENCODER || M2_ENCODER -->
</#if><#-- FOC -->

  <#if (M1_HALL_SENSOR == true) || (M2_HALL_SENSOR == true)>
#include "hall_speed_pos_fdbk.h"
  </#if><#-- (M1_HALL_SENSOR == true) || (M2_HALL_SENSOR == true) -->
<#if MC.M1_POTENTIOMETER_ENABLE == true || MC.M2_POTENTIOMETER_ENABLE == true >
#include "speed_potentiometer.h"
</#if><#-- MC.M1_POTENTIOMETER_ENABLE == true || MC.M2_POTENTIOMETER_ENABLE == true -->
<#if MC.ESC_ENABLE == true>
#include "esc.h"
</#if><#-- MC.ESC_ENABLE == true -->

<#if FOC>
#include "ramp_ext_mngr.h"
#include "circle_limitation.h"

  <#if (MC.M1_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || 
       (MC.M1_SPEED_SENSOR == "STO_CORDIC") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
#include "sto_speed_pos_fdbk.h"
  </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || 
       (MC.M1_SPEED_SENSOR == "STO_CORDIC") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->
  <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||
       (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_PLL")>
#include "sto_pll_speed_pos_fdbk.h"
  </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||
             (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_PLL") -->
  <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") ||
       (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")>
#include "sto_cordic_speed_pos_fdbk.h"
  </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") ||
             (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->
  <#-- PFC feature usage -->
  <#if MC.PFC_ENABLED == true>
#include "pfc.h"
  </#if><#-- MC.PFC_ENABLED == true -->
/* USER CODE BEGIN Additional include */

/* USER CODE END Additional include */ 
</#if><#-- FOC -->

<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
extern RevUpCtrl_Handle_t RevUpControlM1;
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") 
        || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->

extern PID_Handle_t PIDSpeedHandle_M1;
<#if FOC>
extern PID_Handle_t PIDIqHandle_M1;
extern PID_Handle_t PIDIdHandle_M1;
</#if><#-- FOC -->
<#if (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true)>
extern RegConv_t TempRegConv_M1;
</#if><#-- (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true) -->
extern NTC_Handle_t TempSensor_M1;
<#if FOC>
  <#-- Flux Weakening feature usage -->
  <#if MC.M1_FLUX_WEAKENING_ENABLING == true>
extern PID_Handle_t PIDFluxWeakeningHandle_M1;
extern FW_Handle_t FW_M1;
  </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING == true -->
  <#if MC.M1_POSITION_CTRL_ENABLING == true>
extern PID_Handle_t PID_PosParamsM1;
extern PosCtrl_Handle_t PosCtrlM1;
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true -->
  
  <#if (CondFamily_STM32H7 == false) && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern PWMC_R1_Handle_t PWM_Handle_M1;
  </#if><#-- (CondFamily_STM32H7 == false) && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') 
  || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    
  <#if CondFamily_STM32H7 == false && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
extern PWMC_R3_1_Handle_t PWM_Handle_M1;
  </#if><#-- CondFamily_STM32H7 == false && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  <#if (MC.DRIVE_NUMBER != "1") && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '1' ))>
extern PWMC_R3_1_Handle_t PWM_Handle_M2;
  </#if><#-- (MC.DRIVE_NUMBER != "1") && ((MC.M2_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M2_CS_ADC_NUM == '1' )) -->


  <#if (CondFamily_STM32F0 == false || CondFamily_STM32G0 == false || CondFamily_STM32C0 == false || CondFamily_STM32H7 == false)
       && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
extern PWMC_ICS_Handle_t PWM_Handle_M1;
  </#if><#-- (CondFamily_STM32F0 == false || CondFamily_STM32G0 == false || CondFamily_STM32C0 == false || CondFamily_STM32H7 == false)
          && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  <#if ((CondFamily_STM32F4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32F3 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
     || (CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
     || (CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
     || (CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
     || (CondFamily_STM32G4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))))>
extern PWMC_R3_2_Handle_t PWM_Handle_M1;
  </#if><#-- ((CondFamily_STM32F4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
           || (CondFamily_STM32F3 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
           || (CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))
           || (CondFamily_STM32F7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
           || (CondFamily_STM32H5 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
           || (CondFamily_STM32H7 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' ))) 
           || (CondFamily_STM32G4 && ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '2' )))) -->
  
  <#-- Specific to F4 family usage -->
  <#if CondFamily_STM32F4>
    <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern PWMC_R1_Handle_t PWM_Handle_M2;
    <#elseif MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
extern PWMC_ICS_Handle_t PWM_Handle_M2;
    </#if><#--  ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32F4 -->
   <#-- Specific to G4 family usage -->
  <#if CondFamily_STM32G4>
    <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern PWMC_R1_Handle_t PWM_Handle_M2;
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
extern PWMC_ICS_Handle_t PWM_Handle_M2;
    </#if><#-- MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
  </#if><#-- CondFamily_STM32G4 -->
 </#if><#-- FOC -->
<#if SIX_STEP>
  <#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
extern PWMC_SixPwm_Handle_t PWM_Handle_M1;
  <#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
extern PWMC_ThreePwm_Handle_t PWM_Handle_M1;
  </#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
  <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
extern Bemf_ADC_Handle_t Bemf_ADC_M1;
  </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
  <#if MC.DRIVE_MODE == "CM">
extern CurrentRef_Handle_t CurrentRef_M1;
  </#if><#-- MC.DRIVE_MODE == "CM" -->
</#if><#-- SIX_STEP -->
<#if FOC>
  <#-- Specific to Dual Drive feature usage -->
  <#if MC.DRIVE_NUMBER != "1">
extern SpeednTorqCtrl_Handle_t SpeednTorqCtrlM2;
extern PID_Handle_t PIDSpeedHandle_M2;
extern PID_Handle_t PIDIqHandle_M2;
extern PID_Handle_t PIDIdHandle_M2;
<#if (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true) >
extern RegConv_t TempRegConv_M2;
</#if><#--(MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true) -->
extern NTC_Handle_t TempSensor_M2;
    <#if MC.M2_FLUX_WEAKENING_ENABLING == true>
extern PID_Handle_t PIDFluxWeakeningHandle_M2;
extern FW_Handle_t FW_M2;
    </#if><#-- MC.M2_FLUX_WEAKENING_ENABLING == true -->
    <#if MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true>
extern FF_Handle_t FF_M2;
    </#if><#-- MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
    <#if MC.M2_POSITION_CTRL_ENABLING == true>
extern PID_Handle_t PID_PosParamsM2;
extern PosCtrl_Handle_t PosCtrlM2;
    </#if><#-- MC.M2_POSITION_CTRL_ENABLING == true -->
    <#if CondFamily_STM32F3 && ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
extern PWMC_R1_Handle_t PWM_Handle_M2;
    </#if><#-- CondFamily_STM32F3 && ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
extern PWMC_R3_2_Handle_t PWM_Handle_M2;
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if CondFamily_STM32F3 && (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
extern PWMC_ICS_Handle_t PWM_Handle_M2;
    </#if><#-- CondFamily_STM32F3 && (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->
extern SpeednTorqCtrl_Handle_t SpeednTorqCtrlM1;
<#if FOC>
extern PQD_MotorPowMeas_Handle_t PQD_MotorPowMeasM1;
extern PQD_MotorPowMeas_Handle_t *pPQD_MotorPowMeasM1; 
  <#if MC.USE_STGAP1S>
extern GAP_Handle_t STGAP_M1;
  </#if><#-- MC.USE_STGAP1S -->
  <#if MC.DRIVE_NUMBER != "1">
extern PQD_MotorPowMeas_Handle_t PQD_MotorPowMeasM2;
extern PQD_MotorPowMeas_Handle_t *pPQD_MotorPowMeasM2; 
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->
<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") 
  || MC.M1_VIEW_ENCODER_FEEDBACK == true || (MC.M1_SPEED_SENSOR == "SENSORLESS_ADC")>
extern VirtualSpeedSensor_Handle_t VirtualSpeedSensorM1;
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z")
        || MC.M1_VIEW_ENCODER_FEEDBACK == true || (MC.M1_SPEED_SENSOR == "SENSORLESS_ADC") -->
<#if FOC>
  <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")
    || MC.M2_VIEW_ENCODER_FEEDBACK == true>
extern VirtualSpeedSensor_Handle_t VirtualSpeedSensorM2;
  </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")
    || MC.M2_VIEW_ENCODER_FEEDBACK == true -->
  <#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")>
extern STO_Handle_t STO_M1;
  </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") -->
  <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")>
extern RevUpCtrl_Handle_t RevUpControlM2;
extern STO_Handle_t STO_M2;
  </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->
  <#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL")>
extern STO_PLL_Handle_t STO_PLL_M1;
  </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") -->
  <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL")>
extern STO_PLL_Handle_t STO_PLL_M2;
  </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") -->
  <#if (MC.M2_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
extern STO_CR_Handle_t STO_CR_M2;
  </#if><#-- (MC.M2_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->
  <#if (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
extern STO_CR_Handle_t STO_CR_M1;
  </#if><#-- (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->
  <#if M1_ENCODER>
extern ENCODER_Handle_t ENCODER_M1;
extern EncAlign_Handle_t EncAlignCtrlM1;
  </#if><#-- M1_ENCODER -->
  <#if M2_ENCODER>
extern ENCODER_Handle_t ENCODER_M2;
extern EncAlign_Handle_t EncAlignCtrlM2;
  </#if><#-- M2_ENCODER -->
  <#if M2_HALL_SENSOR == true>
extern HALL_Handle_t HALL_M2;
  </#if><#-- M2_HALL_SENSOR == true -->
</#if><#-- FOC -->
  <#if M1_HALL_SENSOR == true>
extern HALL_Handle_t HALL_M1;
  </#if><#-- M1_HALL_SENSOR == true -->
<#if FOC>
  <#if MC.M1_ICL_ENABLED == true>
extern ICL_Handle_t ICL_M1;
  </#if><#-- MC.M1_ICL_ENABLED == true -->
  <#if MC.M2_ICL_ENABLED == true && MC.DRIVE_NUMBER != "1">
extern ICL_Handle_t ICL_M2;
  </#if><#-- MC.M2_ICL_ENABLED == true && MC.DRIVE_NUMBER > 1 -->
</#if><#-- FOC -->
<#if MC.M1_BUS_VOLTAGE_READING == true>
extern RegConv_t VbusRegConv_M1;
extern RDivider_Handle_t BusVoltageSensor_M1;
<#else><#-- MC.M1_BUS_VOLTAGE_READING == false -->
extern VirtualBusVoltageSensor_Handle_t BusVoltageSensor_M1;
</#if><#-- MC.M1_BUS_VOLTAGE_READING == true -->
<#if MC.DRIVE_NUMBER != "1">
  <#if MC.M2_BUS_VOLTAGE_READING == true>
extern RegConv_t VbusRegConv_M2;
extern RDivider_Handle_t BusVoltageSensor_M2;
  <#else><#-- MC.M2_BUS_VOLTAGE_READING == false -->
extern VirtualBusVoltageSensor_Handle_t BusVoltageSensor_M2;
  </#if><#-- MC.M2_BUS_VOLTAGE_READING == true -->
</#if><#-- MC.DRIVE_NUMBER > 1 -->
<#if FOC>
extern CircleLimitation_Handle_t CircleLimitationM1;
  <#if MC.DRIVE_NUMBER != "1">
extern CircleLimitation_Handle_t CircleLimitationM2;
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
extern RampExtMngr_Handle_t RampExtMngrHFParamsM1;
  <#if MC.DRIVE_NUMBER != "1">
extern RampExtMngr_Handle_t RampExtMngrHFParamsM2;
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
  <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true>
extern FF_Handle_t FF_M1;
  </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
  <#if MC.M1_MTPA_ENABLING == true>
extern MTPA_Handle_t MTPARegM1;
  </#if><#-- MC.M1_MTPA_ENABLING == true -->
  <#if MC.DRIVE_NUMBER != "1" && MC.M2_MTPA_ENABLING == true>
extern MTPA_Handle_t MTPARegM2;
  </#if><#-- MC.DRIVE_NUMBER > 1 && MC.M2_MTPA_ENABLING == true -->
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
extern OpenLoop_Handle_t OpenLoop_ParamsM1;
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
  <#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
extern OpenLoop_Handle_t OpenLoop_ParamsM2;
  </#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
  <#if MC.DRIVE_NUMBER != "1">
    <#if MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
extern DOUT_handle_t R_BrakeParamsM2;
    </#if><#-- MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->
    <#if MC.M2_HW_OV_CURRENT_PROT_BYPASS == true && MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES">
extern DOUT_handle_t DOUT_OCPDisablingParamsM2;
    </#if><#-- MC.M2_HW_OV_CURRENT_PROT_BYPASS == true && MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" -->
    <#if MC.M2_ICL_ENABLED == true>
extern DOUT_handle_t ICLDOUTParamsM2;
    </#if><#-- MC.M2_ICL_ENABLED == true -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
  <#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
extern DOUT_handle_t R_BrakeParamsM1;
  </#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->
  <#if MC.M1_HW_OV_CURRENT_PROT_BYPASS == true && MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES">
extern DOUT_handle_t DOUT_OCPDisablingParamsM1;
  </#if><#-- MC.M1_HW_OV_CURRENT_PROT_BYPASS == true && MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" -->
  <#if MC.M1_ICL_ENABLED == true>
extern DOUT_handle_t ICLDOUTParamsM1;
  </#if><#-- MC.M1_ICL_ENABLED == true -->
</#if><#-- FOC -->




<#if FOC>
  <#-- PFC feature usage -->
  <#if MC.PFC_ENABLED == true>
extern PFC_Handle_t PFC;
  </#if><#-- MC.PFC_ENABLED == true -->
  
  <#-- Motor Profiler feature usage -->
  <#if MC.MOTOR_PROFILER == true>
extern RampExtMngr_Handle_t RampExtMngrParamsSCC;
extern RampExtMngr_Handle_t RampExtMngrParamsOTT;
extern SCC_Handle_t SCC;
extern OTT_Handle_t OTT;
    <#if MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
extern HT_Handle_t HT;  
    </#if><#-- MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR" -->
  </#if><#-- MC.MOTOR_PROFILER == true -->
</#if><#-- FOC -->
<#if MC.M1_POTENTIOMETER_ENABLE == true>
extern SpeedPotentiometer_Handle_t SpeedPotentiometer_M1;
extern RegConv_t PotRegConv_M1;
</#if><#-- MC.M1_POTENTIOMETER_ENABLE == true -->
<#if MC.DRIVE_NUMBER != "1">
<#if MC.M2_POTENTIOMETER_ENABLE == true>
extern SpeedPotentiometer_Handle_t SpeedPotentiometer_M2;
extern RegConv_t PotRegConv_M2;
</#if><#-- MC.M2_POTENTIOMETER_ENABLE == true -->
</#if><#-- MC.DRIVE_NUMBER > 1 -->
extern MCI_Handle_t Mci[NBR_OF_MOTORS];
extern SpeednTorqCtrl_Handle_t *pSTC[NBR_OF_MOTORS];
<#if FOC>
extern PID_Handle_t *pPIDIq[NBR_OF_MOTORS];
extern PID_Handle_t *pPIDId[NBR_OF_MOTORS];
</#if><#-- FOC -->
extern NTC_Handle_t *pTemperatureSensor[NBR_OF_MOTORS];
<#if FOC>
extern PQD_MotorPowMeas_Handle_t *pMPM[NBR_OF_MOTORS]; 
  <#if MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true>
extern PosCtrl_Handle_t *pPosCtrl[NBR_OF_MOTORS];
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true -->
  <#if MC.M1_FLUX_WEAKENING_ENABLING==true || MC.M2_FLUX_WEAKENING_ENABLING==true>
extern FW_Handle_t *pFW[NBR_OF_MOTORS];
  </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING==true || MC.M2_FLUX_WEAKENING_ENABLING==true -->
  <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true>
extern FF_Handle_t *pFF[NBR_OF_MOTORS];
  </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
</#if><#-- FOC -->
extern MCI_Handle_t* pMCI[NBR_OF_MOTORS];
<#if MC.STSPIN32G4 == true>
extern STSPIN32G4_HandleTypeDef HdlSTSPING4;
</#if><#-- MC.STSPIN32G4 == true -->
<#if MC.ESC_ENABLE == true>
extern ESC_Handle_t ESC_M1;
</#if><#-- MC.ESC_ENABLE == true -->
/* USER CODE BEGIN Additional extern */

/* USER CODE END Additional extern */  
 
  

#endif /* MC_CONFIG_H */
/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
