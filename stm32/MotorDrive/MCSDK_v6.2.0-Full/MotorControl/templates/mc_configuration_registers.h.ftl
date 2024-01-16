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

<#function GetSpeedSensor motor aux=false> 
      <#local SENSORS =
        [ {"name": "NULL", "val": "ENO_SENSOR"} 
        , {"name": "PLL", "val": "EPLL"} 
        , {"name": "CORDIC", "val": "ECORDIC"} 
        , {"name": "ENCODER", "val": "EENCODER"}
        , {"name": "HALL", "val": "EHALL"}
        , {"name": "HSO", "val": "EHSO"}        
        , {"name": "ZEST", "val": "EZEST"}
        ] >
      <#if motor == "M1" > 
        <#if !aux > <#assign entry = MC.M1_SPEED_SENSOR> <#else> 
        <#assign entry = MC.AUXILIARY_SPEED_MEASUREMENT_SELECTION > </#if>
      <#else>
        <#if !aux > <#assign entry = MC.M2_SPEED_SENSOR> <#else> 
        <#assign entry = MC.AUXILIARY_SPEED_MEASUREMENT_SELECTION2 > </#if>
      </#if>
      <#list SENSORS as sensor >
        <#if entry?contains(sensor.name) >
         <#return  sensor.val >
        </#if>
      </#list>
     <#return 0 >
</#function>

<#function GetTopology motor> 
      <#local TOPOLOGIES =
        [ {"name": "THREE_SHUNT", "val": 0} 
        , {"name": "SINGLE_SHUNT_PHASE_SHIFT", "val": 2}
        , {"name": "SINGLE_SHUNT_ACTIVE_WIN", "val": 1} 
        , {"name": "ICS_SENSORS", "val": 3}
        , {"name": "NONE", "val": 4}
        ] >
      <#if motor == "M1" > 
        <#assign entry = MC.M1_CURRENT_SENSING_TOPO > 
      <#else>
        <#assign entry = MC.M2_CURRENT_SENSING_TOPO >
      </#if>
      <#list TOPOLOGIES as topology >
        <#if entry?contains(topology.name) >
         <#return  topology.val >
        </#if>
      </#list>
     <#return 0 >
</#function>

<#function GetConfigurationFlag motor Flag>
  <#local result =""> 
  <#if Flag == 1 >
    <#if motor == "M1" >
      <#if MC.M1_FLUX_WEAKENING_ENABLING > <#local result+="|FLUX_WEAKENING_FLAG"> </#if>
      <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING > <#local result+="|FEED_FORWARD_FLAG"> </#if>
      <#if MC.M1_MTPA_ENABLING > <#local result+="|MTPA_FLAG"> </#if>
      <#if MC.PFC_ENABLED >   <#local result+="|PFC_FLAG">  </#if>
      <#if MC.M1_ICL_ENABLED >   <#local result+="|ICL_FLAG"> </#if>  
      <#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" >  <#local result+="|RESISTIVE_BREAK_FLAG"> </#if>  
      <#if MC.M1_HW_OV_CURRENT_PROT_BYPASS >  <#local result+="|OCP_DISABLE_FLAG"> </#if>  
      <#if MC.USE_STGAP1S >  <#local result+="|STGAP_FLAG"> </#if>  
      <#if MC.M1_POSITION_CTRL_ENABLING >  <#local result+="|POSITION_CTRL_FLAG"> </#if>  
      <#if MC.M1_BUS_VOLTAGE_READING >  <#local result+="|VBUS_SENSING_FLAG"> </#if>  
      <#if MC.M1_TEMPERATURE_READING >  <#local result+="|TEMP_SENSING_FLAG"> </#if>  
    /* Voltage sensing to be defined in data model*/
    /* Flash Config to be defined in data model*/
      <#if MC.DEBUG_DAC_CH1_EN >  <#local result+="|DAC_CH1_FLAG"> </#if>  
      <#if MC.DEBUG_DAC_CH2_EN >  <#local result+="|DAC_CH2_FLAG"> </#if>
      <#if MC.M1_OTF_STARTUP >  <#local result+="|OTF_STARTUP_FLAG"> </#if>    
    <#else> 
      <#if MC.M2_FLUX_WEAKENING_ENABLING > <#local result+="|FLUX_WEAKENING_FLAG"> </#if>
      <#if MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING > <#local result+="|FEED_FORWARD_FLAG"> </#if>
      <#if MC.M2_MTPA_ENABLING > <#local result+="|MTPA_FLAG"> </#if>
      <#if MC.PFC_ENABLED >   <#local result+="|PFC_FLAG">  </#if>
      <#if MC.M2_ICL_ENABLED >   <#local result+="|ICL_FLAG"> </#if>  
      <#if MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" >  <#local result+="|RESISTIVE_BREAK_FLAG"> </#if>  
      <#if MC.M2_HW_OV_CURRENT_PROT_BYPASS >  <#local result+="|OCP_DISABLE_FLAG"> </#if>  
      <#if MC.USE_STGAP1S >  <#local result+="|STGAP_FLAG"> </#if>  
      <#if MC.M2_POSITION_CTRL_ENABLING >  <#local result+="|POSITION_CTRL_FLAG"> </#if>  
      <#if MC.M2_BUS_VOLTAGE_READING >  <#local result+="|VBUS_SENSING_FLAG"> </#if>  
      <#if MC.M2_TEMPERATURE_READING >  <#local result+="|TEMP_SENSING_FLAG"> </#if>  
      <#if MC.DEBUG_DAC_CH1_EN >  <#local result+="|DAC_CH1_FLAG"> </#if>  
      <#if MC.DEBUG_DAC_CH2_EN >  <#local result+="|DAC_CH2_FLAG"> </#if>
      <#if MC.M2_OTF_STARTUP >  <#local result+="|OTF_STARTUP_FLAG"> </#if> 
    </#if>
  <#elseif Flag == 2 >
    <#if motor == "M1" >  
      <#if MC.M1_OVERMODULATION >  <#local result+="|OVERMODULATION_FLAG"> </#if> 
      <#if MC.M1_DISCONTINUOUS_PWM > <#local result+="|DISCONTINUOUS_PWM_FLAG"> </#if>
      <#if MC.MOTOR_PROFILER==true> <#local result+="|PROFILER_FLAG"> </#if>
      <#if MC.DBG_MCU_LOAD_MEASURE > <#local result+="|DBG_MCU_LOAD_MEASURE_FLAG"> </#if>
      <#if MC.M1_DBG_OPEN_LOOP_ENABLE >   <#local result+="|DBG_OPEN_LOOP_FLAG">  </#if>    
    <#else> 
      <#if MC.M2_OVERMODULATION >  <#local result+="|OVERMODULATION_FLAG"> </#if> 
      <#if MC.M2_DISCONTINUOUS_PWM > <#local result+="|DISCONTINUOUS_PWM_FLAG"> </#if>
      <#if MC.DBG_MCU_LOAD_MEASURE > <#local result+="|DBG_MCU_LOAD_MEASURE_FLAG"> </#if>
      <#if MC.M2_DBG_OPEN_LOOP_ENABLE >   <#local result+="|DBG_OPEN_LOOP_FLAG">  </#if> 
    </#if>
  </#if> <#-- Flag number -->
  <#if result =="">
    <#return "0U" >
  <#else>
    <#return result[1..] >
   </#if>
</#function>
/**
  ******************************************************************************
  * @file    mc_configuration_registers.h
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file provides the definitions needed to build the project 
  *          configuration information registers.
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

#ifndef MC_CONFIGURATION_REGISTERS_H
#define MC_CONFIGURATION_REGISTERS_H

#include "mc_type.h"

typedef struct
{
  uint32_t SDKVersion;
  uint8_t MotorNumber;
  uint8_t MCP_Flag;
  uint8_t MCPA_UARTA_LOG;
  uint8_t MCPA_UARTB_LOG;
  uint8_t MCPA_STLNK_LOG;
  uint8_t Padding;
} __attribute__ ((packed)) GlobalConfig_reg_t;

typedef struct
{
    float_t polePairs;
    float_t ratedFlux;
    float_t rs;
    float_t rsSkinFactor;
    float_t ls;
    float_t ld; 
    float_t maxCurrent;
    float_t mass_copper_kg;
    float_t cooling_tau_s;
    char_t name[24];
} __attribute__ ((packed)) MotorConfig_reg_t;

typedef  struct
{
  uint32_t maxMechanicalSpeed;
  float_t maxReadableCurrent;
  float_t nominalCurrent;
  uint16_t nominalVoltage;
  uint8_t driveType;
  uint8_t padding;
} __attribute__ ((packed)) ApplicationConfig_reg_t;

<#if FOC  | ACIM>
typedef struct
{
  uint8_t primarySensor;
  uint8_t auxiliarySensor;
  uint8_t topology;
  uint8_t FOCRate;
  uint32_t PWMFrequency;
  uint16_t MediumFrequency;
  uint16_t configurationFlag1;
  uint16_t configurationFlag2;
} __attribute__ ((packed)) FOCFwConfig_reg_t;
</#if><#-- FOC  | ACIM -->
<#if SIX_STEP>
typedef struct
{
  uint8_t primarySensor;
  uint8_t topology;
  uint32_t PWMFrequency;
  uint16_t MediumFrequency;
  uint16_t configurationFlag1;
  uint8_t driveMode;  
} __attribute__ ((packed)) SixStepFwConfig_reg_t;
</#if><#-- SIX_STEP -->

#define ENO_SENSOR                0
#define EPLL                      1
#define ECORDIC                   2
#define EENCODER                  3
#define EHALL                     4
#define EHSO                      5
#define EZEST                     6

#define SDK_VERSION_MAIN   (0x6) /*!< [31:24] main version */
#define SDK_VERSION_SUB1   (0x2) /*!< [23:16] sub1 version */
#define SDK_VERSION_SUB2   (0x0) /*!< [15:8]  sub2 version */
#define SDK_VERSION_RC     (0x0) /*!< [7:0]  release candidate */
#define SDK_VERSION               ((SDK_VERSION_MAIN << 24U)\
                                  |(SDK_VERSION_SUB1 << 16U)\
                                  |(SDK_VERSION_SUB2 << 8U )\
                                  |(SDK_VERSION_RC))
/* configurationFlag1 definition */
#define FLUX_WEAKENING_FLAG       (1U)
#define FEED_FORWARD_FLAG         (1U << 1U)
#define MTPA_FLAG                 (1U << 2U)
#define PFC_FLAG                  (1U << 3U)
#define ICL_FLAG                  (1U << 4U)
#define RESISTIVE_BREAK_FLAG      (1U << 5U)
#define OCP_DISABLE_FLAG          (1U << 6U)
#define STGAP_FLAG                (1U << 7U)
#define POSITION_CTRL_FLAG        (1U << 8U)
#define VBUS_SENSING_FLAG         (1U << 9U)
#define TEMP_SENSING_FLAG         (1U << 10U)
#define VOLTAGE_SENSING_FLAG      (1U << 11U)
#define FLASH_CONFIG_FLAG         (1U << 12U)
#define DAC_CH1_FLAG              (1U << 13U)
#define DAC_CH2_FLAG              (1U << 14U)
#define OTF_STARTUP_FLAG          (1U << 15U)

/* configurationFlag2 definition */
#define OVERMODULATION_FLAG       (1U)
#define DISCONTINUOUS_PWM_FLAG    (1U << 1U)
#define PROFILER_FLAG             (1U << 13U)
#define DBG_MCU_LOAD_MEASURE_FLAG (1U << 14U)
#define DBG_OPEN_LOOP_FLAG        (1U << 15U)

/* MCP_Flag definition */
#define FLAG_MCP_OVER_STLINK      <#if MC.MCP_OVER_STLNK_EN > 1U <#else> 0U </#if>
#define FLAG_MCP_OVER_UARTA       <#if MC.MCP_OVER_UART_A_EN > (1U << 1U) <#else> 0U </#if>
#define FLAG_MCP_OVER_UARTB       <#if MC.MCP_OVER_UART_B_EN > (1U << 2U) <#else> 0U </#if>


#define configurationFlag1_M1     (${GetConfigurationFlag ("M1",1)})
#define configurationFlag2_M1     (${GetConfigurationFlag ("M1",2)})

#define DRIVE_TYPE_M1             <#if FOC > 0 <#elseif SIX_STEP> 1 
                                  <#else> 2 </#if>
#define PRIM_SENSOR_M1            ${GetSpeedSensor ("M1",false)}
#define AUX_SENSOR_M1             ${GetSpeedSensor ("M1",true)}
#define TOPOLOGY_M1               ${GetTopology ("M1")}
#define FOC_RATE_M1               ${MC.M1_REGULATION_EXECUTION_RATE}
#define PWM_FREQ_M1               ${MC.M1_PWM_FREQUENCY}
 
<#if MC.DRIVE_NUMBER != "1">
#define configurationFlag1_M2     (${GetConfigurationFlag ("M2",1)})
#define configurationFlag2_M2     (${GetConfigurationFlag ("M2",2)})

#define DRIVE_TYPE_M2             <#if FOC > 0 <#elseif SIX_STEP> 1 
                                  <#else> 2 </#if>
#define PRIM_SENSOR_M2            ${GetSpeedSensor ("M2",false)}
#define AUX_SENSOR_M2             ${GetSpeedSensor ("M2",true)}
#define TOPOLOGY_M2               ${GetTopology ("M2")}
#define FOC_RATE_M2               ${MC.M2_REGULATION_EXECUTION_RATE}
#define PWM_FREQ_M2               ${MC.M2_PWM_FREQUENCY}
</#if><#-- MC.DRIVE_NUMBER > 1 -->

extern const char_t FIRMWARE_NAME[]; //cstat !MISRAC2012-Rule-18.8 !MISRAC2012-Rule-8.11
extern const char_t CTL_BOARD[]; //cstat !MISRAC2012-Rule-18.8 !MISRAC2012-Rule-8.11
extern const char_t *PWR_BOARD_NAME[NBR_OF_MOTORS];
extern const char_t *MOTOR_NAME[NBR_OF_MOTORS];
extern const GlobalConfig_reg_t globalConfig_reg;
<#if FOC ||ACIM>
extern const FOCFwConfig_reg_t *FOCConfig_reg[NBR_OF_MOTORS];
</#if><#-- FOC ||ACIM -->
<#if SIX_STEP>
extern const SixStepFwConfig_reg_t *SixStepConfig_reg[NBR_OF_MOTORS];
</#if><#-- SIX_STEP -->
extern const MotorConfig_reg_t *MotorConfig_reg[NBR_OF_MOTORS];
extern const ApplicationConfig_reg_t *ApplicationConfig_reg[NBR_OF_MOTORS];

#endif /* MC_CONFIGURATION_REGISTERS_H */
/************************ (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
