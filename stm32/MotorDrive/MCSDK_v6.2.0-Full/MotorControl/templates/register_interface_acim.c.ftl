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
<#assign AUX_SPEED_FDBK_M1 = ((MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC"))>
<#assign AUX_SPEED_FDBK_M2 = ((MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC"))>
<#assign M1_IS_STO_CORDIC = false>
<#assign M2_IS_STO_CORDIC = false >
<#assign M1_IS_STO_PLL = false >
<#assign M2_IS_STO_PLL = false >
<#assign M1_IS_SENSORLESS = false >
<#assign M2_IS_SENSORLESS = false >
<#assign M1_IS_ENCODER = (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M2_IS_ENCODER = (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") ||  (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M1_IS_HALL_SENSOR = (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") >
<#assign M2_IS_HALL_SENSOR = (MC.M2_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") >

<#assign CondFamily_STM32F0 = (FamilyName?? && FamilyName=="STM32F0")>
<#-- Condition for STM32G0 Family -->
<#assign CondFamily_STM32G0 = (FamilyName?? && FamilyName=="STM32G0") >
<#assign CondFamily_STM32C0 = (FamilyName?? && FamilyName=="STM32C0") >

<#assign UNALIGNMENT_SUPPORTED = !(CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0) >

<#function GetSpeedSensor motor aux=false> 
      <#local SENSORS =
        [ {"name": "PLL", "variable": "&STO_PLL"} 
        , {"name": "CORDIC", "variable": "&STO_CR"} 
        , {"name": "HALL", "variable": "&HALL"}
        , {"name": "ENCODER", "variable": "&ENCODER"}
        ] >
      <#if motor == "M1" > 
        <#if !aux > <#assign entry = MC.M1_SPEED_SENSOR> <#else> <#assign entry = MC.AUXILIARY_SPEED_MEASUREMENT_SELECTION > </#if>
      <#else>
        <#if !aux > <#assign entry = MC.M2_SPEED_SENSOR> <#else> <#assign entry = MC.AUXILIARY_SPEED_MEASUREMENT_SELECTION2 > </#if>
      </#if>
      <#list SENSORS as sensor >
        <#if entry?contains(sensor.name) >
         <#return  sensor.variable+"_"+motor >
        </#if>
      </#list>
     <#return "MC_NULL" >
</#function>
/**
  ******************************************************************************
  * @file    register_interface.c
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file provides firmware functions that implement the register access for the MCP protocol
  *
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
  
#include "mc_type.h"
#include "string.h"
#include "register_interface.h"
#include "mc_config.h"
<#if MC.MCP_EN>
#include "mcp.h"
#include "mcp_config.h"
<#if MC.MCP_ASYNC_EN >
#include "mcpa.h"
  </#if>
</#if>
<#if MC.DEBUG_DAC_FUNCTIONALITY_EN>
#include "dac_ui.h"
</#if>
#include "mc_configuration_registers.h"

BusVoltageSensor_Handle_t* BusVoltageSensor[NBR_OF_MOTORS]={ &BusVoltageSensor_M1._Super};
<#if M1_IS_SENSORLESS >
RevUpCtrl_Handle_t *RevUpControl[NBR_OF_MOTORS] = { &RevUpControlM1 };
</#if>

<#if  M1_IS_STO_CORDIC>
STO_CR_Handle_t * stoCRSensor [NBR_OF_MOTORS] = { &STO_CR_M1 };
</#if>
<#if  M1_IS_STO_PLL>  
STO_PLL_Handle_t * stoPLLSensor [NBR_OF_MOTORS] = { &STO_PLL_M1 };
</#if>
<#if MC.ACIM_CONFIG == "LSO_FOC">
PID_Handle_t *pPIDSpeed[NBR_OF_MOTORS] = { &PIDSpeedHandle_M1 };
</#if>
<#if MC.M1_POSITION_CTRL_ENABLING>      
PID_Handle_t *pPIDPosCtrl[NBR_OF_MOTORS] = { &PID_PosParamsM1 };
</#if>
<#if MC.M1_FLUX_WEAKENING_ENABLING==true > 
PID_Handle_t *pPIDFW[NBR_OF_MOTORS] = { &PIDFluxWeakeningHandle_M1};
</#if>
<#if M1_IS_ENCODER >
ENCODER_Handle_t *pEncoder[NBR_OF_MOTORS] = {&ENCODER_M1};
</#if>
<#if M1_IS_HALL_SENSOR >
HALL_Handle_t *pHallSensor[NBR_OF_MOTORS] = {&HALL_M1};
</#if>


__weak uint8_t RI_SetRegCommandParser (MCP_Handle_t * pHandle, uint16_t txSyncFreeSpace)
{
  uint16_t * dataElementID;
  uint8_t * rxData = pHandle->rxBuffer;
  uint8_t * txData = pHandle->txBuffer;
  int16_t rxLength = pHandle->rxLength;
  uint16_t size;
  uint8_t retVal=MCP_CMD_OK;
  uint8_t accessResult;
  uint8_t number_of_item =0;
  pHandle->txLength = 0;
  while (rxLength > 0)
  {
     number_of_item ++;
     dataElementID = (uint16_t *) rxData;
     rxLength = rxLength-MCP_ID_SIZE; // We consume 2 byte in the DataID
     rxData = rxData+MCP_ID_SIZE; // Shift buffer to the next data
     accessResult = RI_SetReg (*dataElementID,rxData,&size,rxLength);
     
     /* Prepare next data*/
     rxLength = (int16_t) (rxLength - size);
     rxData = rxData+size;
     /* If there is only one CMD in the buffer, we do not store the result */
     if (number_of_item == 1 && rxLength == 0)
     {
       retVal = accessResult;
     }
     else
     {/* Store the result for each access to be able to report failling access */
       if (txSyncFreeSpace !=0) 
       {
         *txData = accessResult;
         txData = txData+1;
         pHandle->txLength++;
         txSyncFreeSpace--; /* decrement one by one no wraparound possible */
         retVal = (accessResult != MCP_CMD_OK) ? MCP_CMD_NOK : retVal;
         if ((accessResult == MCP_ERROR_BAD_DATA_TYPE) || (accessResult == MCP_ERROR_BAD_RAW_FORMAT))
         { /* From this point we are not able to continue to decode CMD buffer*/
           /* We stop the parsing */
           rxLength = 0;
         }
       }
       else
       {
         /* Stop parsing the cmd buffer as no space to answer */
         /* If we reach this state, chances are high the command was badly formated or received */
         rxLength = 0; 
         retVal = MCP_ERROR_NO_TXSYNC_SPACE;
       }
     }     
  }
  /* If all accesses are fine, just one global MCP_CMD_OK is required*/  
  if (retVal == MCP_CMD_OK)
  {
    pHandle->txLength = 0;
  }
  return retVal;
}

__weak uint8_t RI_GetRegCommandParser (MCP_Handle_t * pHandle, uint16_t txSyncFreeSpace)
{
  uint16_t * dataElementID;
  uint8_t * rxData = pHandle->rxBuffer;
  uint8_t * txData = pHandle->txBuffer;
  uint16_t size = 0;
  uint16_t rxLength = pHandle->rxLength;
  int16_t freeSpaceS16 = (int16_t) txSyncFreeSpace;
  uint8_t retVal = MCP_CMD_NOK;
  pHandle->txLength = 0;
  while (rxLength > 0)
  {
     dataElementID = (uint16_t *) rxData;
     rxLength = rxLength-MCP_ID_SIZE; 
     rxData = rxData+MCP_ID_SIZE; // Shift buffer to the next MCP_ID 
     retVal = RI_GetReg (*dataElementID,txData, &size, freeSpaceS16); 
     if (retVal == MCP_CMD_OK )
     {
       txData = txData+size;
       pHandle->txLength += size;
       freeSpaceS16 = freeSpaceS16-size;
     }
     else 
     { 
       rxLength = 0;
     }
  }
  return retVal;
}

uint8_t RI_SetReg (uint16_t dataID, uint8_t * data, uint16_t *size, int16_t dataAvailable)
{
  uint8_t typeID;
  uint8_t motorID;
  uint8_t retVal = MCP_CMD_OK;
  uint16_t regID = dataID & REG_MASK;
  typeID = dataID & TYPE_MASK;
  motorID = (dataID & MOTOR_MASK)-1;
  
<#if MC.M1_DRIVE_TYPE == "ACIM" ><#-- Specific to FOC algorithm usage -->
  MCI_Handle_t * pMCI = &Mci[motorID];
<#elseif  MC.M1_DRIVE_TYPE == "SIX_STEP" ><#-- Specific to 6_STEP algorithm usage -->
  MC_Handle_t * pMCI = MC_Core_GetMotorControlHandle( motorID );
</#if><#-- MC.M1_DRIVE_TYPE -->
  
  switch (typeID)
  {
    case TYPE_DATA_8BIT:
    {
      uint8_t regdata8 = *data;
    
      switch (regID)
      {
        case MC_REG_STATUS:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }
        
<#if MC.M1_DRIVE_TYPE == "ACIM">
        case MC_REG_CONTROL_MODE:
        {
          if ((MC_ControlMode_t)regdata8 == MCM_TORQUE_MODE)
          {
            MCI_ExecTorqueRamp(pMCI, MCI_GetTeref(pMCI),0);
          }
          else
          {
            /* Nothing to do */
          }
          if ((MC_ControlMode_t)regdata8 == MCM_SPEED_MODE)
          {
            MCI_ExecSpeedRamp(pMCI, MCI_GetMecSpeedRefUnit(pMCI),0);
          }
          else
          {
            /* Nothinf to do */
          }
          break;
        }

  <#if M1_IS_SENSORLESS || M2_IS_SENSORLESS >      
        case MC_REG_RUC_STAGE_NBR:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }
  </#if>  
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->
<#if MC.PFC_ENABLED>

        case MC_REG_PFC_STATUS:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }
       
        case MC_REG_PFC_ENABLED:
          break;
</#if><#-- MC.PFC_ENABLED -->

<#if MC.MOTOR_PROFILER>      
        case MC_REG_SC_CHECK:
        case MC_REG_SC_STATE:
        case MC_REG_SC_STEPS:
        case MC_REG_SC_PP:
        case MC_REG_SC_FOC_REP_RATE:
        case MC_REG_SC_COMPLETED:
          break;
</#if><#-- MC.MOTOR_PROFILER -->

<#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING>  
        case MC_REG_POSITION_CTRL_STATE:
        case MC_REG_POSITION_ALIGN_STATE:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }
</#if><#-- MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING -->

        default:
        {
          retVal = MCP_ERROR_UNKNOWN_REG;
          break;
        }
      }
      *size = 1;
      break;
    }
    
    case TYPE_DATA_16BIT:
    {
      uint16_t regdata16 = *(uint16_t *)data;
      switch (regID) 
      {
<#if MC.ACIM_CONFIG == "LSO_FOC">
        case MC_REG_SPEED_KP:
        {
          PID_SetKP(pPIDSpeed[motorID], regdata16);
          break;
        }

        case MC_REG_SPEED_KI:
        {
          PID_SetKI(pPIDSpeed[motorID], regdata16);
          break;
        }

        case MC_REG_SPEED_KD:
        {
          PID_SetKD(pPIDSpeed[motorID], regdata16);
          break;
        }
</#if><#-- MC.ACIM_CONFIG == "LSO_FOC" -->

<#if MC.M1_DRIVE_TYPE == "ACIM">
  <#if MC.ACIM_CONFIG == "LSO_FOC">
        case MC_REG_I_Q_KP:
        {
          PID_SetKP(pPIDIq[motorID], regdata16);
          break;
        }

        case MC_REG_I_Q_KI:
        {
          PID_SetKI(pPIDIq[motorID], regdata16);
          break;
        }

        case MC_REG_I_Q_KD:
        {
          PID_SetKD(pPIDIq[motorID], regdata16);
          break;
        }

        case MC_REG_I_D_KP:
        {
          PID_SetKP(pPIDId[motorID], regdata16);
          break;
        }

        case MC_REG_I_D_KI:
        {
          PID_SetKI(pPIDId[motorID], regdata16);
          break;
        }

        case MC_REG_I_D_KD:
        {
          PID_SetKD(pPIDId[motorID], regdata16);
          break;
        }
  </#if>

  <#if MC.M1_FLUX_WEAKENING_ENABLING==true || MC.M2_FLUX_WEAKENING_ENABLING==true>     
        case MC_REG_FLUXWK_KP:
        {
          PID_SetKP(pPIDFW[motorID], regdata16);
          break;
        }

        case MC_REG_FLUXWK_KI:
        {
          PID_SetKI(pPIDFW[motorID], regdata16);
          break;
        }

        case MC_REG_FLUXWK_BUS:
        {
          FW_SetVref(pFW[motorID], regdata16);
          break;
        }
  </#if> <#-- M1_FLUX_WEAKENING_ENABLING -->     

        case MC_REG_BUS_VOLTAGE:
        case MC_REG_HEATS_TEMP:
        case MC_REG_MOTOR_POWER:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }
</#if><#-- MC.M1_DRIVE_TYPE -->

<#if MC.DEBUG_DAC_FUNCTIONALITY_EN>     
        case MC_REG_DAC_OUT1:
        {
          DAC_SetChannelConfig(&DAC_Handle , DAC_CH1, regdata16 );
          break;
        }

        case MC_REG_DAC_OUT2:
        {
          DAC_SetChannelConfig(&DAC_Handle , DAC_CH2, regdata16 );
          break;
        }
</#if><#-- MC.DEBUG_DAC_FUNCTIONALITY_EN -->

        case MC_REG_FLUXWK_BUS_MEAS:    
        case MC_REG_I_A:
        case MC_REG_I_B:
        case MC_REG_I_ALPHA_MEAS:
        case MC_REG_I_BETA_MEAS:
        case MC_REG_I_Q_MEAS:
        case MC_REG_I_D_MEAS:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }

        case MC_REG_I_Q_REF:
        {
          qd_t currComp;
          currComp = MCI_GetIqdref(pMCI);
          currComp.q = (int16_t)regdata16;
          MCI_SetCurrentReferences(pMCI,currComp);
          break;
        }

        case MC_REG_I_D_REF:
        {
          qd_t currComp;
          currComp = MCI_GetIqdref(pMCI);
          currComp.d = regdata16;
          MCI_SetCurrentReferences(pMCI,currComp);
          break;
        }

        case MC_REG_V_Q:
        case MC_REG_V_D:
        case MC_REG_V_ALPHA:
        case MC_REG_V_BETA:
        case MC_REG_ENCODER_EL_ANGLE:
        case MC_REG_ENCODER_SPEED:
        case MC_REG_HALL_EL_ANGLE:
        case MC_REG_HALL_SPEED:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }
        
<#if M1_IS_STO_PLL || M2_IS_STO_PLL>
        case MC_REG_STOPLL_C1:
        {
          int16_t hC1,hC2;
          STO_PLL_GetObserverGains(stoPLLSensor[motorID],&hC1,&hC2);
          STO_PLL_SetObserverGains(stoPLLSensor[motorID],regdata16,hC2);
          break;
        }

        case MC_REG_STOPLL_C2:
        {
          int16_t hC1,hC2;
          STO_PLL_GetObserverGains(stoPLLSensor[motorID],&hC1,&hC2);
          STO_PLL_SetObserverGains(stoPLLSensor[motorID],hC1,regdata16);
          break;
        }

        case MC_REG_STOPLL_KI:
        {
          PID_SetKI (&stoPLLSensor[motorID]->PIRegulator,regdata16);
          break;
        }

        case MC_REG_STOPLL_KP:
        {
          PID_SetKP (&stoPLLSensor[motorID]->PIRegulator,regdata16);
          break;
        }
 
        case MC_REG_STOPLL_EL_ANGLE:
        case MC_REG_STOPLL_ROT_SPEED:
        case MC_REG_STOPLL_I_ALPHA:
        case MC_REG_STOPLL_I_BETA:
        case MC_REG_STOPLL_BEMF_ALPHA:
        case MC_REG_STOPLL_BEMF_BETA:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }
</#if><#-- M1_IS_STO_PLL || M2_IS_STO_PLL -->

<#if M1_IS_STO_CORDIC || M2_IS_STO_CORDIC>
        case MC_REG_STOCORDIC_EL_ANGLE:  
        case MC_REG_STOCORDIC_ROT_SPEED:
        case MC_REG_STOCORDIC_I_ALPHA:
        case MC_REG_STOCORDIC_I_BETA:
        case MC_REG_STOCORDIC_BEMF_ALPHA:
        case MC_REG_STOCORDIC_BEMF_BETA:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }

        case MC_REG_STOCORDIC_C1:
        {
          int16_t hC1,hC2;
          STO_CR_GetObserverGains(stoCRSensor[motorID],&hC1,&hC2);
          STO_CR_SetObserverGains(stoCRSensor[motorID],regdata16,hC2);
          break;
        }

        case MC_REG_STOCORDIC_C2:
        {
          int16_t hC1,hC2;
          STO_CR_GetObserverGains(stoCRSensor[motorID],&hC1,&hC2);
          STO_CR_SetObserverGains(stoCRSensor[motorID],hC1,regdata16);
          break;
        }

</#if><#-- M1_IS_STO_CORDIC || M2_IS_STO_CORDIC -->
        case MC_REG_DAC_USER1:
        case MC_REG_DAC_USER2:
          break;
          
<#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING>
        case MC_REG_FF_VQ:
        case MC_REG_FF_VD:
        case MC_REG_FF_VQ_PIOUT:
        case MC_REG_FF_VD_PIOUT:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }
</#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING -->

<#if MC.PFC_ENABLED>
        case MC_REG_PFC_DCBUS_REF:
        case MC_REG_PFC_DCBUS_MEAS:
        case MC_REG_PFC_ACBUS_FREQ:
        case MC_REG_PFC_ACBUS_RMS:
        case MC_REG_PFC_I_KP:
        case MC_REG_PFC_I_KI:
        case MC_REG_PFC_I_KD:
        case MC_REG_PFC_V_KP:
        case MC_REG_PFC_V_KI:
        case MC_REG_PFC_V_KD:
        case MC_REG_PFC_STARTUP_DURATION:
         break;
</#if><#-- MC.PFC_ENABLED -->

<#if MC.MOTOR_PROFILER>
        case MC_REG_SC_PWM_FREQUENCY:
          break;
</#if><#-- MC.MOTOR_PROFILER -->

<#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING>
        case MC_REG_POSITION_KP:
        {
          PID_SetKP(pPIDPosCtrl[motorID], regdata16);  
          break;
        }

        case MC_REG_POSITION_KI:
        {
          PID_SetKI(pPIDPosCtrl[motorID], regdata16);  
          break;
        }

        case MC_REG_POSITION_KD:
        {
          PID_SetKD(pPIDPosCtrl[motorID], regdata16);  
          break;
        }
</#if><#-- MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING -->

<#if MC.ACIM_CONFIG == "LSO_FOC">
        case MC_REG_SPEED_KP_DIV:
        {
          PID_SetKPDivisorPOW2(pPIDSpeed[motorID], regdata16);
          break;
        }

        case MC_REG_SPEED_KI_DIV:
        {
          PID_SetKIDivisorPOW2(pPIDSpeed[motorID], regdata16);
          break;
        }

        case MC_REG_SPEED_KD_DIV:
        {
          PID_SetKDDivisorPOW2(pPIDSpeed[motorID], regdata16);
          break;
        }

        case MC_REG_I_D_KP_DIV:
        {
          PID_SetKPDivisorPOW2(pPIDId[motorID], regdata16);
          break;
        }

        case MC_REG_I_D_KI_DIV:
        {
          PID_SetKIDivisorPOW2(pPIDId[motorID], regdata16);
          break;
        }

        case MC_REG_I_D_KD_DIV:
        {
          PID_SetKDDivisorPOW2(pPIDId[motorID], regdata16);
          break;
        }

        case MC_REG_I_Q_KP_DIV:
        {
          PID_SetKPDivisorPOW2(pPIDIq[motorID], regdata16);
          break;
        }

        case MC_REG_I_Q_KI_DIV:
        {
          PID_SetKIDivisorPOW2(pPIDIq[motorID], regdata16);
          break;
        }

        case MC_REG_I_Q_KD_DIV:
        {
          PID_SetKDDivisorPOW2(pPIDIq[motorID], regdata16);
          break;
        }
</#if><#-- MC.ACIM_CONFIG == "LSO_FOC" -->

<#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING>
        case MC_REG_POSITION_KP_DIV:
        {
          PID_SetKPDivisorPOW2(pPIDPosCtrl[motorID], regdata16);
          break;
        }

        case MC_REG_POSITION_KI_DIV:
        {
          PID_SetKIDivisorPOW2(pPIDPosCtrl[motorID], regdata16);
          break;
        }

        case MC_REG_POSITION_KD_DIV:
        {
          PID_SetKDDivisorPOW2(pPIDPosCtrl[motorID], regdata16);
          break;
        }
</#if><#-- MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING -->

<#if MC.PFC_ENABLED>
        case MC_REG_PFC_I_KP_DIV:   
        case MC_REG_PFC_I_KI_DIV:   
        case MC_REG_PFC_I_KD_DIV:   
        case MC_REG_PFC_V_KP_DIV:   
        case MC_REG_PFC_V_KI_DIV:   
        case MC_REG_PFC_V_KD_DIV:
</#if><#-- MC.PFC_ENABLED -->

<#if M1_IS_STO_PLL || M2_IS_STO_PLL>
        case MC_REG_STOPLL_KI_DIV:
        {
          PID_SetKIDivisorPOW2 (&stoPLLSensor[motorID]->PIRegulator,regdata16);
          break;
        }

        case MC_REG_STOPLL_KP_DIV:
        {
          PID_SetKPDivisorPOW2 (&stoPLLSensor[motorID]->PIRegulator,regdata16);
          break;
        }
</#if><#-- M1_IS_STO_PLL || M2_IS_STO_PLL -->

<#if MC.M1_FLUX_WEAKENING_ENABLING==true || MC.M2_FLUX_WEAKENING_ENABLING==true>
        case MC_REG_FLUXWK_KP_DIV: 
        {
          PID_SetKPDivisorPOW2 (pPIDFW[motorID],regdata16);
          break;
        }

        case MC_REG_FLUXWK_KI_DIV:
        {
          PID_SetKIDivisorPOW2 (pPIDFW[motorID],regdata16);
          break;
        }
</#if><#-- MC.M1_FLUX_WEAKENING_ENABLING==true || MC.M2_FLUX_WEAKENING_ENABLING==true -->

        default:
        {
          retVal = MCP_ERROR_UNKNOWN_REG;
          break;
        }
      }
      *size = 2;
      break;
    }
      
    case TYPE_DATA_32BIT:
    {
<#if UNALIGNMENT_SUPPORTED>
      int32_t regdata32 = *(int32_t *)data;
<#else><#-- NO_UNALIGNMENT_SUPPORTED -->
      int32_t regdata32 = ((int32_t)(*(int16_t *)&data[2]))<<16 | *(uint16_t *)data;
</#if><#-- UNALIGNMENT_SUPPORTED -->

      switch (regID)
      {
        case MC_REG_FAULTS_FLAGS:
        case MC_REG_SPEED_MEAS:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }

        case MC_REG_SPEED_REF:
<#if MC.M1_DRIVE_TYPE == "ACIM">
        {
          MCI_ExecSpeedRamp(pMCI,(int16_t)((regdata32*SPEED_UNIT)/U_RPM),0);
<#elseif MC.M1_DRIVE_TYPE == "SIX_STEP">
          pMCI->speed_target_value = regdataS32;
</#if><#-- MC.M1_DRIVE_TYPE -->
          break;
        }

        case MC_REG_STOPLL_EST_BEMF:
        case MC_REG_STOPLL_OBS_BEMF:
        case MC_REG_STOCORDIC_EST_BEMF:
        case MC_REG_STOCORDIC_OBS_BEMF:
        {
          retVal = MCP_ERROR_RO_REG;
          break;
        }

<#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING>
        case MC_REG_FF_1Q:
        {
          pFF[motorID]->wConstant_1Q = regdata32;
          break;
        }

        case MC_REG_FF_1D:
        {
          pFF[motorID]->wConstant_1D = regdata32;
          break;
        }

        case MC_REG_FF_2:
        {
          pFF[motorID]->wConstant_2 = regdata32;
          break;
        }
</#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING -->

<#if MC.PFC_ENABLED> 
        case MC_REG_PFC_FAULTS:
          break;
</#if><#-- MC.PFC_ENABLED -->

<#if MC.MOTOR_PROFILER>
        case MC_REG_SC_RS:
        case MC_REG_SC_LS:
        case MC_REG_SC_KE:
        case MC_REG_SC_VBUS:
        case MC_REG_SC_MEAS_NOMINALSPEED:
        case MC_REG_SC_CURRENT:
        case MC_REG_SC_SPDBANDWIDTH:
        case MC_REG_SC_LDLQRATIO:
        case MC_REG_SC_NOMINAL_SPEED:
        case MC_REG_SC_CURRBANDWIDTH:
        case MC_REG_SC_J:
        case MC_REG_SC_F:
        case MC_REG_SC_MAX_CURRENT:
        case MC_REG_SC_STARTUP_SPEED:
        case MC_REG_SC_STARTUP_ACC:
          break;
</#if><#-- MC.MOTOR_PROFILER -->
    
        default:
        {
          retVal = MCP_ERROR_UNKNOWN_REG;
          break;
        }
      }
      *size = 4;
      break;
    }

    case TYPE_DATA_STRING:
    {
      const char_t *charData = (const char_t *) data;
      char_t *dummy = (char_t *) data ;
      retVal = MCP_ERROR_RO_REG;
      /* Used to compute String length stored in RXBUFF even if Reg does not exist*/
      /* It allows to jump to the next command in the buffer */
      RI_MovString (charData, dummy, size, dataAvailable);
      break;
    }

    case TYPE_DATA_RAW:
    {
      uint16_t rawSize = *(uint16_t *) data;
      *size = rawSize+2; /* The size consumed by the structure is the structure size + 2 bytes used to store the size*/
      uint8_t * rawData = data+2; /* rawData points to the first data (after size extraction) */
      if (*size > dataAvailable )
      { /* The decoded size of the raw structure can not match with transmitted buffer, error in buffer construction*/
        *size =0; 
        retVal = MCP_ERROR_BAD_RAW_FORMAT; /* this error stop the parsing of the CMD buffer */
      }
      else
      {
        switch (regID)
        {
          case MC_REG_GLOBAL_CONFIG:
          case MC_REG_MOTOR_CONFIG:
          case MC_REG_APPLICATION_CONFIG:
          case MC_REG_FOCFW_CONFIG:
          {
            retVal = MCP_ERROR_RO_REG;
            break;
          }

          case MC_REG_SPEED_RAMP:
          {
<#if MC.M1_DRIVE_TYPE == "ACIM">
            int32_t rpm;
            uint16_t duration;
  <#if UNALIGNMENT_SUPPORTED >
            rpm = *(int32_t *)rawData;
  <#else><#-- NO_UNALIGNMENT_SUPPORTED -->
            rpm = ((int32_t)(*(int16_t *)&rawData[2]))<<16 | *(uint16_t *)rawData; /* 32 bits access are split into 2x16 bits access */
  </#if><#-- UNALIGNMENT_SUPPORTED -->
            duration = *(uint16_t *)&rawData[4]; 
            MCI_ExecSpeedRamp(pMCI,(int16_t)((rpm*SPEED_UNIT)/U_RPM),duration);
<#elseif MC.M1_DRIVE_TYPE == "SIX_STEP">
            int32_t rpm;
  <#if UNALIGNMENT_SUPPORTED >
            rpm = *(int32_t *)rawData;
  <#else><#-- NO_UNALIGNMENT_SUPPORTED -->
            rpm = ((int32_t)(*(int16_t *)&rawData[2]))<<16 | *(uint16_t *)rawData; /* 32 bits access are split into 2x16 bits access */
  </#if><#-- UNALIGNMENT_SUPPORTED -->
            pMCI->speed_target_command = (rpm);
</#if><#-- MC.M1_DRIVE_TYPE -->
            break;
          }

<#if MC.M1_DRIVE_TYPE == "ACIM">
          case MC_REG_TORQUE_RAMP:
          {
            uint32_t torque; 
            uint16_t duration; 
  <#if UNALIGNMENT_SUPPORTED >
            torque = *(int32_t *)rawData;
  <#else><#-- NO_UNALIGNMENT_SUPPORTED -->
            torque = ((int32_t)(*(int16_t *)&rawData[2]))<<16 | *(uint16_t *)rawData; /* 32 bits access are split into 2x16 bits access */
  </#if><#-- UNALIGNMENT_SUPPORTED -->
            duration = *(uint16_t *)&rawData[4];
            MCI_ExecTorqueRamp(pMCI,torque,duration);
            break;
          }

  <#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING>      
          case MC_REG_POSITION_RAMP:
          {
            FloatToU32 Position;
            FloatToU32 Duration;
    <#if UNALIGNMENT_SUPPORTED >
            Position.U32_Val = *(int32_t *)rawData;
            Duration.U32_Val = *(int32_t *)&rawData[4];
    <#else><#-- NO_UNALIGNMENT_SUPPORTED -->
            Position.U32_Val = ((int32_t)(*(int16_t *)&rawData[2]))<<16 | *(uint16_t *)rawData; /* 32 bits access are split into 2x16 bits access */
            Duration.U32_Val = ((int32_t)(*(int16_t *)&rawData[6]))<<16 | *(uint16_t *)&rawData[4]; /* 32 bits access are split into 2x16 bits access */
    </#if><#-- UNALIGNMENT_SUPPORTED -->
            MCI_ExecPositionCommand(pMCI, Position.Float_Val, Duration.Float_Val);
            break;
          }
  </#if><#--  MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING -->

  <#if M1_IS_SENSORLESS || M2_IS_SENSORLESS>
          case MC_REG_REVUP_DATA:
          {
            int32_t rpm;
            RevUpCtrl_PhaseParams_t revUpPhase;
            uint8_t i;
            uint8_t nbrOfPhase = rawSize/ 8;
            if ((rawSize % 8) || (nbrOfPhase > RUC_MAX_PHASE_NUMBER) != 0 )
            {
              retVal = MCP_ERROR_BAD_RAW_FORMAT;
            }
            else
            {          
              for (i = 0; i <nbrOfPhase; i++)
              {
    <#if UNALIGNMENT_SUPPORTED >
                rpm = *(int32_t *) &rawData[i*8];
    <#else><#-- NO_UNALIGNMENT_SUPPORTED -->
                /* &rawData is guarantee to be 16 bits aligned, so 32 bits access are to be splitted */
                rpm = ((int32_t)(*(int16_t *)&rawData[2+i*8]))<<16 | *(uint16_t *)&rawData[i*8];
    </#if><#-- UNALIGNMENT_SUPPORTED -->
                revUpPhase.hFinalMecSpeedUnit = (uint16_t) (rpm * SPEED_UNIT ) / U_RPM ;
                revUpPhase.hFinalTorque = *((uint16_t *) &rawData[4+i*8]);
                revUpPhase.hDurationms  = *((uint16_t *) &rawData[6+i*8]);
                RUC_SetPhase( RevUpControl[motorID] ,i, &revUpPhase);
              }
            }
            break;
          }
  </#if><#-- M1_IS_SENSORLESS || M2_IS_SENSORLESS -->
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->

<#if MC.MCP_ASYNC_OVER_UART_A_EN>
          case MC_REG_ASYNC_UARTA:
          {          
            retVal =  MCPA_cfgLog ( &MCPA_UART_A, rawData );
            break;
          }
</#if><#-- MC.MCP_ASYNC_OVER_UART_A_EN -->

<#if MC.MCP_ASYNC_OVER_UART_B_EN >
          case MC_REG_ASYNC_UARTB:
          {
            retVal =  MCPA_cfgLog ( &MCPA_UART_B, rawData );
            break;
          }
</#if><#-- MC.MCP_ASYNC_OVER_UART_B_EN -->

<#if MC.MCP_ASYNC_OVER_STLNK_EN >
          case MC_REG_ASYNC_STLNK:
          {
            retVal =  MCPA_cfgLog ( &MCPA_STLNK, rawData );
            break;
          }
</#if><#-- MC.MCP_ASYNC_OVER_STLNK_EN -->

<#if MC.M1_DRIVE_TYPE == "ACIM">
          case MC_REG_CURRENT_REF:
          {
            qd_t currComp;
            currComp.q = *((uint16_t *) rawData);
            currComp.d = *((uint16_t *) &rawData[2]);
            MCI_SetCurrentReferences(pMCI,currComp);
            break;
          }
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->

          default:
          {
            retVal = MCP_ERROR_UNKNOWN_REG;
            break;
          }
        }
      }
    }

    default:
    {
      retVal = MCP_ERROR_BAD_DATA_TYPE;
      *size =0; /* From this point we are not able anymore to decode the RX buffer*/
      break;
    }
  }
  return retVal;
}

uint8_t RI_GetReg (uint16_t dataID, uint8_t * data, uint16_t *size, int16_t freeSpace)
{
  uint8_t typeID = dataID & TYPE_MASK;
  uint8_t motorID = (dataID & MOTOR_MASK)-1;
  uint16_t regID = dataID & REG_MASK;
  uint8_t retVal = MCP_CMD_OK;
  
<#if MC.M1_DRIVE_TYPE == "ACIM" ><#-- Specific to FOC algorithm usage -->
  MCI_Handle_t * pMCI = &Mci[motorID];
<#elseif  MC.M1_DRIVE_TYPE == "SIX_STEP" ><#-- Specific to 6_STEP algorithm usage -->
  MC_Handle_t * pMCI = MC_Core_GetMotorControlHandle( motorID );
</#if><#-- MC.M1_DRIVE_TYPE -->
  switch (typeID)
  {
    case TYPE_DATA_8BIT:
    {
      if (freeSpace > 0)
      {
        switch (regID)
        {
          case MC_REG_STATUS:
          {
<#if MC.M1_DRIVE_TYPE == "ACIM" >
            *data = (uint8_t)MCI_GetSTMState(pMCI);
<#elseif  MC.M1_DRIVE_TYPE == "SIX_STEP" >
            *data = MC_Core_GetState(pMCI);
</#if><#-- MC.M1_DRIVE_TYPE -->
            break;
          }
        
<#if MC.M1_DRIVE_TYPE == "ACIM" >
          case MC_REG_CONTROL_MODE:
          {
            *data =  MCI_GetControlMode(pMCI);
            break;
          }
          
  <#if M1_IS_SENSORLESS || M2_IS_SENSORLESS>
          case MC_REG_RUC_STAGE_NBR:
          {
            *data = (RevUpControl[motorID] != MC_NULL ) ? RUC_GetNumberOfPhases(RevUpControl[motorID]) : 0;
            break;
          }
  </#if><#-- M1_IS_SENSORLESS || M2_IS_SENSORLESS -->
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->

<#if MC.PFC_ENABLED >          
          case MC_REG_PFC_STATUS:
          case MC_REG_PFC_ENABLED:
            break;
</#if><#-- MC.PFC_ENABLE -->
  
<#if MC.MOTOR_PROFILER >
          case MC_REG_SC_CHECK:
          case MC_REG_SC_STATE:
          case MC_REG_SC_STEPS:
          case MC_REG_SC_PP:
          case MC_REG_SC_FOC_REP_RATE:
          case MC_REG_SC_COMPLETED:
            break;
</#if><#-- MC.MOTOR_PROFILER -->
  
<#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING >
          case MC_REG_POSITION_CTRL_STATE:
          {
            *data = (uint8_t) TC_GetControlPositionStatus(pPosCtrl[motorID]);
            break;
          }

          case MC_REG_POSITION_ALIGN_STATE:
          {
            *data = (uint8_t) TC_GetAlignmentStatus(pPosCtrl[motorID]);
            break;
          }
</#if><#-- MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING --> 
        
          default:
          {
            retVal = MCP_ERROR_UNKNOWN_REG;
            break;
          }
        }
          *size = 1;
      }

      else
      {
        retVal = MCP_ERROR_NO_TXSYNC_SPACE;
      }
      break;
    }
      
    case TYPE_DATA_16BIT:
    {
      uint16_t * regdataU16 = (uint16_t *)data;
      int16_t * regdata16 = (int16_t *) data;
      if (freeSpace >= 2)
      {
        switch (regID) 
        {
<#if MC.ACIM_CONFIG == "LSO_FOC">
          case MC_REG_SPEED_KP:
          {
            *regdata16 = PID_GetKP(pPIDSpeed[motorID]);
            break;
          }

          case MC_REG_SPEED_KI:
          {
            *regdata16 = PID_GetKI(pPIDSpeed[motorID]);
            break;
          }

          case MC_REG_SPEED_KD:
          {
            *regdata16 = PID_GetKD(pPIDSpeed[motorID]);
           break;
          }

          case MC_REG_I_Q_KP:
          {
            *regdata16 = PID_GetKP(pPIDIq[motorID]);
             break;
          }

          case MC_REG_I_Q_KI:
          {
            *regdata16 = PID_GetKI(pPIDIq[motorID]);
            break;
          }

          case MC_REG_I_Q_KD:
          {
            *regdata16 = PID_GetKD(pPIDIq[motorID]);
            break;
          }

          case MC_REG_I_D_KP:
          {
            *regdata16 = PID_GetKP(pPIDId[motorID]);
            break;
          }

          case MC_REG_I_D_KI:
          {
            *regdata16 = PID_GetKI(pPIDId[motorID]);
            break;
          }

          case MC_REG_I_D_KD:
          {
            *regdata16 = PID_GetKD(pPIDId[motorID]);
            break;
          }
</#if><#-- MC.ACIM_CONFIG == "LSO_FOC" -->

<#if MC.M1_FLUX_WEAKENING_ENABLING || MC.M2_FLUX_WEAKENING_ENABLING>
          case MC_REG_FLUXWK_KP:
          {
            *regdata16 = PID_GetKP(pPIDFW[motorID]);
            break;
          }

          case MC_REG_FLUXWK_KI:
          {
            *regdata16 = PID_GetKI(pPIDFW[motorID]);
            break;
          }

          case MC_REG_FLUXWK_BUS:
          {
            *regdataU16 = FW_GetVref(pFW[motorID]);
            break;
          }

          case MC_REG_FLUXWK_BUS_MEAS:
          {
            *regdata16 = FW_GetAvVPercentage(pFW[motorID]);
            break;
          }
</#if><#-- MC.M1_FLUX_WEAKENING_ENABLING || MC.M1_FLUX_WEAKENING_ENABLING -->

<#if MC.M1_DRIVE_TYPE == "ACIM" ><#-- TODO: These registers should also be supported identically in 6S configurations in public releases... and some be made optional -->
          case MC_REG_BUS_VOLTAGE:
          {
            *regdataU16 = VBS_GetAvBusVoltage_V(BusVoltageSensor[motorID]);
            break;
          }

          case MC_REG_HEATS_TEMP:
          {
            *regdata16 = NTC_GetAvTemp_C(pTemperatureSensor[motorID]);
            break;
          }
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->

<#if MC.DEBUG_DAC_FUNCTIONALITY_EN>
          case MC_REG_DAC_OUT1:
          {
            *regdata16 = DAC_GetChannelConfig(&DAC_Handle , DAC_CH1);
             break;
          }

          case MC_REG_DAC_OUT2:
          {
            *regdata16 = DAC_GetChannelConfig(&DAC_Handle , DAC_CH2);
            break;
          }
</#if><#-- MC.DEBUG_DAC_FUNCTIONALITY_EN -->

<#if MC.M1_DRIVE_TYPE == "ACIM" > 
          case MC_REG_I_A:
          {
            *regdata16 = MCI_GetIab(pMCI).a;
            break;
          }

          case MC_REG_I_B:
          {
            *regdata16 = MCI_GetIab(pMCI).b;
            break;
          }

          case MC_REG_I_ALPHA_MEAS:
          {
            *regdata16 = MCI_GetIalphabeta(pMCI).alpha;
            break;
          }

          case MC_REG_I_BETA_MEAS:
          {
            *regdata16 = MCI_GetIalphabeta(pMCI).beta;
            break;
          }

          case MC_REG_I_Q_MEAS:
          {
            *regdata16 = MCI_GetIqd(pMCI).q;
            break;
          }

          case MC_REG_I_D_MEAS:
          {
            *regdata16 = MCI_GetIqd(pMCI).d;
            break;
          }

          case MC_REG_I_Q_REF:
          {
            *regdata16 = MCI_GetIqdref(pMCI).q;
            break;
          }

          case MC_REG_I_D_REF:
          {
            *regdata16 = MCI_GetIqdref(pMCI).d;
            break;
          }

          case MC_REG_V_Q:
          {
            *regdata16 = MCI_GetVqd(pMCI).q;
            break;
          }

          case MC_REG_V_D:
          {
            *regdata16 = MCI_GetVqd(pMCI).d;
            break;
          }

          case MC_REG_V_ALPHA:
          {
            *regdata16 = MCI_GetValphabeta(pMCI).alpha;
            break;
          }

          case MC_REG_V_BETA:
          {
            *regdata16 = MCI_GetValphabeta(pMCI).beta;
            break;
          }

  <#if M1_IS_ENCODER || M2_IS_ENCODER>
          case MC_REG_ENCODER_EL_ANGLE:
          {
            *regdata16 = SPD_GetElAngle ((SpeednPosFdbk_Handle_t*) pEncoder[motorID]);
            break;
          }

          case MC_REG_ENCODER_SPEED:
          {
            *regdata16 = SPD_GetS16Speed ((SpeednPosFdbk_Handle_t*) pEncoder[motorID]);
            break;
          }
  </#if><#-- M1_IS_ENCODER || M2_IS_ENCODER -->

  <#if M1_IS_HALL_SENSOR || M2_IS_HALL_SENSOR>
          case MC_REG_HALL_EL_ANGLE:
          {
            *regdata16 = SPD_GetElAngle ((SpeednPosFdbk_Handle_t*) pHallSensor[motorID]);
             break;
          }

          case MC_REG_HALL_SPEED:
          {
            *regdata16 = SPD_GetS16Speed ((SpeednPosFdbk_Handle_t*) pHallSensor[motorID]);
            break;
          }
  </#if><#-- M1_IS_HALL_SENSOR || M2_IS_HALL_SENSOR -->

  <#if M1_IS_STO_PLL || M2_IS_STO_PLL>
          case MC_REG_STOPLL_EL_ANGLE:
          {
            *regdata16 = SPD_GetElAngle( (SpeednPosFdbk_Handle_t*) stoPLLSensor[motorID]);
            break;
          }

          case MC_REG_STOPLL_ROT_SPEED:
          {
            *regdata16 = SPD_GetS16Speed((SpeednPosFdbk_Handle_t*) stoPLLSensor[motorID]);
            break;
          }

          case MC_REG_STOPLL_I_ALPHA:
          {
            *regdata16 = STO_PLL_GetEstimatedCurrent(stoPLLSensor[motorID]).alpha;
            break;
          }

          case MC_REG_STOPLL_I_BETA:
          {
            *regdata16 = STO_PLL_GetEstimatedCurrent(stoPLLSensor[motorID]).beta;
            break;
          }

          case MC_REG_STOPLL_BEMF_ALPHA:
          {
            *regdata16 = STO_PLL_GetEstimatedBemf(stoPLLSensor[motorID]).alpha;
            break;
          }

          case MC_REG_STOPLL_BEMF_BETA:
          {
            *regdata16 = STO_PLL_GetEstimatedBemf(stoPLLSensor[motorID]).beta;
            break;
          }

          case MC_REG_STOPLL_C1:
          {
            int16_t hC1,hC2;
            STO_PLL_GetObserverGains(stoPLLSensor[motorID],&hC1,&hC2);
            *regdata16 = hC1;
            break;
          }

          case MC_REG_STOPLL_C2:
          {
            int16_t hC1,hC2;
            STO_PLL_GetObserverGains(stoPLLSensor[motorID],&hC1,&hC2);
            *regdata16 = hC2;
            break;
          }

          case MC_REG_STOPLL_KI:
          {
            *regdata16 = PID_GetKI (&stoPLLSensor[motorID]->PIRegulator);
            break;
          }

          case MC_REG_STOPLL_KP:
          {
            *regdata16 = PID_GetKP (&stoPLLSensor[motorID]->PIRegulator);
            break;
          }
  </#if><#-- M1_IS_STO_PLL || M2_IS_STO_PLL -->

  <#if M1_IS_STO_CORDIC || M2_IS_STO_CORDIC>
          case MC_REG_STOCORDIC_EL_ANGLE:
          {
            if (stoCRSensor[motorID] != MC_NULL)
            {
              *regdata16 = SPD_GetElAngle((SpeednPosFdbk_Handle_t*) stoCRSensor[motorID]);
            }
            else 
            {
              retVal = MCP_ERROR_UNKNOWN_REG;
            }
            break;
          }

          case MC_REG_STOCORDIC_ROT_SPEED:
          {
            *regdata16 = SPD_GetS16Speed((SpeednPosFdbk_Handle_t*) stoCRSensor[motorID]);
             break;
          }

          case MC_REG_STOCORDIC_I_ALPHA:
          {
            *regdata16 = STO_CR_GetEstimatedCurrent(stoCRSensor[motorID]).alpha;
            break;
          }

          case MC_REG_STOCORDIC_I_BETA:
          {
            *regdata16 = STO_CR_GetEstimatedCurrent(stoCRSensor[motorID]).beta;
            break;
          }

          case MC_REG_STOCORDIC_BEMF_ALPHA:
          {
            *regdata16 = STO_CR_GetEstimatedBemf(stoCRSensor[motorID]).alpha; 
            break;
          }

          case MC_REG_STOCORDIC_BEMF_BETA:
          {
            *regdata16 = STO_CR_GetEstimatedBemf(stoCRSensor[motorID]).beta;
            break;
          }

          case MC_REG_STOCORDIC_C1:
          {
            int16_t hC1,hC2;
            STO_CR_GetObserverGains(stoCRSensor[motorID],&hC1,&hC2);
            *regdata16 = hC1;
            break;
          }

          case MC_REG_STOCORDIC_C2:
          {
            int16_t hC1,hC2;
            STO_CR_GetObserverGains(stoCRSensor[motorID],&hC1,&hC2);
            *regdata16 = hC2;
            break;
          }
  </#if><#-- M1_IS_STO_CORDIC || M2_IS_STO_CORDIC -->

          case MC_REG_DAC_USER1:
          case MC_REG_DAC_USER2:
            break;

  <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING>
          case MC_REG_FF_VQ:
          {
            *regdata16 = FF_GetVqdff(pFF[motorID]).q;
            break;
          }

          case MC_REG_FF_VD:
          {
            *regdata16 = FF_GetVqdff(pFF[motorID]).d;
            break;
          }

          case MC_REG_FF_VQ_PIOUT:
          {
            *regdata16 = FF_GetVqdAvPIout(pFF[motorID]).q;
            break;
          }

          case MC_REG_FF_VD_PIOUT:
          {
            *regdata16 = FF_GetVqdAvPIout(pFF[motorID]).d;
            break;
          }
  </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING -->
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->

<#if MC.PFC_ENABLED>
          case MC_REG_PFC_DCBUS_REF:
          case MC_REG_PFC_DCBUS_MEAS:
          case MC_REG_PFC_ACBUS_FREQ:
          case MC_REG_PFC_ACBUS_RMS:
          case MC_REG_PFC_I_KP:
          case MC_REG_PFC_I_KI:
          case MC_REG_PFC_I_KD:
          case MC_REG_PFC_V_KP:
          case MC_REG_PFC_V_KI:
          case MC_REG_PFC_V_KD:
          case MC_REG_PFC_STARTUP_DURATION:
            break;
</#if><#-- MC.PFC_ENABLED -->

<#if MC.MOTOR_PROFILER >
          case MC_REG_SC_PWM_FREQUENCY:
            break;
</#if><#-- MC.MOTOR_PROFILER -->

<#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING>
          case MC_REG_POSITION_KP:
          {
            *regdata16 = PID_GetKP( pPIDPosCtrl[motorID]); 
            break;
          }

          case MC_REG_POSITION_KI:
          {
            *regdata16 = PID_GetKI( pPIDPosCtrl[motorID]);
            break;
          }

          case MC_REG_POSITION_KD:
          {
            *regdata16 = PID_GetKD( pPIDPosCtrl[motorID]);
            break;
          }
</#if><#-- MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING -->

<#if MC.ACIM_CONFIG == "LSO_FOC">
          case MC_REG_SPEED_KP_DIV:
          {
            *regdataU16 = PID_GetKPDivisor(pPIDSpeed[motorID]);
            break;
          }

          case MC_REG_SPEED_KI_DIV:
          {
            *regdataU16 = PID_GetKIDivisor(pPIDSpeed[motorID]);
            break;
          }
 
          case MC_REG_SPEED_KD_DIV:
          {
            *regdataU16 = PID_GetKDDivisor(pPIDSpeed[motorID]);
            break;
          }

          case MC_REG_I_D_KP_DIV:
          {
            *regdataU16 = PID_GetKPDivisor(pPIDId[motorID]);
            break;
          }

          case MC_REG_I_D_KI_DIV:
          {
            *regdataU16 = PID_GetKIDivisor(pPIDId[motorID]);
            break;
          }

          case MC_REG_I_D_KD_DIV:
          {
            *regdataU16 = PID_GetKDDivisor(pPIDId[motorID]);
            break;
          }

          case MC_REG_I_Q_KP_DIV:
          {
            *regdataU16 = PID_GetKPDivisor(pPIDIq[motorID]);
            break;
          }

          case MC_REG_I_Q_KI_DIV:
          {
            *regdataU16 = PID_GetKIDivisor(pPIDIq[motorID]);
            break;
          }

          case MC_REG_I_Q_KD_DIV:
          {
            *regdataU16 = PID_GetKDDivisor(pPIDIq[motorID]);
            break;
          }
</#if><#-- MC.ACIM_CONFIG == "LSO_FOC" -->

<#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING>
          case MC_REG_POSITION_KP_DIV:
          {
            *regdataU16 = PID_GetKPDivisor(pPIDPosCtrl[motorID]);
            break;
          }

          case MC_REG_POSITION_KI_DIV: 
          {
            *regdataU16 = PID_GetKIDivisor(pPIDPosCtrl[motorID]);
            break;
          }

          case MC_REG_POSITION_KD_DIV:
          {
            *regdataU16 = PID_GetKDDivisor(pPIDPosCtrl[motorID]);
            break;
          }
</#if><#-- MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING -->

<#if MC.PFC_ENABLED>
          case MC_REG_PFC_I_KP_DIV:
          case MC_REG_PFC_I_KI_DIV:
          case MC_REG_PFC_I_KD_DIV:
          case MC_REG_PFC_V_KP_DIV:
          case MC_REG_PFC_V_KI_DIV:
          case MC_REG_PFC_V_KD_DIV:
            break;
</#if><#-- MC.PFC_ENABLED -->

<#if M1_IS_STO_PLL || M2_IS_STO_PLL>
          case MC_REG_STOPLL_KI_DIV:
          {
            *regdataU16 = PID_GetKIDivisor(&stoPLLSensor[motorID]->PIRegulator);
            break;
          }

          case MC_REG_STOPLL_KP_DIV:
          {
            *regdataU16 = PID_GetKPDivisor(&stoPLLSensor[motorID]->PIRegulator);
            break;
          }
</#if><#-- M1_IS_STO_PLL || M2_IS_STO_PLL -->

<#if MC.M1_FLUX_WEAKENING_ENABLING || MC.M2_FLUX_WEAKENING_ENABLING>
          case MC_REG_FLUXWK_KP_DIV:
          {
            *regdataU16 = PID_GetKPDivisor(pPIDFW[motorID]);
            break;
          }
        
          case MC_REG_FLUXWK_KI_DIV:
          {
            *regdataU16 = PID_GetKIDivisor(pPIDFW[motorID]);
            break;
          }
</#if><#-- MC.M1_FLUX_WEAKENING_ENABLING || MC.M2_FLUX_WEAKENING_ENABLING -->

          default:
          {
            retVal = MCP_ERROR_UNKNOWN_REG;
            break;
          }
        }
        *size = 2;
      }
      else
      {
        retVal = MCP_ERROR_NO_TXSYNC_SPACE; 
      }
      break;
    }

    case TYPE_DATA_32BIT:
    {
      uint32_t *regdataU32 = (uint32_t *) data;
      int32_t *regdata32 = (int32_t *) data;
      if (freeSpace >= 4)
      {
        switch (regID)
        {
          case MC_REG_FAULTS_FLAGS:
          {
            *regdataU32 = MCI_GetFaultState(pMCI);
            break;
          }

          case MC_REG_SPEED_MEAS:
          {
            *regdata32 = (((int32_t)MCI_GetAvrgMecSpeedUnit(pMCI) * U_RPM) / SPEED_UNIT);
            break;
          }
          
          case MC_REG_SPEED_REF: 
          { 
            *regdata32 = (((int32_t)MCI_GetMecSpeedRefUnit(pMCI) * U_RPM) / SPEED_UNIT);
            break;
          }

<#if M1_IS_STO_PLL || M2_IS_STO_PLL>
          case MC_REG_STOPLL_EST_BEMF:
          {
            *regdata32 = STO_PLL_GetEstimatedBemfLevel(stoPLLSensor[motorID]);
            break;
          }

          case MC_REG_STOPLL_OBS_BEMF:
          {
            *regdata32 = STO_PLL_GetObservedBemfLevel(stoPLLSensor[motorID]);
            break;
          }
</#if><#-- M1_IS_STO_PLL || M2_IS_STO_PLL -->

<#if M1_IS_STO_CORDIC || M2_IS_STO_CORDIC > 
          case MC_REG_STOCORDIC_EST_BEMF:
          {
            *regdata32 = STO_CR_GetEstimatedBemfLevel(stoCRSensor[motorID]);
            break;
          }

          case MC_REG_STOCORDIC_OBS_BEMF:
          {
            *regdata32 = STO_CR_GetObservedBemfLevel(stoCRSensor[motorID]);
            break;
          }
</#if><#-- M1_IS_STO_CORDIC || M2_IS_STO_CORDIC -->

<#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING>
          case MC_REG_FF_1Q:
          {
            *regdata32 = pFF[motorID]->wConstant_1Q;
            break;
          }

          case MC_REG_FF_1D:
          { 
            *regdata32 = pFF[motorID]->wConstant_1D;
            break;
          }

          case MC_REG_FF_2:
          {
            *regdata32 = pFF[motorID]->wConstant_2;
            break;
          }
</#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING || MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING -->

<#if MC.PFC_ENABLED>
          case MC_REG_PFC_FAULTS:
            break;     
</#if><#-- MC.PFC_ENABLED -->

<#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING>
          case MC_REG_CURRENT_POSITION:
          {
            FloatToU32 ReadVal;
            ReadVal.Float_Val = MCI_GetCurrentPosition(pMCI);
            *regdataU32 = ReadVal.U32_Val;
            break;
          } 
</#if><#-- MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING -->

<#if MC.M1_DRIVE_TYPE == "ACIM"><#-- TODO: These registers should also be supported identically in 6S configurations in public releases... and some be made optional -->
          case MC_REG_MOTOR_POWER:
          {
            FloatToU32 ReadVal;
            ReadVal.Float_Val = PQD_GetAvrgElMotorPowerW(pMPM[M1]);
            *regdataU32 = ReadVal.U32_Val;
            break;
          }
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->

<#if MC.MOTOR_PROFILER>
          case MC_REG_SC_RS:
          case MC_REG_SC_LS:
          case MC_REG_SC_KE:
          case MC_REG_SC_VBUS:
          case MC_REG_SC_MEAS_NOMINALSPEED:
          case MC_REG_SC_CURRENT:
          case MC_REG_SC_SPDBANDWIDTH:
          case MC_REG_SC_LDLQRATIO:
          case MC_REG_SC_NOMINAL_SPEED:
          case MC_REG_SC_CURRBANDWIDTH:
          case MC_REG_SC_J:
          case MC_REG_SC_F:
          case MC_REG_SC_MAX_CURRENT:
          case MC_REG_SC_STARTUP_SPEED:
          case MC_REG_SC_STARTUP_ACC:
            break;
</#if><#-- MC.MOTOR_PROFILER -->

          default:
          {
            retVal = MCP_ERROR_UNKNOWN_REG;
            break;
          }
        }
        *size = 4;
      }
      else
      {
        retVal = MCP_ERROR_NO_TXSYNC_SPACE; 
      }
      break;
    }

    case TYPE_DATA_STRING:
    {
      char_t *charData = (char_t *) data;
      switch (regID)
      {
        case MC_REG_FW_NAME:
        {
          retVal = RI_MovString (FIRMWARE_NAME ,charData, size, freeSpace);
          break;
        }

        case MC_REG_CTRL_STAGE_NAME:
        {
          retVal = RI_MovString (CTL_BOARD ,charData, size, freeSpace);
          break;
        }

        case MC_REG_PWR_STAGE_NAME:
        {
          retVal = RI_MovString (PWR_BOARD_NAME[motorID] ,charData, size, freeSpace);
          break;
        }

        case MC_REG_MOTOR_NAME:
        {
          retVal = RI_MovString (MotorConfig_reg[motorID]->name ,charData, size, freeSpace);
          break;
        }

        default:
        {
          retVal = MCP_ERROR_UNKNOWN_REG;
          *size= 0 ; /* */
          break;
        }
      }
      break;
    }

    case TYPE_DATA_RAW:
    {
      uint16_t *rawSize = (uint16_t *) data; /* First 2 bytes of the answer is reserved to the size */
      uint8_t * rawData = data+2;

      switch (regID) 
      {
        case MC_REG_GLOBAL_CONFIG:
        {
          *rawSize = sizeof(GlobalConfig_reg_t); 
          if ((*rawSize) +2  > freeSpace) 
          {
            retVal = MCP_ERROR_NO_TXSYNC_SPACE;
          }
          else 
          {
            memcpy(rawData, &globalConfig_reg, sizeof(GlobalConfig_reg_t) );
          }
          break;
        }

        case MC_REG_MOTOR_CONFIG:
        {
          *rawSize = sizeof(MotorConfig_reg_t);
          if ((*rawSize) +2  > freeSpace) 
          {
            retVal = MCP_ERROR_NO_TXSYNC_SPACE;
          }
          else 
          { 
            memcpy(rawData, MotorConfig_reg[motorID], sizeof(MotorConfig_reg_t) );
          }
          break;
        }

        case MC_REG_APPLICATION_CONFIG:
        {  
          *rawSize = sizeof(ApplicationConfig_reg_t);
          if ((*rawSize) +2  > freeSpace) 
          {
            retVal = MCP_ERROR_NO_TXSYNC_SPACE;
          }
          else 
          { 
            memcpy(rawData, ApplicationConfig_reg[motorID], sizeof(ApplicationConfig_reg_t));
          }
          break;
        }
 
        case MC_REG_FOCFW_CONFIG:
<#if MC.M1_DRIVE_TYPE == "ACIM" ><#-- TODO: Create an equivalent register for 6S -->
        {
          *rawSize = sizeof(FOCFwConfig_reg_t); 
          if ((*rawSize) +2  > freeSpace) 
          {
            retVal = MCP_ERROR_NO_TXSYNC_SPACE;
          }
          else 
          {
            memcpy(rawData, FOCConfig_reg[motorID], sizeof(FOCFwConfig_reg_t) );
          }
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->
          break;
        }

        case MC_REG_SPEED_RAMP:
        {          
<#if UNALIGNMENT_SUPPORTED>
          int32_t *rpm = (int32_t *) rawData; 
          uint16_t *duration = (uint16_t *) &rawData[4];
          *rpm = (int32_t)((MCI_GetLastRampFinalSpeed(pMCI) * U_RPM)/SPEED_UNIT) ;
<#else><#-- NO_UNALIGNMENT_SUPPORTED -->
          uint16_t *rpm16p = (uint16_t *) rawData; 
          uint16_t *duration = (uint16_t *) &rawData[4];
          int32_t rpm32 = (int32_t)((MCI_GetLastRampFinalSpeed(pMCI) * U_RPM)/SPEED_UNIT) ;
          *rpm16p = (uint16_t) rpm32;
          *(rpm16p+1) = (uint16_t) (rpm32>>16);
</#if><#-- UNALIGNMENT_SUPPORTED -->
          *duration = MCI_GetLastRampFinalDuration(pMCI) ;
          *rawSize = 6;
          break;
        }

        case MC_REG_TORQUE_RAMP:
        {
          int16_t *torque = (int16_t *) rawData;
          uint16_t *duration = (uint16_t *) &rawData[2];
          *rawSize = 4;
          *torque = MCI_GetLastRampFinalTorque(pMCI);
          *duration = MCI_GetLastRampFinalDuration(pMCI);
          break;
        }

<#if M1_IS_SENSORLESS || M2_IS_SENSORLESS>
      case MC_REG_REVUP_DATA:
      {
  <#if UNALIGNMENT_SUPPORTED >
        int32_t *rpm;
  <#else><#-- NO_UNALIGNMENT_SUPPORTED -->
        int32_t rpm32;
        int16_t *rpm16p;
  </#if><#-- UNALIGNMENT_SUPPORTED -->
        uint16_t *finalTorque;
        uint16_t *durationms;
        RevUpCtrl_PhaseParams_t revUpPhase;
        uint8_t i;

        *rawSize = RUC_MAX_PHASE_NUMBER*8; 
        if ((*rawSize) +2  > freeSpace) 
        {
          retVal = MCP_ERROR_NO_TXSYNC_SPACE;
        }
        else
        {
          for (i = 0; i <RUC_MAX_PHASE_NUMBER; i++)
          {
            RUC_GetPhase( RevUpControl[motorID] ,i, &revUpPhase);
  <#if UNALIGNMENT_SUPPORTED >
            rpm = (int32_t *) &data[2+i*8];
            *rpm = (revUpPhase.hFinalMecSpeedUnit * U_RPM) / SPEED_UNIT;
  <#else><#-- NO_UNALIGNMENT_SUPPORTED -->
            rpm32 = (revUpPhase.hFinalMecSpeedUnit * U_RPM) / SPEED_UNIT;
            rpm16p = (int16_t *) &data[2+i*8];
            *rpm16p = (uint16_t) rpm32; /* 16 LSB access */
            *(rpm16p+1) = (uint16_t) (rpm32 >> 16); /* 16 MSB access */
  </#if><#-- UNALIGNMENT_SUPPORTED -->
            finalTorque = (uint16_t *) &data[6+i*8];
            *finalTorque = revUpPhase.hFinalTorque;
            durationms  = (uint16_t *) &data[8+i*8];
            *durationms  = revUpPhase.hDurationms;
          }
        }
        break;
      }
</#if><#-- M1_IS_SENSORLESS || M2_IS_SENSORLESS -->

        case MC_REG_CURRENT_REF: 
        { 
          uint16_t *iqref = (uint16_t *) rawData;
          uint16_t *idref = (uint16_t *) &rawData[2];
         *rawSize = 4;
          *iqref = MCI_GetIqdref(pMCI).q;
          *idref = MCI_GetIqdref(pMCI).d;
           break;
        } 

<#if MC.M1_POSITION_CTRL_ENABLING || MC.M2_POSITION_CTRL_ENABLING>           
        case MC_REG_POSITION_RAMP:
        {
          float Position;
          float Duration;
          *rawSize = 8;
          Position = TC_GetMoveDuration(pPosCtrl[motorID]);   /* Does this duration make sense ? */
          Duration = TC_GetTargetPosition(pPosCtrl[motorID]);
          memcpy(rawData, &Position, 4 );
          memcpy(&rawData[4], &Duration, 4 );
          break;
        }
</#if>
        case MC_REG_ASYNC_UARTA:
        case MC_REG_ASYNC_UARTB:
        case MC_REG_ASYNC_STLNK:
        default:
        {
          retVal = MCP_ERROR_UNKNOWN_REG;
          break;
        }
      }
      /* Size of the answer is size of the data + 2 bytes containing data size*/
      *size = (*rawSize)+2;
      break;
    }
    default:
    {
      retVal = MCP_ERROR_BAD_DATA_TYPE;
      break;
    }
  }
  return (retVal);
}

uint8_t RI_MovString (const char_t * srcString, char_t * destString, uint16_t *size, int16_t maxSize)
{
  uint8_t retVal = MCP_CMD_OK;

  *size= 1 ; /* /0 is the min String size */
  while ((*srcString != 0) && (*size < maxSize) )
  {
    *destString = *srcString ;
    srcString = srcString+1;
    destString = destString+1;
    *size=*size+1;
  }
  if (*srcString != 0)
  { /* Last string char must be 0 */   
    retVal = MCP_ERROR_STRING_FORMAT;
  }
  else
  { 
    *destString = 0;
  }

  return (retVal);
}

uint8_t RI_GetIDSize (uint16_t dataID)
{
  uint8_t typeID = dataID & TYPE_MASK;
  uint8_t result;
  
  switch (typeID)
  {
    case TYPE_DATA_8BIT:
    {
      result = 1;
      break;
    }

    case TYPE_DATA_16BIT:
    {
      result = 2;
      break;
    }

    case TYPE_DATA_32BIT:
    {
      result = 4;
      break;
    }

    default:
    {
      result = 0;
      break;
    }
  }
  
  return (result);
}
__weak uint8_t RI_GetPtrReg (uint16_t dataID, void ** dataPtr)
{
<#if MC.M1_DRIVE_TYPE == "ACIM"><#-- The datalog feature is not supported by 6S yet -->
  uint8_t typeID = dataID & TYPE_MASK;
  uint8_t motorID = (dataID & MOTOR_MASK)-1;
  uint16_t regID = dataID & REG_MASK;
  uint8_t retVal = MCP_CMD_OK;
  
  MCI_Handle_t * pMCI = &Mci[motorID];

  switch (typeID)
  {
    case TYPE_DATA_16BIT:
    {
      switch (regID)
      {
        case MC_REG_I_A:
        {
          *dataPtr = &(pMCI->pFOCVars->Iab.a);
          break;
        }

        case MC_REG_I_B:
        {
          *dataPtr = &(pMCI->pFOCVars->Iab.b);
          break;
        }

        case MC_REG_I_ALPHA_MEAS:
        {
          *dataPtr = &(pMCI->pFOCVars->Ialphabeta.alpha);
          break;
        }

        case MC_REG_I_BETA_MEAS:
        {
          *dataPtr = &(pMCI->pFOCVars->Ialphabeta.beta);
          break;
        }

        case MC_REG_I_Q_MEAS:
        {
          *dataPtr = &(pMCI->pFOCVars->Iqd.q);
          break;
        }

        case MC_REG_I_D_MEAS:
        {
          *dataPtr = &(pMCI->pFOCVars->Iqd.d);
          break;
        }

        case MC_REG_I_Q_REF:
        {
          *dataPtr = &(pMCI->pFOCVars->Iqdref.q);
          break;
        }

        case MC_REG_I_D_REF:
        {
          *dataPtr = &(pMCI->pFOCVars->Iqdref.d);
          break;
        }

        case MC_REG_V_Q:
        {
          *dataPtr = &(pMCI->pFOCVars->Vqd.q);
          break;
        }

        case MC_REG_V_D:
        {
          *dataPtr = &(pMCI->pFOCVars->Vqd.d);
          break;
        }

        case MC_REG_V_ALPHA:
        {
          *dataPtr = &(pMCI->pFOCVars->Valphabeta.alpha);
          break;
        }

        case MC_REG_V_BETA:
        {
          *dataPtr = &(pMCI->pFOCVars->Valphabeta.beta);
          break;
        }

#if NOT_IMPLEMENTER
        case MC_REG_PRIMSENS_ANGLE:
        {
          *dataPtr = &(pMCI->pFOCVars->hElAngle);
          break;
        }

        case MC_REG_PRIMSENS_SPEED:
        case MC_REG_AUXSENS_ANGLE:
        case MC_REG_AUXSENS_SPEED:
          break;

  <#if M1_IS_STO_PLL || M2_IS_STO_PLL>
         // stoPLLSensor[motorID];

        case MC_REG_STOPLL_I_ALPHA:
        case MC_REG_STOPLL_I_BETA:
        case MC_REG_STOPLL_BEMF_ALPHA:
        case MC_REG_STOPLL_BEMF_BETA:
          break;
  </#if><#-- M1_IS_STO_PLL || M2_IS_STO_PLL -->
#endif

  <#if M1_IS_STO_CORDIC || M2_IS_STO_CORDIC > 
      // stoCRSensor[motorID];       
        case MC_REG_STOCORDIC_I_ALPHA:
        case MC_REG_STOCORDIC_I_BETA:
        case MC_REG_STOCORDIC_BEMF_ALPHA:
        case MC_REG_STOCORDIC_BEMF_BETA:
          break;
  </#if><#-- M1_IS_STO_CORDIC || M2_IS_STO_CORDIC -->

        default:
          break;
      }
      break;
    }

    default:
      break;
  }
  
  return (retVal);
<#else><#-- MC.M1_DRIVE_TYPE == "ACIM" -->
  return (MCP_CMD_NOK);
</#if><#-- MC.M1_DRIVE_TYPE == "ACIM" -->
}

/************************ (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
