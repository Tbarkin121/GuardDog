
/**
  ******************************************************************************
  * @file    mc_parameters.c
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file provides definitions of HW parameters specific to the
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

/* Includes ------------------------------------------------------------------*/
//cstat -MISRAC2012-Rule-21.1
#include "main.h" //cstat !MISRAC2012-Rule-21.1
//cstat +MISRAC2012-Rule-21.1
#include "parameters_conversion.h"

#include "r3_1_g4xx_pwm_curr_fdbk.h"

#include "mp_self_com_ctrl.h"
#include "mp_one_touch_tuning.h"
/* USER CODE BEGIN Additional include */

/* USER CODE END Additional include */

#define FREQ_RATIO 1                /* Dummy value for single drive */
#define FREQ_RELATION HIGHEST_FREQ  /* Dummy value for single drive */

/**
  * @brief  Current sensor parameters Motor 1 - three shunt - G4
  */
const R3_1_Params_t R3_1_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio       = FREQ_RATIO,
  .IsHigherFreqTim = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx            = ADC1,

  .ADCConfig = {
                 (7U << ADC_JSQR_JSQ1_Pos)
               | (6U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_TIM1_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(1U << ADC_JSQR_JSQ1_Pos)
               | (6U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_TIM1_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(1U << ADC_JSQR_JSQ1_Pos)
               | (6U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_TIM1_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(1U << ADC_JSQR_JSQ1_Pos)
               | (7U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_TIM1_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(1U << ADC_JSQR_JSQ1_Pos)
               | (7U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_TIM1_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(7U << ADC_JSQR_JSQ1_Pos)
               | (6U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_TIM1_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
               },

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .Tafter            = TW_AFTER,
  .Tbefore           = TW_BEFORE_R3_1,
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME)/2u,
  .TIMx               = TIM1,

/* Internal OPAMP common settings --------------------------------------------*/
  .OPAMPParams     = MC_NULL,
/* Internal COMP settings ----------------------------------------------------*/

  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t) 0,
  .DAC_Channel_OCPB      = (uint32_t) 0,
  .DAC_Channel_OCPC      = (uint32_t) 0,

  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t) 0,

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold =  0,
  .DAC_OVP_Threshold =  23830,

};

/*** Motor Profiler ***/

SCC_Params_t SCC_Params =
{
  {
    .FrequencyHz = TF_REGULATION_RATE,
  },
  .fRshunt                 = RSHUNT,
  .fAmplificationGain      = AMPLIFICATION_GAIN,
  .fVbusConvFactor         = BUS_VOLTAGE_CONVERSION_FACTOR,
  .fVbusPartitioningFactor = VBUS_PARTITIONING_FACTOR,

  .fRVNK                   = (float)(RESISTOR_OFFSET),

  .fRSMeasCurrLevelMax     = (float)(DC_CURRENT_RS_MEAS),

  .hDutyRampDuration       = (uint16_t)8000,

  .hAlignmentDuration      = (uint16_t)(1000),
  .hRSDetectionDuration    = (uint16_t)(500),
  .fLdLqRatio              = (float)(LDLQ_RATIO),
  .fCurrentBW              = (float)(CURRENT_REGULATOR_BANDWIDTH),
  .bPBCharacterization     = false,

  .wNominalSpeed           = MOTOR_MAX_SPEED_RPM,
  .hPWMFreqHz              = (uint16_t)(PWM_FREQUENCY),
  .bFOCRepRate             = (uint8_t)(REGULATION_EXECUTION_RATE),
  .fMCUPowerSupply         = (float)ADC_REFERENCE_VOLTAGE,
  .IThreshold              = I_THRESHOLD

};

OTT_Params_t OTT_Params =
{
  {
    .FrequencyHz        = MEDIUM_FREQUENCY_TASK_RATE,         /*!< Frequency expressed in Hz at which the user
                                                                  clocks the OTT calling OTT_MF method */
  },
  .fBWdef               = (float)(SPEED_REGULATOR_BANDWIDTH), /*!< Default bandwidth of speed regulator.*/
  .fMeasWin             = 1.0f,                               /*!< Duration of measurement window for speed and
                                                                  current Iq, expressed in seconds.*/
  .bPolesPairs          = POLE_PAIR_NUM,                      /*!< Number of motor poles pairs.*/
  .hMaxPositiveTorque   = (int16_t)NOMINAL_CURRENT,           /*!< Maximum positive value of motor
                                                                   torque. This value represents
                                                                   actually the maximum Iq current
                                                                   expressed in digit.*/
  .fCurrtRegStabTimeSec = 10.0f,                              /*!< Current regulation stabilization time in seconds.*/
  .fOttLowSpeedPerc     = 0.6f,                               /*!< OTT lower speed percentage.*/
  .fOttHighSpeedPerc    = 0.8f,                               /*!< OTT higher speed percentage.*/
  .fSpeedStabTimeSec    = 20.0f,                              /*!< Speed stabilization time in seconds.*/
  .fTimeOutSec          = 10.0f,                              /*!< Timeout for speed stabilization.*/
  .fSpeedMargin         = 0.90f,                              /*!< Speed margin percentage to validate speed ctrl.*/
  .wNominalSpeed        = MOTOR_MAX_SPEED_RPM,                /*!< Nominal speed set by the user expressed in RPM.*/
  .spdKp                = MP_KP,                              /*!< Initial KP factor of the speed regulator to be
                                                                   tuned.*/
  .spdKi                = MP_KI,                              /*!< Initial KI factor of the speed regulator to be
                                                                   tuned.*/
  .spdKs                = 0.1f,                               /*!< Initial antiwindup factor of the speed regulator to
                                                                   be tuned.*/
  .fRshunt              = (float)(RSHUNT),                    /*!< Value of shunt resistor.*/
  .fAmplificationGain   = (float)(AMPLIFICATION_GAIN)         /*!< Current sensing amplification gain.*/

};

ScaleParams_t scaleParams_M1 =
{
 .voltage = NOMINAL_BUS_VOLTAGE_V/(1.73205 * 32767), /* sqrt(3) = 1.73205 */
 .current = CURRENT_CONV_FACTOR_INV,
 .frequency = (1.15 * MAX_APPLICATION_SPEED_UNIT * U_RPM)/(32768* SPEED_UNIT)
};

/* USER CODE BEGIN Additional parameters */

/* USER CODE END Additional parameters */

/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/

