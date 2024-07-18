
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
#define FREQ_RATIO 1                /* Dummy value for single drive */
#define FREQ_RELATION HIGHEST_FREQ  /* Dummy value for single drive */

#include "pqd_motor_power_measurement.h"

/* USER CODE BEGIN Additional define */

/* USER CODE END Additional define */

PQD_MotorPowMeas_Handle_t PQD_MotorPowMeasM1 =
{
  .ConvFact = PQD_CONVERSION_FACTOR
};

/**
  * @brief  PI / PID Speed loop parameters Motor 1.
  */
PID_Handle_t PIDSpeedHandle_M1 =
{
  .hDefKpGain          = (int16_t)PID_SPEED_KP_DEFAULT,
  .hDefKiGain          = (int16_t)PID_SPEED_KI_DEFAULT,
  .wUpperIntegralLimit = (int32_t)IQMAX * (int32_t)SP_KIDIV,
  .wLowerIntegralLimit = -(int32_t)IQMAX * (int32_t)SP_KIDIV,
  .hUpperOutputLimit   = (int16_t)IQMAX,
  .hLowerOutputLimit   = -(int16_t)IQMAX,
  .hKpDivisor          = (uint16_t)SP_KPDIV,
  .hKiDivisor          = (uint16_t)SP_KIDIV,
  .hKpDivisorPOW2      = (uint16_t)SP_KPDIV_LOG,
  .hKiDivisorPOW2      = (uint16_t)SP_KIDIV_LOG,
  .hDefKdGain          = 0x0000U,
  .hKdDivisor          = 0x0000U,
  .hKdDivisorPOW2      = 0x0000U,
};

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
  .AlignmentCfg  = TC_ABSOLUTE_ALIGNMENT_SUPPORTED,
};

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
  .MaxPositiveTorque          = (int16_t)NOMINAL_CURRENT,
  .MinNegativeTorque          = -(int16_t)NOMINAL_CURRENT,
  .ModeDefault                = DEFAULT_CONTROL_MODE,
  .MecSpeedRefUnitDefault     = (int16_t)(DEFAULT_TARGET_SPEED_UNIT),
  .TorqueRefDefault           = (int16_t)DEFAULT_TORQUE_COMPONENT,
  .IdrefDefault               = (int16_t)DEFAULT_FLUX_COMPONENT,
};

/**
  * @brief  PWM parameters Motor 1 for one ADC.
  */
PWMC_R3_1_Handle_t PWM_Handle_M1 =
{
  {
    .pFctGetPhaseCurrents       = &R3_1_GetPhaseCurrents,
    .pFctSetADCSampPointSectX   = &R3_1_SetADCSampPointSectX,
    .pFctSetOffsetCalib         = &R3_1_SetOffsetCalib,
    .pFctGetOffsetCalib         = &R3_1_GetOffsetCalib,
    .pFctSwitchOffPwm           = &R3_1_SwitchOffPWM,
    .pFctSwitchOnPwm            = &R3_1_SwitchOnPWM,
    .pFctCurrReadingCalib       = &R3_1_CurrentReadingCalibration,
    .pFctTurnOnLowSides         = &R3_1_TurnOnLowSides,
    .pFctOCPSetReferenceVoltage = MC_NULL,

    .pFctRLDetectionModeEnable  = &R3_1_RLDetectionModeEnable,
    .pFctRLDetectionModeDisable = &R3_1_RLDetectionModeDisable,
    .pFctRLDetectionModeSetDuty = &R3_1_RLDetectionModeSetDuty,
    .pFctRLTurnOnLowSidesAndStart = &R3_1_RLTurnOnLowSidesAndStart,
    .LowSideOutputs    = (LowSideOutputsFunction_t)LOW_SIDE_SIGNALS_ENABLING,
    .pwm_en_u_port     = MC_NULL,
    .pwm_en_u_pin      = (uint16_t)0,
    .pwm_en_v_port     = MC_NULL,
    .pwm_en_v_pin      = (uint16_t)0,
    .pwm_en_w_port     = MC_NULL,
    .pwm_en_w_pin      = (uint16_t)0,
    .hT_Sqrt3                   = (PWM_PERIOD_CYCLES*SQRT3FACTOR)/16384u,
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
  .PhaseCOffset                 = 0,
  .Half_PWMPeriod               = PWM_PERIOD_CYCLES / 2u,
  .PolarizationCounter          = 0,
  .ADC_ExternalTriggerInjected  = 0,
  .ADCTriggerEdge               = 0,

  .pParams_str                  = &R3_1_ParamsM1
};

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
  .TIMx                        = TIM2,
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

/**
  * Virtual temperature sensor parameters Motor 1.
  */
NTC_Handle_t TempSensor_M1 =
{
  .bSensorType     = VIRTUAL_SENSOR,
  .hExpectedTemp_d = 555,
  .hExpectedTemp_C = M1_VIRTUAL_HEAT_SINK_TEMPERATURE_VALUE,
};

/* Bus voltage sensor value filter buffer */
static uint16_t RealBusVoltageSensorFilterBufferM1[M1_VBUS_SW_FILTER_BW_FACTOR];

/**
  * Bus voltage sensor parameters Motor 1.
  */
RegConv_t VbusRegConv_M1 =
{
    .regADC                   = ADC1,
    .channel                  = MC_ADC_CHANNEL_1,
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
  .OverVoltageThresholdLow    = OVERVOLTAGE_THRESHOLD_d,
  .OverVoltageHysteresisUpDir = true,
  .UnderVoltageThreshold      =  UNDERVOLTAGE_THRESHOLD_d,
  .aBuffer                    = RealBusVoltageSensorFilterBufferM1,
};

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
  .MaxVd     = (uint16_t)((MAX_MODULE * 950) / 1000),
};

MCI_Handle_t Mci[NBR_OF_MOTORS];
SpeednTorqCtrl_Handle_t *pSTC[NBR_OF_MOTORS]    = {&SpeednTorqCtrlM1};
NTC_Handle_t *pTemperatureSensor[NBR_OF_MOTORS] = {&TempSensor_M1};
PID_Handle_t *pPIDIq[NBR_OF_MOTORS]             = {&PIDIqHandle_M1};
PID_Handle_t *pPIDId[NBR_OF_MOTORS]             = {&PIDIdHandle_M1};
PQD_MotorPowMeas_Handle_t *pMPM[NBR_OF_MOTORS]  = {&PQD_MotorPowMeasM1};
PosCtrl_Handle_t *pPosCtrl[NBR_OF_MOTORS]       = {&PosCtrlM1};
/* USER CODE BEGIN Additional configuration */
/* USER CODE END Additional configuration */

/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
