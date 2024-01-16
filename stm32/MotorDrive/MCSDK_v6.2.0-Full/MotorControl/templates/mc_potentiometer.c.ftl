/**
 ******************************************************************************
 * @file    mc_potentiometer.c
 * @author  Motor Control SDK Team, ST Microelectronics
 * @brief   This file implements potentiometer application
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
 * @ingroup mc_potentiometer
 */

/* Includes ------------------------------------------------------------------*/
#include "mc_potentiometer.h"

/** @addtogroup MCSDK
  * @{
  */

/** @defgroup mc_potentiometer MC Potentiometer
  *
  * @brief todo
  *
  * @{
  */
  
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* USER CODE BEGIN Private define */
/* Private define ------------------------------------------------------------*/

/* USER CODE END Private define */

/* Private variables----------------------------------------------------------*/

/* Performs the CPU load measure of FOC main tasks. */

/* USER CODE BEGIN Private Variables */

/* USER CODE END Private Variables */

/* Private functions ---------------------------------------------------------*/
/* USER CODE BEGIN Private Functions */

/* USER CODE END Private Functions */

/**
  * @brief  todo
  */
void PotentiometerGetPosition(Potentiometer_Handle_t *pHandle )
{
  /* Read potentiometer ADC value */
  PWMC_GetAuxAdcMeasurement(pHandle->pPWM, &pHandle->adcVal);

  /* Convert potentiometer ADC to position */
  {
    fixp30_t pos = pHandle->adcVal - FIXP30(pHandle->offset); // offset
    pos = FIXP_MPY(pos, FIXP24(pHandle->gain), 24); // gain
    pHandle->position = FIXP_sat(pos, FIXP30(1.0f), FIXP30(0.0f)) * pHandle->direction;
  }
}

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/