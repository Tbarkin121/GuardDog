/*
 * as5147u.c
 *
 *  Created on: Jul 21, 2024
 *      Author: Plutonium
 */

#include "as5147u.h"
#include "spi.h"


uint8_t AS5147U_Init(void)
{
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_15, GPIO_PIN_SET); // Pull nSCS high
	HAL_Delay(1);
	return 0;
}
