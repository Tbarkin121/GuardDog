/*
 * drv8323.c
 *
 *  Created on: Jul 19, 2024
 *      Author: Plutonium
 */


#include "drv8323.h"
#include "spi.h"

uint8_t Set_CSA_Gain(CSA_GAIN_t gain)
{
    // Ensure the gain value is valid (0 to 3)
    if (gain > CSA_GAIN_40_VV)
        return 1; // Invalid gain value

    // Prepare the SPI data
    SPI_Data_t spiData;
    CSA_ControlRegister_t csaData;
    uint16_t receivedWord;
    spiData.bits.rw = 1;         // Set as read operation
    spiData.bits.address = 0x6;  // CSA Control Register
    spiData.bits.payload = 0;  // We are readings sooo.... 0s?

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);   // CS Enable
    HAL_SPI_TransmitReceive(&hspi1, (uint8_t*)&spiData.value, (uint8_t*)&receivedWord, 1, 1000);
    csaData.value = receivedWord;
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
    HAL_Delay(1);

    spiData.bits.rw = 0; // Set as write operation
    csaData.bits.CSA_GAIN = gain; // change gain settings
    spiData.bits.payload = csaData.value;

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);
    HAL_SPI_TransmitReceive(&hspi1, (uint8_t*)&spiData.value, (uint8_t*)&receivedWord, 1, 1000);
    csaData.value = receivedWord;
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
    HAL_Delay(1);

    // Checking that the write took
    HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_8);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);   // CS Enable?
    HAL_SPI_TransmitReceive(&hspi1, (uint8_t*)&spiData.value, (uint8_t*)&receivedWord, 1, 1000);
    csaData.value = receivedWord;
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
    HAL_Delay(1);

    if(csaData.bits.CSA_GAIN == gain)
    {
        return 0; // Success
    }
    else
    {
    	return 1; // Failure
    }
//	return 0;

}










