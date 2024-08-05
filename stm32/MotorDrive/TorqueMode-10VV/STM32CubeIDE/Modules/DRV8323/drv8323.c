/*
 * drv8323.c
 *
 *  Created on: Jul 19, 2024
 *      Author: Plutonium
 */


#include "drv8323.h"
#include "spi.h"

uint8_t DRV8323_Init(void)
{
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET); // Pull nSCS high
	HAL_Delay(1);
	return 0;
}


uint8_t Set_CSA_Gain(CSA_GAIN_t gain)
{
    // Ensure the gain value is valid (0 to 3)
    if (gain > CSA_GAIN_40_VV)
        return 1; // Invalid gain value

    // Prepare the SPI data
    DRV8323_SPI_Data_t spiData;
    CSA_ControlRegister_t csaData;
    uint16_t receivedWord;
    spiData.bits.rw = DRV8323_READ;
    spiData.bits.address = CSA_REG;
    spiData.bits.payload = 0;  // We are readings sooo.... 0s?

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);   // CS Enable
    HAL_SPI_TransmitReceive(&hspi1, (uint8_t*)&spiData.value, (uint8_t*)&receivedWord, 1, 1000);
    csaData.value = receivedWord;
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
    HAL_Delay(1);

    spiData.bits.rw = DRV8323_WRITE;
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

}

FaultStatus Get_Fault_Status()
{
	// Prepare the SPI data
	DRV8323_SPI_Data_t spiData;

	FaultStatus status;

	uint16_t receivedWord;
	spiData.bits.rw = DRV8323_READ;
	spiData.bits.address = FAULT1_REG;
	spiData.bits.payload = 0;

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);   // CS Enable
    HAL_SPI_TransmitReceive(&hspi1, (uint8_t*)&spiData.value, (uint8_t*)&receivedWord, 1, 1000);
    status.faultReg1.value = receivedWord;
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
    HAL_Delay(1);

	spiData.bits.address = FAULT2_REG;

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);   // CS Enable
    HAL_SPI_TransmitReceive(&hspi1, (uint8_t*)&spiData.value, (uint8_t*)&receivedWord, 1, 1000);
    status.faultReg2.value = receivedWord;
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
    HAL_Delay(1);

    return status;

}









