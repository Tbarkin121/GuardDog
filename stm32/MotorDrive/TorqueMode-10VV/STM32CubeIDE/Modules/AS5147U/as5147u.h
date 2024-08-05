/*
 * as5147u.h
 *
 *  Created on: Jul 21, 2024
 *      Author: Plutonium
 */

#ifndef MODULES_AS5147U_AS5147U_H_
#define MODULES_AS5147U_AS5147U_H_


/* USER CODE BEGIN Includes */
#include <stdint.h>
/* USER CODE END Includes */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define NOP_REG 		0x0000
#define ERRFL_REG 		0x0001
#define PROG_REG		0x0003
#define DIA_REG 		0x3FF5
#define AGC_REG 		0x3FF9
#define SIN_REG 		0x3FFA
#define COS_REG 		0x3FFB
#define VEL_REG 		0x3FFC
#define MAG_REG			0x3FFD
#define ANGLEUNC_REG	0x3FFE
#define ANGLECOM_REG	0x3FFF
#define ECC_CHECKSUM	0x00D1

/* USER CODE END PD */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

typedef union {
    struct {
        uint16_t payload : 11; // 11 bits for the payload
        uint16_t address : 4;  // 4 bits for the address
        uint16_t rw : 1;       // 1 bit for the read/write flag
    } bits;
    uint16_t value; // 16-bit value representing the entire word
} AS5147U_SPI_Data_t;

typedef union {
    struct {
        uint16_t VDS_LC : 1;
        uint16_t VDS_HC : 1;
        uint16_t VDS_LB : 1;
        uint16_t VDS_HB : 1;
        uint16_t VDS_LA : 1;
        uint16_t VDS_HA : 1;
        uint16_t OTSD   : 1;
        uint16_t UVLO   : 1;
        uint16_t GDF    : 1;
        uint16_t VDS_OCP: 1;
        uint16_t FAULT  : 1;
    } bits;
    uint16_t value : 11; // 11-bit value representing the entire payload
} ERRFLRegister_t;

/* USER CODE BEGIN PFP */
uint8_t AS5147U_Init(void);
/* USER CODE END PFP */

/* USER CODE END PV */



#endif /* MODULES_AS5147U_AS5147U_H_ */
