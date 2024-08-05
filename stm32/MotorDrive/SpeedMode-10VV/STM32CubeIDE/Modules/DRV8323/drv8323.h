/*
 * drv8323.h
 *
 *  Created on: Jul 19, 2024
 *      Author: Plutonium
 */

#ifndef DRV8323_DRV8323_H_
#define DRV8323_DRV8323_H_

/* USER CODE BEGIN Includes */
#include <stdint.h>
/* USER CODE END Includes */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define FAULT_REG 0x0
#define FAULT2_REG 0x1
#define CONTROL_REG 0x2
#define GATE_DRIVE_HS_REG 0x3
#define GATE_DRIVE_LS_REG 0x4
#define OCP_REG 0x5
#define CSA_REG 0x6
#define FAULT_REG 0x7

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
} SPI_Data_t;

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
} FaultStatusRegister_t;


typedef union {
    struct {
        uint16_t VGS_LC : 1;  // Bit 0
        uint16_t VGS_HC : 1;  // Bit 1
        uint16_t VGS_LB : 1;  // Bit 2
        uint16_t VGS_HB : 1;  // Bit 3
        uint16_t VGS_LA : 1;  // Bit 4
        uint16_t VGS_HA : 1;  // Bit 5
        uint16_t CPUV   : 1;  // Bit 6
        uint16_t OTW    : 1;  // Bit 7
        uint16_t SC_OC  : 1;  // Bit 8
        uint16_t SB_OC  : 1;  // Bit 9
        uint16_t SA_OC  : 1;  // Bit 10
    } bits;
    uint16_t value : 11;  // 11-bit value representing the entire register
} FaultStatusRegister2_t;

typedef union {
    struct {
        uint16_t CLR_FLT   : 1; // Bit 0
        uint16_t BRAKE     : 1; // Bit 1
        uint16_t COAST     : 1; // Bit 2
        uint16_t PWM_DIR   : 1; // Bit 3
        uint16_t PWM_COM   : 1; // Bit 4
        uint16_t PWM_MODE  : 2; // Bits 5-6
        uint16_t OTW_REP   : 1; // Bit 7
        uint16_t DIS_GDF   : 1; // Bit 8
        uint16_t DIS_CPUV  : 1; // Bit 9
        uint16_t Reserved  : 1; // Bit 10
    } bits;
    uint16_t value : 11; // 11-bit value representing the entire register
} DriverControlRegister_t;


typedef union {
    struct {
        uint16_t IDRIVEN_HS  : 4; // Bits 0-3
        uint16_t IDRIVEP_HS  : 4; // Bits 4-7
        uint16_t LOCK        : 3; // Bits 8-10
    } bits;
    uint16_t value : 11; // 11-bit value representing the entire register
} GateDriveHSRegister_t;

typedef union {
    struct {
        uint16_t IDRIVEN_LS  : 4; // Bits 0-3
        uint16_t IDRIVEP_LS  : 4; // Bits 4-7
        uint16_t TDRIVE      : 2; // Bits 8-9
        uint16_t CBC         : 1; // Bit 10
    } bits;
    uint16_t value : 11; // 11-bit value representing the entire register
} GateDriveLSRegister_t;

typedef union {
    struct {
        uint16_t VDS_LVL   : 4; // Bits 0-3
        uint16_t OCP_DEG   : 2; // Bits 4-5
        uint16_t OCP_MODE  : 2; // Bits 6-7
        uint16_t DEAD_TIME : 2; // Bits 8-9
        uint16_t TRETRY    : 1; // Bit 10
    } bits;
    uint16_t value : 11; // 11-bit value representing the entire register
} OCPControlRegister_t;

typedef union {
    struct {
        uint16_t SEN_LVL    : 2; // Bits 0-1
        uint16_t CSA_CAL_C  : 1; // Bit 2
        uint16_t CSA_CAL_B  : 1; // Bit 3
        uint16_t CSA_CAL_A  : 1; // Bit 4
        uint16_t DIS_SEN    : 1; // Bit 5
        uint16_t CSA_GAIN   : 2; // Bits 6-7
        uint16_t LS_REF     : 1; // Bit 8
        uint16_t VREF_DIV   : 1; // Bit 9
        uint16_t CSA_FET    : 1; // Bit 10
    } bits;
    uint16_t value : 11; // 11-bit value representing the entire payload
} CSA_ControlRegister_t;

typedef enum {
    CSA_GAIN_5_VV = 0x00,  // 00b = 5-V/V shunt amplifier gain
    CSA_GAIN_10_VV = 0x01, // 01b = 10-V/V shunt amplifier gain
    CSA_GAIN_20_VV = 0x02, // 10b = 20-V/V shunt amplifier gain
    CSA_GAIN_40_VV = 0x03  // 11b = 40-V/V shunt amplifier gain
} CSA_GAIN_t;

/* USER CODE BEGIN PFP */
uint8_t Set_CSA_Gain(CSA_GAIN_t gain);
/* USER CODE END PFP */

/* USER CODE END PV */



#endif /* DRV8323_DRV8323_H_ */
