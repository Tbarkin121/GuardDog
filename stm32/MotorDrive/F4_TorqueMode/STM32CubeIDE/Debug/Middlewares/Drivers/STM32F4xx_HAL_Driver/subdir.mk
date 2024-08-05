################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (12.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
OBJS += \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc_ex.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_cortex.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma_ex.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_exti.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ex.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ramfunc.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_gpio.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr_ex.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc_ex.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim_ex.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_uart.o \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_ll_adc.o 

C_DEPS += \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc_ex.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_cortex.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma_ex.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_exti.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ex.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ramfunc.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_gpio.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr_ex.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc_ex.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim_ex.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_uart.d \
./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_ll_adc.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_adc.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc_ex.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_adc_ex.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_cortex.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_cortex.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_dma.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma_ex.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_dma_ex.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_exti.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_exti.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_flash.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ex.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_flash_ex.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ramfunc.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_flash_ramfunc.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_gpio.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_gpio.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_pwr.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr_ex.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_pwr_ex.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_rcc.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc_ex.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_rcc_ex.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_tim.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim_ex.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_tim_ex.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_uart.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_uart.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_ll_adc.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_ll_adc.c Middlewares/Drivers/STM32F4xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-Drivers-2f-STM32F4xx_HAL_Driver

clean-Middlewares-2f-Drivers-2f-STM32F4xx_HAL_Driver:
	-$(RM) ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc_ex.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc_ex.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc_ex.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_adc_ex.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_cortex.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_cortex.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_cortex.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_cortex.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma_ex.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma_ex.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma_ex.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_dma_ex.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_exti.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_exti.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_exti.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_exti.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ex.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ex.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ex.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ex.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ramfunc.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ramfunc.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ramfunc.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_flash_ramfunc.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_gpio.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_gpio.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_gpio.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_gpio.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr_ex.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr_ex.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr_ex.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_pwr_ex.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc_ex.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc_ex.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc_ex.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_rcc_ex.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim_ex.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim_ex.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim_ex.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_tim_ex.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_uart.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_uart.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_uart.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_hal_uart.su ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_ll_adc.cyclo ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_ll_adc.d ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_ll_adc.o ./Middlewares/Drivers/STM32F4xx_HAL_Driver/stm32f4xx_ll_adc.su

.PHONY: clean-Middlewares-2f-Drivers-2f-STM32F4xx_HAL_Driver

