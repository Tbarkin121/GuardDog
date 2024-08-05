################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (12.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Modules/DRV8323/drv8323.c 

OBJS += \
./Modules/DRV8323/drv8323.o 

C_DEPS += \
./Modules/DRV8323/drv8323.d 


# Each subdirectory must supply rules for building sources it contributes
Modules/DRV8323/%.o Modules/DRV8323/%.su Modules/DRV8323/%.cyclo: ../Modules/DRV8323/%.c Modules/DRV8323/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I"C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/TorqueMode-10VV/STM32CubeIDE/Modules" -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Modules-2f-DRV8323

clean-Modules-2f-DRV8323:
	-$(RM) ./Modules/DRV8323/drv8323.cyclo ./Modules/DRV8323/drv8323.d ./Modules/DRV8323/drv8323.o ./Modules/DRV8323/drv8323.su

.PHONY: clean-Modules-2f-DRV8323

