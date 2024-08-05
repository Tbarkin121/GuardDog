################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (12.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
OBJS += \
./Middlewares/Drivers/CMSIS/system_stm32f4xx.o 

C_DEPS += \
./Middlewares/Drivers/CMSIS/system_stm32f4xx.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/Drivers/CMSIS/system_stm32f4xx.o: C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/F4_TorqueMode/Src/system_stm32f4xx.c Middlewares/Drivers/CMSIS/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-Drivers-2f-CMSIS

clean-Middlewares-2f-Drivers-2f-CMSIS:
	-$(RM) ./Middlewares/Drivers/CMSIS/system_stm32f4xx.cyclo ./Middlewares/Drivers/CMSIS/system_stm32f4xx.d ./Middlewares/Drivers/CMSIS/system_stm32f4xx.o ./Middlewares/Drivers/CMSIS/system_stm32f4xx.su

.PHONY: clean-Middlewares-2f-Drivers-2f-CMSIS

