################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (12.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Modules/AS5147U/as5147u.c 

OBJS += \
./Modules/AS5147U/as5147u.o 

C_DEPS += \
./Modules/AS5147U/as5147u.d 


# Each subdirectory must supply rules for building sources it contributes
Modules/AS5147U/%.o Modules/AS5147U/%.su Modules/AS5147U/%.cyclo: ../Modules/AS5147U/%.c Modules/AS5147U/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DARM_MATH_CM4 -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../../Inc -I"C:/Users/Plutonium/MyProjects/GuardDog/stm32/MotorDrive/TorqueMode-10VV/STM32CubeIDE/Modules" -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/Any/Inc -I../../../MCSDK_v6.2.0-Full/MotorControl/MCSDK/MCLib/F4xx/Inc -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -I../../Drivers/CMSIS/DSP/Include -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Modules-2f-AS5147U

clean-Modules-2f-AS5147U:
	-$(RM) ./Modules/AS5147U/as5147u.cyclo ./Modules/AS5147U/as5147u.d ./Modules/AS5147U/as5147u.o ./Modules/AS5147U/as5147u.su

.PHONY: clean-Modules-2f-AS5147U

