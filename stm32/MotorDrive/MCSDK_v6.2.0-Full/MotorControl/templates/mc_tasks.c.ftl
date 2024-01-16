<#ftl strip_whitespace = true>
<#if !MC??>
  <#if SWIPdatas??>
  <#list SWIPdatas as SWIP>
    <#if SWIP.ipName == "MotorControl">
      <#if SWIP.parameters??>
      <#assign MC = SWIP.parameters>
      <#break>
      </#if>
    </#if>
  </#list>
  </#if>
  <#if MC??>
  <#else>
  <#stop "No MotorControl SW IP data found">
  </#if>
</#if>
<#function _last_word text sep="_"><#return text?split(sep)?last></#function>
<#assign FOC = MC.M1_DRIVE_TYPE == "FOC" || MC.M2_DRIVE_TYPE == "FOC">
<#assign SIX_STEP = MC.M1_DRIVE_TYPE == "SIX_STEP" || MC.M2_DRIVE_TYPE == "SIX_STEP">
<#assign M1_ENCODER = (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
<#assign M2_ENCODER = (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>


<#-- Condition for STM32F302x8x MCU -->
<#assign CondMcu_STM32F302x8x = (McuName?? && McuName?matches("STM32F302.8.*"))>
<#-- Condition for STM32F072xxx MCU -->
<#assign CondMcu_STM32F072xxx = (McuName?? && McuName?matches("STM32F072.*"))>
<#-- Condition for STM32F446xCx or STM32F446xEx -->
<#assign CondMcu_STM32F446xCEx = (McuName?? && McuName?matches("STM32F446.(C|E).*"))>
<#-- Condition for STM32F0 Family -->
<#assign CondFamily_STM32F0 = (FamilyName?? && FamilyName=="STM32F0")>
<#-- Condition for STM32G0 Family -->
<#assign CondFamily_STM32G0 = (FamilyName?? && FamilyName=="STM32G0") >
<#-- Condition for STM32C0 Family -->
<#assign CondFamily_STM32C0 = (FamilyName?? && FamilyName=="STM32C0") >
<#-- Condition for STM32F3 Family -->
<#assign CondFamily_STM32F3 = (FamilyName?? && FamilyName == "STM32F3")>
<#-- Condition for STM32F4 Family -->
<#assign CondFamily_STM32F4 = (FamilyName?? && FamilyName == "STM32F4") >
<#-- Condition for STM32G4 Family -->
<#assign CondFamily_STM32G4 = (FamilyName?? && FamilyName == "STM32G4") >
<#-- Condition for STM32L4 Family -->
<#assign CondFamily_STM32L4 = (FamilyName?? && FamilyName == "STM32L4") >
<#-- Condition for STM32F7 Family -->
<#assign CondFamily_STM32F7 = (FamilyName?? && FamilyName == "STM32F7") >
<#-- Condition for STM32H5 Family -->
<#assign CondFamily_STM32H5 = (FamilyName?? && FamilyName == "STM32H5") >
<#-- Condition for STM32H7 Family -->
<#assign CondFamily_STM32H7 = (FamilyName?? && FamilyName == "STM32H7") >
<#-- Define some helper symbols -->
<#assign AUX_SPEED_FDBK_M1 = ((MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC"))>
<#assign AUX_SPEED_FDBK_M2 = ((MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC"))>
<#assign G4_Cut2_2_patch = CondFamily_STM32G4 >
<#assign NoInjectedChannel = (CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0 || G4_Cut2_2_patch ) >
<#assign DWT_CYCCNT_SUPPORTED = !(CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0) >

  <#if  MC.M1_SPEED_SENSOR == "STO_PLL">
    <#assign SPD_M1   = "&STO_PLL_M1">
    <#assign SPD_init_M1 = "STO_PLL_Init" >
    <#assign SPD_calcAvrgMecSpeedUnit_M1 = "STO_PLL_CalcAvrgMecSpeedUnit" >
  <#assign SPD_calcElAngle_M1 = "( void )STO_PLL_CalcElAngle" >
  <#assign SPD_calcAvergElSpeedDpp_M1 = "STO_PLL_CalcAvrgElSpeedDpp">
  <#assign SPD_clear_M1 = "STO_PLL_Clear">
  <#elseif  MC.M1_SPEED_SENSOR == "STO_CORDIC">
    <#assign SPD_M1 = "&STO_CR_M1" >
    <#assign SPD_init_M1 = "STO_CR_Init" >
    <#assign SPD_calcElAngle_M1 = "STO_CR_CalcElAngle" >
    <#assign SPD_calcAvrgMecSpeedUnit_M1 = "STO_CR_CalcAvrgMecSpeedUnit" >
    <#assign SPD_calcAvergElSpeedDpp_M1 = "STO_CR_CalcAvrgElSpeedDpp">
  <#assign SPD_clear_M1 = "STO_CR_Clear">
  <#elseif  MC.M1_SPEED_SENSOR == "HALL_SENSOR">
    <#assign SPD_M1 = "&HALL_M1" >
    <#assign SPD_init_M1 = "HALL_Init" >
    <#assign SPD_calcElAngle_M1 = "HALL_CalcElAngle" >
    <#assign SPD_calcAvrgMecSpeedUnit_M1 = "HALL_CalcAvrgMecSpeedUnit" >
  <#assign SPD_clear_M1 = "HALL_Clear">
  <#elseif (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z")>
    <#assign SPD_M1 = "&ENCODER_M1" >
  <#assign SPD_init_M1 = "ENC_Init" >
    <#assign SPD_calcElAngle_M1 = "ENC_CalcElAngle" >
    <#assign SPD_calcAvrgMecSpeedUnit_M1 = "ENC_CalcAvrgMecSpeedUnit" >
  <#assign SPD_clear_M1 = "ENC_Clear">
  <#elseif  MC.M1_SPEED_SENSOR == "SENSORLESS_ADC"  >
    <#assign SPD_M1 = "&Bemf_ADC_M1" >
  <#assign SPD_init_M1 = "BADC_Init" >
    <#assign SPD_calcElAngle_M1 = "BADC_CalcElAngle" >
    <#assign SPD_calcAvrgMecSpeedUnit_M1 = "BADC_CalcAvrgMecSpeedUnit" >
  <#assign SPD_clear_M1 = "BADC_Clear">
  </#if>
  <#if  AUX_SPEED_FDBK_M1 == true>
    <#if   MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL">
      <#assign SPD_AUX_M1 = "&STO_PLL_M1">
    <#assign SPD_aux_init_M1 = "STO_PLL_Init">
    <#assign SPD_aux_calcAvrgMecSpeedUnit_M1 ="STO_PLL_CalcAvrgMecSpeedUnit">
    <#assign SPD_aux_calcAvrgElSpeedDpp_M1 = "STO_PLL_CalcAvrgElSpeedDpp">
    <#assign SPD_aux_calcElAngle_M1 = "( void )STO_PLL_CalcElAngle">
    <#assign SPD_aux_clear_M1 = "STO_PLL_Clear">
    <#elseif  MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC">
      <#assign SPD_AUX_M1 = "&STO_CR_M1">
    <#assign SPD_aux_init_M1 = "STO_CR_Init">
    <#assign SPD_aux_calcAvrgMecSpeedUnit_M1 = "STO_CR_CalcAvrgMecSpeedUnit">
    <#assign SPD_aux_calcAvrgElSpeedDpp_M1 = "STO_CR_CalcAvrgElSpeedDpp">
    <#assign SPD_aux_calcElAngle_M1 = "STO_CR_CalcElAngle">
    <#assign SPD_aux_clear_M1 = "STO_CR_Clear">
  <#elseif  MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
      <#assign SPD_AUX_M1 = "&HALL_M1">
    <#assign SPD_aux_init_M1 = "HALL_Init" >
    <#assign SPD_aux_calcAvrgMecSpeedUnit_M1 = "HALL_CalcAvrgMecSpeedUnit">
    <#assign SPD_aux_clear_M1 = "HALL_Clear">
    <#elseif  (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
      <#assign SPD_AUX_M1 = "&ENCODER_M1">
    <#assign SPD_aux_init_M1 = "ENC_Init" >
    <#assign SPD_aux_calcAvrgMecSpeedUnit_M1 = "ENC_CalcAvrgMecSpeedUnit">
    <#assign SPD_aux_clear_M1 = "ENC_Clear">
    </#if>
  </#if>
    <#if  MC.M2_SPEED_SENSOR == "STO_PLL">
    <#assign SPD_M2   = "&STO_PLL_M2">
    <#assign SPD_init_M2 = "STO_PLL_Init" >
    <#assign SPD_calcAvrgMecSpeedUnit_M2 = "STO_PLL_CalcAvrgMecSpeedUnit" >
  <#assign SPD_calcElAngle_M2 = "( void )STO_PLL_CalcElAngle" >   /* if not sensorless then 2nd parameter is MC_NULL*/
    <#assign SPD_calcAvergElSpeedDpp_M2 = "STO_PLL_CalcAvrgElSpeedDpp">
    <#assign SPD_clear_M2 = "STO_PLL_Clear">
  <#elseif  MC.M2_SPEED_SENSOR == "STO_CORDIC">
    <#assign SPD_M2 = "&STO_CR_M2" >
    <#assign SPD_init_M2 = "STO_CR_Init" >
    <#assign SPD_calcElAngle_M2 = "STO_CR_CalcElAngle" >
    <#assign SPD_calcAvrgMecSpeedUnit_M2 = "STO_CR_CalcAvrgMecSpeedUnit" >
    <#assign SPD_calcAvergElSpeedDpp_M2 = "STO_CR_CalcAvrgElSpeedDpp">
    <#assign SPD_clear_M2 = "STO_CR_Clear">
  <#elseif  MC.M2_SPEED_SENSOR == "HALL_SENSOR">
    <#assign SPD_M2 = "&HALL_M2" >
    <#assign SPD_init_M2 = "HALL_Init" >
    <#assign SPD_calcElAngle_M2 = "HALL_CalcElAngle" >
    <#assign SPD_calcAvrgMecSpeedUnit_M2 = "HALL_CalcAvrgMecSpeedUnit" >
    <#assign SPD_clear_M2 = "HALL_Clear">
  <#elseif  (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")>
    <#assign SPD_M2 = "&ENCODER_M2" >
    <#assign SPD_init_M2 = "ENC_Init" >
    <#assign SPD_calcElAngle_M2 = "ENC_CalcElAngle" >
    <#assign SPD_calcAvrgMecSpeedUnit_M2 = "ENC_CalcAvrgMecSpeedUnit" >
    <#assign SPD_clear_M2 = "ENC_Clear">
  </#if>
  <#if  AUX_SPEED_FDBK_M2 == true>
    <#if   MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL">
      <#assign SPD_AUX_M2 = "&STO_PLL_M2">
      <#assign SPD_aux_init_M2 = "STO_PLL_Init" >
      <#assign SPD_aux_calcAvrgMecSpeedUnit_M2 ="STO_PLL_CalcAvrgMecSpeedUnit">
      <#assign SPD_aux_calcAvrgElSpeedDpp_M2 = "STO_PLL_CalcAvrgElSpeedDpp">
    <#assign SPD_aux_calcElAngle_M2 = "( void )STO_PLL_CalcElAngle">
      <#assign SPD_aux_clear_M2 = "STO_PLL_Clear">
    <#elseif  MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC">
      <#assign SPD_AUX_M2 = "&STO_CR_M2">
      <#assign SPD_aux_init_M2 = "STO_CR_Init" >
      <#assign SPD_aux_calcAvrgMecSpeedUnit_M2 = "STO_CR_CalcAvrgMecSpeedUnit">
      <#assign SPD_aux_calcAvrgElSpeedDpp_M2 = "STO_CR_CalcAvrgElSpeedDpp">
      <#assign SPD_aux_calcElAngle_M2 = "STO_CR_CalcElAngle">
      <#assign SPD_aux_clear_M2 = "STO_CR_Clear">
  <#elseif  MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
      <#assign SPD_AUX_M2 = "&HALL_M2">
      <#assign SPD_aux_init_M2 = "HALL_Init" >
      <#assign SPD_aux_calcElAngle_M2 = "HALL_CalcElAngle">
      <#assign SPD_aux_calcAvrgMecSpeedUnit_M2 = "HALL_CalcAvrgMecSpeedUnit">
      <#assign SPD_aux_clear_M2 = "HALL_Clear">
    <#elseif  (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
      <#assign SPD_AUX_M2 = "&ENCODER_M2">
      <#assign SPD_aux_init_M2 = "ENC_Init" >
      <#assign SPD_aux_calcElAngle_M2 = "ENC_CalcElAngle">
      <#assign SPD_aux_calcAvrgMecSpeedUnit_M2 = "ENC_CalcAvrgMecSpeedUnit">
      <#assign SPD_aux_clear_M2 = "ENC_Clear">
    </#if>
  </#if>
<#if FOC>
  <#if CondFamily_STM32F3 && (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
  <#assign PWM_Init = "R3_1_Init">
  <#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">
<#elseif CondFamily_STM32F3 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R1_Handle_t">
<#elseif CondFamily_STM32F3 && ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
  <#assign PWM_Init = "R3_2_Init">
  <#assign PWM_TurnOnLowSides = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_2_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_2_Handle_t">
<#elseif CondFamily_STM32F4 && (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
  <#assign PWM_Init = "R3_1_Init">
  <#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">
<#elseif CondFamily_STM32F4 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R1_Handle_t">
<#elseif CondFamily_STM32F4 &&  ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
  <#assign PWM_Init = "R3_2_Init">
  <#assign PWM_TurnOnLowSides = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_2_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_2_Handle_t">
<#elseif CondFamily_STM32F0 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">
<#elseif CondFamily_STM32F0 && (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
  <#assign PWM_Init = "R3_1_Init">
  <#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32G0 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32G0 && (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
  <#assign PWM_Init = "R3_1_Init">
  <#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32C0 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32C0 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
  <#assign PWM_Init = "R3_1_Init">
  <#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32F3 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
  <#assign PWM_Init = "ICS_Init">
  <#assign PWM_TurnOnLowSides = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="ICS_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_ICS_Handle_t">
<#elseif CondFamily_STM32F4 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
  <#assign PWM_Init = "ICS_Init">
  <#assign PWM_TurnOnLowSides = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="ICS_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_ICS_Handle_t">
<#elseif CondFamily_STM32G4 && ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
   <#assign PWM_Init = "R3_2_Init">
  <#assign PWM_TurnOnLowSides = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_2_GetPhaseCurrents">
	<#assign PWM_GetCalibStatus ="R3_2_GetCalibrationStatus">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_2_Handle_t"> 	 	
<#elseif CondFamily_STM32G4 && ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1'))>   
 	<#assign PWM_Init = "R3_1_Init">  
	<#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
	<#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
	<#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
	<#assign PWM_GetPhaseCurrents ="R3_1_GetPhaseCurrents">
	<#assign PWM_GetCalibStatus ="R3_1_GetCalibrationStatus">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32G4 && ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
   <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
	<#assign PWM_GetPhaseCurrents ="R1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R1_Handle_t">   
<#elseif CondFamily_STM32G4 && (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
   <#assign PWM_Init = "ICS_Init">
  <#assign PWM_TurnOnLowSides = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="ICS_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_ICS_Handle_t">
</#if>
<#if CondFamily_STM32L4 && ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
  <#assign PWM_Init = "R3_2_Init">
  <#assign PWM_TurnOnLowSides = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_2_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_2_Handle_t">	
<#elseif CondFamily_STM32L4 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R1L4XX_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R1_Handle_t">	
<#elseif CondFamily_STM32L4 && (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
  <#assign PWM_Init = "R3_1_Init">
  <#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32L4 &&  (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
  <#assign PWM_Init = "ICS_Init">
  <#assign PWM_TurnOnLowSides = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="ICS_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_ICS_Handle_t">	
</#if>
<#if CondFamily_STM32F7 &&  ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
  <#assign PWM_Init = "R3_2_Init">
  <#assign PWM_TurnOnLowSides = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_2_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_2_Handle_t">	
<#elseif CondFamily_STM32F7 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R1F7XX_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R1_Handle_t">	
<#elseif CondFamily_STM32F7 && (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
  <#assign PWM_Init = "R3_1_Init">
  <#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32F7 &&  (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
  <#assign PWM_Init = "ICS_Init">
  <#assign PWM_TurnOnLowSides = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="ICS_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_ICS_Handle_t">	
</#if>
<#if CondFamily_STM32H7 &&  ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
  <#assign PWM_Init = "R3_2_Init">
  <#assign PWM_TurnOnLowSides = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_2_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_2_Handle_t">	
<#elseif CondFamily_STM32H7 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init = "R1_Init">
  <#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R1_Handle_t">	
<#elseif CondFamily_STM32H7 && (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
  <#assign PWM_Init = "R3_1_Init">
  <#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="R3_1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32H7 &&  (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
  <#assign PWM_Init = "ICS_Init">
  <#assign PWM_TurnOnLowSides = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="ICS_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_ICS_Handle_t">	
</#if>
<#if CondFamily_STM32H5 &&  ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
	<#assign PWM_Init = "R1_Init"> 
	<#assign PWM_TurnOnLowSides = "R1_TurnOnLowSides">
	<#assign PWM_SwitchOn = "R1_SwitchOnPWM">
	<#assign PWM_SwitchOff = "R1_SwitchOffPWM">	
	<#assign PWM_GetPhaseCurrents ="R1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R1_Handle_t">	
<#elseif CondFamily_STM32H5 &&  (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
	<#assign PWM_Init = "R3_1_Init"> 
	<#assign PWM_TurnOnLowSides = "R3_1_TurnOnLowSides">
	<#assign PWM_SwitchOn = "R3_1_SwitchOnPWM">
	<#assign PWM_SwitchOff = "R3_1_SwitchOffPWM">	
	<#assign PWM_GetPhaseCurrents ="R3_1_GetPhaseCurrents">	
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32H5 &&  (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')>
	<#assign PWM_Init = "R3_2_Init">
	<#assign PWM_TurnOnLowSides = "R3_2_TurnOnLowSides">
	<#assign PWM_SwitchOn = "R3_2_SwitchOnPWM">
	<#assign PWM_SwitchOff = "R3_2_SwitchOffPWM">
	<#assign PWM_GetPhaseCurrents ="R3_2_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M1 ="PWMC_R3_2_Handle_t">	
<#elseif CondFamily_STM32H5 &&  (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
	<#assign PWM_Init = "ICS_Init"> 
	<#assign PWM_TurnOnLowSides = "ICS_TurnOnLowSides">
	<#assign PWM_SwitchOn = "ICS_SwitchOnPWM">
	<#assign PWM_SwitchOff = "ICS_SwitchOffPWM">	
	<#assign PWM_GetPhaseCurrents ="ICS_GetPhaseCurrents">	
  <#assign PWM_Handle_Type_M1 ="PWMC_ICS_Handle_t">	
</#if>
<#if CondFamily_STM32F3 &&  ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init_M2 = "R1_Init">
  <#assign PWM_TurnOnLowSides_M2 = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="">
  <#assign PWM_Handle_Type_M2 ="PWMC_R1_Handle_t">	
<#elseif CondFamily_STM32F3 && ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
  <#assign PWM_Init_M2 = "R3_2_Init">
  <#assign PWM_TurnOnLowSides_M2 = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="">
  <#assign PWM_Handle_Type_M2 ="PWMC_R3_2_Handle_t">	
<#elseif CondFamily_STM32F3 && (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')>
  <#assign PWM_Init_M2 = "R3_1_Init">
  <#assign PWM_TurnOnLowSides_M2 = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="R3_1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M2 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32F4 &&  ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init_M2 = "R1_Init">
  <#assign PWM_TurnOnLowSides_M2 = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="">
  <#assign PWM_Handle_Type_M2 ="PWMC_R1_Handle_t">	
<#elseif CondFamily_STM32F4 &&  ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
  <#assign PWM_Init_M2 = "R3_2_Init">
  <#assign PWM_TurnOnLowSides_M2 = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="">
  <#assign PWM_Handle_Type_M2 ="PWMC_R3_2_Handle_t">	
<#elseif CondFamily_STM32F4 && (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')>
  <#assign PWM_Init_M2 = "R3_1_Init">
  <#assign PWM_TurnOnLowSides_M2 = "R3_1_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "R3_1_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "R3_1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="R3_1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M2 ="PWMC_R3_1_Handle_t">	
<#elseif CondFamily_STM32F3 && (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
  <#assign PWM_Init_M2 = "ICS_Init">
  <#assign PWM_TurnOnLowSides_M2 = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="">
  <#assign PWM_Handle_Type_M2 ="PWMC_ICS_Handle_t">	
<#elseif CondFamily_STM32F4 && (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
  <#assign PWM_Init_M2 = "ICS_Init">
  <#assign PWM_TurnOnLowSides_M2 = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="">
  <#assign PWM_Handle_Type_M2 ="PWMC_ICS_Handle_t">	
<#elseif CondFamily_STM32G4 && ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
  <#assign PWM_Init_M2 = "R3_2_Init">
  <#assign PWM_TurnOnLowSides_M2 = "R3_2_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "R3_2_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "R3_2_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="R3_2_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M2 ="PWMC_R3_2_Handle_t"> 	 	
<#elseif CondFamily_STM32G4 && ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1'))>   
 	<#assign PWM_Init_M2 = "R3_1_Init">  
	<#assign PWM_TurnOnLowSides_M2 = "R3_1_TurnOnLowSides">
	<#assign PWM_SwitchOn_M2 = "R3_1_SwitchOnPWM">
	<#assign PWM_SwitchOff_M2 = "R3_1_SwitchOffPWM">
	<#assign PWM_GetPhaseCurrents_M2 ="R3_1_GetPhaseCurrents">
	<#assign PWM_GetCalibStatus_M2 ="R3_1_GetCalibrationStatus">
  <#assign PWM_Handle_Type_M2 ="PWMC_R3_1_Handle_t">
<#elseif CondFamily_STM32G4 && ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
  <#assign PWM_Init_M2 = "R1_Init">
  <#assign PWM_TurnOnLowSides_M2 = "R1_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "R1_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "R1_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="R1_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M2 ="PWMC_R1_Handle_t">
<#elseif CondFamily_STM32G4 && (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
  <#assign PWM_Init_M2 = "ICS_Init">
  <#assign PWM_TurnOnLowSides_M2 = "ICS_TurnOnLowSides">
  <#assign PWM_SwitchOn_M2 = "ICS_SwitchOnPWM">
  <#assign PWM_SwitchOff_M2 = "ICS_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents_M2 ="ICS_GetPhaseCurrents">
  <#assign PWM_Handle_Type_M2 ="PWMC_ICS_Handle_t"> 	
</#if>
</#if><#-- FOC -->
<#if SIX_STEP>
<#if  MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
  <#assign PWM_Init = "SixPwm_Init">
  <#assign PWM_TurnOnLowSides = "SixPwm_TurnOnLowSides">
  <#assign PWM_SwitchOn = "SixPwm_SwitchOnPWM">
  <#assign PWM_SwitchOff = "SixPwm_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="">
<#else>
  <#assign PWM_Init = "ThreePwm_Init">
  <#assign PWM_TurnOnLowSides = "ThreePwm_TurnOnLowSides">
  <#assign PWM_SwitchOn = "ThreePwm_SwitchOnPWM">
  <#assign PWM_SwitchOff = "ThreePwm_SwitchOffPWM">
  <#assign PWM_GetPhaseCurrents ="">
</#if>
</#if><#-- SIX_STEP -->
<#-- Charge Boot Cap enable condition -->
<#assign CHARGE_BOOT_CAP_ENABLING = (! MC.M1_OTF_STARTUP) || ((MC.M1_OTF_STARTUP) && (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE"))>
<#assign CHARGE_BOOT_CAP_ENABLING2 = (! MC.M2_OTF_STARTUP) || ((MC.M2_OTF_STARTUP) && (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE"))>

<#if MC.M1_OVERMODULATION == true> <#assign OVM ="_OVM"> <#else> <#assign OVM =""> </#if>
<#if MC.M2_OVERMODULATION == true> <#assign OVM2 ="_OVM"> <#else>  <#assign OVM2 =""> </#if>

/**
  ******************************************************************************
  * @file    mc_tasks.c
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file implements tasks definition
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
  */

/* Includes ------------------------------------------------------------------*/
//cstat -MISRAC2012-Rule-21.1
#include "main.h"
//cstat +MISRAC2012-Rule-21.1 
#include "mc_type.h"
#include "mc_math.h"
#include "motorcontrol.h"
#include "regular_conversion_manager.h"
<#if MC.RTOS == "FREERTOS">
#include "cmsis_os.h"
</#if><#-- MC.RTOS == "FREERTOS" -->
#include "mc_interface.h"
#include "digital_output.h"
<#if FOC>
#include "pwm_common.h"
</#if><#-- FOC -->
<#if SIX_STEP>
#include "pwm_common_sixstep.h"
</#if><#-- SIX_STEP -->
#include "mc_tasks.h"
#include "parameters_conversion.h"
<#if MC.MCP_EN == true>
#include "mcp_config.h"
</#if><#--  MC.MCP_EN == true -->
<#if MC.DEBUG_DAC_FUNCTIONALITY_EN>
#include "dac_ui.h"
</#if><#--  MC.DEBUG_DAC_FUNCTIONALITY_EN -->
#include "mc_app_hooks.h"

<#if MC.TESTENV == true>
  <#if FOC && MC.PFC_ENABLED == false >
#include "mc_testenv.h"
  </#if>
  <#if SIX_STEP>
#include "mc_testenv_6step.h"
  </#if>
</#if>

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* USER CODE BEGIN Private define */
/* Private define ------------------------------------------------------------*/
/* Un-Comment this macro define in order to activate the smooth
   braking action on over voltage */
/* #define  MC.SMOOTH_BRAKING_ACTION_ON_OVERVOLTAGE */

<#if FOC>
</#if><#-- FOC -->
#define STOPPERMANENCY_MS              ((uint16_t)400)
#define STOPPERMANENCY_MS2             ((uint16_t)400)
#define STOPPERMANENCY_TICKS           (uint16_t)((SYS_TICK_FREQUENCY * STOPPERMANENCY_MS)  / ((uint16_t)1000))
#define STOPPERMANENCY_TICKS2          (uint16_t)((SYS_TICK_FREQUENCY * STOPPERMANENCY_MS2) / ((uint16_t)1000))
/* USER CODE END Private define */

<#if MC.M1_OV_TEMPERATURE_PROT_ENABLING == true &&  MC.M1_UV_VOLTAGE_PROT_ENABLING == true
  && MC.M1_OV_VOLTAGE_PROT_ENABLING == true>
#define VBUS_TEMP_ERR_MASK (MC_OVER_VOLT| MC_UNDER_VOLT| MC_OVER_TEMP)
<#else><#-- MC.M1_OV_TEMPERATURE_PROT_ENABLING == false ||  MC.M1_UV_VOLTAGE_PROT_ENABLING == false
|| MC.M1_OV_VOLTAGE_PROT_ENABLING == false -->
  <#if MC.M1_UV_VOLTAGE_PROT_ENABLING == false>
<#assign UV_ERR = "MC_UNDER_VOLT">
  <#else><#-- MC.M1_UV_VOLTAGE_PROT_ENABLING == true -->
<#assign UV_ERR = "0">
  </#if><#--  MC.M1_UV_VOLTAGE_PROT_ENABLING == false -->
  <#if MC.M1_OV_VOLTAGE_PROT_ENABLING == false>
<#assign OV_ERR = "MC_OVER_VOLT">
  <#else><#-- MC.M1_OV_VOLTAGE_PROT_ENABLING == true -->
<#assign OV_ERR = "0">
  </#if><#-- MC.M1_OV_VOLTAGE_PROT_ENABLING == false -->
  <#if MC.M1_OV_TEMPERATURE_PROT_ENABLING == false>
<#assign OT_ERR = "MC_OVER_TEMP">
  <#else><#-- MC.M1_OV_TEMPERATURE_PROT_ENABLING == true -->
<#assign OT_ERR = "0">
  </#if><#-- MC.M1_OV_TEMPERATURE_PROT_ENABLING == false -->
#define VBUS_TEMP_ERR_MASK ~(${OV_ERR} | ${UV_ERR} | ${OT_ERR})
</#if><#-- MC.M1_OV_TEMPERATURE_PROT_ENABLING == true &&  MC.M1_UV_VOLTAGE_PROT_ENABLING == true
        && MC.M1_OV_VOLTAGE_PROT_ENABLING == true -->
<#if MC.DRIVE_NUMBER != "1">
  <#if MC.M2_OV_TEMPERATURE_PROT_ENABLING == true &&  MC.M2_UV_VOLTAGE_PROT_ENABLING == true
    && MC.M2_OV_VOLTAGE_PROT_ENABLING == true>
#define VBUS_TEMP_ERR_MASK2 (MC_OVER_VOLT| MC_UNDER_VOLT| MC_OVER_TEMP)
  <#else><#-- MC.M2_OV_TEMPERATURE_PROT_ENABLING == false ||  MC.M2_UV_VOLTAGE_PROT_ENABLING == false
         || MC.M2_OV_VOLTAGE_PROT_ENABLING == false -->
    <#if MC.M2_UV_VOLTAGE_PROT_ENABLING == false>
<#assign UV_ERR2 = "MC_UNDER_VOLT">
    <#else><#-- MC.M2_UV_VOLTAGE_PROT_ENABLING == true -->
<#assign UV_ERR2 = "0">
    </#if><#-- MC.M2_UV_VOLTAGE_PROT_ENABLING == false -->
    <#if MC.M2_OV_VOLTAGE_PROT_ENABLING == false>
<#assign OV_ERR2 = "MC_OVER_VOLT">
    <#else><#-- MC.M2_OV_VOLTAGE_PROT_ENABLING == true -->
<#assign OV_ERR2 = "0">
    </#if><#-- MC.M2_OV_VOLTAGE_PROT_ENABLING == false -->
    <#if MC.M1_OV_TEMPERATURE_PROT_ENABLING == false>
<#assign OT_ERR2 = "MC_OVER_TEMP">
    <#else><#-- MC.M1_OV_TEMPERATURE_PROT_ENABLING == true -->
<#assign OT_ERR2 = "0">
    </#if><#-- MC.M1_OV_TEMPERATURE_PROT_ENABLING == false -->
#define VBUS_TEMP_ERR_MASK2 ~(${OV_ERR2} | ${UV_ERR2} | ${OT_ERR2})
  </#if><#-- MC.M2_OV_TEMPERATURE_PROT_ENABLING == true &&  MC.M2_UV_VOLTAGE_PROT_ENABLING == true
          && MC.M2_OV_VOLTAGE_PROT_ENABLING == true -->
</#if><#-- MC.DRIVE_NUMBER > 1 -->
/* Private variables----------------------------------------------------------*/

<#if FOC>
static FOCVars_t FOCVars[NBR_OF_MOTORS];
</#if><#-- FOC -->
<#if SIX_STEP>
static SixStepVars_t SixStepVars[NBR_OF_MOTORS];
</#if><#-- SIX_STEP -->
<#if M1_ENCODER || M2_ENCODER>
static EncAlign_Handle_t *pEAC[NBR_OF_MOTORS];
</#if><#-- M1_ENCODER || M2_ENCODER -->

<#if MC.SMOOTH_BRAKING_ACTION_ON_OVERVOLTAGE == true>
  <#if MC.DRIVE_NUMBER == "1">
static uint16_t nominalBusd[1] = {0u};
static uint16_t ovthd[1] = {OVERVOLTAGE_THRESHOLD_d};
  <#else><#-- MC.DRIVE_NUMBER != 1 -->
static uint16_t nominalBusd[2] = {0u,0u};
static uint16_t ovthd[2] = {OVERVOLTAGE_THRESHOLD_d,OVERVOLTAGE_THRESHOLD_d2};
  </#if><#-- MC.DRIVE_NUMBER == 1 -->
</#if><#-- MC.SMOOTH_BRAKING_ACTION_ON_OVERVOLTAGE == true -->
static PWMC_Handle_t *pwmcHandle[NBR_OF_MOTORS];
<#if (MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE") || (MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE")>
static DOUT_handle_t *pR_Brake[NBR_OF_MOTORS];
</#if><#-- (MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE") || (MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE") -->
<#if (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true) &&  (MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES")
 || (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true)>
static DOUT_handle_t *pOCPDisabling[NBR_OF_MOTORS];
</#if><#-- (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true) &&  (MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES")
        || (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true) -->
<#if FOC>
//cstat !MISRAC2012-Rule-8.9_a
static RampExtMngr_Handle_t *pREMNG[NBR_OF_MOTORS];   /*!< Ramp manager used to modify the Iq ref
                                                    during the start-up switch over. */
</#if><#-- FOC -->
<#if MC.M1_MTPA_ENABLING == true || MC.M2_MTPA_ENABLING == true>
static MTPA_Handle_t *pMaxTorquePerAmpere[NBR_OF_MOTORS] = {<#list 1..(MC.DRIVE_NUMBER?number) as NUM>MC_NULL<#sep>,</#sep></#list>};
</#if><#-- MC.M1_MTPA_ENABLING == true || MC.M2_MTPA_ENABLING == true -->
<#if MC.DRIVE_NUMBER != "1" && (MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true)>
OpenLoop_Handle_t *pOpenLoop[2] = {MC_NULL,MC_NULL};  /* Only if M1 or M2 has OPEN LOOP */
<#elseif (MC.DRIVE_NUMBER == "1" &&  MC.M1_DBG_OPEN_LOOP_ENABLE == true)>
OpenLoop_Handle_t *pOpenLoop[1] = {MC_NULL};          /* Only if M1 has OPEN LOOP */
</#if><#-- (MC.DRIVE_NUMBER > 1 &&  MC.M2_DBG_OPEN_LOOP_ENABLE == true) -->

static uint16_t hMFTaskCounterM1 = 0; //cstat !MISRAC2012-Rule-8.9_a
static volatile uint16_t hBootCapDelayCounterM1 = ((uint16_t)0);
static volatile uint16_t hStopPermanencyCounterM1 = ((uint16_t)0);
<#if MC.DRIVE_NUMBER != "1">
static volatile uint16_t hMFTaskCounterM2 = ((uint16_t)0);
static volatile uint16_t hBootCapDelayCounterM2 = ((uint16_t)0);
static volatile uint16_t hStopPermanencyCounterM2 = ((uint16_t)0);
</#if><#-- MC.DRIVE_NUMBER > 1 -->

static volatile uint8_t bMCBootCompleted = ((uint8_t)0);

<#if DWT_CYCCNT_SUPPORTED>
  <#if MC.DBG_MCU_LOAD_MEASURE == true>
/* Performs the CPU load measure of FOC main tasks */
MC_Perf_Handle_t PerfTraces;
  </#if><#--  MC.DBG_MCU_LOAD_MEASURE == true -->
</#if><#-- DWT_CYCCNT_SUPPORTED -->

<#if (MC.M1_ICL_ENABLED == true)>
static volatile bool ICLFaultTreatedM1 = true;
</#if><#-- (MC.M1_ICL_ENABLED == true) -->
<#if (MC.M2_ICL_ENABLED == true)>
static volatile bool ICLFaultTreatedM2 = true;
</#if><#-- (MC.M2_ICL_ENABLED == true) -->


<#if CHARGE_BOOT_CAP_ENABLING == true>
#define M1_CHARGE_BOOT_CAP_TICKS          (((uint16_t)SYS_TICK_FREQUENCY * (uint16_t)${MC.M1_PWM_CHARGE_BOOT_CAP_MS}) / 1000U)
#define M1_CHARGE_BOOT_CAP_DUTY_CYCLES ((uint32_t)${MC.M1_PWM_CHARGE_BOOT_CAP_DUTY_CYCLES}\
                                      * ((uint32_t)PWM_PERIOD_CYCLES / 2U))
</#if><#-- CHARGE_BOOT_CAP_ENABLING == true -->
<#if CHARGE_BOOT_CAP_ENABLING2 == true>
#define M2_CHARGE_BOOT_CAP_TICKS         (((uint16_t)SYS_TICK_FREQUENCY * (uint16_t)${MC.M2_PWM_CHARGE_BOOT_CAP_MS}) / 1000U)
#define M2_CHARGE_BOOT_CAP_DUTY_CYCLES ((uint32_t)${MC.M2_PWM_CHARGE_BOOT_CAP_DUTY_CYCLES}\
                                      * ((uint32_t)PWM_PERIOD_CYCLES2 / 2U))
</#if><#-- CHARGE_BOOT_CAP_ENABLING2 == true -->
<#if SIX_STEP>
#define S16_90_PHASE_SHIFT             (int16_t)(65536/4)
</#if><#-- SIX_STEP -->

/* USER CODE BEGIN Private Variables */

/* USER CODE END Private Variables */

/* Private functions ---------------------------------------------------------*/
void TSK_MediumFrequencyTaskM1(void);
<#if FOC>
void FOC_Clear(uint8_t bMotor);
void FOC_InitAdditionalMethods(uint8_t bMotor);
void FOC_CalcCurrRef(uint8_t bMotor);
</#if><#-- FOC -->
void TSK_MF_StopProcessing(uint8_t motor);
MCI_Handle_t *GetMCI(uint8_t bMotor);
<#if FOC>
  <#if MC.MOTOR_PROFILER != true>
static uint16_t FOC_CurrControllerM1(void);
    <#if MC.DRIVE_NUMBER != "1">
static uint16_t FOC_CurrControllerM2(void);
    </#if><#-- MC.DRIVE_NUMBER > 1 -->
  <#else><#-- MC.MOTOR_PROFILER == true -->
bool SCC_DetectBemf( SCC_Handle_t * pHandle );
  </#if><#-- MC.MOTOR_PROFILER != true -->
</#if><#-- FOC -->
<#if SIX_STEP>
void SixStep_Clear(uint8_t bMotor);
void SixStep_InitAdditionalMethods(uint8_t bMotor);
void SixStep_CalcSpeedRef(uint8_t bMotor);
static uint16_t SixStep_StatorController(void);
</#if><#-- SIX_STEP -->
void TSK_SetChargeBootCapDelayM1(uint16_t hTickCount);
bool TSK_ChargeBootCapDelayHasElapsedM1(void);
void TSK_SetStopPermanencyTimeM1(uint16_t hTickCount);
bool TSK_StopPermanencyTimeHasElapsedM1(void);
<#if MC.M1_ON_OVER_VOLTAGE == "TURN_OFF_PWM" || MC.M2_ON_OVER_VOLTAGE == "TURN_OFF_PWM">
void TSK_SafetyTask_PWMOFF(uint8_t motor);
</#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_OFF_PWM" || MC.M2_ON_OVER_VOLTAGE == "TURN_OFF_PWM" -->
<#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
void TSK_SafetyTask_RBRK(uint8_t motor);
</#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->
<#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES">
void TSK_SafetyTask_LSON(uint8_t motor);
</#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" -->
<#if MC.DRIVE_NUMBER != "1">
void TSK_MediumFrequencyTaskM2(void);
void TSK_SetChargeBootCapDelayM2(uint16_t hTickCount);
bool TSK_ChargeBootCapDelayHasElapsedM2(void);
void TSK_SetStopPermanencyTimeM2(uint16_t SysTickCount);
bool TSK_StopPermanencyTimeHasElapsedM2(void);

#define FOC_ARRAY_LENGTH 2
static uint8_t FOC_array[FOC_ARRAY_LENGTH]={0, 0};
static uint8_t FOC_array_head = 0; /* Next obj to be executed */
static uint8_t FOC_array_tail = 0; /* Last arrived */
</#if><#-- MC.DRIVE_NUMBER > 1 -->
<#if MC.PFC_ENABLED == true>
void PFC_Scheduler(void);
</#if><#-- MC.PFC_ENABLED == true -->

<#if MC.EXAMPLE_SPEEDMONITOR == true>
/****************************** USE ONLY FOR SDK 4.0 EXAMPLES *************/
void ARR_TIM5_update(SpeednPosFdbk_Handle_t pSPD);
/**************************************************************************/
</#if><#-- MC.EXAMPLE_SPEEDMONITOR == true -->
/* USER CODE BEGIN Private Functions */

/* USER CODE END Private Functions */
/**
  * @brief  It initializes the whole MC core according to user defined
  *         parameters.
  * @param  pMCIList pointer to the vector of MCInterface objects that will be
  *         created and initialized. The vector must have length equal to the
  *         number of motor drives.
  */
__weak void MCboot( MCI_Handle_t* pMCIList[NBR_OF_MOTORS] )
{
  /* USER CODE BEGIN MCboot 0 */

  /* USER CODE END MCboot 0 */
  
  if (MC_NULL == pMCIList)
  {
    /* Nothing to do */
  }
  else
  {
<#if MC.TESTENV == true && FOC && MC.PFC_ENABLED == false>
    mc_testenv_init();
</#if><#-- MC.TESTENV == true && FOC && MC.PFC_ENABLED == false -->
   
<#if MC.USE_STGAP1S>
    /**************************************/
    /*    STGAP1AS initialization         */
    /**************************************/
    if (false == GAP_Configuration(&STGAP_M1))
    {
      MCI_FaultProcessing(&Mci[M1], MC_SW_ERROR, 0);
    }
    else
    {
      /* Nothing to do */
    }
</#if><#-- MC.USE_STGAP1S -->

    bMCBootCompleted = (uint8_t )0;
<#if MC.M1_MTPA_ENABLING == true>
    pMaxTorquePerAmpere[M1] = &MTPARegM1;
</#if><#-- MC.M1_MTPA_ENABLING == true -->
<#if MC.M1_HW_OV_CURRENT_PROT_BYPASS == true &&  MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES">
    pOCPDisabling[M1] = &DOUT_OCPDisablingParamsM1;
    DOUT_SetOutputState(pOCPDisabling[M1],INACTIVE);
</#if><#-- MC.M1_HW_OV_CURRENT_PROT_BYPASS == true &&  MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" -->
<#if MC.DRIVE_NUMBER != "1">
  <#if MC.M2_MTPA_ENABLING == true>
    pMaxTorquePerAmpere[M2] = &MTPARegM2;
  </#if><#-- MC.M2_MTPA_ENABLING == true -->
  <#if MC.M2_HW_OV_CURRENT_PROT_BYPASS == true &&  MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES">
    pOCPDisabling[M2] = &DOUT_OCPDisablingParamsM2;
    DOUT_SetOutputState(pOCPDisabling[M2],INACTIVE);
  </#if><#-- MC.M2_HW_OV_CURRENT_PROT_BYPASS == true &&  MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" == true -->
</#if><#-- MC.DRIVE_NUMBER > 1 -->

    /**********************************************************/
    /*    PWM and current sensing component initialization    */
    /**********************************************************/
    pwmcHandle[M1] = &PWM_Handle_M1._Super;
    ${PWM_Init}(&PWM_Handle_M1);
<#if MC.DRIVE_NUMBER != "1">
    pwmcHandle[M2] = &PWM_Handle_M2._Super;
    ${PWM_Init_M2}(&PWM_Handle_M2);
</#if><#-- MC.DRIVE_NUMBER > 1 -->
<#if MC.MCP_OVER_UART_A_EN>
    ASPEP_start(&aspepOverUartA);
</#if><#-- MC.MCP_OVER_UART_A_EN -->
<#if MC.MCP_OVER_UART_B_EN>
    ASPEP_start(&aspepOverUartB);
</#if><#-- MC.MCP_OVER_UART_B_EN -->
<#if MC.MCP_OVER_STLNK_EN>
    STLNK_init(&STLNK);
</#if><#-- MC.MCP_OVER_STLNK_EN -->
<#if SIX_STEP && MC.DRIVE_MODE == "CM">
    CRM_Init(&CurrentRef_M1);
</#if><#-- SIX_STEP && MC.DRIVE_MODE == "CM -->

    /* USER CODE BEGIN MCboot 1 */
  
    /* USER CODE END MCboot 1 */

<#if FOC>
  <#if !CondFamily_STM32F0 && !CondFamily_STM32G0 && !CondFamily_STM32C0>
    /**************************************/
    /*    Start timers synchronously      */
    /**************************************/
    startTimers();
  </#if><#-- !CondFamily_STM32F0 && !CondFamily_STM32G -->
</#if><#-- FOC -->
  
    /******************************************************/
    /*   PID component initialization: speed regulation   */
    /******************************************************/
    PID_HandleInit(&PIDSpeedHandle_M1);
    
    /******************************************************/
    /*   Main speed sensor component initialization       */
    /******************************************************/
    ${SPD_init_M1} (${SPD_M1});

<#if FOC>
  <#if (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z")>
    /******************************************************/
    /*   Main encoder alignment component initialization  */
    /******************************************************/
    EAC_Init(&EncAlignCtrlM1,pSTC[M1],&VirtualSpeedSensorM1,${SPD_M1});
    pEAC[M1] = &EncAlignCtrlM1;
  </#if><#-- (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") -->
  
  <#if MC.M1_POSITION_CTRL_ENABLING == true>
    /******************************************************/
    /*   Position Control component initialization        */
    /******************************************************/
    PID_HandleInit(&PID_PosParamsM1);
    TC_Init(&PosCtrlM1, &PID_PosParamsM1, &SpeednTorqCtrlM1, &ENCODER_M1);
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true -->
</#if><#-- FOC -->
  
    /******************************************************/
    /*   Speed & torque component initialization          */
    /******************************************************/
    STC_Init(pSTC[M1],&PIDSpeedHandle_M1, ${SPD_M1}._Super);

<#if FOC>
   <#if AUX_SPEED_FDBK_M1  == true>
    /******************************************************/
    /*   Auxiliary speed sensor component initialization  */
    /******************************************************/
    ${SPD_aux_init_M1} (${SPD_AUX_M1});
    
     <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
    /***********************************************************/
    /*   Auxiliary encoder alignment component initialization  */
    /***********************************************************/
    EAC_Init(&EncAlignCtrlM1,pSTC[M1],&VirtualSpeedSensorM1,${SPD_AUX_M1});
    pEAC[M1] = &EncAlignCtrlM1;
     </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z") -->
   </#if><#-- AUX_SPEED_FDBK_M1  == true -->
</#if><#-- FOC -->

<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || M1_ENCODER
  || (MC.M1_SPEED_SENSOR == "SENSORLESS_ADC")>
    /****************************************************/
    /*   Virtual speed sensor component initialization  */
    /****************************************************/ 
    VSS_Init(&VirtualSpeedSensorM1);
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || M1_ENCODER || (MC.M1_SPEED_SENSOR == "SENSORLESS_ADC") -->

<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
    /**************************************/
    /*   Rev-up component initialization  */
    /**************************************/
  <#if FOC>
    RUC_Init(&RevUpControlM1, pSTC[M1], &VirtualSpeedSensorM1, &STO_M1, pwmcHandle[M1]);
  </#if><#-- FOC -->
  <#if SIX_STEP>
    RUC_Init(&RevUpControlM1,pSTC[M1],&VirtualSpeedSensorM1);
  </#if><#-- SIX_STEP -->
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")
        || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->

<#if FOC>
    /********************************************************/
    /*   PID component initialization: current regulation   */
    /********************************************************/
    PID_HandleInit(&PIDIqHandle_M1);
    PID_HandleInit(&PIDIdHandle_M1);
</#if><#-- FOC -->

<#if MC.M1_BUS_VOLTAGE_READING == true>
    /********************************************************/
    /*   Bus voltage sensor component initialization        */
    /********************************************************/
    (void)RCM_RegisterRegConv(&VbusRegConv_M1);
    RVBS_Init(&BusVoltageSensor_M1);
<#else><#-- MC.M1_BUS_VOLTAGE_READING == false -->
    /**********************************************************/
    /*   Virtual bus voltage sensor component initialization  */
    /**********************************************************/
    VVBS_Init(&BusVoltageSensor_M1);
</#if><#-- MC.M1_BUS_VOLTAGE_READING == true -->

<#if FOC>
    /*************************************************/
    /*   Power measurement component initialization  */
    /*************************************************/
    pMPM[M1]->pVBS = &(BusVoltageSensor_M1._Super);
    pMPM[M1]->pFOCVars = &FOCVars[M1];
  <#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
    pR_Brake[M1] = &R_BrakeParamsM1;
    DOUT_SetOutputState(pR_Brake[M1],INACTIVE);
  </#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->
</#if><#-- FOC -->

    /*******************************************************/
    /*   Temperature measurement component initialization  */
    /*******************************************************/
<#if (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true)>
    (void)RCM_RegisterRegConv(&TempRegConv_M1);
</#if>
    NTC_Init(&TempSensor_M1);

<#if FOC>
  <#if MC.M1_FLUX_WEAKENING_ENABLING == true>
    /*******************************************************/
    /*   Flux weakening component initialization           */
    /*******************************************************/
    PID_HandleInit(&PIDFluxWeakeningHandle_M1);
    FW_Init(pFW[M1],&PIDSpeedHandle_M1,&PIDFluxWeakeningHandle_M1);
  </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING == true -->

  <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true>
    /*******************************************************/
    /*   Feed forward component initialization             */
    /*******************************************************/
    FF_Init(pFF[M1],&(BusVoltageSensor_M1._Super),pPIDId[M1],pPIDIq[M1]);
  </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true -->

  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
    OL_Init(&OpenLoop_ParamsM1, &VirtualSpeedSensorM1);     /* Only if M1 has open loop */
    pOpenLoop[M1] = &OpenLoop_ParamsM1;
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->

    pREMNG[M1] = &RampExtMngrHFParamsM1;
    REMNG_Init(pREMNG[M1]);

  <#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
    SCC.pPWMC = pwmcHandle[M1];
    SCC.pVBS = &BusVoltageSensor_M1;
    SCC.pFOCVars = &FOCVars[M1];
    SCC.pMCI = &Mci[M1];
    SCC.pVSS = &VirtualSpeedSensorM1;
    SCC.pCLM = &CircleLimitationM1;
    SCC.pPIDIq = pPIDIq[M1];
    SCC.pPIDId = pPIDId[M1];
    SCC.pRevupCtrl = &RevUpControlM1;
    SCC.pSTO = &STO_PLL_M1;
    SCC.pSTC = &SpeednTorqCtrlM1;
    SCC.pOTT = &OTT;
    <#if MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
    SCC.pHT = &HT;
    <#else><#-- MC.M1_AUXILIARY_SPEED_SENSOR != "HALL_SENSOR" -->
    SCC.pHT = MC_NULL;
    </#if><#-- MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR" -->
    SCC_Init(&SCC);

    OTT.pSpeedSensor = &STO_PLL_M1._Super;
    OTT.pFOCVars = &FOCVars[M1];
    OTT.pPIDSpeed = &PIDSpeedHandle_M1;
    OTT.pSTC = &SpeednTorqCtrlM1;
    OTT_Init(&OTT);
  </#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->

    FOC_Clear(M1);
    FOCVars[M1].bDriveInput = EXTERNAL;
    FOCVars[M1].Iqdref = STC_GetDefaultIqdref(pSTC[M1]);
    FOCVars[M1].UserIdref = STC_GetDefaultIqdref(pSTC[M1]).d;
  <#if MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true>
    <#if MC.M1_POSITION_CTRL_ENABLING == true>
    MCI_Init(&Mci[M1], pSTC[M1], &FOCVars[M1], pPosCtrl[M1], pwmcHandle[M1]);
    <#else><#-- MC.M1_POSITION_CTRL_ENABLING == false -->
    MCI_Init(&Mci[M1], pSTC[M1], &FOCVars[M1], MC_NULL,pwmcHandle[M1]);
    </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true -->
  <#else><#-- MC.M1_POSITION_CTRL_ENABLING == false || MC.M2_POSITION_CTRL_ENABLING == false -->
    MCI_Init(&Mci[M1], pSTC[M1], &FOCVars[M1],pwmcHandle[M1] );
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true -->
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true>
    Mci[M1].pVSS =  &VirtualSpeedSensorM1;
    MCI_SetSpeedMode(&Mci[M1]);
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
   Mci[M1].pScale = &scaleParams_M1;
  </#if><#-- FOC -->
   


<#if SIX_STEP>
    SixStep_Clear(M1);
    SixStepVars[M1].bDriveInput = EXTERNAL;
    MCI_Init(&Mci[M1], pSTC[M1], &SixStepVars[M1], pwmcHandle[M1] );
</#if><#-- SIX_STEP -->
<#if MC.M1_DEFAULT_CONTROL_MODE == "STC_TORQUE_MODE">
    MCI_ExecTorqueRamp(&Mci[M1], STC_GetDefaultIqdref(pSTC[M1]).q, 0);
<#else><#-- MC.M1_DEFAULT_CONTROL_MODE != "STC_TORQUE_MODE" -->
    MCI_ExecSpeedRamp(&Mci[M1],
    STC_GetMecSpeedRefUnitDefault(pSTC[M1]),0); /* First command to STC */
</#if><#-- MC.M1_DEFAULT_CONTROL_MODE == "STC_TORQUE_MODE"-->
<#if DWT_CYCCNT_SUPPORTED>
  <#if MC.DBG_MCU_LOAD_MEASURE == true>
    Mci[M1].pPerfMeasure = &PerfTraces;
    MC_Perf_Measure_Init(&PerfTraces);
  </#if><#-- MC.DBG_MCU_LOAD_MEASURE == true -->
</#if><#-- DWT_CYCCNT_SUPPORTED -->
    pMCIList[M1] = &Mci[M1];

<#if MC.MOTOR_PROFILER && (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
    HT.pOTT = &OTT;
    HT.pMCI = &Mci[M1];
    HT.pHALL_M1 = &HALL_M1;
    HT.pSTO_PLL_M1 = &STO_PLL_M1;
    HT_Init(&HT, false);
</#if><#-- MC.MOTOR_PROFILER && (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") -->
   
<#if MC.DRIVE_NUMBER != "1">
    /******************************************************/
    /*   Motor 2 features initialization                  */
    /******************************************************/
    
    /******************************************************/
    /*   PID component initialization: speed regulation   */
    /******************************************************/  
    PID_HandleInit(&PIDSpeedHandle_M2);
    
    /***********************************************************/
    /*   Main speed  sensor initialization: speed regulation   */
    /***********************************************************/ 
    ${SPD_init_M2} (${SPD_M2});
    
  <#if (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")>
    /******************************************************/
    /*   Main encoder alignment component initialization  */
    /******************************************************/  
    EAC_Init(&EncAlignCtrlM2,pSTC[M2],&VirtualSpeedSensorM2,${SPD_M2});
    pEAC[M2] = &EncAlignCtrlM2;
  </#if><#-- (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") -->
  
  <#if MC.M2_POSITION_CTRL_ENABLING == true>
    /******************************************************/
    /*   Position Control component initialization        */
    /******************************************************/
    PID_HandleInit(&PID_PosParamsM2);
    TC_Init(&PosCtrlM2, &PID_PosParamsM2, &SpeednTorqCtrlM2, ${SPD_M2});
  </#if><#-- MC.M2_POSITION_CTRL_ENABLING == true -->

    /******************************************************/
    /*   Speed & torque component initialization          */
    /******************************************************/
    STC_Init(pSTC[M2], &PIDSpeedHandle_M2, ${SPD_M2}._Super);

  <#if AUX_SPEED_FDBK_M2>
    /***********************************************************/
    /*   Auxiliary speed sensor component initialization       */
    /***********************************************************/ 
    ${SPD_aux_init_M2} (${SPD_AUX_M2});

    <#if (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
    /***********************************************************/
    /*   Auxiliary encoder alignment component initialization  */
    /***********************************************************/ 
    EAC_Init(&EncAlignCtrlM2,pSTC[M2],&VirtualSpeedSensorM2,${SPD_AUX_M2});
    pEAC[M2] = &EncAlignCtrlM2;
    </#if><#-- (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z") -->
  </#if><#-- AUX_SPEED_FDBK_M2 -->

  <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || M2_ENCODER>
    /****************************************************/
    /*   Virtual speed sensor component initialization  */
    /****************************************************/ 
    VSS_Init(&VirtualSpeedSensorM2);
  </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || M2_ENCODER -->

  <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")>
    /****************************************************/
    /*   Rev-up component initialization                */
    /****************************************************/ 
    RUC_Init(&RevUpControlM2, pSTC[M2], &VirtualSpeedSensorM2, &STO_M2, pwmcHandle[M2]); /* Only if sensorless */
  </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->

    /********************************************************/
    /*   PID component initialization: current regulation   */
    /********************************************************/
    PID_HandleInit(&PIDIqHandle_M2);
    PID_HandleInit(&PIDIdHandle_M2);

  <#if MC.M2_BUS_VOLTAGE_READING == true>
    /**********************************************************/
    /*   Bus voltage sensor component initialization          */
    /**********************************************************/
    (void)RCM_RegisterRegConv(&VbusRegConv_M2);
    RVBS_Init(&BusVoltageSensor_M2);
  <#else><#-- MC.M2_BUS_VOLTAGE_READING == false -->
    /**********************************************************/
    /*   Virtual bus voltage sensor component initialization  */
    /**********************************************************/
    VVBS_Init(&BusVoltageSensor_M2);
  </#if><#-- MC.M2_BUS_VOLTAGE_READING == true -->

    /*************************************************/
    /*   Power measurement component initialization  */
    /*************************************************/
    pMPM[M2]->pVBS = &(BusVoltageSensor_M2._Super);
    pMPM[M2]->pFOCVars = &FOCVars[M2];
  <#if MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
    pR_Brake[M2] = &R_BrakeParamsM2;
    DOUT_SetOutputState(pR_Brake[M2],INACTIVE);
  </#if><#-- MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->

    /*******************************************************/
    /*   Temperature measurement component initialization  */
    /*******************************************************/
<#if (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true)>
    (void)RCM_RegisterRegConv(&TempRegConv_M2);
</#if>
    NTC_Init(&TempSensor_M2);

  <#if MC.M2_FLUX_WEAKENING_ENABLING == true>
    /*************************************************/
    /*   Flux weakening component initialization     */
    /*************************************************/
    PID_HandleInit(&PIDFluxWeakeningHandle_M2);
    FW_Init(pFW[M2], &PIDSpeedHandle_M2, &PIDFluxWeakeningHandle_M2); /* Only if M2 has FW */
  </#if><#-- MC.M2_FLUX_WEAKENING_ENABLING == true -->

  <#if MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true>
    /*************************************************/
    /*   Feed forward component initialization       */
    /*************************************************/
    FF_Init(pFF[M2], &(BusVoltageSensor_M2._Super), pPIDId[M2], pPIDIq[M2]); /* Only if M2 has FF */
  </#if><#-- MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true -->

  <#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
    OL_Init(&OpenLoop_ParamsM2, &VirtualSpeedSensorM2._Super); /* Only if M2 has open loop */
    pOpenLoop[M2] = &OpenLoop_ParamsM2;
  </#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->

    pREMNG[M2] = &RampExtMngrHFParamsM2;
    REMNG_Init(pREMNG[M2]);
    FOC_Clear(M2);
    FOCVars[M2].bDriveInput = EXTERNAL;
    FOCVars[M2].Iqdref = STC_GetDefaultIqdref(pSTC[M2]);
    FOCVars[M2].UserIdref = STC_GetDefaultIqdref(pSTC[M2]).d;
  <#if MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true>
    <#if MC.M2_POSITION_CTRL_ENABLING == true>
    MCI_Init(&Mci[M2], pSTC[M2], &FOCVars[M2], pPosCtrl[M2]);
    <#else><#-- MC.M2_POSITION_CTRL_ENABLING == false -->
    MCI_Init(&Mci[M2], pSTC[M2], &FOCVars[M2], MC_NULL);
    </#if><#-- MC.M2_POSITION_CTRL_ENABLING == true -->
  <#else><#-- MC.M1_POSITION_CTRL_ENABLING == false && MC.M2_POSITION_CTRL_ENABLING == false -->
    MCI_Init(&Mci[M2], pSTC[M2], &FOCVars[M2],pwmcHandle[M2] );
  </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true || MC.M2_POSITION_CTRL_ENABLING == true -->
  <#if MC.M2_DEFAULT_CONTROL_MODE == 'STC_TORQUE_MODE'>
    MCI_ExecTorqueRamp(&Mci[M2], STC_GetDefaultIqdref(pSTC[M2]).q, 0);
<#else><#-- MC.M2_DEFAULT_CONTROL_MODE != "STC_TORQUE_MODE" -->
    MCI_ExecSpeedRamp(&Mci[M2],
    STC_GetMecSpeedRefUnitDefault(pSTC[M2]),0); /* First command to STC */

  </#if><#-- MC.M2_DEFAULT_CONTROL_MODE == 'STC_TORQUE_MODE -->
    pMCIList[M2] = &Mci[M2];
<#if FOC >
    Mci[M2].pScale = &scaleParams_M2;
</#if><#-- FOC -->


</#if><#-- MC.DRIVE_NUMBER > !1 -->

<#if MC.M1_ICL_ENABLED == true>
    ICL_Init(&ICL_M1, &(BusVoltageSensor_M1._Super), &ICLDOUTParamsM1);
    Mci[M1].State = ICLWAIT;
</#if><#-- MC.M1_ICL_ENABLED == true -->
<#if MC.M2_ICL_ENABLED == true>
    ICL_Init(&ICL_M2, &(BusVoltageSensor_M2._Super), &ICLDOUTParamsM2);
    Mci[M2].State = ICLWAIT;
</#if><#-- MC.M2_ICL_ENABLED == true -->

<#if MC.PFC_ENABLED == true>
    /* Initializing the PFC component */
    PFC_Init(&PFC);
</#if><#-- MC.PFC_ENABLED == true -->

<#if MC.DEBUG_DAC_FUNCTIONALITY_EN>
    DAC_Init(&DAC_Handle);
</#if><#-- MC.PFC_ENABLED == true -->

<#if MC.STSPIN32G4 == true>
    /*************************************************/
    /*   STSPIN32G4 driver component initialization  */
    /*************************************************/
    STSPIN32G4_init(&HdlSTSPING4);
    STSPIN32G4_reset(&HdlSTSPING4);
    STSPIN32G4_setVCC(&HdlSTSPING4, (STSPIN32G4_confVCC){.voltage = _12V,
                                                         .useNFAULT = true,
                                                         .useREADY = false });
    STSPIN32G4_setVDSP(&HdlSTSPING4, (STSPIN32G4_confVDSP){.deglitchTime = _4us,
                                                           .useNFAULT = true });
    STSPIN32G4_clearFaults(&HdlSTSPING4);
</#if><#-- MC.STSPIN32G4 == true -->

    /* Applicative hook in MCBoot() */
    MC_APP_BootHook();

    /* USER CODE BEGIN MCboot 2 */

    /* USER CODE END MCboot 2 */
  
    bMCBootCompleted = 1U;
  }
}

/**
 * @brief Runs all the Tasks of the Motor Control cockpit
 *
 * This function is to be called periodically at least at the Medium Frequency task
 * rate (It is typically called on the Systick interrupt). Exact invokation rate is 
 * the Speed regulator execution rate set in the Motor Contorl Workbench.
 *
 * The following tasks are executed in this order:
 *
 * - Medium Frequency Tasks of each motors.
 * - Safety Task.
 * - Power Factor Correction Task (if enabled).
 * - User Interface task.
 */
__weak void MC_RunMotorControlTasks(void)
{
  if (0U == bMCBootCompleted)
  {
    /* Nothing to do */
  }
  else
  {
    /* ** Medium Frequency Tasks ** */
    MC_Scheduler();
<#if MC.RTOS == "NONE">

    /* Safety task is run after Medium Frequency task so that
     * it can overcome actions they initiated if needed */
    TSK_SafetyTask();
    
</#if><#-- MC.RTOS == "NONE" -->
<#if MC.PFC_ENABLED == true>
    /* ** Power Factor Correction Task ** */ 
    PFC_Scheduler();
</#if><#-- MC.PFC_ENABLED == true -->
  }
}

/**
 * @brief Performs stop process and update the state machine.This function 
 *        shall be called only during medium frequency task.
 */
void TSK_MF_StopProcessing(uint8_t motor)
{
  <#if MC.DRIVE_NUMBER != "1">
  if (M1 == motor)
  {
    ${PWM_SwitchOff}(pwmcHandle[motor]);
  }
  else
  {
    ${PWM_SwitchOff_M2}(pwmcHandle[motor]);
  }
  <#else>
    ${PWM_SwitchOff}(pwmcHandle[motor]);
  </#if>

<#if MC.MOTOR_PROFILER == true && MC.DRIVE_NUMBER == "1">
  SCC_Stop(&SCC);
  OTT_Stop(&OTT);
</#if><#-- MC.MOTOR_PROFILER == true && MC.DRIVE_NUMBER == 1 -->
<#if FOC>
  FOC_Clear(motor);
  PQD_Clear(pMPM[motor]);
</#if><#-- FOC -->
<#if MC.M1_DISCONTINUOUS_PWM == true  || MC.M2_DISCONTINUOUS_PWM == true>
  /* Disable DPWM mode */
  PWMC_DPWM_ModeDisable(pwmcHandle[motor]);
</#if><#-- MC.M1_DISCONTINUOUS_PWM == true  || MC.M2_DISCONTINUOUS_PWM == true -->
<#if SIX_STEP>
  SixStep_Clear(motor);
</#if><#-- SIX_STEP -->
<#if MC.DRIVE_NUMBER != "1">
 if (M1 == motor)
  {
    TSK_SetStopPermanencyTimeM1(STOPPERMANENCY_TICKS);
  }
  else
  {
    TSK_SetStopPermanencyTimeM2(STOPPERMANENCY_TICKS);
  }
<#else>
  TSK_SetStopPermanencyTimeM1(STOPPERMANENCY_TICKS);
</#if>  
  Mci[motor].State = STOP;
}


/**
 * @brief  Executes the Medium Frequency Task functions for each drive instance. 
 *
 * It is to be clocked at the Systick frequency.
 */
__weak void MC_Scheduler(void)
{
/* USER CODE BEGIN MC_Scheduler 0 */

/* USER CODE END MC_Scheduler 0 */

  if (((uint8_t)1) == bMCBootCompleted)
  {
    if(hMFTaskCounterM1 > 0u)
    {
      hMFTaskCounterM1--;
    }
    else
    {
      TSK_MediumFrequencyTaskM1();

      /* Applicative hook at end of Medium Frequency for Motor 1 */
      MC_APP_PostMediumFrequencyHook_M1();

<#if MC.MCP_OVER_UART_A_EN>
      MCP_Over_UartA.rxBuffer = MCP_Over_UartA.pTransportLayer->fRXPacketProcess(MCP_Over_UartA.pTransportLayer, 
                                                                                &MCP_Over_UartA.rxLength);
      if ( 0U == MCP_Over_UartA.rxBuffer)
      {
        /* Nothing to do */
      }
      else
      {
        /* Synchronous answer */
        if (0U == MCP_Over_UartA.pTransportLayer->fGetBuffer(MCP_Over_UartA.pTransportLayer, 
                                                     (void **) &MCP_Over_UartA.txBuffer, //cstat !MISRAC2012-Rule-11.3
                                                     MCTL_SYNC)) 
        {
          /* No buffer available to build the answer ... should not occur */
        }
        else
        {
          MCP_ReceivedPacket(&MCP_Over_UartA);
          MCP_Over_UartA.pTransportLayer->fSendPacket(MCP_Over_UartA.pTransportLayer, MCP_Over_UartA.txBuffer, 
                                                      MCP_Over_UartA.txLength, MCTL_SYNC);
          /* No buffer available to build the answer ... should not occur */
        }
      }
</#if><#-- MC.MCP_OVER_UART_A_EN -->
<#if MC.MCP_OVER_UART_B_EN>
      MCP_Over_UartB.rxBuffer = MCP_Over_UartB.pTransportLayer->fRXPacketProcess(MCP_Over_UartB.pTransportLayer,
                                                                                 &MCP_Over_UartB.rxLength);
      if (MCP_Over_UartB.rxBuffer)
      {
        /* Synchronous answer */
        if (MCP_Over_UartB.pTransportLayer->fGetBuffer(MCP_Over_UartB.pTransportLayer,
                                                       (void **) &MCP_Over_UartB.txBuffer, MCTL_SYNC))
        {
          MCP_ReceivedPacket(&MCP_Over_UartB);
          MCP_Over_UartB.pTransportLayer->fSendPacket(MCP_Over_UartB.pTransportLayer, MCP_Over_UartB.txBuffer,
                                                      MCP_Over_UartB.txLength, MCTL_SYNC);
        }
        else 
        {
          /* No buffer available to build the answer ... should not occur */
        }
      }
</#if><#-- MC.MCP_OVER_UART_B_EN -->
<#if MC.MCP_OVER_STLNK_EN>
      MCP_Over_STLNK.rxBuffer = MCP_Over_STLNK.pTransportLayer->fRXPacketProcess( MCP_Over_STLNK.pTransportLayer,
                                                                                  &MCP_Over_STLNK.rxLength);
      if (0U == MCP_Over_STLNK.rxBuffer)
      {
        /* Nothing to do */
      }
      else
      {
        /* Synchronous answer */
        if (0U == MCP_Over_STLNK.pTransportLayer->fGetBuffer(MCP_Over_STLNK.pTransportLayer,
                                      (void **) &MCP_Over_STLNK.txBuffer, MCTL_SYNC)) //cstat !MISRAC2012-Rule-11.3
        {
          /* No buffer available to build the answer ... should not occur */
        }
        else 
        {
          MCP_ReceivedPacket(&MCP_Over_STLNK);
          MCP_Over_STLNK.pTransportLayer->fSendPacket (MCP_Over_STLNK.pTransportLayer, MCP_Over_STLNK.txBuffer,
                                                       MCP_Over_STLNK.txLength, MCTL_SYNC);
        }
      }
</#if><#-- MC.MCP_OVER_STLNK_EN -->

      /* USER CODE BEGIN MC_Scheduler 1 */

      /* USER CODE END MC_Scheduler 1 */
<#if MC.EXAMPLE_SPEEDMONITOR == true>
      /****************************** USE ONLY FOR SDK 4.0 EXAMPLES *************/
      ARR_TIM5_update(${SPD_M1});

      /**************************************************************************/
</#if><#-- MC.EXAMPLE_SPEEDMONITOR == true -->
      hMFTaskCounterM1 = (uint16_t)MF_TASK_OCCURENCE_TICKS;
    }
<#if MC.DRIVE_NUMBER != "1">
    if(hMFTaskCounterM2 > ((uint16_t )0))
    {
      hMFTaskCounterM2--;
    }
    else
    {
      TSK_MediumFrequencyTaskM2();

      /* Applicative hook at end of Medium Frequency for Motor 2 */
      MC_APP_PostMediumFrequencyHook_M2();

      /* USER CODE BEGIN MC_Scheduler MediumFrequencyTask M2 */

      /* USER CODE END MC_Scheduler MediumFrequencyTask M2 */
      hMFTaskCounterM2 = MF_TASK_OCCURENCE_TICKS2;
    }
</#if><#-- MC.DRIVE_NUMBER > 1 -->
    if(hBootCapDelayCounterM1 > 0U)
    {
      hBootCapDelayCounterM1--;
    }
    else
    {
      /* Nothing to do */
    }
    if(hStopPermanencyCounterM1 > 0U)
    {
      hStopPermanencyCounterM1--;
    }
    else
    {
      /* Nothing to do */
    }
<#if MC.DRIVE_NUMBER != "1">
    if(hBootCapDelayCounterM2 > 0U)
    {
      hBootCapDelayCounterM2--;
    }
    else
    {
      /* Nothing to do */
    }
    if(hStopPermanencyCounterM2 > 0U)
    {
      hStopPermanencyCounterM2--;
    }
    else
    {
      /* Nothing to do */
    }
</#if><#-- MC.DRIVE_NUMBER > 1 -->
  }
  else
  {
    /* Nothing to do */
  }
  /* USER CODE BEGIN MC_Scheduler 2 */

  /* USER CODE END MC_Scheduler 2 */
}

/**
  * @brief Executes medium frequency periodic Motor Control tasks
  *
  * This function performs some of the control duties on Motor 1 according to the 
  * present state of its state machine. In particular, duties requiring a periodic 
  * execution at a medium frequency rate (such as the speed controller for instance) 
  * are executed here.
  */
__weak void TSK_MediumFrequencyTaskM1(void)
{
<#if DWT_CYCCNT_SUPPORTED>
  <#if MC.DBG_MCU_LOAD_MEASURE == true>
  MC_BG_Perf_Measure_Start(&PerfTraces, MEASURE_TSK_MediumFrequencyTaskM1);
  </#if><#-- DWT_CYCCNT_SUPPORTED -->
</#if><#-- MC.DBG_MCU_LOAD_MEASURE == true -->
  /* USER CODE BEGIN MediumFrequencyTask M1 0 */

  /* USER CODE END MediumFrequencyTask M1 0 */

  int16_t wAux = 0;
<#if (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE") && (MC.M1_OTF_STARTUP == true)>
  ${PWM_Handle_Type_M1} *p_Handle = (${PWM_Handle_Type_M1} *)pwmcHandle[M1];
</#if><#-- (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE") && (MC.M1_OTF_STARTUP == true) -->
<#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
  MC_ControlMode_t mode;
  
  mode = MCI_GetControlMode(&Mci[M1]);
</#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
<#if MC.M1_ICL_ENABLED == true>
  ICL_State_t ICLstate = ICL_Exec(&ICL_M1);
</#if><#-- MC.M1_ICL_ENABLED == true -->
<#if AUX_SPEED_FDBK_M1 == true>
  (void)${SPD_aux_calcAvrgMecSpeedUnit_M1}(${SPD_AUX_M1}, &wAux);
</#if><#-- AUX_SPEED_FDBK_M1 == true -->
<#if MC.M1_SPEED_FEEDBACK_CHECK == true || (MC.M1_SPEED_SENSOR == "HALL_SENSOR")>
  bool IsSpeedReliable = ${SPD_calcAvrgMecSpeedUnit_M1}(${SPD_M1}, &wAux);
<#else><#-- MC.M1_SPEED_FEEDBACK_CHECK = false || (MC.M1_SPEED_SENSOR != "HALL_SENSOR") -->
  (void)${SPD_calcAvrgMecSpeedUnit_M1}(${SPD_M1}, &wAux);
</#if><#-- MC.M1_SPEED_FEEDBACK_CHECK == true || (MC.M1_SPEED_SENSOR == "HALL_SENSOR") -->
<#if FOC>
  PQD_CalcElMotorPower(pMPM[M1]);
</#if><#-- FOC -->

<#if MC.M1_ICL_ENABLED == true>
  if ( !ICLFaultTreatedM1 && (ICLstate == ICL_ACTIVE))
  {
    ICLFaultTreatedM1 = true;
  }
  else
  {
    /* Nothing to do */
  }
</#if><#-- MC.M1_ICL_ENABLED == true -->

<#if MC.M1_ICL_ENABLED == true>
  if ((MCI_GetCurrentFaults(&Mci[M1]) == MC_NO_FAULTS) && ICLFaultTreatedM1)
<#else><#-- MC.M1_ICL_ENABLED == false -->
  if (MCI_GetCurrentFaults(&Mci[M1]) == MC_NO_FAULTS)
</#if><#-- MC.M1_ICL_ENABLED == true -->
  {
    if (MCI_GetOccurredFaults(&Mci[M1]) == MC_NO_FAULTS)
    {
      switch (Mci[M1].State)
      {
<#if MC.M1_ICL_ENABLED == true>
        case ICLWAIT:
        {
          if (ICL_INACTIVE == ICLstate)
          {
            /* If ICL is Inactive, move to IDLE */
            Mci[M1].State = IDLE;
          }
          break;
        }
</#if><#-- MC.M1_ICL_ENABLED == true -->
        
        case IDLE:
        {
          if ((MCI_START == Mci[M1].DirectCommand) || (MCI_MEASURE_OFFSETS == Mci[M1].DirectCommand))
          {
<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
            if ( mode != MCM_OPEN_LOOP_VOLTAGE_MODE && mode != MCM_OPEN_LOOP_CURRENT_MODE) 
            {
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
              RUC_Clear(&RevUpControlM1, MCI_GetImposedMotorDirection(&Mci[M1]));
  <#if SIX_STEP>
              RUC_UpdatePulse(&RevUpControlM1, &BusVoltageSensor_M1._Super);
  </#if><#-- SIX_STEP -->
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
            }
            else
            {
              /* Nothing to do */
            }
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")
        || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC"  -->
        <#if MC.TESTENV == true && SIX_STEP>
              mc_testenv_init();
        </#if>
        <#if FOC>
            if (pwmcHandle[M1]->offsetCalibStatus == false)
            {
              (void)PWMC_CurrentReadingCalibr(pwmcHandle[M1], CRC_START);
              Mci[M1].State = OFFSET_CALIB;
            }
            else
            {
              /* Calibration already done. Enables only TIM channels */
              pwmcHandle[M1]->OffCalibrWaitTimeCounter = 1u;
              (void)PWMC_CurrentReadingCalibr(pwmcHandle[M1], CRC_EXEC);
</#if><#-- FOC -->
<#if CHARGE_BOOT_CAP_ENABLING == true>
  <#if (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE")>
    <#if MC.M1_DP_DESTINATION == "TIM_BKIN">
              LL_TIM_DisableBRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
    <#elseif MC.M1_DP_DESTINATION == "TIM_BKIN2">
              LL_TIM_DisableBRK2(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
    </#if><#-- MC.M1_DP_DESTINATION == "TIM_BKIN" -->
  </#if><#-- (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE") -->
  <#if (MC.M1_OTF_STARTUP == true)>
              ${PWM_TurnOnLowSides}(pwmcHandle[M1],p_Handle->Half_PWMPeriod-1);
  <#else><#-- !M1_OTF_STARTUP -->
              ${PWM_TurnOnLowSides}(pwmcHandle[M1],M1_CHARGE_BOOT_CAP_DUTY_CYCLES);
  </#if><#-- M1_OTF_STARTUP -->
              TSK_SetChargeBootCapDelayM1(M1_CHARGE_BOOT_CAP_TICKS);
              Mci[M1].State = CHARGE_BOOT_CAP;
<#else><#-- CHARGE_BOOT_CAP_ENABLING == false -->
<#-- test sensorless -->
              FOCVars[M1].bDriveInput = EXTERNAL;
              STC_SetSpeedSensor(pSTC[M1], &VirtualSpeedSensorM1._Super);
              ${SPD_clear_M1}(${SPD_M1});
              FOC_Clear(M1);
              ${PWM_SwitchOn}(pwmcHandle[M1]);
              Mci[M1].State = START;
</#if><#-- CHARGE_BOOT_CAP_ENABLING == true -->
<#if FOC>
            }
</#if><#-- FOC -->
<#if (MC.MOTOR_PROFILER == true) || (MC.ONE_TOUCH_TUNING == true)>
            OTT_Clear(&OTT);
</#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
          }
          else
          {
        <#if MC.TESTENV == true && SIX_STEP>
              mc_testenv_clear();
        <#else>
            /* Nothing to be done, FW stays in IDLE state */
        </#if>
          }
          break;
        }

<#if FOC>
        case OFFSET_CALIB:
        {
          if (MCI_STOP == Mci[M1].DirectCommand)
          {
            TSK_MF_StopProcessing(M1);
          }
          else
          {
            if (PWMC_CurrentReadingCalibr(pwmcHandle[M1], CRC_EXEC))
            {
              if (MCI_MEASURE_OFFSETS == Mci[M1].DirectCommand)
              {
                FOC_Clear(M1);
                PQD_Clear(pMPM[M1]);
                Mci[M1].DirectCommand = MCI_NO_COMMAND;
                Mci[M1].State = IDLE;
              }
              else
              {
  <#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
                Mci[M1].State = WAIT_STOP_MOTOR;
  <#else><#-- MC.MOTOR_PROFILER == false || MC.ONE_TOUCH_TUNING == false -->
    <#if CHARGE_BOOT_CAP_ENABLING == true>
      <#if (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE")>
        <#if MC.M1_DP_DESTINATION == "TIM_BKIN">
                LL_TIM_DisableBRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
        <#elseif MC.M1_DP_DESTINATION == "TIM_BKIN2">
                LL_TIM_DisableBRK2(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
        </#if><#-- MC.M1_DP_DESTINATION == "TIM_BKIN" -->
      </#if><#-- (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE") -->
      <#if (MC.M1_OTF_STARTUP == true)>
                ${PWM_TurnOnLowSides}(pwmcHandle[M1],p_Handle->Half_PWMPeriod-1);
      <#else><#-- !M1_OTF_STARTUP -->
                ${PWM_TurnOnLowSides}(pwmcHandle[M1],M1_CHARGE_BOOT_CAP_DUTY_CYCLES);
      </#if><#-- M1_OTF_STARTUP -->
                TSK_SetChargeBootCapDelayM1(M1_CHARGE_BOOT_CAP_TICKS);
                Mci[M1].State = CHARGE_BOOT_CAP;
    <#else><#-- CHARGE_BOOT_CAP_ENABLING == false -->
                FOCVars[M1].bDriveInput = EXTERNAL;
                STC_SetSpeedSensor(pSTC[M1], &VirtualSpeedSensorM1._Super);
                ${SPD_clear_M1}(${SPD_M1});
                FOC_Clear(M1);
      <#if MC.M1_DISCONTINUOUS_PWM == true>
                /* Enable DPWM mode before Start */
                PWMC_DPWM_ModeEnable(pwmcHandle[M1]);
      </#if><#-- MC.M1_DISCONTINUOUS_PWM == true -->
                ${PWM_SwitchOn}(pwmcHandle[M1]);
                Mci[M1].State = START;
    </#if><#-- CHARGE_BOOT_CAP_ENABLING == true -->
  </#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
              }
            }
            else
            {
              /* Nothing to be done, FW waits for offset calibration to finish */
            }
          }  
          break;
        }
</#if><#-- FOC -->

<#if (CHARGE_BOOT_CAP_ENABLING == true)>
  
        case CHARGE_BOOT_CAP:
        {
          if (MCI_STOP == Mci[M1].DirectCommand)
          {
            TSK_MF_StopProcessing(M1);
          }
          else
          {
            if (TSK_ChargeBootCapDelayHasElapsedM1())
            {
              ${PWM_SwitchOff}(pwmcHandle[M1]);
  <#if (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE")>
    <#if MC.M1_DP_DESTINATION == "TIM_BKIN">
              LL_TIM_ClearFlag_BRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
              LL_TIM_EnableBRK(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
    <#elseif MC.M1_DP_DESTINATION == "TIM_BKIN2">
              LL_TIM_ClearFlag_BRK2(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
              LL_TIM_EnableBRK2(${_last_word(MC.M1_PWM_TIMER_SELECTION)});
    </#if><#-- MC.M1_DP_DESTINATION == "TIM_BKIN" -->
  </#if><#-- (MC.M1_PWM_DRIVER_PN == "STDRIVE101") && (MC.M1_DP_TOPOLOGY != "NONE") -->
<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || M1_ENCODER>
              FOCVars[M1].bDriveInput = EXTERNAL;
              STC_SetSpeedSensor( pSTC[M1], &VirtualSpeedSensorM1._Super );
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || M1_ENCODER -->
<#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
              SixStepVars[M1].bDriveInput = EXTERNAL;
              STC_SetSpeedSensor( pSTC[M1], &VirtualSpeedSensorM1._Super );
</#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
              ${SPD_clear_M1}(${SPD_M1});
<#if AUX_SPEED_FDBK_M1 == true>
              ${SPD_aux_clear_M1}(${SPD_AUX_M1});
</#if><#-- AUX_SPEED_FDBK_M1 == true -->
<#if MC.M1_OVERMODULATION == true>
              PWMC_Clear(pwmcHandle[M1]);
</#if><#-- MC.M1_OVERMODULATION == true -->
<#if MC.M1_DISCONTINUOUS_PWM == true>
              /* Enable DPWM mode before Start */
              PWMC_DPWM_ModeEnable( pwmcHandle[M1]);
</#if><#-- MC.M1_DISCONTINUOUS_PWM == true -->
<#if FOC>
              FOC_Clear( M1 );
  <#if (MC.M1_OTF_STARTUP == true)>
              ${PWM_SwitchOn}(pwmcHandle[M1]);
  </#if><#-- MC.M1_OTF_STARTUP == true -->
</#if><#-- FOC -->
<#if SIX_STEP>
              SixStep_Clear( M1 );
</#if><#-- SIX_STEP -->
<#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
              BADC_SetDirection(&Bemf_ADC_M1, MCI_GetImposedMotorDirection( &Mci[M1]));
              BADC_SetSamplingPoint(&Bemf_ADC_M1, &PWM_Handle_M1._Super, pSTC[M1] );
</#if>		  
<#if (MC.MOTOR_PROFILER == true)>
        SCC_Start(&SCC);
              /* The generic function needs to be called here as the undelying   
               * implementation changes in time depending on the Profiler's state 
               * machine. Calling the generic function ensures that the correct
               * implementation is invoked */
              PWMC_SwitchOnPWM(pwmcHandle[M1]);
              Mci[M1].State = START;
<#else><#-- MC.MOTOR_PROFILER == false -->
 
  <#if (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z")>
              if (EAC_IsAligned(&EncAlignCtrlM1) == false)
              {
                EAC_StartAlignment(&EncAlignCtrlM1);
                Mci[M1].State = ALIGNMENT;
              }
              else
              {
                STC_SetControlMode(pSTC[M1], MCM_SPEED_MODE);
                STC_SetSpeedSensor(pSTC[M1], &ENCODER_M1._Super);
                FOC_InitAdditionalMethods(M1);
                FOC_CalcCurrRef(M1);
                STC_ForceSpeedReferenceToCurrentSpeed(pSTC[M1]); /* Init the reference speed to current speed */
                MCI_ExecBufferedCommands(&Mci[M1]); /* Exec the speed ramp after changing of the speed sensor */
                Mci[M1].State = RUN;
              }
  <#elseif  (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
              if (EAC_IsAligned(&EncAlignCtrlM1) == false)
              {
                EAC_StartAlignment(&EncAlignCtrlM1);
                Mci[M1].State = ALIGNMENT;
              }
              else
              {
                VSS_Clear(&VirtualSpeedSensorM1); /* Reset measured speed in IDLE */
                FOC_Clear(M1);
                Mci[M1].State = START;
              }
  <#elseif MC.M1_SPEED_SENSOR == "HALL_SENSOR">
    <#if FOC>
              FOC_InitAdditionalMethods(M1);
              FOC_CalcCurrRef(M1);
    </#if><#-- FOC -->
    <#if SIX_STEP>
              SixStep_InitAdditionalMethods(M1);
              SixStep_CalcSpeedRef(M1);
      <#if MC.CURRENT_LIMITER_OFFSET == true>
#if (PID_SPEED_INTEGRAL_INIT_DIV == 0)
              PID_SetIntegralTerm(&PIDSpeedHandle_M1, 0);
#else
              PID_SetIntegralTerm(&PIDSpeedHandle_M1,
                                 (((int32_t)SixStepVars[M1].DutyCycleRef * (int16_t)PID_GetKIDivisor(&PIDSpeedHandle_M1))
                                 / PID_SPEED_INTEGRAL_INIT_DIV));
#endif
      </#if><#-- MC.CURRENT_LIMITER_OFFSET == true -->
    </#if><#-- SIX_STEP -->
              STC_ForceSpeedReferenceToCurrentSpeed( pSTC[M1]); /* Init the reference speed to current speed */
              MCI_ExecBufferedCommands(&Mci[M1]); /* Exec the speed ramp after changing of the speed sensor */
              Mci[M1].State = RUN;
  <#else><#-- sensorless mode only -->
    <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
              if (MCM_OPEN_LOOP_VOLTAGE_MODE == mode || MCM_OPEN_LOOP_CURRENT_MODE == mode)
              {
                Mci[M1].State = RUN;
              }
              else
              {
    </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
                Mci[M1].State = START;
    <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
              }
    </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
  </#if><#-- (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") -->
</#if><#-- MC.MOTOR_PROFILER == true -->
<#if (MC.M1_OTF_STARTUP == false)>
              PWMC_SwitchOnPWM(pwmcHandle[M1]);
</#if><#-- (MC.M1_OTF_STARTUP == false) -->
            }
            else
            {
              /* Nothing to be done, FW waits for bootstrap capacitor to charge */
            }
          }
          break;
        }
</#if><#-- CHARGE_BOOT_CAP_ENABLING == true -->

<#if M1_ENCODER><#-- only for encoder -->
        case ALIGNMENT:
        {
          if (MCI_STOP == Mci[M1].DirectCommand)
          {
            TSK_MF_StopProcessing(M1);
          }
          else
          {
            bool isAligned = EAC_IsAligned(&EncAlignCtrlM1);
            bool EACDone = EAC_Exec(&EncAlignCtrlM1);
            if ((isAligned == false)  && (EACDone == false))
            {
              qd_t IqdRef;
              IqdRef.q = 0;
              IqdRef.d = STC_CalcTorqueReference(pSTC[M1]);
              FOCVars[M1].Iqdref = IqdRef;
            }
            else
            {
              ${PWM_SwitchOff}( pwmcHandle[M1] );
  <#if (MC.M1_AUXILIARY_SPEED_SENSOR != "QUAD_ENCODER") && (MC.M1_AUXILIARY_SPEED_SENSOR != "QUAD_ENCODER_Z")>
              STC_SetControlMode(pSTC[M1], MCM_SPEED_MODE);
              STC_SetSpeedSensor(pSTC[M1], &ENCODER_M1._Super);
  </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR != "QUAD_ENCODER") && (MC.M1_AUXILIARY_SPEED_SENSOR != "QUAD_ENCODER_Z") -->
              FOC_Clear(M1);
  <#if (CHARGE_BOOT_CAP_ENABLING == true)>
              ${PWM_TurnOnLowSides}(pwmcHandle[M1],M1_CHARGE_BOOT_CAP_DUTY_CYCLES);
  </#if><#-- CHARGE_BOOT_CAP_ENABLING == true -->
              TSK_SetStopPermanencyTimeM1(STOPPERMANENCY_TICKS);
              Mci[M1].State = WAIT_STOP_MOTOR;
              /* USER CODE BEGIN MediumFrequencyTask M1 EndOfEncAlignment */
            
              /* USER CODE END MediumFrequencyTask M1 EndOfEncAlignment */
            }
          }
          break;
        }
</#if><#-- M1_ENCODER -->

<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
        case START:
        {
          if (MCI_STOP == Mci[M1].DirectCommand)
          {
            TSK_MF_StopProcessing(M1);
          }
          else
          {
          <#-- only for sensor-less control -->
            /* Mechanical speed as imposed by the Virtual Speed Sensor during the Rev Up phase. */
            int16_t hForcedMecSpeedUnit;
  <#if FOC>
            qd_t IqdRef;
  </#if><#-- FOC -->
            bool ObserverConverged;

            /* Execute the Rev Up procedure */
  <#if MC.M1_OTF_STARTUP == true>
            if (! RUC_OTF_Exec(&RevUpControlM1))
  <#else><#-- MC.M1_OTF_STARTUP == false -->
            if(! RUC_Exec(&RevUpControlM1))
  </#if><#-- MC.M1_OTF_STARTUP == true -->
            {
            /* The time allowed for the startup sequence has expired */
  <#if MC.MOTOR_PROFILER == true>
              /* However, no error is generated when OPEN LOOP is enabled 
               * since then the system does not try to close the loop... */
  <#else><#-- MC.MOTOR_PROFILER == false -->
              MCI_FaultProcessing(&Mci[M1], MC_START_UP, 0);
  </#if><#-- MC.MOTOR_PROFILER == true -->
            }
            else
            {
              /* Execute the torque open loop current start-up ramp:
               * Compute the Iq reference current as configured in the Rev Up sequence */
  <#if FOC>
              IqdRef.q = STC_CalcTorqueReference(pSTC[M1]);
              IqdRef.d = FOCVars[M1].UserIdref;
              /* Iqd reference current used by the High Frequency Loop to generate the PWM output */
              FOCVars[M1].Iqdref = IqdRef;
  </#if><#-- FOC -->
  <#if SIX_STEP>
            (void) BADC_CalcRevUpDemagTime (&Bemf_ADC_M1);
            PWMC_ForceFastDemagTime (pwmcHandle[M1], Bemf_ADC_M1.DemagCounterThreshold);
            SixStepVars[M1].DutyCycleRef = STC_CalcSpeedReference( pSTC[M1] );
  </#if>
           }
          
            (void)VSS_CalcAvrgMecSpeedUnit(&VirtualSpeedSensorM1, &hForcedMecSpeedUnit);
         
  <#if MC.M1_OTF_STARTUP == false && MC.MOTOR_PROFILER == false>
            /* Check that startup stage where the observer has to be used has been reached */
            if (true == RUC_FirstAccelerationStageReached(&RevUpControlM1))
            {
  </#if><#-- MC.M1_OTF_STARTUP == false) && (MC.MOTOR_PROFILER == false -->
  <#if MC.M1_SPEED_SENSOR == "STO_PLL">
              ObserverConverged = STO_PLL_IsObserverConverged(&STO_PLL_M1, &hForcedMecSpeedUnit);
              STO_SetDirection(&STO_PLL_M1, (int8_t)MCI_GetImposedMotorDirection(&Mci[M1]));
  <#elseif MC.M1_SPEED_SENSOR == "STO_CORDIC">
              ObserverConverged = STO_CR_IsObserverConverged(&STO_CR_M1, hForcedMecSpeedUnit);
  <#elseif MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
             ObserverConverged = BADC_IsObserverConverged( &Bemf_ADC_M1);
  </#if> <#-- MC.M1_SPEED_SENSOR == "STO_PLL"     -->
              (void)VSS_SetStartTransition(&VirtualSpeedSensorM1, ObserverConverged);
  <#if MC.M1_OTF_STARTUP == false && MC.MOTOR_PROFILER == false>
            }
            else
            {
              ObserverConverged = false;
            }
  </#if><#-- MC.M1_OTF_STARTUP == false) && (MC.MOTOR_PROFILER == false -->
            if (ObserverConverged)
            {
  <#if FOC>
              qd_t StatorCurrent = MCM_Park(FOCVars[M1].Ialphabeta, SPD_GetElAngle(${SPD_M1}._Super));

              /* Start switch over ramp. This ramp will transition from the revup to the closed loop FOC */
              REMNG_Init(pREMNG[M1]);
              (void)REMNG_ExecRamp(pREMNG[M1], FOCVars[M1].Iqdref.q, 0);
              (void)REMNG_ExecRamp(pREMNG[M1], StatorCurrent.q, TRANSITION_DURATION);
  </#if><#-- FOC -->
              Mci[M1].State = SWITCH_OVER;
            }
          }
          break;
        }

        case SWITCH_OVER:
        {
          if (MCI_STOP == Mci[M1].DirectCommand)
          {
            TSK_MF_StopProcessing(M1);
          }
          else
          {
            bool LoopClosed;
            int16_t hForcedMecSpeedUnit;

  <#if MC.MOTOR_PROFILER == false><#-- No need to call RUC_Exec() when in MP in this state -->
    <#if MC.M1_OTF_STARTUP == true>
            if (! RUC_OTF_Exec(&RevUpControlM1))
    <#else><#-- MC.M1_OTF_STARTUP == false -->
            if (! RUC_Exec(&RevUpControlM1))
    </#if><#-- MC.M1_OTF_STARTUP == true -->
            {
              /* The time allowed for the startup sequence has expired */
              MCI_FaultProcessing(&Mci[M1], MC_START_UP, 0);
            } 
            else
            {
  </#if><#--  MC.MOTOR_PROFILER == false -->
              /* Compute the virtual speed and positions of the rotor. 
                 The function returns true if the virtual speed is in the reliability range */
              LoopClosed = VSS_CalcAvrgMecSpeedUnit(&VirtualSpeedSensorM1, &hForcedMecSpeedUnit);
              /* Check if the transition ramp has completed. */
              bool tempBool;
              tempBool = VSS_TransitionEnded(&VirtualSpeedSensorM1);
              LoopClosed = LoopClosed || tempBool;
              
              /* If any of the above conditions is true, the loop is considered closed. 
                 The state machine transitions to the RUN state */
              if (true ==  LoopClosed) 
              {
#if PID_SPEED_INTEGRAL_INIT_DIV == 0
                PID_SetIntegralTerm(&PIDSpeedHandle_M1, 0);
#else
  <#if FOC>
                PID_SetIntegralTerm(&PIDSpeedHandle_M1,
                                    (((int32_t)FOCVars[M1].Iqdref.q * (int16_t)PID_GetKIDivisor(&PIDSpeedHandle_M1)) 
                                    / PID_SPEED_INTEGRAL_INIT_DIV));
  </#if><#-- FOC -->
  <#if SIX_STEP>
                PID_SetIntegralTerm(&PIDSpeedHandle_M1,
                                    (((int32_t)SixStepVars[M1].DutyCycleRef * (int16_t)PID_GetKIDivisor(&PIDSpeedHandle_M1)) 
                                    / PID_SPEED_INTEGRAL_INIT_DIV));
  </#if><#-- SIX_STEP -->
#endif
  <#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
                OTT_SR(&OTT);
  </#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
                /* USER CODE BEGIN MediumFrequencyTask M1 1 */

                /* USER CODE END MediumFrequencyTask M1 1 */ 
                STC_SetSpeedSensor(pSTC[M1], ${SPD_M1}._Super); /* Observer has converged */
  <#if FOC>
                FOC_InitAdditionalMethods(M1);
                FOC_CalcCurrRef(M1);
  </#if><#-- FOC -->
  <#if SIX_STEP>
                SixStep_InitAdditionalMethods(M1);
                SixStep_CalcSpeedRef(M1);
    <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
                BADC_SetLoopClosed(&Bemf_ADC_M1);
                BADC_SpeedMeasureOn(&Bemf_ADC_M1);
    </#if>
  </#if><#-- SIX_STEP -->
                STC_ForceSpeedReferenceToCurrentSpeed(pSTC[M1]); /* Init the reference speed to current speed */
                MCI_ExecBufferedCommands(&Mci[M1]); /* Exec the speed ramp after changing of the speed sensor */
                Mci[M1].State = RUN;
              }
  <#if MC.MOTOR_PROFILER == false>
            }
  </#if><#--  MC.MOTOR_PROFILER == false -->
          }
          break;
        }
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")
        || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->

        case RUN:
        {
          if (MCI_STOP == Mci[M1].DirectCommand)
          {
            TSK_MF_StopProcessing(M1);
          }
          else
          {
            /* USER CODE BEGIN MediumFrequencyTask M1 2 */
            
            /* USER CODE END MediumFrequencyTask M1 2 */
       
<#if  MC.M1_POSITION_CTRL_ENABLING == true >
            TC_PositionRegulation(pPosCtrl[M1]);
</#if><#-- MC.M1_POSITION_CTRL_ENABLING == true -->
            MCI_ExecBufferedCommands(&Mci[M1]);
<#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
            if (mode != MCM_OPEN_LOOP_VOLTAGE_MODE && mode != MCM_OPEN_LOOP_CURRENT_MODE)
            {
</#if> <#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
<#if FOC>
              FOC_CalcCurrRef(M1);
  </#if><#-- FOC -->
  <#if SIX_STEP>
            SixStep_CalcSpeedRef( M1 );
  <#if  MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
            BADC_SetSamplingPoint(&Bemf_ADC_M1, &PWM_Handle_M1._Super, pSTC[M1] );
            (void) BADC_CalcRunDemagTime (&Bemf_ADC_M1);
            PWMC_ForceFastDemagTime (pwmcHandle[M1], Bemf_ADC_M1.DemagCounterThreshold);
  </#if>	
  </#if><#-- SIX_STEP -->
         
<#if MC.M1_SPEED_FEEDBACK_CHECK == true || (MC.M1_SPEED_SENSOR == "HALL_SENSOR")>
              if(!IsSpeedReliable)
              {
                MCI_FaultProcessing(&Mci[M1], MC_SPEED_FDBK, 0);
              }
              else
              {
                /* Nothing to do */
              }
</#if><#-- MC.M1_SPEED_FEEDBACK_CHECK == true || (MC.M1_SPEED_SENSOR == "HALL_SENSOR") -->
<#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
            }
            else
            {
              int16_t hForcedMecSpeedUnit;
              /* Open Loop */
              VSS_CalcAvrgMecSpeedUnit( &VirtualSpeedSensorM1, &hForcedMecSpeedUnit);
              OL_Calc(pOpenLoop[M1]);
            }   
</#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
<#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
            OTT_MF(&OTT);
</#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
          }
          break;
        }

        case STOP:
        {
          if (TSK_StopPermanencyTimeHasElapsedM1())
          {

<#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
            STC_SetSpeedSensor(pSTC[M1], &VirtualSpeedSensorM1._Super);    /* Sensor-less */
            VSS_Clear(&VirtualSpeedSensorM1); /* Reset measured speed in IDLE */
  <#if SIX_STEP>
            BADC_Clear(&Bemf_ADC_M1);
  </#if><#-- SIX_STEP -->
</#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
            /* USER CODE BEGIN MediumFrequencyTask M1 5 */
    
            /* USER CODE END MediumFrequencyTask M1 5 */
            Mci[M1].DirectCommand = MCI_NO_COMMAND;
            Mci[M1].State = IDLE;
          }
          else
          {
            /* Nothing to do, FW waits for to stop */
          }
          break;
        }

        case FAULT_OVER:
        {
          if (MCI_ACK_FAULTS == Mci[M1].DirectCommand)
          {
            Mci[M1].DirectCommand = MCI_NO_COMMAND;
            Mci[M1].State = IDLE;
          }
          else
          {
            /* Nothing to do, FW stays in FAULT_OVER state until acknowledgement */
          }
          break;
        }

        
        case FAULT_NOW:
        {
          Mci[M1].State = FAULT_OVER;
          break;
        }

        
<#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true || M1_ENCODER>
        case WAIT_STOP_MOTOR:
        {
          if (MCI_STOP == Mci[M1].DirectCommand)
          {
            TSK_MF_StopProcessing(M1);
          }
          else
          {
  <#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
            if (0 == SCC_DetectBemf(&SCC))
            {
              /* In a sensorless configuration. Initiate the Revup procedure */
              FOCVars[M1].bDriveInput = EXTERNAL;
              STC_SetSpeedSensor(pSTC[M1], &VirtualSpeedSensorM1._Super);
               ${SPD_clear_M1}(${SPD_M1});
              FOC_Clear(M1);
              SCC_Start(&SCC);
              /* The generic function needs to be called here as the undelying   
               * implementation changes in time depending on the Profiler's state 
               * machine. Calling the generic function ensures that the correct
               * implementation is invoked */
              PWMC_SwitchOnPWM(pwmcHandle[M1]);
              Mci[M1].State = START;
            }
            else
            {
              /* Nothing to do */
            }
  <#elseif (MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z")>
            if (TSK_StopPermanencyTimeHasElapsedM1())
            {
              ENC_Clear(&ENCODER_M1);
              ${PWM_SwitchOn}(pwmcHandle[M1]);
    <#if MC.M1_POSITION_CTRL_ENABLING == true>
              TC_EncAlignmentCommand(pPosCtrl[M1]);
    </#if><#-- MC.M1_POSITION_CTRL_ENABLING == true -->
              FOC_InitAdditionalMethods(M1);
              STC_ForceSpeedReferenceToCurrentSpeed(pSTC[M1]); /* Init the reference speed to current speed */
              MCI_ExecBufferedCommands(&Mci[M1]); /* Exec the speed ramp after changing of the speed sensor */
              FOC_CalcCurrRef(M1);
              Mci[M1].State = RUN;
            } 
            else
            {
              /* Nothing to do */
            }
  <#elseif (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
            if (TSK_StopPermanencyTimeHasElapsedM1())
            {
              RUC_Clear(&RevUpControlM1, MCI_GetImposedMotorDirection(&Mci[M1]));
              ${SPD_clear_M1}(${SPD_M1});
              ENC_Clear(&ENCODER_M1);
              VSS_Clear(&VirtualSpeedSensorM1);
              ${PWM_SwitchOn}(pwmcHandle[M1]);
              Mci[M1].State = START;
            } 
            else
            {
              /* Nothing to do */
            }
  </#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
          }
          break;
        }
</#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true || M1_ENCODER -->

        default:
          break;
       }
    }  
    else
    {
      Mci[M1].State = FAULT_OVER;
    }
  }
  else
  {
    Mci[M1].State = FAULT_NOW;
  }
<#if MC.MOTOR_PROFILER>
   <#if MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
  HT_MF(&HT);
   </#if><#-- MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR" -->
  SCC_MF(&SCC);
</#if><#-- MC.MOTOR_PROFILER == true -->
  /* USER CODE BEGIN MediumFrequencyTask M1 6 */

  /* USER CODE END MediumFrequencyTask M1 6 */
<#if DWT_CYCCNT_SUPPORTED>
  <#if MC.DBG_MCU_LOAD_MEASURE == true>
  MC_BG_Perf_Measure_Stop(&PerfTraces, MEASURE_TSK_MediumFrequencyTaskM1);
  </#if><#-- DWT_CYCCNT_SUPPORTED -->
</#if><#-- MC.DBG_MCU_LOAD_MEASURE == true -->
}

<#if FOC>
/**
  * @brief  It re-initializes the current and voltage variables. Moreover
  *         it clears qd currents PI controllers, voltage sensor and SpeednTorque
  *         controller. It must be called before each motor restart.
  *         It does not clear speed sensor.
  * @param  bMotor related motor it can be M1 or M2.
  */
__weak void FOC_Clear(uint8_t bMotor)
{
  /* USER CODE BEGIN FOC_Clear 0 */

  /* USER CODE END FOC_Clear 0 */
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true>
  MC_ControlMode_t mode;
  
  mode = MCI_GetControlMode( &Mci[bMotor] );
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
    
  ab_t NULL_ab = {((int16_t)0), ((int16_t)0)};
  qd_t NULL_qd = {((int16_t)0), ((int16_t)0)};
  alphabeta_t NULL_alphabeta = {((int16_t)0), ((int16_t)0)};
  
  FOCVars[bMotor].Iab = NULL_ab;
  FOCVars[bMotor].Ialphabeta = NULL_alphabeta;
  FOCVars[bMotor].Iqd = NULL_qd;
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true>
  if ( mode != MCM_OPEN_LOOP_VOLTAGE_MODE && mode != MCM_OPEN_LOOP_CURRENT_MODE)
  {
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
    FOCVars[bMotor].Iqdref = NULL_qd;
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true>
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
  FOCVars[bMotor].hTeref = (int16_t)0;
  FOCVars[bMotor].Vqd = NULL_qd;
  FOCVars[bMotor].Valphabeta = NULL_alphabeta;
  FOCVars[bMotor].hElAngle = (int16_t)0;

  PID_SetIntegralTerm(pPIDIq[bMotor], ((int32_t)0));
  PID_SetIntegralTerm(pPIDId[bMotor], ((int32_t)0));

  STC_Clear(pSTC[bMotor]);

  PWMC_SwitchOffPWM(pwmcHandle[bMotor]);

  <#if (MC.M1_FLUX_WEAKENING_ENABLING == true) || (MC.M2_FLUX_WEAKENING_ENABLING == true)>
  if (NULL == pFW[bMotor])
  {
    /* Nothing to do */
  }
  else
  {
    FW_Clear(pFW[bMotor]);
  }
  </#if><#-- (MC.M1_FLUX_WEAKENING_ENABLING == true) || (MC.M2_FLUX_WEAKENING_ENABLING == true) -->
  <#if (MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true) || ( MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true)>
  if (NULL == pFF[bMotor])
  {
    /* Nothing to do */
  }
  else
  {
    FF_Clear(pFF[bMotor]);
  }
  </#if><#-- (MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true) || ( MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true) -->

  <#if DWT_CYCCNT_SUPPORTED>
    <#if MC.DBG_MCU_LOAD_MEASURE == true>
  MC_Perf_Clear(&PerfTraces,bMotor);
    </#if><#-- MC.DBG_MCU_LOAD_MEASURE == true -->
  </#if><#-- DWT_CYCCNT_SUPPORTED -->
  /* USER CODE BEGIN FOC_Clear 1 */

  /* USER CODE END FOC_Clear 1 */
}

/**
  * @brief  Use this method to initialize additional methods (if any) in
  *         START_TO_RUN state.
  * @param  bMotor related motor it can be M1 or M2.
  */
__weak void FOC_InitAdditionalMethods(uint8_t bMotor) //cstat !RED-func-no-effect
{
    if (M_NONE == bMotor)
    {
      /* Nothing to do */
    }
    else
    {
  <#if (MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true) || ( MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true)>
      if (NULL == pFF[bMotor])
      {
        /* Nothing to do */
      }
      else
      {
        FF_InitFOCAdditionalMethods(pFF[bMotor]);
      }
  </#if><#-- (MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true) || ( MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true) -->
  /* USER CODE BEGIN FOC_InitAdditionalMethods 0 */

  /* USER CODE END FOC_InitAdditionalMethods 0 */
    }
}

/**
  * @brief  It computes the new values of Iqdref (current references on qd
  *         reference frame) based on the required electrical torque information
  *         provided by oTSC object (internally clocked).
  *         If implemented in the derived class it executes flux weakening and/or
  *         MTPA algorithm(s). It must be called with the periodicity specified
  *         in oTSC parameters.
  * @param  bMotor related motor it can be M1 or M2.
  */
__weak void FOC_CalcCurrRef(uint8_t bMotor)
{
  <#if ((MC.M1_MTPA_ENABLING == false) &&  (MC.M1_FLUX_WEAKENING_ENABLING == true)) ||
       ((MC.M2_MTPA_ENABLING == false) &&  (MC.M2_FLUX_WEAKENING_ENABLING == true))>
  qd_t IqdTmp;
  </#if><#-- ((MC.M1_MTPA_ENABLING == false) &&  (MC.M1_FLUX_WEAKENING_ENABLING == true)) ||
             ((MC.M2_MTPA_ENABLING == false) &&  (MC.M2_FLUX_WEAKENING_ENABLING == true)) -->
    
  /* USER CODE BEGIN FOC_CalcCurrRef 0 */

  /* USER CODE END FOC_CalcCurrRef 0 */
  <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true>
  MC_ControlMode_t mode;
  
  mode = MCI_GetControlMode( &Mci[bMotor] );
  if (INTERNAL == FOCVars[bMotor].bDriveInput
               && (mode != MCM_OPEN_LOOP_VOLTAGE_MODE && mode != MCM_OPEN_LOOP_CURRENT_MODE))
  <#else><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == false && MC.M2_DBG_OPEN_LOOP_ENABLE == false -->
  if (INTERNAL == FOCVars[bMotor].bDriveInput)
  </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true || MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
  {
    FOCVars[bMotor].hTeref = STC_CalcTorqueReference(pSTC[bMotor]);
    FOCVars[bMotor].Iqdref.q = FOCVars[bMotor].hTeref;
  <#if (MC.M1_MTPA_ENABLING == true) || (MC.M2_MTPA_ENABLING == true)>
    if (0 == pMaxTorquePerAmpere[bMotor])
    {
      /* Nothing to do */
    }
    else
    {
      MTPA_CalcCurrRefFromIq(pMaxTorquePerAmpere[bMotor], &FOCVars[bMotor].Iqdref);
    }
  </#if><#-- (MC.M1_MTPA_ENABLING == true) || (MC.M2_MTPA_ENABLING == true) -->

  <#if (MC.M1_FLUX_WEAKENING_ENABLING == true) || (MC.M2_FLUX_WEAKENING_ENABLING == true)>
    if (NULL == pFW[bMotor])
    {
      /* Nothing to do */
    }
    else
    {
    <#if MC.DRIVE_NUMBER == "1">
      <#if (MC.M1_MTPA_ENABLING == true)>
      FOCVars[bMotor].Iqdref = FW_CalcCurrRef(pFW[bMotor], FOCVars[bMotor].Iqdref);
      <#else><#-- (MC.M1_MTPA_ENABLING == false) -->
      IqdTmp.q = FOCVars[bMotor].Iqdref.q;
      IqdTmp.d = FOCVars[bMotor].UserIdref;
      FOCVars[bMotor].Iqdref = FW_CalcCurrRef(pFW[bMotor], IqdTmp);
      </#if><#-- (MC.M1_MTPA_ENABLING == true) -->
    <#else><#-- MC.DRIVE_NUMBER == 1 -->
      <#if (MC.M1_MTPA_ENABLING == true) &&  (MC.M2_MTPA_ENABLING == true)>
      FOCVars[bMotor].Iqdref = FW_CalcCurrRef(pFW[bMotor], FOCVars[bMotor].Iqdref);
      <#elseif (MC.M1_MTPA_ENABLING == false) && (MC.M2_MTPA_ENABLING == false)>
      IqdTmp.q = FOCVars[bMotor].Iqdref.q;
      IqdTmp.d = FOCVars[bMotor].UserIdref;
      FOCVars[bMotor].Iqdref = FW_CalcCurrRef(pFW[bMotor], IqdTmp);
      <#else><#-- ((MC.M1_MTPA_ENABLING == true) &&  (MC.M2_MTPA_ENABLING == false)
                || (MC.M1_MTPA_ENABLING == false) && (MC.M2_MTPA_ENABLING == true)) -->
      if (0 == pMaxTorquePerAmpere[bMotor])
      {
        IqdTmp.q = FOCVars[bMotor].Iqdref.q;
        IqdTmp.d = FOCVars[bMotor].UserIdref;
        FOCVars[bMotor].Iqdref = FW_CalcCurrRef(pFW[bMotor], IqdTmp);
      }
      else
      {
        FOCVars[bMotor].Iqdref = FW_CalcCurrRef(pFW[bMotor], FOCVars[bMotor].Iqdref);
      }     
      </#if><#-- (MC.M1_MTPA_ENABLING == true) &&  (MC.M2_MTPA_ENABLING == true) -->
    </#if><#-- MC.DRIVE_NUMBER == 1 -->
    }
  </#if><#-- (MC.M1_FLUX_WEAKENING_ENABLING == true) || (MC.M2_FLUX_WEAKENING_ENABLING == true) -->
  <#if (MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true) || ( MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true)>
    if (NULL == pFF[bMotor])
    {
      /* Nothing to do */
    }
    else
    {
      FF_VqdffComputation(pFF[bMotor], FOCVars[bMotor].Iqdref, pSTC[bMotor]);
    }
  </#if><#-- (MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true) || ( MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true) -->
  }
  else
  {
    /* Nothing to do */
  }
  /* USER CODE BEGIN FOC_CalcCurrRef 1 */

  /* USER CODE END FOC_CalcCurrRef 1 */
}
</#if><#-- FOC -->

<#if SIX_STEP>
/**
  * @brief  It re-initializes the current and voltage variables. Moreover
  *         it clears qd currents PI controllers, voltage sensor and SpeednTorque
  *         controller. It must be called before each motor restart.
  *         It does not clear speed sensor.
  * @param  bMotor related motor it can be M1 or M2.
  */
__weak void SixStep_Clear(uint8_t bMotor)
{
  /* USER CODE BEGIN SixStep_Clear 0 */

  /* USER CODE END SixStep_Clear 0 */

  STC_Clear(pSTC[bMotor]);
  SixStepVars[bMotor].DutyCycleRef = STC_GetDutyCycleRef(pSTC[bMotor]);
  <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
  BADC_Stop( &Bemf_ADC_M1 );
  BADC_Clear( &Bemf_ADC_M1 );
  BADC_SpeedMeasureOff(&Bemf_ADC_M1);
</#if>
<#if  MC.CURRENT_LIMITER_OFFSET == true >
  #if ( PID_SPEED_INTEGRAL_INIT_DIV == 0 )
    PID_SetIntegralTerm(&PIDSpeedHandle_M1, 0);
  #else
    PID_SetIntegralTerm(&PIDSpeedHandle_M1,
         (((int32_t)SixStepVars[M1].DutyCycleRef * (int16_t)PID_GetKIDivisor(&PIDSpeedHandle_M1))
            / PID_SPEED_INTEGRAL_INIT_DIV));
  #endif
  </#if><#-- MC.CURRENT_LIMITER_OFFSET == true -->
  PWMC_SwitchOffPWM(pwmcHandle[bMotor]);

  /* USER CODE BEGIN SixStep_Clear 1 */

  /* USER CODE END SixStep_Clear 1 */
}

/**
  * @brief  Use this method to initialize additional methods (if any) in
  *         START_TO_RUN state.
  * @param  bMotor related motor it can be M1 or M2.
  */
__weak void SixStep_InitAdditionalMethods(uint8_t bMotor)
{
  /* USER CODE BEGIN FOC_InitAdditionalMethods 0 */

  /* USER CODE END FOC_InitAdditionalMethods 0 */
}

/**
  * @brief  It computes the new values of Iqdref (current references on qd
  *         reference frame) based on the required electrical torque information
  *         provided by oTSC object (internally clocked).
  *         If implemented in the derived class it executes flux weakening and/or
  *         MTPA algorithm(s). It must be called with the periodicity specified
  *         in oTSC parameters.
  * @param  bMotor related motor it can be M1 or M2.
  */
__weak void SixStep_CalcSpeedRef(uint8_t bMotor)
{

  /* USER CODE BEGIN FOC_CalcCurrRef 0 */

  /* USER CODE END FOC_CalcCurrRef 0 */
  if(SixStepVars[bMotor].bDriveInput == INTERNAL)
  {
    SixStepVars[bMotor].DutyCycleRef = STC_CalcSpeedReference(pSTC[bMotor]);
  }
  else
  {
    /* Nothing to do */
  }
  /* USER CODE BEGIN FOC_CalcCurrRef 1 */

  /* USER CODE END FOC_CalcCurrRef 1 */
}
</#if><#-- SIX_STEP -->

/**
  * @brief  It set a counter intended to be used for counting the delay required
  *         for drivers boot capacitors charging of motor 1.
  * @param  hTickCount number of ticks to be counted.
  * @retval void
  */
__weak void TSK_SetChargeBootCapDelayM1(uint16_t hTickCount)
{
   hBootCapDelayCounterM1 = hTickCount;
}

/**
  * @brief  Use this function to know whether the time required to charge boot
  *         capacitors of motor 1 has elapsed.
  * @param  none
  * @retval bool true if time has elapsed, false otherwise.
  */
__weak bool TSK_ChargeBootCapDelayHasElapsedM1(void)
{
  bool retVal = false;
  if (((uint16_t)0) == hBootCapDelayCounterM1)
  {
    retVal = true;
  }
  return (retVal);
}

/**
  * @brief  It set a counter intended to be used for counting the permanency
  *         time in STOP state of motor 1.
  * @param  hTickCount number of ticks to be counted.
  * @retval void
  */
__weak void TSK_SetStopPermanencyTimeM1(uint16_t hTickCount)
{
  hStopPermanencyCounterM1 = hTickCount;
}

/**
  * @brief  Use this function to know whether the permanency time in STOP state
  *         of motor 1 has elapsed.
  * @param  none
  * @retval bool true if time is elapsed, false otherwise.
  */
__weak bool TSK_StopPermanencyTimeHasElapsedM1(void)
{
  bool retVal = false;
  if (((uint16_t)0) == hStopPermanencyCounterM1)
  {
    retVal = true;
  }
  return (retVal);
}

<#if MC.DRIVE_NUMBER != "1">
#if defined (CCMRAM_ENABLED)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief Executes medium frequency periodic Motor Control tasks
  *
  * This function performs some of the control duties on Motor 2 according to the 
  * present state of its state machine. In particular, duties requiring a periodic 
  * execution at a medium frequency rate (such as the speed controller for instance) 
  * are executed here.
  */
__weak void TSK_MediumFrequencyTaskM2(void)
{
<#if DWT_CYCCNT_SUPPORTED>
  <#if MC.DBG_MCU_LOAD_MEASURE == true>
  MC_BG_Perf_Measure_Start(&PerfTraces, MEASURE_TSK_MediumFrequencyTaskM2);
  </#if><#-- DWT_CYCCNT_SUPPORTED -->
</#if><#-- MC.DBG_MCU_LOAD_MEASURE == true -->
  /* USER CODE BEGIN MediumFrequencyTask M2 0 */

  /* USER CODE END MediumFrequencyTask M2 0 */

  int16_t wAux = 0;
<#if (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE") && (MC.M2_OTF_STARTUP == true)>
  ${PWM_Handle_Type_M2} *p_Handle = (${PWM_Handle_Type_M2} *)pwmcHandle[M2];
</#if><#-- (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE") && (MC.M2_OTF_STARTUP == true) -->
<#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
  MC_ControlMode_t mode;
  mode = MCI_GetControlMode(&Mci[M2]);
</#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->

<#if MC.M2_ICL_ENABLED == true>
  ICL_State_t ICLstate = ICL_Exec(&ICL_M2);
</#if><#-- MC.M2_ICL_ENABLED == true -->
<#if AUX_SPEED_FDBK_M2 == true>
  (void)${SPD_aux_calcAvrgMecSpeedUnit_M2}(${SPD_AUX_M2}, &wAux);
</#if><#-- AUX_SPEED_FDBK_M2 == true -->
<#if MC.M2_SPEED_FEEDBACK_CHECK == true || MC.M2_SPEED_SENSOR == "HALL_SENSOR">
  bool IsSpeedReliable = ${SPD_calcAvrgMecSpeedUnit_M2}(${SPD_M2}, &wAux);
<#else><#-- MC.M2_SPEED_FEEDBACK_CHECK = false || MC.M2_SPEED_SENSOR != "HALL_SENSOR" -->
  (void)${SPD_calcAvrgMecSpeedUnit_M2}(${SPD_M2}, &wAux);
</#if><#-- MC.M2_SPEED_FEEDBACK_CHECK == true || MC.M2_SPEED_SENSOR == "HALL_SENSOR" -->
<#if FOC>
  PQD_CalcElMotorPower(pMPM[M2]);
</#if><#-- FOC -->

<#if MC.M2_ICL_ENABLED == true>
  if ( !ICLFaultTreatedM2 && (ICLstate == ICL_ACTIVE))
  {
    ICLFaultTreatedM2 = true;
  }
  else
  {
    /* Nothing to do */
  }
</#if><#-- MC.M2_ICL_ENABLED == true -->

<#if MC.M2_ICL_ENABLED == true>
  if ((MCI_GetCurrentFaults(&Mci[M2]) == MC_NO_FAULTS) && ICLFaultTreatedM2)
<#else><#-- MC.M2_ICL_ENABLED == false -->
  if (MCI_GetCurrentFaults(&Mci[M2]) == MC_NO_FAULTS)
</#if><#-- MC.M2_ICL_ENABLED == true -->
  {
    if (MCI_GetOccurredFaults(&Mci[M2]) == MC_NO_FAULTS)
    {
      switch (Mci[M2].State)
      {
<#if MC.M2_ICL_ENABLED == true>
        case ICLWAIT:
        {
          if (ICL_INACTIVE == ICLstate)
          {
            /* If ICL is Inactive, move to IDLE */
            Mci[M2].State = IDLE;
          }
          break;
        }
</#if><#-- MC.M2_ICL_ENABLED == true -->
        
        case IDLE:
        {
          if ((MCI_START == Mci[M2].DirectCommand) || (MCI_MEASURE_OFFSETS == Mci[M2].DirectCommand))
          {
<#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || MC.M2_SPEED_SENSOR == "SENSORLESS_ADC">
  <#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
            if ( mode != MCM_OPEN_LOOP_VOLTAGE_MODE && mode != MCM_OPEN_LOOP_CURRENT_MODE) 
            {
  </#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
              RUC_Clear(&RevUpControlM2, MCI_GetImposedMotorDirection(&Mci[M2]));
  <#if SIX_STEP>
              RUC_UpdatePulse(&RevUpControlM2, &BusVoltageSensor_M2._Super);
  </#if><#-- SIX_STEP -->
  <#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
            }
            else
            {
              /* Nothing to do */
            }
  </#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
</#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")
        || MC.M2_SPEED_SENSOR == "SENSORLESS_ADC"  -->

        <#if FOC>
            if (pwmcHandle[M2]->offsetCalibStatus == false)
            {
              (void)PWMC_CurrentReadingCalibr(pwmcHandle[M2], CRC_START);
              Mci[M2].State = OFFSET_CALIB;
            }
            else
            {
             /* Calibration already done. Enables only TIM channels */
             pwmcHandle[M2]->OffCalibrWaitTimeCounter = 1u;
              (void)PWMC_CurrentReadingCalibr(pwmcHandle[M2], CRC_EXEC);
</#if><#-- FOC -->
<#if CHARGE_BOOT_CAP_ENABLING2 == true>
  <#if (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE")>  
    <#if MC.M2_DP_DESTINATION == "TIM_BKIN">
              LL_TIM_DisableBRK(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
    <#elseif MC.M2_DP_DESTINATION == "TIM_BKIN2" >
              LL_TIM_DisableBRK2(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
    </#if><#-- MC.M2_DP_DESTINATION == "TIM_BKIN" -->
  </#if> <#-- (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE") -->
  <#if (MC.M2_OTF_STARTUP == true)>
              ${PWM_TurnOnLowSides_M2}(pwmcHandle[M2],p_Handle->Half_PWMPeriod-1);
  <#else><#-- !M2_OTF_STARTUP -->
              ${PWM_TurnOnLowSides_M2}(pwmcHandle[M2],M2_CHARGE_BOOT_CAP_DUTY_CYCLES);
  </#if><#-- M2_OTF_STARTUP -->
              TSK_SetChargeBootCapDelayM2(M2_CHARGE_BOOT_CAP_TICKS);
              Mci[M2].State = CHARGE_BOOT_CAP;
<#else><#-- CHARGE_BOOT_CAP_ENABLING2 == false -->
<#-- test sensorless -->
              FOCVars[M2].bDriveInput = EXTERNAL;
              STC_SetSpeedSensor(pSTC[M2], &VirtualSpeedSensorM2._Super);
              ${SPD_clear_M2}(${SPD_M2});
              FOC_Clear(M2);
              ${PWM_SwitchOn_M2}(pwmcHandle[M2]);
              Mci[M2].State = START;
</#if><#-- CHARGE_BOOT_CAP_ENABLING2 == true -->
<#if FOC>
            }
</#if><#-- FOC -->
<#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
            OTT_Clear(&OTT);
</#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
          }
          else
          {
            /* Nothing to be done, FW stays in IDLE state */
          }
          break;
        }

<#if FOC>
        case OFFSET_CALIB:
        {
          if (MCI_STOP == Mci[M2].DirectCommand)
          {
            TSK_MF_StopProcessing(M2);
          }
          else
          {
            if (PWMC_CurrentReadingCalibr(pwmcHandle[M2], CRC_EXEC))
            {
              if (MCI_MEASURE_OFFSETS == Mci[M2].DirectCommand)
              {
                FOC_Clear(M2);
                PQD_Clear(pMPM[M2]);
                Mci[M2].DirectCommand = MCI_NO_COMMAND;
                Mci[M2].State = IDLE;
              }
              else
                {
  <#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
                Mci[M2].State = WAIT_STOP_MOTOR;
  <#else><#-- MC.MOTOR_PROFILER == false || MC.ONE_TOUCH_TUNING == false -->
    <#if CHARGE_BOOT_CAP_ENABLING2 == true>
      <#if (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE")>
        <#if MC.M2_DP_DESTINATION == "TIM_BKIN">
                LL_TIM_DisableBRK(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
        <#elseif MC.M2_DP_DESTINATION == "TIM_BKIN2" >
                LL_TIM_DisableBRK2(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
        </#if><#-- MC.M2_DP_DESTINATION == "TIM_BKIN" -->
      </#if>   <#-- (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE") -->
      <#if (MC.M2_OTF_STARTUP == true)>
                ${PWM_TurnOnLowSides_M2}(pwmcHandle[M2],p_Handle->Half_PWMPeriod-1);
      <#else><#-- !M2_OTF_STARTUP -->
                ${PWM_TurnOnLowSides_M2}(pwmcHandle[M2],M2_CHARGE_BOOT_CAP_DUTY_CYCLES);
      </#if><#-- M2_OTF_STARTUP -->
                TSK_SetChargeBootCapDelayM2(M2_CHARGE_BOOT_CAP_TICKS);
                Mci[M2].State = CHARGE_BOOT_CAP;
    <#else><#-- CHARGE_BOOT_CAP_ENABLING2 == true -->
                FOCVars[M2].bDriveInput = EXTERNAL;
                STC_SetSpeedSensor(pSTC[M2], &VirtualSpeedSensorM2._Super);
                ${SPD_clear_M2}(${SPD_M2});
                FOC_Clear(M2);
        <#if MC.M2_DISCONTINUOUS_PWM == true>
                /* Enable DPWM mode before Start */
                PWMC_DPWM_ModeEnable(pwmcHandle[M2]);
        </#if><#-- MC.M2_DISCONTINUOUS_PWM == true -->
                ${PWM_SwitchOn_M2}(pwmcHandle[M2]);
                Mci[M2].State = START;
    </#if><#-- CHARGE_BOOT_CAP_ENABLING2 == true -->
  </#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
              }
            }
            else
            {
              /* Nothing to be done, FW waits for offset calibration to finish */
            }
          }  
          break;
        }
</#if><#-- FOC -->

<#if (CHARGE_BOOT_CAP_ENABLING2 == true)>
        case CHARGE_BOOT_CAP:
        {
          if (MCI_STOP == Mci[M2].DirectCommand)
          {
            TSK_MF_StopProcessing(M2);
          }
          else
          {
            if (TSK_ChargeBootCapDelayHasElapsedM2())
            {
              ${PWM_SwitchOff_M2}(pwmcHandle[M2]);
  <#if (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE")>
    <#if MC.M2_DP_DESTINATION == "TIM_BKIN">
              LL_TIM_ClearFlag_BRK(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
              LL_TIM_EnableBRK(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
    <#elseif MC.M2_DP_DESTINATION == "TIM_BKIN2">
              LL_TIM_ClearFlag_BRK2(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
              LL_TIM_EnableBRK2(${_last_word(MC.M2_PWM_TIMER_SELECTION)});
    </#if><#-- MC.M2_DP_DESTINATION == "TIM_BKIN" -->
  </#if><#-- (MC.M2_PWM_DRIVER_PN == "STDRIVE101") && (MC.M2_DP_TOPOLOGY != "NONE") -->
<#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || M2_ENCODER>
              FOCVars[M2].bDriveInput = EXTERNAL;
              STC_SetSpeedSensor( pSTC[M2], &VirtualSpeedSensorM2._Super );
</#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || M2_ENCODER -->
<#if MC.M2_SPEED_SENSOR == "SENSORLESS_ADC">
              SixStepVars[M2].bDriveInput = EXTERNAL;
              STC_SetSpeedSensor( pSTC[M2], &VirtualSpeedSensorM2._Super );
</#if><#-- MC.M2_SPEED_SENSOR == "SENSORLESS_ADC" -->
              ${SPD_clear_M2}(${SPD_M2});
<#if AUX_SPEED_FDBK_M2 == true>
              ${SPD_aux_clear_M2}(${SPD_AUX_M2});
</#if><#-- AUX_SPEED_FDBK_M2 == true -->
<#if MC.M1_OVERMODULATION == true>
              PWMC_Clear(pwmcHandle[M2]);
</#if><#-- MC.M1_OVERMODULATION == true -->
<#if MC.M2_DISCONTINUOUS_PWM == true>
              /* Enable DPWM mode before Start */
              PWMC_DPWM_ModeEnable( pwmcHandle[M2]);
</#if><#-- MC.M2_DISCONTINUOUS_PWM == true -->
<#if FOC>
		FOC_Clear( M2 );
    <#if (MC.M2_OTF_STARTUP == true)>
              ${PWM_SwitchOn_M2}(pwmcHandle[M2]);
    </#if><#-- MC.M2_OTF_STARTUP == true -->
</#if><#-- FOC -->  
<#if SIX_STEP>
              SixStep_Clear( M2 );
</#if><#-- SIX_STEP -->
<#if MC.M2_SPEED_SENSOR == "SENSORLESS_ADC">
              BADC_SetDirection(&Bemf_ADC_M2, MCI_GetImposedMotorDirection( &Mci[M2]));
              BADC_SetSamplingPoint(&Bemf_ADC_M2, &PWM_Handle_M2._Super, pSTC[M2] );
</#if>         
<#if (MC.MOTOR_PROFILER == true)>
        SCC_Start(&SCC);
              /* The generic function needs to be called here as the undelying   
               * implementation changes in time depending on the Profiler's state 
               * machine. Calling the generic function ensures that the correct
               * implementation is invoked */
              PWMC_SwitchOnPWM(pwmcHandle[M2]);
              Mci[M2].State = START;
<#else><#-- MC.MOTOR_PROFILER == false -->
 
  <#if (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")>
              if (EAC_IsAligned(&EncAlignCtrlM2) == false)
              {
                EAC_StartAlignment(&EncAlignCtrlM2);  
                Mci[M2].State = ALIGNMENT;
              }
              else
              {
                STC_SetControlMode(pSTC[M2], MCM_SPEED_MODE);
                STC_SetSpeedSensor(pSTC[M2], &ENCODER_M2._Super);
                FOC_InitAdditionalMethods(M2);
                FOC_CalcCurrRef(M2);
                STC_ForceSpeedReferenceToCurrentSpeed(pSTC[M2]); /* Init the reference speed to current speed */
                MCI_ExecBufferedCommands(&Mci[M2]); /* Exec the speed ramp after changing of the speed sensor */
                Mci[M2].State = RUN;
              }
  <#elseif  (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
              if (EAC_IsAligned(&EncAlignCtrlM2) == false)
              {
                EAC_StartAlignment(&EncAlignCtrlM2);
                Mci[M2].State = ALIGNMENT;
              }
              else
              {
                VSS_Clear(&VirtualSpeedSensorM2); /* Reset measured speed in IDLE */
                FOC_Clear(M2);
                Mci[M2].State = START;
              }
  <#elseif MC.M2_SPEED_SENSOR == "HALL_SENSOR">
    <#if FOC>
              FOC_InitAdditionalMethods(M2);
              FOC_CalcCurrRef(M2);
    </#if><#-- FOC -->
    <#if SIX_STEP>
              SixStep_InitAdditionalMethods(M2);
              SixStep_CalcSpeedRef(M2);
      <#if MC.CURRENT_LIMITER_OFFSET == true>
#if (PID_SPEED_INTEGRAL_INIT_DIV == 0)
              PID_SetIntegralTerm(&PIDSpeedHandle_M2, 0);
#else
              PID_SetIntegralTerm(&PIDSpeedHandle_M2,
                                 (((int32_t)SixStepVars[M2].DutyCycleRef * (int16_t)PID_GetKIDivisor(&PIDSpeedHandle_M2))
                                 / PID_SPEED_INTEGRAL_INIT_DIV));
#endif
      </#if><#-- MC.CURRENT_LIMITER_OFFSET == true -->
    </#if><#-- SIX_STEP -->
              STC_ForceSpeedReferenceToCurrentSpeed( pSTC[M2]); /* Init the reference speed to current speed */
              MCI_ExecBufferedCommands(&Mci[M2]); /* Exec the speed ramp after changing of the speed sensor */
              Mci[M2].State = RUN;
  <#else><#-- sensorless mode only -->
    <#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
              if (MCM_OPEN_LOOP_VOLTAGE_MODE == mode || MCM_OPEN_LOOP_CURRENT_MODE == mode)
              {
                Mci[M2].State = RUN;
              }
              else
              {
    </#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
                Mci[M2].State = START;
    <#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
              }
    </#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
  </#if><#-- (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z") -->
       
</#if><#-- MC.MOTOR_PROFILER == true -->

<#if (MC.M2_OTF_STARTUP == false)>
              PWMC_SwitchOnPWM(pwmcHandle[M2]);

</#if><#-- (MC.M2_OTF_STARTUP == false) -->
            }
            else
            {
              /* Nothing to be done, FW waits for bootstrap capacitor to charge */
            }
          }
          break;
        }
</#if><#-- CHARGE_BOOT_CAP_ENABLING2 == true -->

<#if M2_ENCODER><#-- only for encoder -->
        case ALIGNMENT:
        {
          if (MCI_STOP == Mci[M2].DirectCommand)
          {
            TSK_MF_StopProcessing(M2);
          }
          else
          {
            bool isAligned = EAC_IsAligned(&EncAlignCtrlM2);
            bool EACDone = EAC_Exec(&EncAlignCtrlM2);
            if ((isAligned == false)  && (EACDone == false))
            {
              qd_t IqdRef;
              IqdRef.q = 0;
              IqdRef.d = STC_CalcTorqueReference(pSTC[M2]);
              FOCVars[M2].Iqdref = IqdRef;
            }
            else
            {
              ${PWM_SwitchOff_M2}( pwmcHandle[M2] );
  <#if (MC.M2_AUXILIARY_SPEED_SENSOR != "QUAD_ENCODER") && (MC.M2_AUXILIARY_SPEED_SENSOR != "QUAD_ENCODER_Z")>
              STC_SetControlMode(pSTC[M2], MCM_SPEED_MODE);
              STC_SetSpeedSensor(pSTC[M2], &ENCODER_M2._Super);
  </#if><#-- (MC.M2_AUXILIARY_SPEED_SENSOR != "QUAD_ENCODER") && (MC.M2_AUXILIARY_SPEED_SENSOR != "QUAD_ENCODER_Z") -->
              FOC_Clear(M2);
              TSK_SetStopPermanencyTimeM2(STOPPERMANENCY_TICKS);
  <#if (CHARGE_BOOT_CAP_ENABLING == true)>
              ${PWM_TurnOnLowSides_M2}(pwmcHandle[M2],M2_CHARGE_BOOT_CAP_DUTY_CYCLES);
  </#if><#-- CHARGE_BOOT_CAP_ENABLING2 == true -->
              Mci[M2].State = WAIT_STOP_MOTOR;
              /* USER CODE BEGIN MediumFrequencyTask M2 EndOfEncAlignment */
            
              /* USER CODE END MediumFrequencyTask M2 EndOfEncAlignment */
            }
          }
          break;
        }
</#if><#-- M2_ENCODER -->

<#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || MC.M2_SPEED_SENSOR == "SENSORLESS_ADC">
        case START:
        {
          if (MCI_STOP == Mci[M2].DirectCommand)
          {
            TSK_MF_StopProcessing(M2);
          }
          else
          {
          <#-- only for sensor-less control -->
            /* Mechanical speed as imposed by the Virtual Speed Sensor during the Rev Up phase. */
            int16_t hForcedMecSpeedUnit;
  <#if FOC>
            qd_t IqdRef;
  </#if><#-- FOC -->
            bool ObserverConverged;

            /* Execute the Rev Up procedure */
  <#if MC.M2_OTF_STARTUP == true>
            if (! RUC_OTF_Exec(&RevUpControlM2))
  <#else><#-- MC.M2_OTF_STARTUP == false -->
            if(! RUC_Exec(&RevUpControlM2))
  </#if><#-- MC.M2_OTF_STARTUP == true -->
            {
            /* The time allowed for the startup sequence has expired */
  <#if MC.MOTOR_PROFILER == true>
              /* However, no error is generated when OPEN LOOP is enabled 
               * since then the system does not try to close the loop... */
  <#else><#-- MC.MOTOR_PROFILER == false -->
              MCI_FaultProcessing(&Mci[M2], MC_START_UP, 0);
  </#if><#-- MC.MOTOR_PROFILER == true -->
            }
            else
            {
              /* Execute the torque open loop current start-up ramp:
               * Compute the Iq reference current as configured in the Rev Up sequence */
  <#if FOC>
              IqdRef.q = STC_CalcTorqueReference(pSTC[M2]);
              IqdRef.d = FOCVars[M2].UserIdref;
              /* Iqd reference current used by the High Frequency Loop to generate the PWM output */
              FOCVars[M2].Iqdref = IqdRef;
  </#if><#-- FOC -->
  <#if SIX_STEP>
            (void) BADC_CalcRevUpDemagTime (&Bemf_ADC_M2);
            PWMC_ForceFastDemagTime (pwmcHandle[M2], Bemf_ADC_M2.DemagCounterThreshold);
            SixStepVars[M2].DutyCycleRef = STC_CalcSpeedReference( pSTC[M2] );
  </#if>
           }
          
            (void)VSS_CalcAvrgMecSpeedUnit(&VirtualSpeedSensorM2, &hForcedMecSpeedUnit);
         
  <#if MC.M2_OTF_STARTUP == false && MC.MOTOR_PROFILER == false>
            /* Check that startup stage where the observer has to be used has been reached */
            if (true == RUC_FirstAccelerationStageReached(&RevUpControlM2))
            {
  </#if><#-- MC.M2_OTF_STARTUP == false) && (MC.MOTOR_PROFILER == false -->
  <#if (MC.M2_SPEED_SENSOR == "STO_PLL")>
              ObserverConverged = STO_PLL_IsObserverConverged(&STO_PLL_M2, &hForcedMecSpeedUnit);
              STO_SetDirection(&STO_PLL_M2, (int8_t)MCI_GetImposedMotorDirection(&Mci[M2]));
  <#elseif MC.M2_SPEED_SENSOR == "STO_CORDIC">
              ObserverConverged = STO_CR_IsObserverConverged(&STO_CR_M2, hForcedMecSpeedUnit);
  <#elseif MC.M2_SPEED_SENSOR == "SENSORLESS_ADC">
             ObserverConverged = BADC_IsObserverConverged( &Bemf_ADC_M2);
  </#if> <#-- (MC.M2_SPEED_SENSOR == "STO_PLL")     -->
              (void)VSS_SetStartTransition(&VirtualSpeedSensorM2, ObserverConverged);
  <#if MC.M2_OTF_STARTUP == false && MC.MOTOR_PROFILER == false>
            }
            else
            {
              ObserverConverged = false;
            }
  </#if><#-- MC.M2_OTF_STARTUP == false) && (MC.MOTOR_PROFILER == false -->
            if (ObserverConverged)
            {
  <#if FOC>
              qd_t StatorCurrent = MCM_Park(FOCVars[M2].Ialphabeta, SPD_GetElAngle(${SPD_M2}._Super));

              /* Start switch over ramp. This ramp will transition from the revup to the closed loop FOC */
              REMNG_Init(pREMNG[M2]);
              (void)REMNG_ExecRamp(pREMNG[M2], FOCVars[M2].Iqdref.q, 0);
              (void)REMNG_ExecRamp(pREMNG[M2], StatorCurrent.q, TRANSITION_DURATION);
  </#if><#-- FOC -->
              Mci[M2].State = SWITCH_OVER;
            }
          }
          break;
        }

        case SWITCH_OVER:
        {
          if (MCI_STOP == Mci[M2].DirectCommand)
          {
            TSK_MF_StopProcessing(M2);
          }
          else
          {
            bool LoopClosed;
            int16_t hForcedMecSpeedUnit;

  <#if MC.MOTOR_PROFILER == false><#-- No need to call RUC_Exec() when in MP in this state -->
    <#if MC.M2_OTF_STARTUP == true>
            if (! RUC_OTF_Exec(&RevUpControlM2))
    <#else><#-- MC.M2_OTF_STARTUP == false -->
            if (! RUC_Exec(&RevUpControlM2))
    </#if><#-- MC.M2_OTF_STARTUP == true -->
            {
              /* The time allowed for the startup sequence has expired */
              MCI_FaultProcessing(&Mci[M2], MC_START_UP, 0);
            } 
            else
            {
  </#if><#--  MC.MOTOR_PROFILER == false -->
              /* Compute the virtual speed and positions of the rotor. 
                 The function returns true if the virtual speed is in the reliability range */
              LoopClosed = VSS_CalcAvrgMecSpeedUnit(&VirtualSpeedSensorM2, &hForcedMecSpeedUnit);
              /* Check if the transition ramp has completed. */
              bool tempBool;
              tempBool = VSS_TransitionEnded(&VirtualSpeedSensorM2);
              LoopClosed = LoopClosed || tempBool;
              
              /* If any of the above conditions is true, the loop is considered closed. 
                 The state machine transitions to the RUN state */
              if (true ==  LoopClosed) 
              {
#if PID_SPEED_INTEGRAL_INIT_DIV == 0
                PID_SetIntegralTerm(&PIDSpeedHandle_M2, 0);
#else
  <#if FOC>
                PID_SetIntegralTerm(&PIDSpeedHandle_M2,
                                    (((int32_t)FOCVars[M2].Iqdref.q * (int16_t)PID_GetKIDivisor(&PIDSpeedHandle_M2)) 
                                    / PID_SPEED_INTEGRAL_INIT_DIV));
  </#if><#-- FOC -->
  <#if SIX_STEP>
                PID_SetIntegralTerm(&PIDSpeedHandle_M2,
                                    (((int32_t)SixStepVars[M2].DutyCycleRef * (int16_t)PID_GetKIDivisor(&PIDSpeedHandle_M2)) 
                                    / PID_SPEED_INTEGRAL_INIT_DIV));
  </#if><#-- SIX_STEP -->
#endif
  <#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
                OTT_SR(&OTT);
  </#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
                /* USER CODE BEGIN MediumFrequencyTask M2 1 */

                /* USER CODE END MediumFrequencyTask M2 1 */ 
                STC_SetSpeedSensor(pSTC[M2], ${SPD_M2}._Super); /* Observer has converged */
  <#if FOC>
                FOC_InitAdditionalMethods(M2);
                FOC_CalcCurrRef(M2);
  </#if><#-- FOC -->
  <#if SIX_STEP>
                SixStep_InitAdditionalMethods(M2);
                SixStep_CalcSpeedRef(M2);
    <#if MC.M2_SPEED_SENSOR == "SENSORLESS_ADC">
                BADC_SetLoopClosed(&Bemf_ADC_M2);
                BADC_SpeedMeasureOn(&Bemf_ADC_M2);
    </#if>
  </#if><#-- SIX_STEP -->
                STC_ForceSpeedReferenceToCurrentSpeed(pSTC[M2]); /* Init the reference speed to current speed */
                MCI_ExecBufferedCommands(&Mci[M2]); /* Exec the speed ramp after changing of the speed sensor */
                Mci[M2].State = RUN;
              }
  <#if MC.MOTOR_PROFILER == false>
            }
  </#if><#--  MC.MOTOR_PROFILER == false -->
          }
          break;
        }
</#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")
        || MC.M2_SPEED_SENSOR == "SENSORLESS_ADC" -->

        case RUN:
        {
          if (MCI_STOP == Mci[M2].DirectCommand)
          {
            TSK_MF_StopProcessing(M2);
          }
          else
          {
            /* USER CODE BEGIN MediumFrequencyTask M2 2 */
            
            /* USER CODE END MediumFrequencyTask M2 2 */
       
<#if  MC.M2_POSITION_CTRL_ENABLING == true >
            TC_PositionRegulation(pPosCtrl[M2]);
</#if><#-- MC.M2_POSITION_CTRL_ENABLING == true -->
            MCI_ExecBufferedCommands(&Mci[M2]);
<#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
            if (mode != MCM_OPEN_LOOP_VOLTAGE_MODE && mode != MCM_OPEN_LOOP_CURRENT_MODE)
            {
</#if> <#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
<#if FOC>
              FOC_CalcCurrRef(M2);
  </#if><#-- FOC -->
  <#if SIX_STEP>
            SixStep_CalcSpeedRef( M2 );
  <#if  MC.M2_SPEED_SENSOR == "SENSORLESS_ADC">
            BADC_SetSamplingPoint(&Bemf_ADC_M2, &PWM_Handle_M2._Super, pSTC[M2] );
            (void) BADC_CalcRunDemagTime (&Bemf_ADC_M2);
            PWMC_ForceFastDemagTime (pwmcHandle[M2], Bemf_ADC_M2.DemagCounterThreshold);
  </#if>	
  </#if><#-- SIX_STEP -->
         
<#if MC.M2_SPEED_FEEDBACK_CHECK == true || (MC.M2_SPEED_SENSOR == "HALL_SENSOR")>
              if(!IsSpeedReliable)
              {
                MCI_FaultProcessing(&Mci[M2], MC_SPEED_FDBK, 0);
              }
              else
              {
                /* Nothing to do */
              }
</#if><#-- MC.M2_SPEED_FEEDBACK_CHECK == true || (MC.M2_SPEED_SENSOR == "HALL_SENSOR") -->
<#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
            }
            else
            {
              int16_t hForcedMecSpeedUnit;
              /* Open Loop */
              VSS_CalcAvrgMecSpeedUnit( &VirtualSpeedSensorM2, &hForcedMecSpeedUnit);
              OL_Calc(pOpenLoop[M2]);
            }   
</#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
<#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
            OTT_MF(&OTT);
</#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
          }
          break;
        }

        case STOP:
        {
          if (TSK_StopPermanencyTimeHasElapsedM2())
          {

<#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || MC.M2_SPEED_SENSOR == "SENSORLESS_ADC">
            STC_SetSpeedSensor(pSTC[M2], &VirtualSpeedSensorM2._Super);    /* Sensor-less */
            VSS_Clear(&VirtualSpeedSensorM2); /* Reset measured speed in IDLE */
  <#if SIX_STEP>
            BADC_Clear(&Bemf_ADC_M2);
  </#if><#-- SIX_STEP -->
</#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") || MC.M2_SPEED_SENSOR == "SENSORLESS_ADC" -->
            /* USER CODE BEGIN MediumFrequencyTask M2 5 */
    
            /* USER CODE END MediumFrequencyTask M2 5 */
            Mci[M2].DirectCommand = MCI_NO_COMMAND;
            Mci[M2].State = IDLE;
          }
          else
          {
            /* Nothing to do, FW waits for to stop */
          }
          break;
        }

        case FAULT_OVER:
        {
          if (MCI_ACK_FAULTS == Mci[M2].DirectCommand)
          {
            Mci[M2].DirectCommand = MCI_NO_COMMAND;
            Mci[M2].State = IDLE;
          }
          else
          {
            /* Nothing to do, FW stays in FAULT_OVER state until acknowledgement */
          }
          break;
        }

        
        case FAULT_NOW:
        {
          Mci[M2].State = FAULT_OVER;
          break;
        }

        
<#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true || M2_ENCODER>
        case WAIT_STOP_MOTOR:
        {
          if (MCI_STOP == Mci[M2].DirectCommand)
          {
            TSK_MF_StopProcessing(M2);
          }
          else
          {
  <#if MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true>
            if (0 == SCC_DetectBemf(&SCC))
            {
              /* In a sensorless configuration. Initiate the Revup procedure */
              FOCVars[M2].bDriveInput = EXTERNAL;
              STC_SetSpeedSensor(pSTC[M2], &VirtualSpeedSensorM2._Super);
               ${SPD_clear_M2}(${SPD_M2});
              FOC_Clear(M2);
              SCC_Start(&SCC);
              /* The generic function needs to be called here as the undelying   
               * implementation changes in time depending on the Profiler's state 
               * machine. Calling the generic function ensures that the correct
               * implementation is invoked */
              PWMC_SwitchOnPWM(pwmcHandle[M2]);
              Mci[M2].State = START;
            }
            else
            {
              /* Nothing to do */
            }
  <#elseif (MC.M2_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_SPEED_SENSOR == "QUAD_ENCODER_Z")>
            if (TSK_StopPermanencyTimeHasElapsedM2())
            {
              ENC_Clear(&ENCODER_M2);
              ${PWM_SwitchOn_M2}(pwmcHandle[M2]);
    <#if MC.M2_POSITION_CTRL_ENABLING == true>
              TC_EncAlignmentCommand(pPosCtrl[M2]);
    </#if><#-- MC.M2_POSITION_CTRL_ENABLING == true -->
              FOC_InitAdditionalMethods(M2);
              STC_ForceSpeedReferenceToCurrentSpeed(pSTC[M2]); /* Init the reference speed to current speed */
              MCI_ExecBufferedCommands(&Mci[M2]); /* Exec the speed ramp after changing of the speed sensor */
              FOC_CalcCurrRef(M2);
              Mci[M2].State = RUN;
            } 
            else
            {
              /* Nothing to do */
            }
  <#elseif (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M2_AUXILIARY_SPEED_SENSOR == "QUAD_ENCODER_Z")>
            if (TSK_StopPermanencyTimeHasElapsedM2())
            {
              RUC_Clear(&RevUpControlM2, MCI_GetImposedMotorDirection(&Mci[M2]));
              ${SPD_clear_M2}(${SPD_M2});
              ENC_Clear(&ENCODER_M2);
              VSS_Clear(&VirtualSpeedSensorM2);
              ${PWM_SwitchOn_M2}(pwmcHandle[M2]);
              Mci[M2].State = START;
            } 
            else
            {
              /* Nothing to do */
            }
  </#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true -->
          }
          break;
        }
</#if><#-- MC.MOTOR_PROFILER == true || MC.ONE_TOUCH_TUNING == true || M2_ENCODER -->

        default:
          break;
       }
    }  
    else
    {
      Mci[M2].State = FAULT_OVER;
    }
  }
  else
  {
    Mci[M2].State = FAULT_NOW;
  }
<#if MC.MOTOR_PROFILER>
   <#if MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
  HT_MF(&HT);
   </#if><#-- MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR" -->
  SCC_MF(&SCC);
</#if><#-- MC.MOTOR_PROFILER == true -->
  /* USER CODE BEGIN MediumFrequencyTask M2 6 */

  /* USER CODE END MediumFrequencyTask M2 6 */
<#if DWT_CYCCNT_SUPPORTED>
  <#if MC.DBG_MCU_LOAD_MEASURE == true>
  MC_BG_Perf_Measure_Stop(&PerfTraces, MEASURE_TSK_MediumFrequencyTaskM2);
  </#if><#-- DWT_CYCCNT_SUPPORTED -->
</#if><#-- MC.DBG_MCU_LOAD_MEASURE == true -->
}



/**
  * @brief  It set a counter intended to be used for counting the delay required
  *         for drivers boot capacitors charging of motor 2.
  * @param  hTickCount number of ticks to be counted.
  * @retval void
  */
__weak void TSK_SetChargeBootCapDelayM2(uint16_t hTickCount)
{
   hBootCapDelayCounterM2 = hTickCount;
}

/**
  * @brief  Use this function to know whether the time required to charge boot
  *         capacitors of motor 2 has elapsed.
  * @param  none
  * @retval bool true if time has elapsed, false otherwise.
  */
__weak bool TSK_ChargeBootCapDelayHasElapsedM2(void)
{
  bool retVal = false;
  if (hBootCapDelayCounterM2 == ((uint16_t )0))
  {
    retVal = true;
  }
  return (retVal);
}

/**
  * @brief  It set a counter intended to be used for counting the permanency
  *         time in STOP state of motor 2.
  * @param  hTickCount number of ticks to be counted.
  * @retval void
  */
__weak void TSK_SetStopPermanencyTimeM2(uint16_t hTickCount)
{
  hStopPermanencyCounterM2 = hTickCount;
}

/**
  * @brief  Use this function to know whether the permanency time in STOP state
  *         of motor 2 has elapsed.
  * @param  none
  * @retval bool true if time is elapsed, false otherwise.
  */
__weak bool TSK_StopPermanencyTimeHasElapsedM2(void)
{
  bool retVal = false;
  if (0U == hStopPermanencyCounterM2)
  {
    retVal = true;
  }
  return (retVal);
}
</#if><#-- MC.DRIVE_NUMBER > 1 -->

<#if MC.MOTOR_PROFILER == false>
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif

/**
  * @brief  Executes the Motor Control duties that require a high frequency rate and a precise timing.
  *
  *  This is mainly the FOC current control loop. It is executed depending on the state of the Motor Control 
  * subsystem (see the state machine(s)).
  *
  * @retval Number of the  motor instance which FOC loop was executed.
  */
__weak uint8_t TSK_HighFrequencyTask(void)
{
  
  <#if FOC>
  uint16_t hFOCreturn;
  uint8_t bMotorNbr = 0; 
  <#if MC.DRIVE_NUMBER != "1">
    <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||  (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||
          (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") ||  (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
  Observer_Inputs_t STO_aux_Inputs;
  </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||  (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||
            (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") ||  (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->
  bMotorNbr = FOC_array[FOC_array_head];
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
  <#if DWT_CYCCNT_SUPPORTED>
    <#if MC.DBG_MCU_LOAD_MEASURE == true>
      <#if MC.DRIVE_NUMBER != "1">
  if(M1 == bMotorNbr){
    MC_Perf_Measure_Start(&PerfTraces, MEASURE_TSK_HighFrequencyTaskM1);
  }
  else{
    MC_Perf_Measure_Start(&PerfTraces, MEASURE_TSK_HighFrequencyTaskM2);
  }
        <#else>
  MC_Perf_Measure_Start(&PerfTraces, MEASURE_TSK_HighFrequencyTaskM1);
        </#if><#-- MC.DRIVE_NUMBER > 1 -->
      </#if><#-- MC.DBG_MCU_LOAD_MEASURE == true -->
    </#if><#-- DWT_CYCCNT_SUPPORTED -->
  /* USER CODE BEGIN HighFrequencyTask 0 */

  /* USER CODE END HighFrequencyTask 0 */
  
  

<#if MC.TESTENV == true && FOC && MC.PFC_ENABLED == false >
  /* Performance Measurement: start measure */
  start_perf_measure();
    </#if><#-- MC.TESTENV == true && MC.PFC_ENABLED == false -->

    <#if MC.DRIVE_NUMBER == "1">
      <#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")>
  Observer_Inputs_t STO_Inputs; /* Only if sensorless main */
      </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") -->
      <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
  Observer_Inputs_t STO_aux_Inputs; /* Only if sensorless aux */
  STO_aux_Inputs.Valfa_beta = FOCVars[M1].Valphabeta;  /* Only if sensorless */
      </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->

      <#if M1_ENCODER>
  (void)ENC_CalcAngle(&ENCODER_M1);   /* If not sensorless then 2nd parameter is MC_NULL */
      </#if><#-- M1_ENCODER -->
      <#if (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")>
  (void)HALL_CalcElAngle(&HALL_M1);
      </#if><#-- (MC.M1_SPEED_SENSOR == "HALL_SENSOR") || (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR") -->

      <#if ((MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC"))>
  STO_Inputs.Valfa_beta = FOCVars[M1].Valphabeta;  /* Only if sensorless */
  if (SWITCH_OVER == Mci[M1].State)
  {
    if (!REMNG_RampCompleted(pREMNG[M1]))
    {
      FOCVars[M1].Iqdref.q = (int16_t)REMNG_Calc(pREMNG[M1]);
    }
    else
    {
      /* Nothing to do */
    }
  }
  else
  {
    /* Nothing to do */
  }
      </#if><#-- ((MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")) -->
      <#if ((MC.M1_OTF_STARTUP == true))>
  if(!RUC_Get_SCLowsideOTF_Status(&RevUpControlM1))
  {
    hFOCreturn = FOC_CurrControllerM1();
  }
  else
  {
    hFOCreturn = MC_NO_ERROR;
  }
      <#else><#-- ((MC.M1_OTF_STARTUP == false)) -->
  /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_1 */

  /* USER CODE END HighFrequencyTask SINGLEDRIVE_1 */
  hFOCreturn = FOC_CurrControllerM1();
  /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_2 */

  /* USER CODE END HighFrequencyTask SINGLEDRIVE_2 */
      </#if><#-- ((MC.M1_OTF_STARTUP == true)) -->
  if(hFOCreturn == MC_DURATION)
  {
    MCI_FaultProcessing(&Mci[M1], MC_DURATION, 0);
  }
  else
  {
      <#if MC.M1_SPEED_SENSOR == "STO_PLL">
    bool IsAccelerationStageReached = RUC_FirstAccelerationStageReached(&RevUpControlM1);
      </#if>
      <#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")>
    STO_Inputs.Ialfa_beta = FOCVars[M1].Ialphabeta; /* Only if sensorless */
    STO_Inputs.Vbus = VBS_GetAvBusVoltage_d(&(BusVoltageSensor_M1._Super)); /* Only for sensorless */
    (void)${SPD_calcElAngle_M1}(${SPD_M1}, &STO_Inputs);
    ${SPD_calcAvergElSpeedDpp_M1}(${SPD_M1}); /* Only in case of Sensor-less */
        <#if MC.M1_SPEED_SENSOR == "STO_PLL">
    if (false == IsAccelerationStageReached)
    {
      STO_ResetPLL(&STO_PLL_M1);
    }
    else
    {
      /* Nothing to do */
    }
        </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") -->
        <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
    /* Only for sensor-less or open loop */
    if((START == Mci[M1].State) || (SWITCH_OVER == Mci[M1].State) || (RUN == Mci[M1].State))
        <#else><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == false -->
    /* Only for sensor-less */
    if((START == Mci[M1].State) || (SWITCH_OVER == Mci[M1].State))
        </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
    {
        <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
      if (START == Mci[M1].State )
      {
        if (0U == RUC_IsAlignStageNow(&RevUpControlM1))
        {
          PWMC_SetAlignFlag(&PWM_Handle_M1._Super, 0);
        }
        else
        {
          PWMC_SetAlignFlag(&PWM_Handle_M1._Super, 1);
        }
      }
      else
      {
        PWMC_SetAlignFlag(&PWM_Handle_M1._Super, 0);
      }
        </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
      int16_t hObsAngle = SPD_GetElAngle(${SPD_M1}._Super);
      (void)VSS_CalcElAngle(&VirtualSpeedSensorM1, &hObsAngle);
    }
      <#else><#-- (MC.M1_SPEED_SENSOR != "STO_PLL") && (MC.M1_SPEED_SENSOR != "STO_CORDIC") -->
        <#if ((MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_SPEED_SENSOR == "HALL_SENSOR")) && MC.M1_DBG_OPEN_LOOP_ENABLE == true>
    if(RUN == Mci[M1].State)
    { 
      int16_t hObsAngle = SPD_GetElAngle(${SPD_M1}._Super);
      (void)VSS_CalcElAngle(&VirtualSpeedSensorM1, &hObsAngle);
    }
    else
    {
      /* Nothing to do */
    }
        </#if><#-- ((MC.M1_SPEED_SENSOR == "QUAD_ENCODER") || (MC.M1_SPEED_SENSOR == "QUAD_ENCODER_Z") || (MC.M1_SPEED_SENSOR == "HALL_SENSOR")) && MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
      </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") -->
      <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
    STO_aux_Inputs.Ialfa_beta = FOCVars[M1].Ialphabeta; /* Only if sensorless */
    STO_aux_Inputs.Vbus = VBS_GetAvBusVoltage_d(&(BusVoltageSensor_M1._Super)); /* Only for sensorless */
    (void)${SPD_aux_calcElAngle_M1} (${SPD_AUX_M1}, &STO_aux_Inputs);
    ${SPD_aux_calcAvrgElSpeedDpp_M1} (${SPD_AUX_M1});
      </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->
    /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_3 */

    /* USER CODE END HighFrequencyTask SINGLEDRIVE_3 */  
  }
      <#if MC.DEBUG_DAC_FUNCTIONALITY_EN>
  DAC_Exec(&DAC_Handle);
      </#if><#-- MC.DEBUG_DAC_FUNCTIONALITY_EN -->
    <#else><#-- MC.DRIVE_NUMBER > 1 -->
      <#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_SPEED_SENSOR == "STO_PLL")
        || (MC.M2_SPEED_SENSOR == "STO_CORDIC") >
  Observer_Inputs_t STO_Inputs; /* Only if sensorless main */
      </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") || (MC.M2_SPEED_SENSOR == "STO_PLL")
              || (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->

  if (M1 == bMotorNbr)
  {
      <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
    STO_aux_Inputs.Valfa_beta = FOCVars[M1].Valphabeta;  /* Only if sensorless */
      </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->

      <#if M1_ENCODER>
    (void)ENC_CalcAngle(&ENCODER_M1);
      </#if><#-- M1_ENCODER -->
      <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")||( MC.M1_SPEED_SENSOR == "HALL_SENSOR")>
    (void)HALL_CalcElAngle(&HALL_M1);
      </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")||( MC.M1_SPEED_SENSOR == "HALL_SENSOR") -->
      <#if ((MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC"))>
    STO_Inputs.Valfa_beta = FOCVars[M1].Valphabeta;        /* Only if motor0 is sensorless */
    if (SWITCH_OVER == Mci[M1].State)
    {
      if (!REMNG_RampCompleted(pREMNG[M1]))
      {
        FOCVars[M1].Iqdref.q = (int16_t)REMNG_Calc(pREMNG[M1]);
      }
      else
      {
        /* Nothing to do */
      }
    }
    else
    {
      /* Nothing to do */
    }
      </#if><#-- ((MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")) -->
      <#if (true == MC.M1_OTF_STARTUP)>
    if(!RUC_Get_SCLowsideOTF_Status(&RevUpControlM1))
    {
      hFOCreturn = FOC_CurrControllerM1();
    }
    else
    {
      hFOCreturn = MC_NO_ERROR;
    }
      <#else><#-- (false == MC.M1_OTF_STARTUP) -->
  /* USER CODE BEGIN HighFrequencyTask DUALDRIVE_1 */

  /* USER CODE END HighFrequencyTask DUALDRIVE_1 */
    hFOCreturn = FOC_CurrControllerM1();
  /* USER CODE BEGIN HighFrequencyTask DUALDRIVE_2 */

  /* USER CODE END HighFrequencyTask DUALDRIVE_2 */
      </#if><#-- (true == MC.M1_OTF_STARTUP) -->
  }
  else /* bMotorNbr != M1 */
  {
      <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")>
    STO_Inputs.Valfa_beta = FOCVars[M2].Valphabeta;      /* Only if motor2 is sensorless */
      </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->
      <#if ((MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL")||(  MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC"))>
    STO_aux_Inputs.Valfa_beta = FOCVars[M2].Valphabeta;   /* Only if motor2 is sensorless */
      </#if><#-- MC.DRIVE_NUMBER == 1 -->
      <#if M2_ENCODER>
    (void)ENC_CalcAngle(&ENCODER_M2);
      </#if><#-- M2_ENCODER -->
      <#if (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")||( MC.M2_SPEED_SENSOR == "HALL_SENSOR")>
    (void)HALL_CalcElAngle(&HALL_M2);
      </#if><#-- (MC.M2_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR")||( MC.M2_SPEED_SENSOR == "HALL_SENSOR") -->
      <#if ((MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC"))>
    if (SWITCH_OVER == Mci[M2].State)
    {
      if (!REMNG_RampCompleted(pREMNG[M2]))
      {
        FOCVars[M2].Iqdref.q = ( int16_t )REMNG_Calc(pREMNG[M2]);
      }
    }
      </#if><#-- ((MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")) -->
      <#if ((MC.M2_OTF_STARTUP == true))>
    if(!RUC_Get_SCLowsideOTF_Status(&RevUpControlM2))
    {
      hFOCreturn = FOC_CurrControllerM2();
    }
    else
    {
      hFOCreturn = MC_NO_ERROR;
    }
      <#else><#-- ((MC.M2_OTF_STARTUP == false)) -->
  /* USER CODE BEGIN HighFrequencyTask DUALDRIVE_3 */

  /* USER CODE END HighFrequencyTask DUALDRIVE_3 */
    hFOCreturn = FOC_CurrControllerM2();
  /* USER CODE BEGIN HighFrequencyTask DUALDRIVE_4 */

  /* USER CODE END HighFrequencyTask DUALDRIVE_4 */
      </#if><#-- ((MC.M2_OTF_STARTUP == true)) -->
  }
  if(MC_DURATION == hFOCreturn)
  {
    MCI_FaultProcessing(&Mci[bMotorNbr], MC_DURATION, 0);
  }
  else
  {
    if (M1 == bMotorNbr)
    {
      <#if MC.M1_SPEED_SENSOR == "STO_PLL">
    bool IsAccelerationStageReached = RUC_FirstAccelerationStageReached(&RevUpControlM1);
      </#if><#-- MC.M1_SPEED_SENSOR == "STO_PLL" -->
      <#if (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC")>
      STO_Inputs.Ialfa_beta = FOCVars[M1].Ialphabeta; /* Only if sensorless*/
      STO_Inputs.Vbus = VBS_GetAvBusVoltage_d(&(BusVoltageSensor_M1._Super)); /* Only for sensorless*/
      (void)${SPD_calcElAngle_M1}(${SPD_M1}, &STO_Inputs);
      ${SPD_calcAvergElSpeedDpp_M1}(${SPD_M1}); /* Only in case of Sensor-less */
        <#if MC.M1_SPEED_SENSOR == "STO_PLL">
      if (false == IsAccelerationStageReached)
      {
        STO_ResetPLL(&STO_PLL_M1);
      }
      else
      {
        /* Nothing to do */
      }
        </#if><#-- MC.M1_SPEED_SENSOR == "STO_PLL" -->

      /* Only for sensor-less */
      if((START == Mci[M1].State) || (SWITCH_OVER == Mci[M1].State))
      {
        <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
        if (START == Mci[M1].State)
        {
          if (0U == RUC_IsAlignStageNow(&RevUpControlM1))
          {
            PWMC_SetAlignFlag(&PWM_Handle_M1._Super, 0);
          }
          else
          {
            PWMC_SetAlignFlag(&PWM_Handle_M1._Super, 1);
          }
        }
        else
        {
          PWMC_SetAlignFlag(&PWM_Handle_M1._Super, 0);
        }
        </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
        int16_t hObsAngle = SPD_GetElAngle(${SPD_M1}._Super);
        (void)VSS_CalcElAngle(&VirtualSpeedSensorM1, &hObsAngle);
      }
      else
      {
        /* Nothing to do */
      }
      </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") || (MC.M1_SPEED_SENSOR == "STO_CORDIC") -->
      <#if (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
      STO_aux_Inputs.Ialfa_beta = FOCVars[M1].Ialphabeta; /* Only if sensorless */
      STO_aux_Inputs.Vbus = VBS_GetAvBusVoltage_d(&(BusVoltageSensor_M1._Super)); /* Only for sensorless */
      (void)${SPD_aux_calcElAngle_M1}(${SPD_AUX_M1}, &STO_aux_Inputs);
      ${SPD_aux_calcAvrgElSpeedDpp_M1}(${SPD_AUX_M1});
      </#if><#-- (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") -->
    }
    else /* bMotorNbr != M1 */
    {
      <#if MC.M2_SPEED_SENSOR == "STO_PLL">
      bool IsAccelerationStageReached = RUC_FirstAccelerationStageReached(&RevUpControlM2);
      </#if><#-- MC.M2_SPEED_SENSOR == "STO_PLL" -->
      <#if (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC")>
      STO_Inputs.Ialfa_beta = FOCVars[M2].Ialphabeta; /* Only if sensorless */
      STO_Inputs.Vbus = VBS_GetAvBusVoltage_d(&(BusVoltageSensor_M2._Super)); /* Only for sensorless */
      ${SPD_calcElAngle_M2}(${SPD_M2}, &STO_Inputs);
      ${SPD_calcAvergElSpeedDpp_M2}(${SPD_M2}); /* Only in case of Sensor-less */
        <#if MC.M2_SPEED_SENSOR == "STO_PLL">
      if (false == IsAccelerationStageReached)
      {
        STO_ResetPLL(&STO_PLL_M2);
      }
      else
      {
        /* Nothing to do */
      }
        </#if><#-- MC.M2_SPEED_SENSOR == "STO_PLL" -->

      /* Only for sensor-less */
      if((START == Mci[M2].State) || (SWITCH_OVER == Mci[M2].State))
      {
        <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
        if (START == Mci[M2].State)
        {
          if (0U == RUC_IsAlignStageNow(&RevUpControlM2))
          {
            PWMC_SetAlignFlag(&PWM_Handle_M2._Super, 0);
          }
          else
          {
            PWMC_SetAlignFlag(&PWM_Handle_M2._Super, 1);
          }
        }
        else
        {
          PWMC_SetAlignFlag(&PWM_Handle_M2._Super, 0);
        }
        </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
        int16_t hObsAngle = SPD_GetElAngle(${SPD_M2}._Super);
        (void)VSS_CalcElAngle(&VirtualSpeedSensorM2, &hObsAngle);
      }
      else
      {
        /* Nothing to do */
      }
      </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") || (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->
  
      <#if (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC")>
      STO_aux_Inputs.Ialfa_beta = FOCVars[M2].Ialphabeta; /* Only if sensorless */
      STO_aux_Inputs.Vbus = VBS_GetAvBusVoltage_d(&(BusVoltageSensor_M2._Super)); /* Only for sensorless */
      ${SPD_aux_calcElAngle_M2}(${SPD_AUX_M2}, &STO_aux_Inputs);
      ${SPD_aux_calcAvrgElSpeedDpp_M2}(${SPD_AUX_M2});
      </#if><#-- (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_PLL") || (MC.M2_AUXILIARY_SPEED_SENSOR == "STO_CORDIC") --->
    }
  }
  FOC_array_head++;
  if (FOC_array_head == FOC_ARRAY_LENGTH)
  {
    FOC_array_head = 0;
  }
  else
  {
    /* Nothing to do */
  }
      <#if MC.DEBUG_DAC_FUNCTIONALITY_EN == true>
  DAC_Exec(&DAC_Handle, bMotorNbr);
      </#if><#-- MC.DEBUG_DAC_FUNCTIONALITY_EN == true -->
    </#if><#-- FOC -->
  /* USER CODE BEGIN HighFrequencyTask 1 */

  /* USER CODE END HighFrequencyTask 1 */

  <#if MC.TESTENV == true && FOC && MC.PFC_ENABLED == false >
  /* Performance Measurement: end measure */
  stop_perf_measure();
    </#if><#-- MC.TESTENV == true && MC.PFC_ENABLED == false -->
    <#if MC.MCP_ASYNC_EN>
  GLOBAL_TIMESTAMP++;
    </#if><#-- MC.MCP_ASYNC_EN -->
    <#if MC.MCP_ASYNC_OVER_UART_A_EN>
  if (0U == MCPA_UART_A.Mark)
  {
    /* Nothing to do */
  }
  else
  {
    MCPA_dataLog (&MCPA_UART_A);
  }
    </#if><#-- MC.MCP_ASYNC_OVER_UART_A_EN -->
    <#if MC.MCP_ASYNC_OVER_UART_B_EN>
  if (0U == MCPA_UART_BM_1.Mark)
  {
    /* Nothing to do */
  }
  else
  {
    MCPA_dataLog (&MCPA_UART_B);
  }
    </#if><#-- MC.MCP_ASYNC_OVER_UART_B_EN -->
    <#if MC.MCP_ASYNC_OVER_STLNK_EN>
  if (0U == MCPA_STLNK.Mark)
  {
    /* Nothing to do */
  }
  else
  {
    MCPA_dataLog (&MCPA_STLNK);
  }
    </#if><#-- MC.MCP_ASYNC_OVER_STLNK_EN -->

    <#if DWT_CYCCNT_SUPPORTED>
      <#if MC.DBG_MCU_LOAD_MEASURE == true>
        <#if MC.DRIVE_NUMBER != "1">
  if(M1 == bMotorNbr){
    MC_Perf_Measure_Stop(&PerfTraces, MEASURE_TSK_HighFrequencyTaskM1);
  }
  else{
    MC_Perf_Measure_Stop(&PerfTraces, MEASURE_TSK_HighFrequencyTaskM2);
  }
        <#else>
  MC_Perf_Measure_Stop(&PerfTraces, MEASURE_TSK_HighFrequencyTaskM1);
        </#if><#-- MC.DRIVE_NUMBER > 1 -->
      </#if><#-- MC.DG_MCU_LOAD_MEASURE == true -->
    </#if><#-- DWT_CYCCNT_SUPPORTED -->
  return (bMotorNbr);
  </#if><#-- FOC -->

  <#if SIX_STEP>
 /* USER CODE BEGIN HighFrequencyTask 0 */

  /* USER CODE END HighFrequencyTask 0 */

  uint8_t bMotorNbr = 0;
  uint16_t hSixStepReturn;
    <#if  MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
      <#if  MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
  if ((true == BADC_ClearStepUpdate(&Bemf_ADC_M1)) || (false == SixPwm_IsFastDemagUpdated(&PWM_Handle_M1)))
      <#else>
  if ((true == BADC_ClearStepUpdate(&Bemf_ADC_M1)) || (false == ThreePwm_IsFastDemagUpdated(&PWM_Handle_M1)))
      </#if>
  {
    if((START == Mci[M1].State) || (SWITCH_OVER == Mci[M1].State )) /*  only for sensor-less*/
    {
      if (START == Mci[M1].State)
      {
        if (0U == RUC_IsAlignStageNow(&RevUpControlM1))
        {
          PWMC_SetAlignFlag(&PWM_Handle_M1._Super, 0);
        }
        else
        {
          PWMC_SetAlignFlag(&PWM_Handle_M1._Super, RUC_GetDirection(&RevUpControlM1));
        }
      }
      else
      {
        PWMC_SetAlignFlag(&PWM_Handle_M1._Super, 0);
      }
      int16_t hObsAngle = SPD_GetElAngle(&VirtualSpeedSensorM1._Super);
      (void)VSS_CalcElAngle(&VirtualSpeedSensorM1, &hObsAngle);
    }
    (void)BADC_CalcElAngle (&Bemf_ADC_M1);
    /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_1 */

    /* USER CODE END HighFrequencyTask SINGLEDRIVE_1 */
      hSixStepReturn = SixStep_StatorController();
    /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_2 */

    /* USER CODE END HighFrequencyTask SINGLEDRIVE_2 */
    if(MC_DURATION == hSixStepReturn)
    {
      MCI_FaultProcessing(&Mci[bMotorNbr], MC_DURATION, 0);
    }
    else
    {
      /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_3 */

      /* USER CODE END HighFrequencyTask SINGLEDRIVE_3 */
    }
      <#if  MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
    SixPwm_UpdatePwmDemagCounter( &PWM_Handle_M1 );
      <#else>
    ThreePwm_UpdatePwmDemagCounter( &PWM_Handle_M1 );
      </#if>
  }
  else
  {
      <#if  MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
    SixPwm_DisablePwmDemagCounter( &PWM_Handle_M1 );
      <#else>
    ThreePwm_DisablePwmDemagCounter( &PWM_Handle_M1 );
      </#if>
  }
    <#elseif MC.M1_SPEED_SENSOR == "HALL_SENSOR">
  (void)HALL_CalcElAngle (&HALL_M1);
  /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_1 */

  /* USER CODE END HighFrequencyTask SINGLEDRIVE_1 */
  hSixStepReturn = SixStep_StatorController();
  /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_2 */

  /* USER CODE END HighFrequencyTask SINGLEDRIVE_2 */
  if(MC_DURATION == hSixStepReturn)
  {
    MCI_FaultProcessing(&Mci[bMotorNbr], MC_DURATION, 0);
  }
  else
  {
    /* USER CODE BEGIN HighFrequencyTask SINGLEDRIVE_3 */

    /* USER CODE END HighFrequencyTask SINGLEDRIVE_3 */
  }
</#if>
  /* USER CODE BEGIN HighFrequencyTask 1 */

  /* USER CODE END HighFrequencyTask 1 */
    <#if MC.MCP_ASYNC_EN>
  GLOBAL_TIMESTAMP++;
    </#if><#-- MC.MCP_ASYNC_EN -->
    <#if MC.MCP_ASYNC_OVER_UART_A_EN>
  if (0U == MCPA_UART_A.Mark)
  {
    /* Nothing to do */
  }
  else
  {
    MCPA_dataLog (&MCPA_UART_A);
  }
    </#if><#-- MC.MCP_ASYNC_OVER_UART_A_EN -->
    <#if MC.MCP_ASYNC_OVER_UART_B_EN>
  if (0U == MCPA_UART_B.Mark)
  {
    /* Nothing to do */
  }
  else
  {
    MCPA_dataLog (&MCPA_UART_B);
  }
    </#if><#-- MC.MCP_ASYNC_OVER_UART_B_EN -->
    <#if MC.MCP_ASYNC_OVER_STLNK_EN>
  if (0U == MCPA_STLNK.Mark)
  {
    /* Nothing to do */
  }
  else
  {
    MCPA_dataLog (&MCPA_STLNK);
  }
    </#if><#-- MC.MCP_ASYNC_OVER_STLNK_EN -->
  return (bMotorNbr);
  </#if><#-- SIX_STEP -->
}
</#if><#-- MC.MOTOR_PROFILER == false -->

<#if (MC.MOTOR_PROFILER == true)>
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief  Motor control profiler HF task
  * @param  None
  * @retval uint8_t It return always 0.
  */
__weak uint8_t TSK_HighFrequencyTask(void)
{
  ab_t Iab;
  
  <#if MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
  HALL_CalcElAngle (&HALL_M1);
  </#if><#-- MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR" -->
  
  if (SWITCH_OVER == Mci[M1].State)
  {
    if (!REMNG_RampCompleted(pREMNG[M1]))
    {
      FOCVars[M1].Iqdref.q = (int16_t)REMNG_Calc(pREMNG[M1]);
    }
    else
    {
      /* Nothing to do */
    }
  }
  else
  {
    /* Nothing to do */
  }
  <#if G4_Cut2_2_patch == true>
  RCM_ReadOngoingConv();
  RCM_ExecNextConv();
  </#if><#-- G4_Cut2_2_patch == true -->
  /* The generic function needs to be called here as the undelying   
   * implementation changes in time depending on the Profiler's state 
   * machine. Calling the generic function ensures that the correct
   * implementation is invoked */
  PWMC_GetPhaseCurrents(pwmcHandle[M1], &Iab);
  FOCVars[M1].Iab = Iab;
  SCC_SetPhaseVoltage(&SCC);
  <#if MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR">
  HT_GetPhaseShift(&HT);
  </#if><#-- MC.M1_AUXILIARY_SPEED_SENSOR == "HALL_SENSOR" -->
  
  return (0); /* Single motor only */
}
</#if><#-- (MC.MOTOR_PROFILER == true) -->

<#if MC.MOTOR_PROFILER == false>
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
  <#if FOC>
/**
  * @brief It executes the core of FOC drive that is the controllers for Iqd
  *        currents regulation. Reference frame transformations are carried out
  *        accordingly to the active speed sensor. It must be called periodically
  *        when new motor currents have been converted
  * @param this related object of class CFOC.
  * @retval int16_t It returns MC_NO_FAULTS if the FOC has been ended before
  *         next PWM Update event, MC_DURATION otherwise
  */
inline uint16_t FOC_CurrControllerM1(void)
{
  qd_t Iqd, Vqd;
  ab_t Iab;
  alphabeta_t Ialphabeta, Valphabeta;
  int16_t hElAngle;
  uint16_t hCodeError;
  SpeednPosFdbk_Handle_t *speedHandle;
    <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
  MC_ControlMode_t mode;
  
  mode = MCI_GetControlMode( &Mci[M1] );
    </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
  speedHandle = STC_GetSpeedSensor(pSTC[M1]);
  hElAngle = SPD_GetElAngle(speedHandle);
    <#if (MC.M1_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_SPEED_SENSOR == "STO_CORDIC")>
  hElAngle += SPD_GetInstElSpeedDpp(speedHandle)*PARK_ANGLE_COMPENSATION_FACTOR;
    </#if><#-- (MC.M1_SPEED_SENSOR == "STO_PLL") ||  (MC.M1_SPEED_SENSOR == "STO_CORDIC") -->
  PWMC_GetPhaseCurrents(pwmcHandle[M1], &Iab);
    <#if NoInjectedChannel>
      <#if G4_Cut2_2_patch>
  RCM_ReadOngoingConv();
      </#if><#-- G4_Cut2_2_patch -->
  RCM_ExecNextConv();
    </#if><#-- NoInjectedChannel -->
    <#if (MC.M1_AMPLIFICATION_GAIN?number <0)>
  /* As the Gain is negative, we invert the current read */
  Iab.a = -Iab.a;
  Iab.b = -Iab.b;
    </#if><#-- (MC.M1_AMPLIFICATION_GAIN?number <0) -->
  Ialphabeta = MCM_Clarke(Iab);
  Iqd = MCM_Park(Ialphabeta, hElAngle);
  Vqd.q = PI_Controller(pPIDIq[M1], (int32_t)(FOCVars[M1].Iqdref.q) - Iqd.q);
  Vqd.d = PI_Controller(pPIDId[M1], (int32_t)(FOCVars[M1].Iqdref.d) - Iqd.d);
    <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true>
  Vqd = FF_VqdConditioning(pFF[M1],Vqd);
    </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
    <#if MC.M1_DBG_OPEN_LOOP_ENABLE == true>
  if (mode == MCM_OPEN_LOOP_VOLTAGE_MODE)
  {
    Vqd = OL_VqdConditioning(pOpenLoop[M1]);
  }
  else
  {
    /* Nothing to do */
  }
    </#if><#-- MC.M1_DBG_OPEN_LOOP_ENABLE == true -->
  Vqd = Circle_Limitation(&CircleLimitationM1, Vqd);
  hElAngle += SPD_GetInstElSpeedDpp(speedHandle)*REV_PARK_ANGLE_COMPENSATION_FACTOR;
  Valphabeta = MCM_Rev_Park(Vqd, hElAngle);
    <#if NoInjectedChannel &&  !G4_Cut2_2_patch>
  RCM_ReadOngoingConv();
    </#if><#-- NoInjectedChannel &&  !G4_Cut2_2_patch -->
  hCodeError = PWMC_SetPhaseVoltage${OVM}(pwmcHandle[M1], Valphabeta);
    <#if (((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || MC.M1_OVERMODULATION == true) && ((MC.M1_CURRENT_SENSING_TOPO != 'ICS_SENSORS'))>
  PWMC_CalcPhaseCurrentsEst(pwmcHandle[M1],Iqd, hElAngle);
    </#if><#-- (((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || MC.M1_OVERMODULATION == true) && (MC.M1_CURRENT_SENSING_TOPO != 'ICS_SENSORS') -->

  FOCVars[M1].Vqd = Vqd;
  FOCVars[M1].Iab = Iab;
  FOCVars[M1].Ialphabeta = Ialphabeta;
  FOCVars[M1].Iqd = Iqd;
  FOCVars[M1].Valphabeta = Valphabeta;
  FOCVars[M1].hElAngle = hElAngle;

    <#if MC.M1_FLUX_WEAKENING_ENABLING == true>
  FW_DataProcess(pFW[M1], Vqd);
    </#if><#-- MC.M1_FLUX_WEAKENING_ENABLING == true -->
    <#if MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true>
  FF_DataProcess(pFF[M1]);
    </#if><#-- MC.M1_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
  return (hCodeError);
}
  </#if><#-- FOC -->

  <#if SIX_STEP>
inline uint16_t SixStep_StatorController(void)
{
  uint16_t hCodeError = MC_NO_ERROR;
    <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
  int16_t hElAngle, hSpeed, hDirection;
  SpeednPosFdbk_Handle_t *speedHandle;
  RCM_ReadOngoingConv();
  RCM_ExecNextConv();
  speedHandle = STC_GetSpeedSensor(pSTC[M1]);
      <#if MC.DRIVE_MODE == "VM">
  if(false == BADC_IsObserverConverged(&Bemf_ADC_M1))
  {
    hElAngle = SPD_GetElAngle(speedHandle);
  }
  else
  {
    hElAngle = SPD_GetElAngle(&Bemf_ADC_M1._Super);
  }
      <#else><#-- MC.DRIVE_MODE != "VM" -->
  hElAngle = SPD_GetElAngle(speedHandle);
      </#if><#-- MC.DRIVE_MODE == "VM" -->
  hSpeed = SPD_GetElSpeedDpp(speedHandle);
  hDirection = RUC_GetDirection(&RevUpControlM1);
      <#if MC.DRIVE_MODE == "VM">
  PWMC_SetPhaseVoltage( pwmcHandle[M1], SixStepVars[M1].DutyCycleRef );
      <#else><#-- MC.DRIVE_MODE != "VM" -->
  PWMC_SetPhaseVoltage( pwmcHandle[M1], PWM_Handle_M1._Super.StartCntPh);
  CRM_SetReference( &CurrentRef_M1, SixStepVars[M1].DutyCycleRef );
      </#if><#-- MC.DRIVE_MODE == "VM" -->
  if (hDirection > 0)
  {
    SixStepVars[M1].qElAngle = hElAngle + S16_90_PHASE_SHIFT;
  }
  else
  {
    SixStepVars[M1].qElAngle = hElAngle - S16_90_PHASE_SHIFT;
  }
  PWM_Handle_M1._Super.hElAngle = SixStepVars[M1].qElAngle;
<#if  MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
  SixPwm_LoadNextStep( &PWM_Handle_M1, hDirection );
  if (true == SixPwm_ApplyNextStep(&PWM_Handle_M1))
      <#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
  ThreePwm_LoadNextStep( &PWM_Handle_M1, hDirection );
  if (true == ThreePwm_ApplyNextStep(&PWM_Handle_M1))
      </#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
  {
    if (false == Bemf_ADC_M1.IsLoopClosed)
    {
      BADC_StepChangeEvent(&Bemf_ADC_M1, hSpeed, &PWM_Handle_M1._Super);
    }
  }
  BADC_Start(&Bemf_ADC_M1, PWM_Handle_M1._Super.Step ); 
    <#elseif MC.M1_SPEED_SENSOR == "HALL_SENSOR">
  int16_t hElAngle, hSpeed;
  SpeednPosFdbk_Handle_t *speedHandle;
  speedHandle = STC_GetSpeedSensor(pSTC[M1]);
  hElAngle = SPD_GetElAngle(speedHandle);
  hSpeed = STC_GetMecSpeedRefUnit(pSTC[M1]);
  if (hSpeed > 0)
  {
    SixStepVars[M1].qElAngle = hElAngle + S16_90_PHASE_SHIFT;
  }
  else
  {
    SixStepVars[M1].qElAngle = hElAngle - S16_90_PHASE_SHIFT;
  }
  PWM_Handle_M1._Super.hElAngle = SixStepVars[M1].qElAngle;
      <#if MC.DRIVE_MODE == "VM">
  PWMC_SetPhaseVoltage( pwmcHandle[M1], SixStepVars[M1].DutyCycleRef );
      <#else><#-- MC.DRIVE_MODE != "VM" -->
  PWMC_SetPhaseVoltage( pwmcHandle[M1], PWM_Handle_M1._Super.StartCntPh);
  CRM_SetReference( &CurrentRef_M1, SixStepVars[M1].DutyCycleRef );
      </#if><#-- MC.DRIVE_MODE == "VM" -->
      <#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
  SixPwm_LoadNextStep(&PWM_Handle_M1, hSpeed);
  SixPwm_ApplyNextStep(&PWM_Handle_M1);
      <#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
  ThreePwm_LoadNextStep(&PWM_Handle_M1, hSpeed);
  ThreePwm_ApplyNextStep(&PWM_Handle_M1);
      </#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->
    </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
  return(hCodeError);
}
  </#if><#-- SIX_STEP -->

  <#if (MC.DRIVE_NUMBER != "1")>
#if defined (CCMRAM)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief It executes the core of FOC drive that is the controllers for Iqd
  *        currents regulation of motor 2. Reference frame transformations are carried out
  *        accordingly to the active speed sensor. It must be called periodically
  *        when new motor currents have been converted.
  * @param This related object of class CFOC.
  * @retval int16_t It returns MC_NO_FAULTS if the FOC has been ended before
  *         next PWM Update event, MC_DURATION otherwise.
  */
inline uint16_t FOC_CurrControllerM2(void)
{
  ab_t Iab;
  alphabeta_t Ialphabeta, Valphabeta;
  qd_t Iqd, Vqd;

  int16_t hElAngle;
  uint16_t hCodeError;
  SpeednPosFdbk_Handle_t *speedHandle;

  speedHandle = STC_GetSpeedSensor(pSTC[M2]);
  hElAngle = SPD_GetElAngle(speedHandle);
    <#if (MC.M2_SPEED_SENSOR == "STO_PLL") ||  (MC.M2_SPEED_SENSOR == "STO_CORDIC")>
  hElAngle += SPD_GetInstElSpeedDpp(speedHandle) * PARK_ANGLE_COMPENSATION_FACTOR2;
    </#if><#-- (MC.M2_SPEED_SENSOR == "STO_PLL") ||  (MC.M2_SPEED_SENSOR == "STO_CORDIC") -->
  PWMC_GetPhaseCurrents(pwmcHandle[M2], &Iab);
    <#if NoInjectedChannel>
      <#if G4_Cut2_2_patch>
  RCM_ReadOngoingConv();
      </#if><#-- G4_Cut2_2_patch -->
  RCM_ExecNextConv();
    </#if><#-- NoInjectedChannel -->
    <#if (MC.M2_AMPLIFICATION_GAIN?number <0)>
  /* As the Gain is negative, we invert the current read */
  Iab.a = -Iab.a;
  Iab.b = -Iab.b;
    </#if><#-- (MC.M2_AMPLIFICATION_GAIN?number <0) -->
  Ialphabeta = MCM_Clarke(Iab);
  Iqd = MCM_Park(Ialphabeta, hElAngle);
  Vqd.q = PI_Controller(pPIDIq[M2], (int32_t)(FOCVars[M2].Iqdref.q) - Iqd.q);
  Vqd.d = PI_Controller(pPIDId[M2], (int32_t)(FOCVars[M2].Iqdref.d) - Iqd.d);
    <#if MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true>
  Vqd = FF_VqdConditioning(pFF[M2],Vqd);
    </#if><#-- MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
  
    <#if MC.M2_DBG_OPEN_LOOP_ENABLE == true>
  Vqd = OL_VqdConditioning(pOpenLoop[M2]);
    </#if><#-- MC.M2_DBG_OPEN_LOOP_ENABLE == true -->
  Vqd = Circle_Limitation(&CircleLimitationM2, Vqd);
  hElAngle += SPD_GetInstElSpeedDpp(speedHandle) * REV_PARK_ANGLE_COMPENSATION_FACTOR2;
  Valphabeta = MCM_Rev_Park(Vqd, hElAngle);
  hCodeError = PWMC_SetPhaseVoltage${OVM2}(pwmcHandle[M2], Valphabeta);
    <#if (((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || MC.M2_OVERMODULATION == true) && (MC.M2_CURRENT_SENSING_TOPO != 'ICS_SENSORS')>
  PWMC_CalcPhaseCurrentsEst(pwmcHandle[M2],Iqd, hElAngle);
    </#if><#-- (((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || MC.M2_OVERMODULATION == true) && (MC.M2_CURRENT_SENSING_TOPO != 'ICS_SENSORS') -->
    <#if NoInjectedChannel && !G4_Cut2_2_patch>
  RCM_ReadOngoingConv();
    </#if><#-- NoInjectedChannel && !G4_Cut2_2_patch -->
  FOCVars[M2].Vqd = Vqd;
  FOCVars[M2].Iab = Iab;
  FOCVars[M2].Ialphabeta = Ialphabeta;
  FOCVars[M2].Iqd = Iqd;
  FOCVars[M2].Valphabeta = Valphabeta;
  FOCVars[M2].hElAngle = hElAngle;
    <#if MC.M2_FLUX_WEAKENING_ENABLING == true>
  FW_DataProcess(pFW[M2], Vqd);
    </#if><#-- MC.M2_FLUX_WEAKENING_ENABLING == true -->
    <#if MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true>
  FF_DataProcess(pFF[M2]);
    </#if><#-- MC.M2_FEED_FORWARD_CURRENT_REG_ENABLING == true -->
  return(hCodeError);
}
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
</#if><#--  MC.MOTOR_PROFILER == false -->

/**
  * @brief  Executes safety checks (e.g. bus voltage and temperature) for all drive instances. 
  *
  * Faults flags are updated here.
  */
__weak void TSK_SafetyTask(void)
{
  /* USER CODE BEGIN TSK_SafetyTask 0 */

  /* USER CODE END TSK_SafetyTask 0 */
  if (1U == bMCBootCompleted)
  {
<#if (MC.MOTOR_PROFILER == true)>
    SCC_CheckOC_RL(&SCC);
</#if><#-- (MC.MOTOR_PROFILER == true) -->
<#if (MC.M1_ON_OVER_VOLTAGE == "TURN_OFF_PWM")>
    TSK_SafetyTask_PWMOFF(M1);
    <#elseif ( MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE")>
    TSK_SafetyTask_RBRK(M1);
    <#elseif ( MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES")>
    TSK_SafetyTask_LSON(M1);
</#if><#-- (MC.M1_ON_OVER_VOLTAGE == "TURN_OFF_PWM") -->
<#if (MC.DRIVE_NUMBER != "1")>
    /* Second drive */
  <#if (MC.M2_ON_OVER_VOLTAGE == "TURN_OFF_PWM")>
    TSK_SafetyTask_PWMOFF(M2);
    <#elseif ( MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE")>
    TSK_SafetyTask_RBRK(M2);
    <#elseif ( MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES")>
    TSK_SafetyTask_LSON(M2);
  </#if><#-- (MC.M2_ON_OVER_VOLTAGE == "TURN_OFF_PWM") -->
</#if><#-- (MC.DRIVE_NUMBER > 1) -->
    /* User conversion execution */
    RCM_ExecUserConv();
  /* USER CODE BEGIN TSK_SafetyTask 1 */

  /* USER CODE END TSK_SafetyTask 1 */
  }
  else
  {
    /* Nothing to do */
  }
}

<#if MC.M1_ON_OVER_VOLTAGE == "TURN_OFF_PWM" || MC.M2_ON_OVER_VOLTAGE == "TURN_OFF_PWM">
/**
  * @brief  Safety task implementation if  MC.M1_ON_OVER_VOLTAGE == TURN_OFF_PWM.
  * @param  bMotor Motor reference number defined
  *         \link Motors_reference_number here \endlink.
  */
__weak void TSK_SafetyTask_PWMOFF(uint8_t bMotor)
{
  /* USER CODE BEGIN TSK_SafetyTask_PWMOFF 0 */

  /* USER CODE END TSK_SafetyTask_PWMOFF 0 */
  uint16_t CodeReturn = MC_NO_ERROR;
  <#if MC.M1_BUS_VOLTAGE_READING == true || (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true) || (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true)>
    <#if MC.DRIVE_NUMBER != "1">
  const uint16_t errMask[NBR_OF_MOTORS] = {VBUS_TEMP_ERR_MASK, VBUS_TEMP_ERR_MASK2};
    <#else><#-- MC.DRIVE_NUMBER == 1 -->
  const uint16_t errMask[NBR_OF_MOTORS] = {VBUS_TEMP_ERR_MASK};
    </#if><#-- MC.DRIVE_NUMBER > 1 -->
  </#if><#-- MC.M1_BUS_VOLTAGE_READING == true || (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true) || (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true) -->
  /* Check for fault if FW protection is activated. It returns MC_OVER_TEMP or MC_NO_ERROR */
<#if (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true)>
  if (M1 == bMotor)
  {
    uint16_t rawValueM1 = RCM_ExecRegularConv(&TempRegConv_M1);
    CodeReturn |= errMask[bMotor] & NTC_CalcAvTemp(&TempSensor_M1, rawValueM1);
  }
  else
  {
    /* Nothing to do */
  }  
</#if>  
<#if (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true)>
  if (M2 == bMotor)
  {
    uint16_t rawValueM2 = RCM_ExecRegularConv(&TempRegConv_M2);
    CodeReturn |= errMask[bMotor] & NTC_CalcAvTemp(&TempSensor_M2, rawValueM2);
  }
  else
  {
    /* Nothing to do */
  }  
</#if>
  CodeReturn |= PWMC_IsFaultOccurred(pwmcHandle[bMotor]);     /* check for fault. It return MC_OVER_CURR or MC_NO_FAULTS
                                                    (for STM32F30x can return MC_OVER_VOLT in case of HW Overvoltage) */
  <#if MC.M1_BUS_VOLTAGE_READING == true>
  if (M1 == bMotor)
  {
    uint16_t rawValueM1 =  RCM_ExecRegularConv(&VbusRegConv_M1);
    CodeReturn |= errMask[bMotor] & RVBS_CalcAvVbus(&BusVoltageSensor_M1, rawValueM1);
  }
  else
  {
    /* Nothing to do */
  }
  <#else><#-- MC.M1_BUS_VOLTAGE_READING == false -->
  <#-- Nothing to do here the virtual voltage does not need computations nor measurement and it cannot fail... -->
  </#if><#-- MC.M1_BUS_VOLTAGE_READING == true -->
  <#if MC.M2_BUS_VOLTAGE_READING == true>
  if (M2 == bMotor)
  {
    uint16_t rawValueM2 =  RCM_ExecRegularConv(&VbusRegConv_M2);
    CodeReturn |= errMask[bMotor] & RVBS_CalcAvVbus(&BusVoltageSensor_M2, rawValueM2);
  }
  else
  {
    /* Nothing to do */
  }
  <#else><#-- MC.M2_BUS_VOLTAGE_READING == false -->
<#-- Nothing to do here the virtual voltage does not need computations nor measurement and it cannot fail... -->
  </#if><#-- MC.M2_BUS_VOLTAGE_READING == true -->
  MCI_FaultProcessing(&Mci[bMotor], CodeReturn, ~CodeReturn); /* Process faults */

  <#if (MC.M1_ICL_ENABLED == true)>
  if ((M1 == bMotor) && (MC_UNDER_VOLT == (CodeReturn & MC_UNDER_VOLT)) && ICLFaultTreatedM1){
    ICLFaultTreatedM1 = false;
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- (MC.M1_ICL_ENABLED == true) -->

  <#if (MC.M2_ICL_ENABLED == true)>
  if ((M2 == bMotor) && (MC_UNDER_VOLT == (CodeReturn & MC_UNDER_VOLT)) && ICLFaultTreatedM2){
    ICLFaultTreatedM2 = false;
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- (MC.M2_ICL_ENABLED == true) -->

  if (MCI_GetFaultState(&Mci[bMotor]) != (uint32_t)MC_NO_FAULTS)
  {
  <#if (MC.MOTOR_PROFILER == true)>
      SCC_Stop(&SCC);
      OTT_Stop(&OTT);
  </#if><#-- (MC.MOTOR_PROFILER == true) -->
  <#if M1_ENCODER || M2_ENCODER>
    /* Reset Encoder state */
    if (pEAC[bMotor] != MC_NULL)
    {
      EAC_SetRestartState(pEAC[bMotor], false);
    }
    else
    {
      /* Nothing to do */
    }
  </#if><#-- M1_ENCODER || M2_ENCODER -->
    PWMC_SwitchOffPWM(pwmcHandle[bMotor]);
  <#if MC.MCP_ASYNC_OVER_UART_A_EN>
    if (MCPA_UART_A.Mark != 0U)
    {
      MCPA_flushDataLog (&MCPA_UART_A);
    }
    else
    {
      /* Nothing to do */
    }
  </#if><#-- MC.MCP_ASYNC_OVER_UART_A_EN -->
  <#if MC.MCP_ASYNC_OVER_UART_B_EN>
    if (MCPA_UART_B.Mark != 0)
    {
      MCPA_flushDataLog (&MCPA_UART_B);
    }
    else
    {
      /* Nothing to do */
    }
  </#if><#-- C.MCP_DATALOG_OVER_UART_B -->
  <#if MC.MCP_ASYNC_OVER_STLNK_EN>
    if (MCPA_STLNK.Mark != 0)
    {
      MCPA_flushDataLog (&MCPA_STLNK);
    }
    else
    {
      /* Nothing to do */
    }
  </#if><#-- MC.MCP_ASYNC_OVER_STLNK_EN -->
  <#if FOC>
    FOC_Clear(bMotor);
    PQD_Clear(pMPM[bMotor]); //cstat !MISRAC2012-Rule-11.3
  </#if><#-- FOC -->
  <#if SIX_STEP>
    SixStep_Clear(bMotor);
  </#if><#-- SIX_STEP -->
    /* USER CODE BEGIN TSK_SafetyTask_PWMOFF 1 */

    /* USER CODE END TSK_SafetyTask_PWMOFF 1 */
  }
  else
  {
    /* No errors */
  }
  <#if  MC.SMOOTH_BRAKING_ACTION_ON_OVERVOLTAGE == true>
  /* Smooth braking action on overvoltage */
  if(M1 == bMotor)
  {
    busd = (uint16_t)VBS_GetAvBusVoltage_d(&(BusVoltageSensor_M1._Super));
  }
  else if(M2 == bMotor)
  {
    busd = (uint16_t)VBS_GetAvBusVoltage_d(&(BusVoltageSensor_M2._Super));
  }

  if ((Mci[bMotor].State == IDLE)||
     ((Mci[bMotor].State== RUN)&&(FOCVars[bMotor].Iqdref.q>0)))
  {
    nominalBusd[bMotor] = busd;
  }
  else
  {
    if((Mci[bMotor].State == RUN) && (FOCVars[bMotor].Iqdref.q<0))
    {
      if (busd > ((ovthd[bMotor] + nominalBusd[bMotor]) >> 1))
      {
        FOCVars[bMotor].Iqdref.q = 0;
        FOCVars[bMotor].Iqdref.d = 0;
      }
      else
      {
        /* Nothing to do */
      }
    }
   else
    {
      /* Nothing to do */
    }
  }
    /* USER CODE BEGIN TSK_SafetyTask_PWMOFF SMOOTH_BREAKING */

    /* USER CODE END TSK_SafetyTask_PWMOFF SMOOTH_BREAKING */
  </#if><#-- MC.SMOOTH_BRAKING_ACTION_ON_OVERVOLTAGE -->
  /* USER CODE BEGIN TSK_SafetyTask_PWMOFF 3 */

  /* USER CODE END TSK_SafetyTask_PWMOFF 3 */
}
</#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_OFF_PWM" || MC.M2_ON_OVER_VOLTAGE == "TURN_OFF_PWM" -->
<#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE">
/**
  * @brief  Safety task implementation if  MC.M1_ON_OVER_VOLTAGE == TURN_ON_R_BRAKE.
  * @param  motor Motor reference number defined
  *         \link Motors_reference_number here \endlink.
  */
__weak void TSK_SafetyTask_RBRK(uint8_t bMotor)
{
  /* USER CODE BEGIN TSK_SafetyTask_RBRK 0 */

  /* USER CODE END TSK_SafetyTask_RBRK 0 */
  uint16_t CodeReturn = MC_NO_ERROR;
  uint16_t BusVoltageFaultsFlag = MC_OVER_VOLT;
  <#if (MC.M1_BUS_VOLTAGE_READING == true) || (MC.M2_BUS_VOLTAGE_READING == true)>
    <#if MC.DRIVE_NUMBER != "1">
  uint16_t errMask[NBR_OF_MOTORS] = {VBUS_TEMP_ERR_MASK, VBUS_TEMP_ERR_MASK2};
    <#else><#-- MC.DRIVE_NUMBER == 1 -->
  uint16_t errMask[NBR_OF_MOTORS] = {VBUS_TEMP_ERR_MASK};
    </#if><#-- MC.DRIVE_NUMBER > 1 -->
  </#if>
  /* Brake resistor management */
  <#if MC.M1_BUS_VOLTAGE_READING == true>
  if (M1 == bMotor)
  {
    uint16_t rawValueM1 =  RCM_ExecRegularConv(&VbusRegConv_M1);
    BusVoltageFaultsFlag =  errMask[bMotor] & RVBS_CalcAvVbus(&BusVoltageSensor_M1, rawValueM1);
  }
  else
  {
    /* Nothing to do */
  }
  <#else><#--MC.M1_BUS_VOLTAGE_READING == true -->
 <#-- Nothing to do here the virtual voltage does not need computations nor measurement and it cannot fail... -->
  </#if><#-- MC.M1_BUS_VOLTAGE_READING == true -->
  <#if MC.M2_BUS_VOLTAGE_READING == true>
  if (M2 == bMotor)
  {
    uint16_t rawValueM2 =  RCM_ExecRegularConv(&VbusRegConv_M2);
    BusVoltageFaultsFlag = errMask[bMotor] & RVBS_CalcAvVbus(&BusVoltageSensor_M2, rawValueM2);
  }
  else
  {
    /* Nothing to do */
  }
  <#else><#-- MC.M2_BUS_VOLTAGE_READING == false -->
<#-- Nothing to do here the virtual voltage does not need computations nor measurement and it cannot fail... -->
  </#if><#-- MC.M2_BUS_VOLTAGE_READING == true -->
  if (MC_OVER_VOLT == BusVoltageFaultsFlag)
  {
    DOUT_SetOutputState(pR_Brake[bMotor], ACTIVE);
  }
  else
  {
    DOUT_SetOutputState(pR_Brake[bMotor], INACTIVE);
  }
  /* Check for fault if FW protection is activated. It returns MC_OVER_TEMP or MC_NO_ERROR */
<#if (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true)>
  if (M1 == bMotor)
  {
    uint16_t rawValueM1 = RCM_ExecRegularConv(&TempRegConv_M1);
    CodeReturn |= errMask[bMotor] & NTC_CalcAvTemp(&TempSensor_M1, rawValueM1);
  }
  else
  {
    /* Nothing to do */
  }  
</#if>  
<#if (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true)>
  if (M2 == bMotor)
  {
    uint16_t rawValueM2 = RCM_ExecRegularConv(&TempRegConv_M2);
    CodeReturn |= errMask[bMotor] & NTC_CalcAvTemp(&TempSensor_M2, rawValueM2);
  }
  else
  {
    /* Nothing to do */
  }  
</#if>
  CodeReturn |= PWMC_IsFaultOccurred(pwmcHandle[bMotor]);     /* Check for fault. It return MC_OVER_CURR or MC_NO_FAULTS
                                                    (for STM32F30x can return MC_OVER_VOLT in case of HW Overvoltage) */
  CodeReturn |= (BusVoltageFaultsFlag & MC_UNDER_VOLT);  /* MC_UNDER_VOLT generates fault if FW protection is activated,
                                                                                  MC_OVER_VOLT doesn't generate fault */
  MCI_FaultProcessing(&Mci[bMotor], CodeReturn, ~CodeReturn);  /* Update the STM according error code */

  <#if (MC.M1_ICL_ENABLED == true)>
  if ((M1 == bMotor) && (MC_UNDER_VOLT == (BusVoltageFaultsFlag & MC_UNDER_VOLT)) && ICLFaultTreatedM1)
  {
    ICLFaultTreatedM1 = false;
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- (MC.M1_ICL_ENABLED == true) -->

  <#if (MC.M2_ICL_ENABLED == true)>
  if ((M2 == bMotor) && (MC_UNDER_VOLT == (BusVoltageFaultsFlag & MC_UNDER_VOLT)) && ICLFaultTreatedM2)
  {
    ICLFaultTreatedM2 = false;
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- (MC.M2_ICL_ENABLED == true) -->
  
  if (MCI_GetFaultState(&Mci[bMotor]) != (uint32_t)MC_NO_FAULTS)
  {
  <#if (MC.MOTOR_PROFILER == true)>
      SCC_Stop(&SCC);
      OTT_Stop(&OTT);
  </#if><#-- (MC.MOTOR_PROFILER == true) -->
  <#if M1_ENCODER || M2_ENCODER>
      /* Reset Encoder state */
      if (pEAC[bMotor] != MC_NULL)
      {
        EAC_SetRestartState( pEAC[bMotor], false );
      }
      else
      {
        /* Nothing to do */
      }
  </#if><#-- M1_ENCODER || M2_ENCODER -->
      PWMC_SwitchOffPWM(pwmcHandle[bMotor]);
  <#if MC.MCP_ASYNC_OVER_UART_A_EN>
    if (MCPA_UART_A.Mark != 0U)
    { /* Dual motor not yet supported */
      MCPA_flushDataLog (&MCPA_UART_A);
    }
    else
    {
      /* Nothing to do */
    }
  </#if><#-- MC.MCP_ASYNC_OVER_UART_A_EN -->
  <#if MC.MCP_ASYNC_OVER_UART_B_EN>
    if (MCPA_UART_B.Mark != 0)
    { /* Dual motor not yet supported */
      MCPA_flushDataLog (&MCPA_UART_B);
    }
    else
    {
      /* Nothing to do */
    }
  </#if><#-- MC.MCP_ASYNC_OVER_UART_B_EN -->
  <#if MC.MCP_ASYNC_OVER_STLNK_EN>
    if (MCPA_STLNK.Mark != 0)
    { /* Dual motor not yet supported */
      MCPA_flushDataLog (&MCPA_STLNK);
    }
    else
    {
      /* Nothing to do */
    }
  </#if><#-- MC.MCP_ASYNC_OVER_STLNK_EN -->
    FOC_Clear(bMotor);
    PQD_Clear(pMPM[bMotor]); //cstat !MISRAC2012-Rule-11.3
    /* USER CODE BEGIN TSK_SafetyTask_RBRK 1 */

    /* USER CODE END TSK_SafetyTask_RBRK 1 */
  }
  else
  {
    /* Nothing to do */
  }
  /* USER CODE BEGIN TSK_SafetyTask_RBRK 2 */

  /* USER CODE END TSK_SafetyTask_RBRK 2 */
}
</#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_R_BRAKE" -->
<#if MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES">
/**
  * @brief  Safety task implementation if  MC.M1_ON_OVER_VOLTAGE == TURN_ON_LOW_SIDES.
  * @param  motor Motor reference number defined
  *         \link Motors_reference_number here \endlink.
  */
__weak void TSK_SafetyTask_LSON(uint8_t bMotor)
{
  /* USER CODE BEGIN TSK_SafetyTask_LSON 0 */

  /* USER CODE END TSK_SafetyTask_LSON 0 */
  uint16_t CodeReturn = MC_NO_ERROR;
  <#if MC.DRIVE_NUMBER != "1">
  uint16_t errMask[NBR_OF_MOTORS] = {VBUS_TEMP_ERR_MASK, VBUS_TEMP_ERR_MASK2};
  <#else><#-- MC.DRIVE_NUMBER == 1 -->
  uint16_t errMask[NBR_OF_MOTORS] = {VBUS_TEMP_ERR_MASK};
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
  bool TurnOnLowSideAction;
  
  TurnOnLowSideAction = PWMC_GetTurnOnLowSidesAction(pwmcHandle[bMotor]);
  /* Check for fault if FW protection is activated */
<#if (MC.M1_TEMPERATURE_READING == true  && MC.M1_OV_TEMPERATURE_PROT_ENABLING == true)>
  if (M1 == bMotor)
  {
    uint16_t rawValueM1 = RCM_ExecRegularConv(&TempRegConv_M1);
    CodeReturn |= errMask[bMotor] & NTC_CalcAvTemp(&TempSensor_M1, rawValueM1);
  }
  else
  {
    /* Nothing to do */
  }  
</#if>  
<#if (MC.M2_TEMPERATURE_READING == true  && MC.M2_OV_TEMPERATURE_PROT_ENABLING == true)>
  if (M2 == bMotor)
  {
    uint16_t rawValueM2 = RCM_ExecRegularConv(&TempRegConv_M2);
    CodeReturn |= errMask[bMotor] & NTC_CalcAvTemp(&TempSensor_M2, rawValueM2);
  }
  else
  {
    /* Nothing to do */
  }  
</#if>
  CodeReturn |= PWMC_IsFaultOccurred(pwmcHandle[bMotor]); /* For fault. It return MC_OVER_CURR or MC_NO_FAULTS
                                                   (for STM32F30x can return MC_OVER_VOLT in case of HW Overvoltage) */
  /* USER CODE BEGIN TSK_SafetyTask_LSON 1 */

  /* USER CODE END TSK_SafetyTask_LSON 1 */
  <#if MC.M1_BUS_VOLTAGE_READING == true>
  if (M1 == bMotor)
  {
    uint16_t rawValueM1 =  RCM_ExecRegularConv(&VbusRegConv_M1);  
    CodeReturn |= errMask[bMotor] & RVBS_CalcAvVbus(&BusVoltageSensor_M1, rawValueM1);
  }
  else
  {
    /* Nothing to do */
  }
  <#else><#-- MC.M1_BUS_VOLTAGE_READING == false -->
  <#-- Nothing to do here the virtual voltage does not need computations nor measurement and it cannot fail... -->
  </#if><#-- MC.M1_BUS_VOLTAGE_READING == true -->
  <#if MC.M2_BUS_VOLTAGE_READING == true>
  if (M2 == bMotor)
  {
    uint16_t rawValueM2 =  RCM_ExecRegularConv(&VbusRegConv_M2);
    CodeReturn |= errMask[bMotor] & RVBS_CalcAvVbus(&BusVoltageSensor_M2, rawValueM2);
  }
  else
  {
    /* Nothing to do */
  }
  <#else><#-- MC.M2_BUS_VOLTAGE_READING == false -->
  <#-- Nothing to do here the virtual voltage does not need computations nor measurement and it cannot fail... -->
  </#if><#-- MC.M2_BUS_VOLTAGE_READING == true -->
  MCI_FaultProcessing(&Mci[bMotor], CodeReturn, ~CodeReturn); /* Update the STM according error code */
  
  <#if (MC.M1_ICL_ENABLED == true)>
  if ((M1 == bMotor) && (MC_UNDER_VOLT == (CodeReturn & MC_UNDER_VOLT)) && ICLFaultTreatedM1){
    ICLFaultTreatedM1 = false;
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- (MC.M1_ICL_ENABLED == true) -->

  <#if (MC.M2_ICL_ENABLED == true)>
  if ((M2 == bMotor) && (MC_UNDER_VOLT == (CodeReturn & MC_UNDER_VOLT)) && ICLFaultTreatedM2)
  {
    ICLFaultTreatedM2 = false;
  }
  else
  {
    /* Nothing to do */
  }
  </#if><#-- (MC.M2_ICL_ENABLED == true) -->
  
  if ((MC_OVER_VOLT == (CodeReturn & MC_OVER_VOLT)) && (false == TurnOnLowSideAction))
  {
  <#if M1_ENCODER || M2_ENCODER>
    /* Reset Encoder state */
    if (pEAC[bMotor] != MC_NULL)
    {
      EAC_SetRestartState(pEAC[bMotor], false);
    }
    else
    {
      /* Nothing to do */
    }
  </#if><#-- M1_ENCODER || M2_ENCODER -->
    /* Start turn on low side action */
    PWMC_SwitchOffPWM(pwmcHandle[bMotor]); /* Required before PWMC_TurnOnLowSides */
  <#if (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true)>
    DOUT_SetOutputState(pOCPDisabling[bMotor], ACTIVE); /* Disable the OCP */
  </#if><#-- (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true) -->
    /* USER CODE BEGIN TSK_SafetyTask_LSON 2 */

    /* USER CODE END TSK_SafetyTask_LSON 2 */
    PWMC_TurnOnLowSides(pwmcHandle[bMotor], 0UL); /* Turn on Low side switches */
  }
  else
  {
    switch (Mci[bMotor].State) /* Is state equal to FAULT_NOW or FAULT_OVER */
    {
    
      case IDLE:
      {
        /* After a OV occurs the turn on low side action become active. It is released just after a fault acknowledge
         * -> state == IDLE */
        if (true == TurnOnLowSideAction)
        {
          /* End of TURN_ON_LOW_SIDES action */
  <#if (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true)>
          DOUT_SetOutputState(pOCPDisabling[bMotor], INACTIVE); /* Re-enable the OCP */
  </#if><#-- (MC.M1_HW_OV_CURRENT_PROT_BYPASS == true) -->
          PWMC_SwitchOffPWM(pwmcHandle[bMotor]);  /* Switch off the PWM */
        }
        else
        {
          /* Nothing to do */
        }
        /* USER CODE BEGIN TSK_SafetyTask_LSON 3 */

        /* USER CODE END TSK_SafetyTask_LSON 3 */
        break;
      }
      
      case FAULT_NOW:
      {
        if (TurnOnLowSideAction == false)
        {
  <#if M1_ENCODER || M2_ENCODER>
          /* Reset Encoder state */
          if (pEAC[bMotor] != MC_NULL)
          {
            EAC_SetRestartState(pEAC[bMotor], false);
          }
          else
          {
            /* Nothing to do */
          }
  </#if><#-- M1_ENCODER || M2_ENCODER -->
          /* Switching off the PWM if fault occurs must be done just if TURN_ON_LOW_SIDES action is not in place */
          PWMC_SwitchOffPWM(pwmcHandle[bMotor]);
  <#if MC.MCP_ASYNC_OVER_UART_A_EN>
          if (MCPA_UART_A.Mark != 0U)
          { /* Dual motor not yet supported */
            MCPA_flushDataLog (&MCPA_UART_A);
          }
          else
          {
            /* Nothing to do */
          }
  </#if><#-- MC.MCP_ASYNC_OVER_UART_A_EN -->
  <#if MC.MCP_ASYNC_OVER_UART_B_EN>
          if (MCPA_UART_B.Mark != 0)
          { /* Dual motor not yet supported */
            MCPA_flushDataLog (&MCPA_UART_B);
          }
          else
          {
            /* Nothing to do */
          }
  </#if><#-- MC.MCP_ASYNC_OVER_UART_B_EN -->
  <#if MC.MCP_ASYNC_OVER_STLNK_EN>
          if (MCPA_STLNK.Mark != 0)
          { /* Dual motor not yet supported */
            MCPA_flushDataLog (&MCPA_STLNK);
          }
          else
          {
            /* Nothing to do */
          }
  </#if><#-- MC.MCP_ASYNC_OVER_STLNK_EN -->
          FOC_Clear(bMotor);
          /* Misra violation Rule 11.3 A cast shall not be performed between a pointer to object 
           * type and a pointer to a different object type. */
          PQD_Clear(pMPM[bMotor]);
        }
        /* USER CODE BEGIN TSK_SafetyTask_LSON 4 */

        /* USER CODE END TSK_SafetyTask_LSON 4 */
        break;
      }
      
      case FAULT_OVER:
      {
        if (TurnOnLowSideAction == false)
        {
          /* Switching off the PWM if fault occurs must be done just if TURN_ON_LOW_SIDES action is not in place */
          PWMC_SwitchOffPWM(pwmcHandle[bMotor]);
  <#if MC.MCP_ASYNC_OVER_UART_A_EN>
          if (MCPA_UART_A.Mark != 0U)
          { /* Dual motor not yet supported */
            MCPA_flushDataLog (&MCPA_UART_A);
          }
          else
          {
            /* Nothing to do */
          }
  </#if><#-- MC.MCP_ASYNC_OVER_UART_A_EN -->
  <#if MC.MCP_ASYNC_OVER_UART_B_EN>
          if (MCPA_UART_B.Mark != 0)
          { /* Dual motor not yet supported */
            MCPA_flushDataLog (&MCPA_UART_B);
          }
          else
          {
            /* Nothing to do */
          }
  </#if><#-- MC.MCP_ASYNC_OVER_UART_B_EN -->
  <#if MC.MCP_ASYNC_OVER_STLNK_EN>
          if (MCPA_STLNK.Mark != 0)
          { /* Dual motor not yet supported */
            MCPA_flushDataLog (&MCPA_STLNK);
          }
          else
          {
            /* Nothing to do */
          }
  </#if><#-- MC.MCP_ASYNC_OVER_STLNK_EN -->
        }
        /* USER CODE BEGIN TSK_SafetyTask_LSON 5 */

        /* USER CODE END TSK_SafetyTask_LSON 5 */
        break;
      }
      
      default:
        break;
    }
  }
  /* USER CODE BEGIN TSK_SafetyTask_LSON 6 */

  /* USER CODE END TSK_SafetyTask_LSON 6 */
}
</#if><#-- MC.M1_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" || MC.M2_ON_OVER_VOLTAGE == "TURN_ON_LOW_SIDES" -->

<#if MC.DRIVE_NUMBER != "1">
#if defined (CCMRAM_ENABLED)
#if defined (__ICCARM__)
#pragma location = ".ccmram"
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".ccmram")))
#endif
#endif
/**
  * @brief Reserves FOC execution on ADC ISR half a PWM period in advance.
  *
  *  This function is called by TIMx_UP_IRQHandler in case of dual MC and
  * it allows to reserve half PWM period in advance the FOC execution on
  * ADC ISR.
  * @param  pDrive Pointer on the FOC Array.
  */
__weak void TSK_DualDriveFIFOUpdate(uint8_t Motor)
{
  FOC_array[FOC_array_tail] = Motor;
  FOC_array_tail++;
  if (FOC_ARRAY_LENGTH == FOC_array_tail)
  {
    FOC_array_tail = 0;
  }
  else
  {
    /* Nothing to do */
  }
}
</#if><#-- MC.DRIVE_NUMBER > 1 -->

<#if FOC>
/**
  * @brief  This function returns the reference of the MCInterface relative to
  *         the selected drive.
  * @param  bMotor Motor reference number defined
  *         \link Motors_reference_number here \endlink.
  * @retval MCI_Handle_t * Reference to MCInterface relative to the selected drive.
  *         Note: it can be MC_NULL if MCInterface of selected drive is not
  *         allocated.
  */
__weak MCI_Handle_t *GetMCI(uint8_t bMotor)
{
  MCI_Handle_t *retVal = MC_NULL; //cstat !MISRAC2012-Rule-8.13
  if (bMotor < (uint8_t)NBR_OF_MOTORS)
  {
    retVal = &Mci[bMotor];
  }
  else
  {
    /* Nothing to do */
  }
  return (retVal);
}
</#if><#-- FOC -->

/**
  * @brief  Puts the Motor Control subsystem in in safety conditions on a Hard Fault
  *
  *  This function is to be executed when a general hardware failure has been detected  
  * by the microcontroller and is used to put the system in safety condition.
  */
__weak void TSK_HardwareFaultTask(void)
{
  /* USER CODE BEGIN TSK_HardwareFaultTask 0 */

  /* USER CODE END TSK_HardwareFaultTask 0 */
<#if FOC>
  <#if (MC.MOTOR_PROFILER == true)>
  SCC_Stop(&SCC);
  OTT_Stop(&OTT);
  </#if><#-- (MC.MOTOR_PROFILER == true) -->
</#if><#-- FOC -->
  ${PWM_SwitchOff}(pwmcHandle[M1]);
  MCI_FaultProcessing(&Mci[M1], MC_SW_ERROR, 0);
<#if MC.DRIVE_NUMBER != "1">
  ${PWM_SwitchOff_M2}(pwmcHandle[M2]);
  MCI_FaultProcessing(&Mci[M2], MC_SW_ERROR, 0);
</#if><#-- MC.DRIVE_NUMBER > 1 -->

  /* USER CODE BEGIN TSK_HardwareFaultTask 1 */

  /* USER CODE END TSK_HardwareFaultTask 1 */
}
<#if MC.PFC_ENABLED == true>

/**
  * @brief  Executes the PFC Task.
  */
void PFC_Scheduler(void)
{
  PFC_Task(&PFC);
}
</#if><#-- MC.PFC_ENABLED == true -->
<#if MC.RTOS == "FREERTOS">

/* startMediumFrequencyTask function */
void startMediumFrequencyTask(void const * argument)
{
  <#if MC.CUBE_MX_VER == "xxx">
  /* Init code for MotorControl */
  MX_MotorControl_Init();
  <#else><#-- MC.CUBE_MX_VER != "xxx" -->
<#assign cubeVersion = MC.CUBE_MX_VER?replace(".","") >
    <#if cubeVersion?number < 540>
  /* Init code for MotorControl */
  MX_MotorControl_Init();
    </#if><#-- cubeVersion?number < 540 -->
  </#if><#-- MC.CUBE_MX_VER == "xxx" -->
  /* USER CODE BEGIN MF task 1 */
  /* Infinite loop */
  for(;;)
  {
    /* Delay of 500us */
    vTaskDelay(1);
    MC_RunMotorControlTasks();
  }
  /* USER CODE END MF task 1 */
}

/* startSafetyTask function */
void StartSafetyTask(void const * argument)
{
  /* USER CODE BEGIN SF task 1 */
  /* Infinite loop */
  for(;;)
  {
    /* Delay of 500us */
    vTaskDelay(1);
    TSK_SafetyTask();
  }
  /* USER CODE END SF task 1 */ 
}

</#if><#-- MC.RTOS == "FREERTOS" -->

<#if MC.START_STOP_BTN == true>
__weak void UI_HandleStartStopButton_cb (void)
{
/* USER CODE BEGIN START_STOP_BTN */
  if (IDLE == MC_GetSTMStateMotor1())
  {
    /* Ramp parameters should be tuned for the actual motor */
    (void)MC_StartMotor1();
  }
  else
  {
    (void)MC_StopMotor1();
  }
/* USER CODE END START_STOP_BTN */
}
</#if><#-- MC.START_STOP_BTN == true -->

 /**
  * @brief  Locks GPIO pins used for Motor Control to prevent accidental reconfiguration.
  */
__weak void mc_lock_pins (void)
{
<#list configs as dt>
<#list dt.peripheralGPIOParams.values() as io>
<#list io.values() as ipIo>
<#list ipIo.entrySet() as e>
<#if (e.getKey().equals("GPIO_Label")) && (e.getValue()?matches("^M[0-9]+_.*$"))>
LL_GPIO_LockPin(${e.getValue()}_GPIO_Port, ${e.getValue()}_Pin);
</#if><#-- (e.getKey().equals("GPIO_Label")) && (e.getValue()?matches("^M[0-9]+_.*$")) -->
</#list>
</#list>
</#list>
</#list>
}
/* USER CODE BEGIN mc_task 0 */

/* USER CODE END mc_task 0 */

/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/
