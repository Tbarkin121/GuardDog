<#ftl>
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

<#assign FOC = MC.M1_DRIVE_TYPE == "FOC" || MC.M2_DRIVE_TYPE == "FOC">
<#assign SIX_STEP = MC.M1_DRIVE_TYPE == "SIX_STEP" || MC.M2_DRIVE_TYPE == "SIX_STEP">
<#assign ACIM = MC.M1_DRIVE_TYPE == "ACIM" || MC.M2_DRIVE_TYPE == "ACIM">

<#-- Motor 1 Tables -->
<#assign M1_OPAMPInputMapp_Shared_U =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W}
  ]>
 <#assign M1_ADCConfig_2OPAMPs_Shared_U =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W}
  ]>

<#assign M1_OPAMPInputMapp_Shared_V =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W}
  ]>

 <#assign M1_ADCConfig_2OPAMPs_Shared_V =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 5, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 6, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W}
  ]>

<#assign M1_OPAMPInputMapp_Shared_W =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W}
  ]>

 <#assign M1_ADCConfig_2OPAMPs_Shared_W =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U},
    {"Sector": 2, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 3, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 4, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U}
  ]>

<#assign M1_OPAMPInputMapp_3_OPAMPS =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_OPAMP_U+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_OPAMP_V+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M1_CS_OPAMP_W+"_NonInvertingInput_"+MC.M1_CS_OPAMP_VPSEL_W}
  ]>

 <#assign M1_ADCConfig_3ADCS =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W}
  ]>

<#assign M1_ADCConfig_Shared_U =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_SHARED,
                  "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_SHARED,
                  "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W}
  ]>

<#assign M1_ADCConfig_Shared_V =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U,
                  "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_SHARED},
    {"Sector": 5, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U,
                  "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_SHARED},
    {"Sector": 6, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_W+"_"+MC.M1_CS_CHANNEL_W}
  ]>

<#assign M1_ADCConfig_Shared_W =
  [ {"Sector": 1, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U,
                  "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_SHARED},
    {"Sector": 3, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U,
                  "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_SHARED},
    {"Sector": 4, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_U, "PHASE_2": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M1_CS_ADC_V+"_"+MC.M1_CS_CHANNEL_V, "PHASE_2": MC.M1_CS_ADC_U+"_"+MC.M1_CS_CHANNEL_W}
  ]>


<#-- Motor 2 Tables -->
<#assign M2_OPAMPInputMapp_Shared_U =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W}
  ]>
 <#assign M2_ADCConfig_2OPAMPs_Shared_U =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W}
  ]>

<#assign M2_OPAMPInputMapp_Shared_V =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W}
  ]>

 <#assign M2_ADCConfig_2OPAMPs_Shared_V =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 5, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 6, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W}
  ]>

<#assign M2_OPAMPInputMapp_Shared_W =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W}
  ]>

 <#assign M2_ADCConfig_2OPAMPs_Shared_W =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U},
    {"Sector": 2, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 3, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 4, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U}
  ]>

<#assign M2_OPAMPInputMapp_3_OPAMPS =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_OPAMP_U+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_U,
                  "PHASE_2": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_OPAMP_V+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_V,
                  "PHASE_2": MC.M2_CS_OPAMP_W+"_NonInvertingInput_"+MC.M2_CS_OPAMP_VPSEL_W}
  ]>

 <#assign M2_ADCConfig_3ADCS =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W}
  ]>

<#assign M2_ADCConfig_Shared_U =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_SHARED,
                  "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_SHARED,
                  "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W}
  ]>

<#assign M2_ADCConfig_Shared_V =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 3, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 4, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U,
                  "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_SHARED},
    {"Sector": 5, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U,
                  "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_SHARED},
    {"Sector": 6, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_W+"_"+MC.M2_CS_CHANNEL_W}
  ]>

<#assign M2_ADCConfig_Shared_W =
  [ {"Sector": 1, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_W},
    {"Sector": 2, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U,
                  "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_SHARED},
    {"Sector": 3, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U,
                  "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_SHARED},
    {"Sector": 4, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 5, "PHASE_1": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_U, "PHASE_2": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V},
    {"Sector": 6, "PHASE_1": MC.M2_CS_ADC_V+"_"+MC.M2_CS_CHANNEL_V, "PHASE_2": MC.M2_CS_ADC_U+"_"+MC.M2_CS_CHANNEL_W}
  ]>


 <#function getOPAMPMap Motor>
   <#if Motor == 1>
     <#switch MC.M1_CS_OPAMP_PHASE_SHARED>
       <#case "U">
         <#local OPAMPMapp = M1_OPAMPInputMapp_Shared_U>
         <#break>
       <#case "V">
         <#local OPAMPMapp = M1_OPAMPInputMapp_Shared_V>
         <#break>
       <#case "W">
         <#local OPAMPMapp = M1_OPAMPInputMapp_Shared_W>
         <#break>
       <#default><#-- No phase shared at Opmap Level -->
         <#local OPAMPMapp = M1_OPAMPInputMapp_3_OPAMPS>
         <#break>
     </#switch>
   <#elseif Motor==2>
   <#-- To be implemented -->
     <#switch MC.M2_CS_OPAMP_PHASE_SHARED>
       <#case "U">
         <#local OPAMPMapp = M2_OPAMPInputMapp_Shared_U>
         <#break>
       <#case "V">
         <#local OPAMPMapp = M2_OPAMPInputMapp_Shared_V>
         <#break>
       <#case "W">
         <#local OPAMPMapp = M2_OPAMPInputMapp_Shared_W>
         <#break>
       <#default><#-- No phase shared at Opmap Level -->
         <#local OPAMPMapp = M2_OPAMPInputMapp_3_OPAMPS>
         <#break>
     </#switch>
   </#if>
   <#return OPAMPMapp>
</#function>

<#assign M1_OPAMPip = {"U":MC.M1_CS_OPAMP_U ,"V":MC.M1_CS_OPAMP_V ,"W":MC.M1_CS_OPAMP_W }>
<#assign M2_OPAMPip = {"U":MC.M2_CS_OPAMP_U ,"V":MC.M2_CS_OPAMP_V ,"W":MC.M2_CS_OPAMP_W }>

<#function getOPAMP config Sector Phase Motor=1>
  <#local OPAMPMap=getOPAMPMap(Motor)>
  <#list OPAMPMap as OPAMPItem>
  <#if Motor == 1>
    <#if OPAMPItem.Sector == Sector>
      <#if MC.M1_CS_OPAMP_DYNAMIC_OUTPUT_SWITCH == false>
        <#if config=="IP">
          <#return _first_word(OPAMPItem[Phase])>
        <#else>
          <#if MC.M1_CS_OPAMP_NUM == '3'>
            <#return "OPAMP_UNCHANGED">
          <#else>  
            <#return OPAMPItem[Phase]>
          </#if>
        </#if>
      <#else><#-- Dynamic opamp to configure -->
        <#if _first_word(OPAMPItem[Phase]) == M1_OPAMPip[MC.M1_CS_ADC_PHASE_SHARED]>
          <#if config=="IP">
            <#return M1_OPAMPip[MC.M1_CS_ADC_PHASE_SHARED]>
          <#else>
            <#if MC.M1_CS_OPAMP_DIRECT_LINK_TO_ADC == getADC("IP",Sector,Phase,1)>
              <#return "DIRECT_CONNECT|"+OPAMPItem[Phase]>
            <#else>
              <#return "PIN_CONNECT|"+OPAMPItem[Phase]>
            </#if>
          </#if>
        <#else><#-- Not an OPAMP we configure return "NULL" -->
          <#if config=="IP">
            <#return _first_word(OPAMPItem[Phase])>
          <#else>
            <#return "OPAMP_UNCHANGED">
          </#if>
        </#if>
      </#if>
    </#if>
  <#elseif Motor == 2>  
    <#if OPAMPItem.Sector == Sector>
      <#if MC.M2_CS_OPAMP_DYNAMIC_OUTPUT_SWITCH == false>
        <#if config=="IP">
          <#return _first_word(OPAMPItem[Phase])>
        <#else>
          <#if MC.M2_CS_OPAMP_NUM == '3'>
            <#return "OPAMP_UNCHANGED">
          <#else>  
            <#return OPAMPItem[Phase]>
          </#if>
        </#if>
      <#else><#-- Dynamic opamp to configure -->
        <#if _first_word(OPAMPItem[Phase]) == M2_OPAMPip[MC.M2_CS_ADC_PHASE_SHARED]>
          <#if config=="IP">
            <#return M2_OPAMPip[MC.M2_CS_ADC_PHASE_SHARED]>
          <#else>
            <#if MC.M2_CS_OPAMP_DIRECT_LINK_TO_ADC == getADC("IP",Sector,Phase,2)>
              <#return "DIRECT_CONNECT|"+OPAMPItem[Phase]>
            <#else>
              <#return "PIN_CONNECT|"+OPAMPItem[Phase]>
             </#if>
          </#if>
        <#else><#-- Not an OPAMP we configure return "NULL" -->
          <#if config=="IP">
            <#return _first_word(OPAMPItem[Phase])>
          <#else>
            <#return "OPAMP_UNCHANGED">
          </#if>
        </#if>
      </#if>
    </#if>
  <#else>
    #error motor number could not exceed 2
  </#if>
  </#list>
</#function>

<#function getADCMap Motor>
   <#if Motor == 1>
     <#switch MC.M1_CS_ADC_PHASE_SHARED>
       <#case "U">
         <#local ADCMap = M1_ADCConfig_Shared_U>
         <#break>
       <#case "V">
         <#local ADCMap = M1_ADCConfig_Shared_V>
         <#break>
       <#case "W">
         <#local ADCMap = M1_ADCConfig_Shared_W>
         <#break>
       <#default><#-- No pahse shared at ADC Level -->
         <#switch MC.M1_CS_OPAMP_PHASE_SHARED>
           <#case "U">
             <#local ADCMap=M1_ADCConfig_2OPAMPs_Shared_U>
             <#break>
           <#case "V">
             <#local ADCMap=M1_ADCConfig_2OPAMPs_Shared_V>
             <#break>
           <#case "W">
             <#local ADCMap=M1_ADCConfig_2OPAMPs_Shared_W>
             <#break>
           <#default>
             <#local ADCMap=M1_ADCConfig_3ADCS><#-- No phase shared at ADC level not OPAMP level -->
            <#break>
         </#switch>
       <#break>
     </#switch>
   <#elseif Motor==2>
   <#-- To be implemented -->
     <#switch MC.M2_CS_ADC_PHASE_SHARED>
       <#case "U">
         <#local ADCMap = M2_ADCConfig_Shared_U>
         <#break>
       <#case "V">
         <#local ADCMap = M2_ADCConfig_Shared_V>
         <#break>
       <#case "W">
         <#local ADCMap = M2_ADCConfig_Shared_W>
         <#break>
       <#default><#-- No pahse shared at ADC Level -->
         <#switch MC.M2_CS_OPAMP_PHASE_SHARED>
           <#case "U">
             <#local ADCMap=M2_ADCConfig_2OPAMPs_Shared_U>
             <#break>
           <#case "V">
             <#local ADCMap=M2_ADCConfig_2OPAMPs_Shared_V>
             <#break>
           <#case "W">
             <#local ADCMap=M2_ADCConfig_2OPAMPs_Shared_W>
             <#break>
           <#default>
             <#local ADCMap=M2_ADCConfig_3ADCS><#-- No phase shared at ADC level not OPAMP level -->
            <#break>
         </#switch>
       <#break>
     </#switch>
   </#if>
   <#return ADCMap>
</#function>

<#assign G4_ADC_Channel_Table =
  [ {"VPOPAMP": "VPOPAMP1", "ADC": "ADC1", "CHANNEL": 13},
    {"VPOPAMP": "VPOPAMP2", "ADC": "ADC2", "CHANNEL": 16},
	{"VPOPAMP": "VPOPAMP3", "ADC": "ADC2", "CHANNEL": 18},
	{"VPOPAMP": "VPOPAMP3", "ADC": "ADC3", "CHANNEL": 13},
	{"VPOPAMP": "VPOPAMP6", "ADC": "ADC4", "CHANNEL": 17},
	{"VPOPAMP": "VPOPAMP5", "ADC": "ADC5", "CHANNEL": 3},
	{"VPOPAMP": "VPOPAMP4", "ADC": "ADC5", "CHANNEL": 5}
  ]>

<#function getADCChannel ADC channelIN>
  <#if CondFamily_STM32G4 >
    <#if channelIN?contains("VPOPAMP")>
      <#local ADCChannelMap = G4_ADC_Channel_Table>
      <#list ADCChannelMap as ADCChannelItem>
        <#if ADCChannelItem.VPOPAMP == channelIN && ADCChannelItem.ADC == ADC>
          <#return ADCChannelItem.CHANNEL>
        </#if>
      <#else> 
        <#return "OPAMP_ADC_CHANNEL_ERROR" >
	  </#list>  
    <#else> <#-- Classic ADC Channel inside G4 family -->
      <#return channelIN>
	</#if>
  <#else> <#-- Only G4 has the OPAMP ADC direct connection capability -->
    <#return channelIN>
  </#if>
</#function>

<#function getADC config sector phase Motor=1>
 <#local ADCMap = getADCMap(Motor)>
 <#if Motor == 1>
  <#list ADCMap as ADCItem>
     <#if ADCItem.Sector==sector>
       <#if config="IP">
         <#return _first_word(ADCItem[phase])>
       <#else>
         <#return getADCChannel (_first_word(ADCItem[phase]),_last_word(ADCItem[phase]))>
       </#if>
     </#if>
  </#list>
 <#elseif Motor==2>
  <#list ADCMap as ADCItem>
     <#if ADCItem.Sector==sector>
       <#if config="IP">
         <#return _first_word(ADCItem[phase])>
       <#else>
         <#return getADCChannel (_first_word(ADCItem[phase]),_last_word(ADCItem[phase]))>
       </#if>
     </#if>
  </#list>
  <#else>
    #error motor number could not exceed 2
  </#if>
</#function>

<#assign PWM_Timer_M1 = _last_word(MC.M1_PWM_TIMER_SELECTION)>
<#assign PWM_Timer_M2 = _last_word(MC.M2_PWM_TIMER_SELECTION)>

<#-- Condition for STM32F302x8x MCU -->
<#assign CondMcu_STM32F302x8x = (McuName?? && McuName?matches("STM32F302.8.*"))>
<#-- Condition for STM32F072xxx MCU -->
<#assign CondMcu_STM32F072xxx = (McuName?? && McuName?matches("STM32F072.*"))>
<#-- Condition for STM32F446xCx or STM32F446xEx -->
<#assign CondMcu_STM32F446xCEx = (McuName?? && McuName?matches("STM32F446.(C|E).*"))>
<#-- Condition for STM32F0 Family -->
<#assign CondFamily_STM32F0 = (FamilyName?? && FamilyName=="STM32F0")>
<#-- Condition for STM32F3 Family -->
<#assign CondFamily_STM32F3 = (FamilyName?? && FamilyName == "STM32F3")>
<#-- Condition for STM32F4 Family -->
<#assign CondFamily_STM32F4 = (FamilyName?? && FamilyName == "STM32F4")>
<#-- Condition for STM32G4 Family -->
<#assign CondFamily_STM32G4 = (FamilyName?? && FamilyName == "STM32G4")>
<#-- Condition for STM32L4 Family -->
<#assign CondFamily_STM32L4 = (FamilyName?? && FamilyName == "STM32L4")>
<#-- Condition for STM32F7 Family -->
<#assign CondFamily_STM32F7 = (FamilyName?? && FamilyName == "STM32F7")>
<#-- Condition for STM32H7 Family -->
<#assign CondFamily_STM32H7 = (FamilyName?? && FamilyName == "STM32H7")>
<#-- Condition for STM32H5 Family -->
<#assign CondFamily_STM32H5 = (FamilyName?? && FamilyName == "STM32H5") >
<#-- Condition for STM32G0 Family -->
<#assign CondFamily_STM32G0 = (FamilyName?? && FamilyName == "STM32G0")>
<#-- Condition for STM32C0 Family -->
<#assign CondFamily_STM32C0 = (FamilyName?? && FamilyName == "STM32C0")>

<#function _last_word text sep="_"><#return text?split(sep)?last></#function>
<#function _first_word text sep="_"><#return text?split(sep)?first></#function>

<#function _filter_opamp opamp>
  <#if opamp == "OPAMP1">
   <#return "OPAMP">
  <#else>
   <#return opamp>
  </#if>
</#function>

<#macro setScandir Ph1 Ph2>
<#if Ph1?number < Ph2?number>
   LL_ADC_REG_SEQ_SCAN_DIR_FORWARD>>ADC_CFGR1_SCANDIR_Pos,
<#else>
   LL_ADC_REG_SEQ_SCAN_DIR_BACKWARD>>ADC_CFGR1_SCANDIR_Pos,
</#if>
<#return>
</#macro>
/**
  ******************************************************************************
  * @file    mc_parameters.c
  * @author  Motor Control SDK Team, ST Microelectronics
  * @brief   This file provides definitions of HW parameters specific to the 
  *          configuration of the subsystem.
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
#include "main.h" //cstat !MISRAC2012-Rule-21.1
//cstat +MISRAC2012-Rule-21.1
#include "parameters_conversion.h"

<#if FOC>
  <#if CondFamily_STM32F4>
    <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_1_f4xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
#include "r3_2_f4xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_f4xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  </#if><#-- CondFamily_STM32F4 -->
  <#if CondFamily_STM32F0>
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_f0xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32F0 -->
  <#if CondFamily_STM32F3><#-- CondFamily_STM32F3 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_f30x_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') ||  (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
#include "r3_2_f30x_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1'))
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1'))>
#include "r3_1_f30x_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32F3 --->
  <#if CondFamily_STM32G4><#-- CondFamily_STM32G4 --->
    <#if MC.M1_SPEED_SENSOR == "HSO" || MC.M1_SPEED_SENSOR == "ZEST">
#include "r3_g4xx_pwm_curr_fdbk.h"
    <#else>
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
#include "r3_2_g4xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
    <#if  ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1'))>
#include "r3_1_g4xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')) -->
      <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#include "ics_g4xx_pwm_curr_fdbk.h"
      </#if>  
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  </#if><#-- CondFamily_STM32G4 --->
  <#if CondFamily_STM32G0><#-- CondFamily_STM32G0 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_g0xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32G4 --->
  <#if CondFamily_STM32C0><#-- CondFamily_STM32C0 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#error C0 single shunt not supported yet
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
#include "r3_c0xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' )) -->
  </#if><#-- CondFamily_STM32C0 --->
  <#if CondFamily_STM32L4><#-- CondFamily_STM32L4 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
#include "ics_l4xx_pwm_curr_fdbk.h"
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
#include "r3_2_l4xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_1_l4xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32L4 --->
  <#if CondFamily_STM32F7><#-- CondFamily_STM32F7 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
#include "ics_f7xx_pwm_curr_fdbk.h"
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))>
#include "r3_2_f7xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
#include "r3_1_f7xx_pwm_curr_fdbk.h"
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32F7 --->
  <#if CondFamily_STM32H7><#-- CondFamily_STM32H7 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#error " H7 Single shunt not supported yet "
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) 
    || ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS')>
#error " H7 ICS not supported yet "
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS') || (MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
#include "r3_2_h7xx_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) 
    || ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
  </#if><#-- CondFamily_STM32H7 --->
  
  <#if CondFamily_STM32H5 > 
/* Include H5 */
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) >
#include "r1_ps_pwm_curr_fdbk.h"
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
    <#if MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT' || MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT'>
      <#if MC.M1_CS_ADC_NUM=='1'>
      /* Include H5 three shunt 1 ADC */
#include "r3_1_h5xx_pwm_curr_fdbk.h"
      <#elseif MC.M1_CS_ADC_NUM=='2'>
      /* Include H5 three shunt 2 ADC */
#include "r3_2_h5xx_pwm_curr_fdbk.h"
      <#else>
#error h5 config not define
      </#if> <#-- MC.M1_CS_ADC_NUM=='1' -->
    </#if> <#-- MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT' || MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT' -->
  </#if> <#-- CondFamily_STM32H5 -->

  <#if MC.PFC_ENABLED == true>
#include "pfc.h"
  </#if><#-- MC.PFC_ENABLED == true -->
  <#if (MC.MOTOR_PROFILER == true) && (MC.M1_SPEED_SENSOR != "HSO"  && MC.M1_SPEED_SENSOR != "ZEST" )>
#include "mp_self_com_ctrl.h"
#include "mp_one_touch_tuning.h"
  </#if><#-- MC.MOTOR_PROFILER == true -->
  <#if MC.ESC_ENABLE>
#include "esc.h"
  </#if><#-- MC.ESC_ENABLE -->
/* USER CODE BEGIN Additional include */

/* USER CODE END Additional include */  

  <#if MC.DRIVE_NUMBER == "1">
#define FREQ_RATIO 1                /* Dummy value for single drive */
#define FREQ_RELATION HIGHEST_FREQ  /* Dummy value for single drive */
  </#if><#-- MC.DRIVE_NUMBER == 1 -->


  <#if CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32H5 || CondFamily_STM32C0 >
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
extern  PWMC_R3_1_Handle_t PWM_Handle_M1;
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->
  </#if><#-- CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32H5 || CondFamily_STM32C0 -->

  <#if CondFamily_STM32F4>
    <#if ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))><#-- inside CondFamily_STM32F4 -->
/**
  * @brief  Current sensor parameters Motor 2 - three shunt
  */
const R3_2_Params_t R3_2_ParamsM2 =
{

/* Dual MC parameters --------------------------------------------------------*/
  .Tw                = MAX_TWAIT2,
  .bFreqRatio        = FREQ_RATIO,
  .bIsHigherFreqTim  = FREQ_RELATION2,

/* PWM generation parameters --------------------------------------------------*/
  .TIMx              = ${_last_word(MC.M2_PWM_TIMER_SELECTION)},
  .hDeadTime         = DEAD_TIME_COUNTS2,
  .RepetitionCounter = REP_COUNTER2,
  .hTafter           = TW_AFTER2,
  .hTbefore          = TW_BEFORE2,

   //cstat -MISRAC2012-Rule-12.1 -MISRAC2012-Rule-10.1_R6 
  .ADCConfig1 = {
                   (uint32_t)(${getADC("CFG",1,"PHASE_1",2)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",2,"PHASE_1",2)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",3,"PHASE_1",2)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",4,"PHASE_1",2)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",5,"PHASE_1",2)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",6,"PHASE_1",2)}U << ADC_JSQR_JSQ4_Pos)
                },
  .ADCConfig2 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",2,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",3,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",4,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",5,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",6,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos)
                },
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",2)}
                  ,${getADC("IP", 2,"PHASE_1",2)}
                  ,${getADC("IP", 3,"PHASE_1",2)}
                  ,${getADC("IP", 4,"PHASE_1",2)}
                  ,${getADC("IP", 5,"PHASE_1",2)}
                  ,${getADC("IP", 6,"PHASE_1",2)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",2)}
                  ,${getADC("IP", 2,"PHASE_2",2)}
                  ,${getADC("IP", 3,"PHASE_2",2)}
                  ,${getADC("IP", 4,"PHASE_2",2)}
                  ,${getADC("IP", 5,"PHASE_2",2)}
                  ,${getADC("IP", 6,"PHASE_2",2)}
                 }
};
    </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2')) -->
 
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')><#-- inside CondFamily_STM32F4 -->
  /**
  * @brief  Current sensor parameters Motor 1 - three shunt - STM32F401x8
  */
const R3_1_Params_t R3_1_ParamsM1 =
{
/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx              = ${MC.M1_CS_ADC_U},

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .hTafter           = TW_AFTER,
  .hTbefore          = TW_BEFORE_R3_1,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,

  .ADCConfig = {
                 (uint32_t)(${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
               }
};
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1') -->

    <#if (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')><#-- inside CondFamily_STM32F4 -->
  /**
  * @brief  Current sensor parameters Motor 2 - three shunt - STM32F401x8
  */
const R3_1_Params_t R3_1_ParamsM2 =
{
/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx              = ${MC.M2_CS_ADC_U},

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER2,
  .hTafter           = TW_AFTER2,
  .hTbefore          = TW_BEFORE_R3_1_2,
  .TIMx              = ${_last_word(MC.M2_PWM_TIMER_SELECTION)},
  .Tsampling         = (uint16_t)SAMPLING_TIME2,
  .Tcase2            = (uint16_t)SAMPLING_TIME2 + (uint16_t)TDEAD2 + (uint16_t)TRISE2,
  .Tcase3            = ((uint16_t)TDEAD2 + (uint16_t)TNOISE2 + (uint16_t)SAMPLING_TIME2) / 2u,

  .ADCConfig = {
                 (uint32_t)(${getADC("CFG",1,"PHASE_1",2)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",1,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",2,"PHASE_1",2)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",2,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",3,"PHASE_1",2)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",3,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",4,"PHASE_1",2)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",4,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",5,"PHASE_1",2)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",5,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",6,"PHASE_1",2)}U << ADC_JSQR_JSQ3_Pos)
                          | ${getADC("CFG",6,"PHASE_2",2)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
               }
};
    </#if><#-- (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1') -->

    <#if ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2'))><#-- inside CondFamily_STM32F4 -->

/**
  * @brief  Current sensor parameters Motor 1 - three shunt
  */
const R3_2_Params_t R3_2_ParamsM1 =
{
  .Tw                = MAX_TWAIT,
  .bFreqRatio        = FREQ_RATIO,
  .bIsHigherFreqTim  = FREQ_RELATION,

/* PWM generation parameters --------------------------------------------------*/
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .RepetitionCounter = REP_COUNTER,
  .hTafter           = TW_AFTER,
  .hTbefore          = TW_BEFORE,
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,

/* Current reading A/D Conversions initialization ----------------------------*/
  .ADCConfig1 = {
                   (uint32_t)(${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos)
                  ,(uint32_t)(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos)
                },
  .ADCConfig2 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos)
                 ,(uint32_t)(${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos)
                },
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",1)}
                  ,${getADC("IP", 2,"PHASE_1",1)}
                  ,${getADC("IP", 3,"PHASE_1",1)}
                  ,${getADC("IP", 4,"PHASE_1",1)}
                  ,${getADC("IP", 5,"PHASE_1",1)}
                  ,${getADC("IP", 6,"PHASE_1",1)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",1)}
                  ,${getADC("IP", 2,"PHASE_2",1)}
                  ,${getADC("IP", 3,"PHASE_2",1)}
                  ,${getADC("IP", 4,"PHASE_2",1)}
                  ,${getADC("IP", 5,"PHASE_2",1)}
                  ,${getADC("IP", 6,"PHASE_2",1)}
                 }
};
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='2')) -->

    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
/**
  * @brief  Current sensor parameters Dual Drive Motor 1 - ICS
  */
const ICS_Params_t ICS_ParamsM1 = {

/* Dual MC parameters --------------------------------------------------------*/
  .InstanceNbr       = 1,
  .Tw                = MAX_TWAIT,
  .FreqRatio         = FREQ_RATIO,
  .IsHigherFreqTim   = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .IaChannel         = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U},
  .IbChannel         = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V},
  
/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)}
  
};
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->

    <#if MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
/**
  * @brief  Current sensor parameters Dual Drive Motor 2 - ICS
  */
const ICS_Params_t ICS_ParamsM2 = {
/* Dual MC parameters --------------------------------------------------------*/
  .InstanceNbr       = 2,
  .Tw                = MAX_TWAIT2,
  .FreqRatio         = FREQ_RATIO,
  .IsHigherFreqTim   = FREQ_RELATION2,
 

/* Current reading A/D Conversions initialization -----------------------------*/
  .IaChannel         = MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_U},
  .IbChannel         = MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_V},

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER2,
  .TIMx              = ${_last_word(MC.M2_PWM_TIMER_SELECTION)}
};
    </#if><#-- MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS') -->
  </#if><#-- <#if CondFamily_STM32F4 -->


  <#if CondFamily_STM32H5 >
    <#if  MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT'> <#-- Inside CondFamily_STM32H5 --->
      <#if MC.M1_CS_ADC_NUM=='1'> <#-- Inside CondFamily_STM32H5 --->

/**
  * @brief  Current sensor parameters Single Drive - three shunt, STM32H5X
  */
const R3_1_Params_t R3_1_ParamsM1 =
{

  .ADCx           = ${MC.M1_CS_ADC_U},
/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .hTafter = TW_AFTER,
  .hTbefore = TW_BEFORE_R3_1, 
  .TIMx = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .Tsampling                  = (uint16_t)SAMPLING_TIME,
  .Tcase2                     = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3                     = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME)/2u,
  
/* PWM Driving signals initialization ----------------------------------------*/
  .ADCConfig = { 
                 ( ${getADC("CFG",1,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | ( ${getADC("CFG",1,"PHASE_2")} << ADC_JSQR_JSQ2_Pos ) | 1 << ADC_JSQR_JL_Pos | ( LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT ),
                 ( ${getADC("CFG",2,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | ( ${getADC("CFG",2,"PHASE_2")} << ADC_JSQR_JSQ2_Pos ) | 1 << ADC_JSQR_JL_Pos | ( LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT ),
                 ( ${getADC("CFG",3,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | ( ${getADC("CFG",3,"PHASE_2")} << ADC_JSQR_JSQ2_Pos ) | 1 << ADC_JSQR_JL_Pos | ( LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT ),
                 ( ${getADC("CFG",4,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | ( ${getADC("CFG",4,"PHASE_2")} << ADC_JSQR_JSQ2_Pos ) | 1 << ADC_JSQR_JL_Pos | ( LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT ),
                 ( ${getADC("CFG",5,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | ( ${getADC("CFG",5,"PHASE_2")} << ADC_JSQR_JSQ2_Pos ) | 1 << ADC_JSQR_JL_Pos | ( LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT ),
                 ( ${getADC("CFG",6,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | ( ${getADC("CFG",6,"PHASE_2")} << ADC_JSQR_JSQ2_Pos ) | 1 << ADC_JSQR_JL_Pos | ( LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT ),                 
               },
};

    <#elseif MC.M1_CS_ADC_NUM=='2' > <#-- inside CondFamily_STM32H5 -->

/**
  * @brief  Current sensor parameters Motor 1 - three shunt 2ADC
  */
const R3_2_Params_t R3_2_ParamsM1 =
{
  .Tw                       = MAX_TWAIT,
  .bFreqRatio               = FREQ_RATIO,
  .bIsHigherFreqTim         = FREQ_RELATION,

/* PWM generation parameters --------------------------------------------------*/
  .TIMx                       =  ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .RepetitionCounter          =  REP_COUNTER,
  .hTafter                    =  TW_AFTER,
  .hTbefore                   =  TW_BEFORE,
  .Tsampling                  = (uint16_t)SAMPLING_TIME,
  .Tcase2                     = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3                     = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME)/2u,
  
/* Current reading A/D Conversions initialization ----------------------------*/
   .ADCConfig1 = { 
                   (uint32_t)( ${getADC("CFG",1,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",2,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",3,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",4,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",5,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",6,"PHASE_1")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                 },
   .ADCConfig2 = { 
                   (uint32_t)( ${getADC("CFG",1,"PHASE_2")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",2,"PHASE_2")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",3,"PHASE_2")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",4,"PHASE_2")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",5,"PHASE_2")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                   (uint32_t)( ${getADC("CFG",6,"PHASE_2")} << ADC_JSQR_JSQ1_Pos ) | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                 },              
  .ADCDataReg1 = { ${getADC("IP", 1,"PHASE_1")}
                 , ${getADC("IP", 2,"PHASE_1")}
                 , ${getADC("IP", 3,"PHASE_1")}
                 , ${getADC("IP", 4,"PHASE_1")}
                 , ${getADC("IP", 5,"PHASE_1")}
                 , ${getADC("IP", 6,"PHASE_1")}                          
                 },
  .ADCDataReg2 =  { ${getADC("IP", 1,"PHASE_2")}
                 , ${getADC("IP", 2,"PHASE_2")}
                 , ${getADC("IP", 3,"PHASE_2")}
                 , ${getADC("IP", 4,"PHASE_2")}
                 , ${getADC("IP", 5,"PHASE_2")}
                 , ${getADC("IP", 6,"PHASE_2")}                            
                  }
};
      </#if> <#--  MC.M1_CS_ADC_NUM=='1' -->
    </#if> <#-- MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT'  -->
  </#if> <#-- CondFamily_STM32H5 -->

  <#if CondFamily_STM32F0>
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
/**
  * @brief  Current sensor parameters Single Drive - three shunt, STM32F0X
  */
const R3_1_Params_t R3_1_Params =
{
/* Current reading A/D Conversions initialization -----------------------------*/
  .b_ISamplingTime   = LL_ADC_SAMPLINGTIME_${MC.M1_CURR_SAMPLING_TIME}<#if MC.M1_CURR_SAMPLING_TIME != "1">CYCLES_5<#else>CYCLE_5</#if>,

/* PWM generation parameters --------------------------------------------------*/
  .hDeadTime         = DEAD_TIME_COUNTS,
  .RepetitionCounter = REP_COUNTER,
  .hTafter           = TW_AFTER,
  .hTbefore          = TW_BEFORE_R3_1,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,

  .ADCConfig = {
                 (uint32_t)(1<< ${getADC("CFG",1,"PHASE_1",1)}U ) | (uint32_t)(1<< ${getADC("CFG",1,"PHASE_2",1)}U ),
                 (uint32_t)(1<< ${getADC("CFG",2,"PHASE_1",1)}U ) | (uint32_t)(1<< ${getADC("CFG",2,"PHASE_2",1)}U ),
                 (uint32_t)(1<< ${getADC("CFG",3,"PHASE_1",1)}U ) | (uint32_t)(1<< ${getADC("CFG",3,"PHASE_2",1)}U ),
                 (uint32_t)(1<< ${getADC("CFG",4,"PHASE_1",1)}U ) | (uint32_t)(1<< ${getADC("CFG",4,"PHASE_2",1)}U ),
                 (uint32_t)(1<< ${getADC("CFG",5,"PHASE_1",1)}U ) | (uint32_t)(1<< ${getADC("CFG",5,"PHASE_2",1)}U ),
                 (uint32_t)(1<< ${getADC("CFG",6,"PHASE_1",1)}U ) | (uint32_t)(1<< ${getADC("CFG",6,"PHASE_2",1)}U ),
               },
  .ADCScandir = {
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_V Ph2=MC.M1_CS_CHANNEL_W/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_U Ph2=MC.M1_CS_CHANNEL_W/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_W Ph2=MC.M1_CS_CHANNEL_U/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_V Ph2=MC.M1_CS_CHANNEL_U/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_U Ph2=MC.M1_CS_CHANNEL_V/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_W Ph2=MC.M1_CS_CHANNEL_V/>
                },
  .ADCDataReg1 = {
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                 },
     
  .ADCDataReg2 = {
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                 },
};
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')  -->
  </#if><#-- CondFamily_STM32F0 -->

  <#if CondFamily_STM32G0>
    <#if (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')>
/**
  * @brief  Current sensor parameters Single Drive - three shunt, STM32G0X
  */
const R3_1_Params_t R3_1_Params =
{
/* PWM generation parameters --------------------------------------------------*/
  .hDeadTime         = DEAD_TIME_COUNTS,
  .RepetitionCounter = REP_COUNTER,
  .hTafter           = TW_AFTER,
  .hTbefore          = TW_BEFORE_R3_1,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,

  .ADCConfig = {
                 (uint32_t)(1<< ${getADC("CFG",1,"PHASE_1",1)}U) | (uint32_t)(1<< ${getADC("CFG",1,"PHASE_2",1)}U),
                 (uint32_t)(1<< ${getADC("CFG",2,"PHASE_1",1)}U) | (uint32_t)(1<< ${getADC("CFG",2,"PHASE_2",1)}U),
                 (uint32_t)(1<< ${getADC("CFG",3,"PHASE_1",1)}U) | (uint32_t)(1<< ${getADC("CFG",3,"PHASE_2",1)}U),
                 (uint32_t)(1<< ${getADC("CFG",4,"PHASE_1",1)}U) | (uint32_t)(1<< ${getADC("CFG",4,"PHASE_2",1)}U),
                 (uint32_t)(1<< ${getADC("CFG",5,"PHASE_1",1)}U) | (uint32_t)(1<< ${getADC("CFG",5,"PHASE_2",1)}U),
                 (uint32_t)(1<< ${getADC("CFG",6,"PHASE_1",1)}U) | (uint32_t)(1<< ${getADC("CFG",6,"PHASE_2",1)}U),
               },
  .ADCScandir = {
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_V Ph2=MC.M1_CS_CHANNEL_W/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_U Ph2=MC.M1_CS_CHANNEL_W/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_W Ph2=MC.M1_CS_CHANNEL_U/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_V Ph2=MC.M1_CS_CHANNEL_U/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_U Ph2=MC.M1_CS_CHANNEL_V/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_W Ph2=MC.M1_CS_CHANNEL_V/>
                },
  .ADCDataReg1 = {
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                 },
     
  .ADCDataReg2 = {
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                 },
};
    </#if><#-- (MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M1_CS_ADC_NUM=='1')  -->
  </#if><#-- CondFamily_STM32G0 -->

<#if CondFamily_STM32C0>
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))>
/**
  * @brief  Current sensor parameters Single Drive - three shunt, STM32F0X
  */
const R3_1_Params_t R3_1_Params =
{
/* Current reading A/D Conversions initialization -----------------------------*/
  .b_ISamplingTime   = LL_ADC_SAMPLINGTIME_${MC.M1_CURR_SAMPLING_TIME}<#if MC.M1_CURR_SAMPLING_TIME != "1">CYCLES_5<#else>CYCLE_5</#if>,

/* PWM generation parameters --------------------------------------------------*/
  .hDeadTime         = DEAD_TIME_COUNTS,
  .RepetitionCounter = REP_COUNTER,
  .hTafter           = TW_AFTER,
  .hTbefore          = TW_BEFORE_R3_1,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,

  .ADCConfig = {
                 (uint32_t)(1<< ${getADC("CFG",1,"PHASE_1")}U ) | (uint32_t)(1<< ${getADC("CFG",1,"PHASE_2")}U ),
                 (uint32_t)(1<< ${getADC("CFG",2,"PHASE_1")}U ) | (uint32_t)(1<< ${getADC("CFG",2,"PHASE_2")}U ),
                 (uint32_t)(1<< ${getADC("CFG",3,"PHASE_1")}U ) | (uint32_t)(1<< ${getADC("CFG",3,"PHASE_2")}U ),
                 (uint32_t)(1<< ${getADC("CFG",4,"PHASE_1")}U ) | (uint32_t)(1<< ${getADC("CFG",4,"PHASE_2")}U ),
                 (uint32_t)(1<< ${getADC("CFG",5,"PHASE_1")}U ) | (uint32_t)(1<< ${getADC("CFG",5,"PHASE_2")}U ),
                 (uint32_t)(1<< ${getADC("CFG",6,"PHASE_1")}U ) | (uint32_t)(1<< ${getADC("CFG",6,"PHASE_2")}U ),
               },
  .ADCScandir = {
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_V Ph2=MC.M1_CS_CHANNEL_W/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_U Ph2=MC.M1_CS_CHANNEL_W/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_W Ph2=MC.M1_CS_CHANNEL_U/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_V Ph2=MC.M1_CS_CHANNEL_U/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_U Ph2=MC.M1_CS_CHANNEL_V/>
                  <@setScandir Ph1=MC.M1_CS_CHANNEL_W Ph2=MC.M1_CS_CHANNEL_V/>
                },
  .ADCDataReg1 = {
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                 },
     
  .ADCDataReg2 = {
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                   &PWM_Handle_M1.ADC1_DMA_converted[1],
                   &PWM_Handle_M1.ADC1_DMA_converted[0],
                 },
};
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'THREE_SHUNT') && ( MC.M1_CS_ADC_NUM == '1' ))  -->
  </#if><#-- CondFamily_STM32C0 -->

  <#if CondFamily_STM32L4><#-- CondFamily_STM32L4 --->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'><#-- Inside CondFamily_STM32L4 --->
const ICS_Params_t ICS_ParamsM1 = 
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio         = FREQ_RATIO,
  .IsHigherFreqTim   = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx_1            = ${MC.M1_CS_ADC_U},
  .ADCx_2            = ${MC.M1_CS_ADC_V},
  .IaChannel         = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U},
  .IbChannel         = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V},

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)}
};
    <#elseif  MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT'><#-- Inside CondFamily_STM32L4 --->
      <#if MC.M1_CS_ADC_NUM =='1'><#-- Inside CondFamily_STM32L4 --->
/**
  * @brief  Current sensor parameters Motor 1 - three shunt 1 ADC 
  */
const R3_1_Params_t R3_1_ParamsM1 =
{
/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx              = ${MC.M1_CS_ADC_U},
  .ADCConfig = {
                 (uint32_t)(${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos) 
               | ${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                 (uint32_t)(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | ${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                 (uint32_t)(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | ${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                 (uint32_t)(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | ${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                 (uint32_t)(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | ${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                 (uint32_t)(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | ${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
               },

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .hTafter           = TW_AFTER,
  .hTbefore          = TW_BEFORE_R3_1,
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
};
      <#elseif MC.M1_CS_ADC_NUM=='2'><#-- Inside CondFamily_STM32L4 --->
/**
  * @brief  Current sensor parameters Motor 1 - three shunt - L4XX - Independent Resources
  */
const R3_2_Params_t R3_2_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .bFreqRatio        = FREQ_RATIO,
  .bIsHigherFreqTim  = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCConfig1 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                },
  .ADCConfig2 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                  (uint32_t)(${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos) | 0U << ADC_JSQR_JL_Pos
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT),
                },
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",1)}
                  ,${getADC("IP", 2,"PHASE_1",1)}
                  ,${getADC("IP", 3,"PHASE_1",1)}
                  ,${getADC("IP", 4,"PHASE_1",1)}
                  ,${getADC("IP", 5,"PHASE_1",1)}
                  ,${getADC("IP", 6,"PHASE_1",1)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",1)}
                  ,${getADC("IP", 2,"PHASE_2",1)}
                  ,${getADC("IP", 3,"PHASE_2",1)}
                  ,${getADC("IP", 4,"PHASE_2",1)}
                  ,${getADC("IP", 5,"PHASE_2",1)}
                  ,${getADC("IP", 6,"PHASE_2",1)}
                 },

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .hTafter           = TW_AFTER,
  .hTbefore          = TW_BEFORE,
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},

};
      </#if><#-- MC.M1_CS_ADC_NUM =='1' --><#-- CondFamily_STM32L4 --->
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' --><#-- CondFamily_STM32L4 --->
  </#if><#-- CondFamily_STM32L4 --->


  <#if CondFamily_STM32F7><#-- CondFamily_STM32F7 --->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'><#-- Inside CondFamily_STM32F7 --->
/**
  * @brief  Current sensor parameters Dual Drive Motor 1 - ICS
  */
const ICS_Params_t ICS_ParamsM1 = {

/* Dual MC parameters --------------------------------------------------------*/
  .InstanceNbr       = 1,
  .Tw                = MAX_TWAIT,
  .FreqRatio         = FREQ_RATIO,
  .IsHigherFreqTim   = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .IaChannel         = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U},
  .IbChannel         = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V},

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)}

}; 
    <#elseif  MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT'><#-- Inside CondFamily_STM32F7 --->
      <#if MC.M1_CS_ADC_NUM=='1'><#-- Inside CondFamily_STM32F7 --->
/**
  * @brief  Current sensor parameters Motor 1 - three shunt 1 ADC
  */
const R3_1_Params_t R3_1_ParamsM1 =
{
/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx = ${MC.M1_CS_ADC_U},
  
  .ADCConfig = {
                 (uint32_t)(${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
               | ${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
               | ${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
               | ${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
               | ${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
               | ${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
                 (uint32_t)(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ3_Pos)
               | ${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos | 1<< ADC_JSQR_JL_Pos,
               },

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .Tafter            = TW_AFTER,
  .Tbefore           = TW_BEFORE_R3_1,
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)}
};  
      <#elseif MC.M1_CS_ADC_NUM=='2'><#-- Inside CondFamily_STM32F7 --->
/**
  * @brief  Current sensor parameters Motor 1 - three shunt
  */
const R3_2_Params_t R3_2_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .Tw                =	MAX_TWAIT,
  .bFreqRatio        =	FREQ_RATIO,
  .bIsHigherFreqTim  =	FREQ_RELATION,

/* Current reading A/D Conversions initialization ----------------------------*/
  .ADCConfig1 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                },
  .ADCConfig2 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                  (uint32_t)(${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ4_Pos) | 0U << ADC_JSQR_JL_Pos,
                },
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",1)}
                  ,${getADC("IP", 2,"PHASE_1",1)}
                  ,${getADC("IP", 3,"PHASE_1",1)}
                  ,${getADC("IP", 4,"PHASE_1",1)}
                  ,${getADC("IP", 5,"PHASE_1",1)}
                  ,${getADC("IP", 6,"PHASE_1",1)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",1)}
                  ,${getADC("IP", 2,"PHASE_2",1)}
                  ,${getADC("IP", 3,"PHASE_2",1)}
                  ,${getADC("IP", 4,"PHASE_2",1)}
                  ,${getADC("IP", 5,"PHASE_2",1)}
                  ,${getADC("IP", 6,"PHASE_2",1)}
                 },

/* PWM generation parameters --------------------------------------------------*/
  .TIMx              =	${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .RepetitionCounter =	REP_COUNTER,
  .hTafter           =	TW_AFTER,
  .hTbefore          =	TW_BEFORE,
  .Tsampling         = (uint16_t)SAMPLING_TIME,
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u
};
      </#if><#-- MC.M1_CS_ADC_NUM=='1' -->
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
  </#if><#-- CondFamily_STM32F7 --->

  <#if CondFamily_STM32H7><#-- CondFamily_STM32H7 --->
    <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
#error " H7 Single shunt not supported yet "
    <#elseif MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'><#-- Inside CondFamily_STM32H7 --->
#error " H7 ICS not supported yet "

    <#elseif MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT'><#-- Inside CondFamily_STM32H7 --->
      <#if MC.M1_CS_ADC_NUM=='2'>
        <#if MC.M1_USE_INTERNAL_OPAMP>
/**
  * @brief  Internal OPAMP parameters Motor 1 - three shunt - H7 
  */
const R3_2_OPAMPParams_t R3_2_OPAMPParamsM1 =
{
  .OPAMPSelect_1 = {
                     ${getOPAMP("IP",1,"PHASE_1",1)}
                    ,${getOPAMP("IP",2,"PHASE_1",1)}
                    ,${getOPAMP("IP",3,"PHASE_1",1)}
                    ,${getOPAMP("IP",4,"PHASE_1",1)}
                    ,${getOPAMP("IP",5,"PHASE_1",1)}
                    ,${getOPAMP("IP",6,"PHASE_1",1)}
                   },
  .OPAMPSelect_2 = {
                     ${getOPAMP("IP",1,"PHASE_2",1)}
                    ,${getOPAMP("IP",2,"PHASE_2",1)}
                    ,${getOPAMP("IP",3,"PHASE_2",1)}
                    ,${getOPAMP("IP",4,"PHASE_2",1)}
                    ,${getOPAMP("IP",5,"PHASE_2",1)}
                    ,${getOPAMP("IP",6,"PHASE_2",1)}
                   },
  .OPAMPConfig1 = {
                    ${getOPAMP("CFG",1,"PHASE_1",1)}
                   ,${getOPAMP("CFG",2,"PHASE_1",1)}
                   ,${getOPAMP("CFG",3,"PHASE_1",1)}
                   ,${getOPAMP("CFG",4,"PHASE_1",1)}
                   ,${getOPAMP("CFG",5,"PHASE_1",1)}
                   ,${getOPAMP("CFG",6,"PHASE_1",1)}
                  },
  .OPAMPConfig2 = {
                    ${getOPAMP("CFG",1,"PHASE_2",1)}
                   ,${getOPAMP("CFG",2,"PHASE_2",1)}
                   ,${getOPAMP("CFG",3,"PHASE_2",1)}
                   ,${getOPAMP("CFG",4,"PHASE_2",1)}
                   ,${getOPAMP("CFG",5,"PHASE_2",1)}
                   ,${getOPAMP("CFG",6,"PHASE_2",1)}
                  },
};
        </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->

  /**
  * @brief  Current sensor parameters Motor 1 - three shunt - H7 - Shared Resources
  */
const R3_2_Params_t R3_2_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCConfig1 = {
                  (${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
  },
  .ADCConfig2 = {
                  (${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                },
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",1)}
                  ,${getADC("IP", 2,"PHASE_1",1)}
                  ,${getADC("IP", 3,"PHASE_1",1)}
                  ,${getADC("IP", 4,"PHASE_1",1)}
                  ,${getADC("IP", 5,"PHASE_1",1)}
                  ,${getADC("IP", 6,"PHASE_1",1)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",1)}
                  ,${getADC("IP", 2,"PHASE_2",1)}
                  ,${getADC("IP", 3,"PHASE_2",1)}
                  ,${getADC("IP", 4,"PHASE_2",1)}
                  ,${getADC("IP", 5,"PHASE_2",1)}
                  ,${getADC("IP", 6,"PHASE_2",1)}
                 },

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter     = REP_COUNTER,
  .Tafter                = TW_AFTER,
  .Tbefore               = TW_BEFORE,
  .TIMx                  = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},

/* Internal OPAMP common settings --------------------------------------------*/
        <#if MC.M1_USE_INTERNAL_OPAMP>
  .OPAMPParams           = &R3_2_OPAMPParamsM1,
        <#else><#-- MC.M1_USE_INTERNAL_OPAMP == false -->
  .OPAMPParams           = MC_NULL,
        </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->

/* Internal COMP settings ----------------------------------------------------*/
        <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M1_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE},
  .CompOCPBSelection     = ${MC.M1_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE},
  .CompOCPCSelection     = ${MC.M1_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE},
        <#else><#-- MC.M1_OCP_TOPOLOGY == "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
        </#if><#-- MC.M1_OCP_TOPOLOGY == "EMBEDDED" -->

        <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
        <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION == flase -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
        </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M1_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M1_OVPREF},
};
      <#elseif MC.M1_CS_ADC_NUM=='1'><#-- Inside CondFamily_STM32H7 --->
    #error " H7 Single ADC not supported yet"
      </#if><#-- MC.M1_CS_ADC_NUM=='2' -->
    </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  </#if><#-- CondFamily_STM32H7 -->


  <#if CondFamily_STM32F3><#-- CondFamily_STM32F3 --->
    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'><#-- Inside CondFamily_STM32F3 --->
const ICS_Params_t ICS_ParamsM1 = 
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio         = FREQ_RATIO,
  .IsHigherFreqTim   = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx_1 = ${MC.M1_CS_ADC_U},
  .ADCx_2 = ${MC.M1_CS_ADC_V},
  .IaChannel         = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U},
  .IbChannel         = MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V},

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)}
  
};

    <#elseif MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT'><#-- Inside CondFamily_STM32F3 --->
      <#if MC.M1_CS_ADC_NUM=='2'>
        <#if MC.M1_USE_INTERNAL_OPAMP>
/**
  * @brief  Internal OPAMP parameters Motor 1 - three shunt - F3xx 
  */
const R3_2_OPAMPParams_t R3_2_OPAMPParamsM1 =
{
  .OPAMPSelect_1 = {
                     ${getOPAMP("IP",1,"PHASE_1",1)}
                    ,${getOPAMP("IP",2,"PHASE_1",1)}
                    ,${getOPAMP("IP",3,"PHASE_1",1)}
                    ,${getOPAMP("IP",4,"PHASE_1",1)}
                    ,${getOPAMP("IP",5,"PHASE_1",1)}
                    ,${getOPAMP("IP",6,"PHASE_1",1)}
                   },
  .OPAMPSelect_2 = {
                     ${getOPAMP("IP",1,"PHASE_2",1)}
                    ,${getOPAMP("IP",2,"PHASE_2",1)}
                    ,${getOPAMP("IP",3,"PHASE_2",1)}
                    ,${getOPAMP("IP",4,"PHASE_2",1)}
                    ,${getOPAMP("IP",5,"PHASE_2",1)}
                    ,${getOPAMP("IP",6,"PHASE_2",1)}
                   },
  
  .OPAMPConfig1 = {
                    ${getOPAMP("CFG",1,"PHASE_1",1)}
                   ,${getOPAMP("CFG",2,"PHASE_1",1)}
                   ,${getOPAMP("CFG",3,"PHASE_1",1)}
                   ,${getOPAMP("CFG",4,"PHASE_1",1)}
                   ,${getOPAMP("CFG",5,"PHASE_1",1)}
                   ,${getOPAMP("CFG",6,"PHASE_1",1)}
                  },
  .OPAMPConfig2 = {
                    ${getOPAMP("CFG",1,"PHASE_2",1)}
                   ,${getOPAMP("CFG",2,"PHASE_2",1)}
                   ,${getOPAMP("CFG",3,"PHASE_2",1)}
                   ,${getOPAMP("CFG",4,"PHASE_2",1)}
                   ,${getOPAMP("CFG",5,"PHASE_2",1)}
                   ,${getOPAMP("CFG",6,"PHASE_2",1)}
                  },
};
        </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->

  /**
  * @brief  Current sensor parameters Motor 1 - three shunt - F30x - Shared Resources
  */
const R3_2_Params_t R3_2_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCConfig1 = {
                  (${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                      
                },
  .ADCConfig2 = {
                  (${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                      
                },
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",1)}
                  ,${getADC("IP", 2,"PHASE_1",1)}
                  ,${getADC("IP", 3,"PHASE_1",1)}
                  ,${getADC("IP", 4,"PHASE_1",1)}
                  ,${getADC("IP", 5,"PHASE_1",1)}
                  ,${getADC("IP", 6,"PHASE_1",1)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",1)}
                  ,${getADC("IP", 2,"PHASE_2",1)}
                  ,${getADC("IP", 3,"PHASE_2",1)}
                  ,${getADC("IP", 4,"PHASE_2",1)}
                  ,${getADC("IP", 5,"PHASE_2",1)}
                  ,${getADC("IP", 6,"PHASE_2",1)}
                 },
/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .Tafter                = TW_AFTER,
  .Tbefore               = TW_BEFORE,
  .Tsampling             = (uint16_t)SAMPLING_TIME,
  .Tcase2                = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3                = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,
  .TIMx                  = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},

/* Internal OPAMP common settings --------------------------------------------*/
        <#if MC.M1_USE_INTERNAL_OPAMP>
  .OPAMPParams           = &R3_2_OPAMPParamsM1,
        <#else><#-- MC.M1_USE_INTERNAL_OPAMP == flase -->
  .OPAMPParams           = MC_NULL,
        </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->

/* Internal COMP settings ----------------------------------------------------*/
        <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M1_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE},
  .CompOCPBSelection     = ${MC.M1_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE},
  .CompOCPCSelection     = ${MC.M1_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE},
        <#else><#-- MC.M1_OCP_TOPOLOGY != "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
        </#if><#-- MC.M1_OCP_TOPOLOGY == "EMBEDDED" -->

        <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
        <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION == false -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
        </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M1_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M1_OVPREF},
};
      <#elseif MC.M1_CS_ADC_NUM=='1'><#-- Inside CondFamily_STM32F3 --->
/**
  * @brief  Current sensor parameters Motor 1 - three shunt 1 ADC (STM32F302x8)
  */
const R3_1_Params_t R3_1_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx                  = ${MC.M1_CS_ADC_U},
  .ADCConfig = {
                 (${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                       
               },

 
/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .Tafter                = TW_AFTER,
  .Tbefore               = TW_BEFORE_R3_1,
  .Tsampling             = (uint16_t)SAMPLING_TIME,
  .Tcase2                = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3                = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,
  .TIMx                  = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},

/* Internal COMP settings ----------------------------------------------------*/
        <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M1_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE},
  .CompOCPBSelection     = ${MC.M1_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE},
  .CompOCPCSelection     = ${MC.M1_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE},
        <#else><#-- MC.M1_OCP_TOPOLOGY != "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
        </#if><#-- MC.M1_OCP_TOPOLOGY == "EMBEDDED" -->

        <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
        <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION == false -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
        </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M1_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M1_OVPREF},
};
      </#if><#--  MC.M1_CS_ADC_NUM=='2' -->
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
    <#if MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'><#-- Inside CondFamily_STM32F3 --->
const ICS_Params_t ICS_ParamsM2 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio         = FREQ_RATIO,
  .IsHigherFreqTim   = FREQ_RELATION2,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx_1 = ${MC.M2_CS_ADC_U},
  .ADCx_2 = ${MC.M2_CS_ADC_V},
  .IaChannel         = MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_U},
  .IbChannel         = MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_V},

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER2,
  .TIMx              = ${_last_word(MC.M2_PWM_TIMER_SELECTION)}
};
    <#elseif ((MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='2'))>
    <#-- Inside CondFamily_STM32F3 --->
      <#if MC.M2_USE_INTERNAL_OPAMP>
/**
  * @brief  Internal OPAMP parameters Motor 2 - three shunt - F3xx 
  */
R3_2_OPAMPParams_t R3_2_OPAMPParamsM2 =
{
  .OPAMPSelect_1 = {
                     ${getOPAMP("IP",1,"PHASE_1",2)}
                    ,${getOPAMP("IP",2,"PHASE_1",2)}
                    ,${getOPAMP("IP",3,"PHASE_1",2)}
                    ,${getOPAMP("IP",4,"PHASE_1",2)}
                    ,${getOPAMP("IP",5,"PHASE_1",2)}
                    ,${getOPAMP("IP",6,"PHASE_1",2)}
                   },
  .OPAMPSelect_2 = {
                     ${getOPAMP("IP",1,"PHASE_2",2)}
                    ,${getOPAMP("IP",2,"PHASE_2",2)}
                    ,${getOPAMP("IP",3,"PHASE_2",2)}
                    ,${getOPAMP("IP",4,"PHASE_2",2)}
                    ,${getOPAMP("IP",5,"PHASE_2",2)}
                    ,${getOPAMP("IP",6,"PHASE_2",2)}
                   },
  
  .OPAMPConfig1 = {
                    ${getOPAMP("CFG",1,"PHASE_1",2)}
                   ,${getOPAMP("CFG",2,"PHASE_1",2)}
                   ,${getOPAMP("CFG",3,"PHASE_1",2)}
                   ,${getOPAMP("CFG",4,"PHASE_1",2)}
                   ,${getOPAMP("CFG",5,"PHASE_1",2)}
                   ,${getOPAMP("CFG",6,"PHASE_1",2)}
                  },
  .OPAMPConfig2 = {
                    ${getOPAMP("CFG",1,"PHASE_2",2)}
                   ,${getOPAMP("CFG",2,"PHASE_2",2)}
                   ,${getOPAMP("CFG",3,"PHASE_2",2)}
                   ,${getOPAMP("CFG",4,"PHASE_2",2)}
                   ,${getOPAMP("CFG",5,"PHASE_2",2)}
                   ,${getOPAMP("CFG",6,"PHASE_2",2)}
                  },
};
      </#if><#-- MC.M2_USE_INTERNAL_OPAMP -->

  /**
  * @brief  Current sensor parameters Motor 2 - three shunt - F30x 
  */
const R3_2_Params_t R3_2_ParamsM2 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION2,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCConfig1 = {
                  (${getADC("CFG",1,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",2,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",3,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",4,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",5,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",6,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                      
                },
  .ADCConfig2 = {
                  (${getADC("CFG",1,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",2,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",3,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",4,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",5,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(${getADC("CFG",6,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                      
                },
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",2)}
                  ,${getADC("IP", 2,"PHASE_1",2)}
                  ,${getADC("IP", 3,"PHASE_1",2)}
                  ,${getADC("IP", 4,"PHASE_1",2)}
                  ,${getADC("IP", 5,"PHASE_1",2)}
                  ,${getADC("IP", 6,"PHASE_1",2)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",2)}
                  ,${getADC("IP", 2,"PHASE_2",2)}
                  ,${getADC("IP", 3,"PHASE_2",2)}
                  ,${getADC("IP", 4,"PHASE_2",2)}
                  ,${getADC("IP", 5,"PHASE_2",2)}
                  ,${getADC("IP", 6,"PHASE_2",2)}
                 },
/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter     = REP_COUNTER2,
  .Tafter                = TW_AFTER2,
  .Tbefore               = TW_BEFORE2,
  .TIMx                  = ${_last_word(MC.M2_PWM_TIMER_SELECTION)},

/* Internal OPAMP common settings --------------------------------------------*/
      <#if MC.M2_USE_INTERNAL_OPAMP>
  .OPAMPParams           = &R3_2_OPAMPParamsM2,
      <#else><#-- MC.M2_USE_INTERNAL_OPAMP == false -->
  .OPAMPParams           = MC_NULL,
      </#if><#-- MC.M2_USE_INTERNAL_OPAMP -->

/* Internal COMP settings ----------------------------------------------------*/
      <#if MC.M2_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M2_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE2},
  .CompOCPBSelection     = ${MC.M2_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE2},
  .CompOCPCSelection     = ${MC.M2_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE2},
      <#else><#-- MC.M2_OCP_TOPOLOGY != "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
      </#if><#-- MC.M2_OCP_TOPOLOGY == "EMBEDDED" -->

      <#if MC.INTERNAL_OVERVOLTAGEPROTECTION2>
  .CompOVPSelection      = OVP_SELECTION2,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE2,
      <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION2 == false -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
      </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION2 -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M2_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M2_OVPREF},
};

    <#elseif (MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT') && (MC.M2_CS_ADC_NUM=='1')><#-- Inside CondFamily_STM32F3 --->
/**
  * @brief  Current sensor parameters Motor 2 - three shunt 1 ADC (STM32F302x8)
  */
const R3_1_Params_t R3_1_ParamsM2 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION2,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx                  = ${MC.M2_CS_ADC_U},
  .ADCConfig = {
                 (${getADC("CFG",1,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",1,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",2,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",2,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",3,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",3,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",4,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",4,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",5,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",5,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",6,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",6,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                       
               },

 
/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER2,
  .Tafter                = TW_AFTER2,
  .Tbefore               = TW_BEFORE_R3_1_2,
  .Tsampling             = (uint16_t)SAMPLING_TIME2,
  .Tcase2                = (uint16_t)SAMPLING_TIME2 + (uint16_t)TDEAD2 + (uint16_t)TRISE2,
  .Tcase3                = ((uint16_t)TDEAD2 + (uint16_t)TNOISE2 + (uint16_t)SAMPLING_TIME2) / 2u,
  .TIMx                  = ${_last_word(MC.M2_PWM_TIMER_SELECTION)},

/* Internal COMP settings ----------------------------------------------------*/
        <#if MC.M2_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M2_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE2},
  .CompOCPBSelection     = ${MC.M2_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE2},
  .CompOCPCSelection     = ${MC.M2_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE2},
        <#else><#-- MC.M2_OCP_TOPOLOGY != "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
        </#if><#-- MC.M2_OCP_TOPOLOGY == "EMBEDDED" -->

        <#if MC.INTERNAL_OVERVOLTAGEPROTECTION2>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
        <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION2 == false -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
        </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION2 -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M2_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M2_OVPREF},
};
    </#if><#-- MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
  </#if><#-- CondFamily_STM32F3 --->

<#if CondFamily_STM32G4>
    <#if MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT' >
      <#if MC.M2_USE_INTERNAL_OPAMP>
R3_3_OPAMPParams_t R3_3_OPAMPParamsM2 = 
{ 
  .OPAMPSelect_1 = {
                     ${getOPAMP("IP",1,"PHASE_1",2)}
                    ,${getOPAMP("IP",2,"PHASE_1",2)}
                    ,${getOPAMP("IP",3,"PHASE_1",2)}
                    ,${getOPAMP("IP",4,"PHASE_1",2)}
                    ,${getOPAMP("IP",5,"PHASE_1",2)}
                    ,${getOPAMP("IP",6,"PHASE_1",2)}
                   },
  .OPAMPSelect_2 = {
                     ${getOPAMP("IP",1,"PHASE_2",2)}
                    ,${getOPAMP("IP",2,"PHASE_2",2)}
                    ,${getOPAMP("IP",3,"PHASE_2",2)}
                    ,${getOPAMP("IP",4,"PHASE_2",2)}
                    ,${getOPAMP("IP",5,"PHASE_2",2)}
                    ,${getOPAMP("IP",6,"PHASE_2",2)}
                   },

  .OPAMPConfig1 = {
                    ${getOPAMP("CFG",1,"PHASE_1",2)}
                   ,${getOPAMP("CFG",2,"PHASE_1",2)}
                   ,${getOPAMP("CFG",3,"PHASE_1",2)}
                   ,${getOPAMP("CFG",4,"PHASE_1",2)}
                   ,${getOPAMP("CFG",5,"PHASE_1",2)}
                   ,${getOPAMP("CFG",6,"PHASE_1",2)}
  },
  .OPAMPConfig2 = {${getOPAMP("CFG",1,"PHASE_2",2)}
                  ,${getOPAMP("CFG",2,"PHASE_2",2)}
                  ,${getOPAMP("CFG",3,"PHASE_2",2)}
                  ,${getOPAMP("CFG",4,"PHASE_2",2)}
                  ,${getOPAMP("CFG",5,"PHASE_2",2)}
                  ,${getOPAMP("CFG",6,"PHASE_2",2)}
                 },
};
      </#if><#-- MC.M2_USE_INTERNAL_OPAMP -->
	  <#if MC.M2_CS_ADC_NUM=='2' >
/**
  * @brief  Current sensor parameters Motor 2 - three shunt - G4 
  */
const R3_2_Params_t R3_2_ParamsM2 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION2,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCConfig1 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",2,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",3,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",4,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",5,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",6,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                },
  .ADCConfig2 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",2,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",3,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",4,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",5,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",6,"PHASE_2",2)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                },
 
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",2)}
                  ,${getADC("IP", 2,"PHASE_1",2)}
                  ,${getADC("IP", 3,"PHASE_1",2)}
                  ,${getADC("IP", 4,"PHASE_1",2)}
                  ,${getADC("IP", 5,"PHASE_1",2)}
                  ,${getADC("IP", 6,"PHASE_1",2)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",2)}
                  ,${getADC("IP", 2,"PHASE_2",2)}
                  ,${getADC("IP", 3,"PHASE_2",2)}
                  ,${getADC("IP", 4,"PHASE_2",2)}
                  ,${getADC("IP", 5,"PHASE_2",2)}
                  ,${getADC("IP", 6,"PHASE_2",2)}
                  },

  /* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter     = REP_COUNTER2,
  .Tafter                = TW_AFTER2,
  .Tbefore               = TW_BEFORE2,
  .Tsampling             = (uint16_t)SAMPLING_TIME2,
  .Tcase2                = (uint16_t)SAMPLING_TIME2 + (uint16_t)TDEAD2 + (uint16_t)TRISE2,
  .Tcase3                = ((uint16_t)TDEAD2 + (uint16_t)TNOISE2 + (uint16_t)SAMPLING_TIME2) / 2u,
  .TIMx                  = ${_last_word(MC.M2_PWM_TIMER_SELECTION)},

/* Internal OPAMP common settings --------------------------------------------*/
        <#if MC.M2_USE_INTERNAL_OPAMP>
  .OPAMPParams           = &R3_3_OPAMPParamsM2,
        <#else><#-- MC.M2_USE_INTERNAL_OPAMP == false -->
  .OPAMPParams           = MC_NULL,
        </#if><#-- MC.M2_USE_INTERNAL_OPAMP -->

/* Internal COMP settings ----------------------------------------------------*/
        <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M2_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE2},
  .CompOCPBSelection     = ${MC.M2_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE2},
  .CompOCPCSelection     = ${MC.M2_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE2},
          <#if MC.M2_OCP_COMP_SRC == "DAC">
  .DAC_OCP_ASelection    = ${MC.M2_OCP_DAC_U},
  .DAC_OCP_BSelection    = ${MC.M2_OCP_DAC_V},
  .DAC_OCP_CSelection    = ${MC.M2_OCP_DAC_W},
  .DAC_Channel_OCPA      = LL_DAC_CHANNEL_${_last_word(MC.M2_OCP_DAC_CHANNEL_U, "OUT")},
  .DAC_Channel_OCPB      = LL_DAC_CHANNEL_${_last_word(MC.M2_OCP_DAC_CHANNEL_V, "OUT")},
  .DAC_Channel_OCPC      = LL_DAC_CHANNEL_${_last_word(MC.M2_OCP_DAC_CHANNEL_W, "OUT")},
          <#else><#-- MC.M2_OCP_COMP_SRC != "DAC" -->
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t)0,
  .DAC_Channel_OCPB      = (uint32_t)0,
  .DAC_Channel_OCPC      = (uint32_t)0,
          </#if><#-- MC.M2_OCP_COMP_SRC == "DAC" -->
        <#else><#-- MC.M2_OCP_TOPOLOGY != "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t)0,
  .DAC_Channel_OCPB      = (uint32_t)0,
  .DAC_Channel_OCPC      = (uint32_t)0,
        </#if><#-- MC.M2_OCP_TOPOLOGY == "EMBEDDED" -->

        <#if MC.INTERNAL_OVERVOLTAGEPROTECTION2>
  .CompOVPSelection      = OVP_SELECTION2,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE2,
          <#if MC.DAC_OVP_M2 != "NOT_USED">
  .DAC_OVP_Selection     = ${_first_word(MC.DAC_OVP_M2)},
  .DAC_Channel_OVP       = LL_DAC_CHANNEL_${_last_word(MC.DAC_OVP_M2, "CH")},
          <#else><#-- MC.DAC_OVP_M2 == "NOT_USED" -->
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t)0,
          </#if><#-- MC.DAC_OVP_M2 != "NOT_USED" -->
        <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION2 == false -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t)0,
        </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION2 -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M2_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M2_OVPREF},

}; 
      </#if><#-- MC.M2_CS_ADC_NUM=='2' -->
      <#if MC.M2_CS_ADC_NUM=='1'>
/**
  * @brief  Current sensor parameters Motor 2 - three shunt - G4 
  */
const R3_1_Params_t R3_1_ParamsM2 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio       = FREQ_RATIO,                         
  .IsHigherFreqTim = FREQ_RELATION2,                      
                                                          
/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx            = ${MC.M2_CS_ADC_U},

  .ADCConfig = {
                 (${getADC("CFG",1,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",1,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",2,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",2,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",3,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",3,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",4,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",4,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",5,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",5,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",6,"PHASE_1",2)}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",6,"PHASE_2",2)}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M2}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                       
               },

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER2,                    
  .Tafter            = TW_AFTER2,                       
  .Tbefore           = TW_BEFORE_R3_1_2,     
  .Tsampling         = (uint16_t)SAMPLING_TIME2,         
  .Tcase2            = (uint16_t)SAMPLING_TIME2 + (uint16_t)TDEAD2 + (uint16_t)TRISE2,
  .Tcase3            = ((uint16_t)TDEAD2 + (uint16_t)TNOISE2 + (uint16_t)SAMPLING_TIME2)/2u,                     
  .TIMx               = ${_last_word(MC.M2_PWM_TIMER_SELECTION)},            

/* Internal OPAMP common settings --------------------------------------------*/
<#if MC.M2_USE_INTERNAL_OPAMP>
  .OPAMPParams     = &R3_3_OPAMPParamsM2,
<#else>  
  .OPAMPParams     = MC_NULL,
</#if>  
/* Internal COMP settings ----------------------------------------------------*/
<#if MC.M2_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M2_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE2},                          
  .CompOCPBSelection     = ${MC.M2_OCP_COMP_V},                
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE2},              
  .CompOCPCSelection     = ${MC.M2_OCP_COMP_W},  
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE2},
<#if MC.M2_OCP_COMP_SRC == "DAC">  
  .DAC_OCP_ASelection    = ${MC.M2_OCP_DAC_U},
  .DAC_OCP_BSelection    = ${MC.M2_OCP_DAC_V},
  .DAC_OCP_CSelection    = ${MC.M2_OCP_DAC_W},
  .DAC_Channel_OCPA      = LL_DAC_CHANNEL_${_last_word(MC.M2_OCP_DAC_CHANNEL_U, "OUT")},
  .DAC_Channel_OCPB      = LL_DAC_CHANNEL_${_last_word(MC.M2_OCP_DAC_CHANNEL_V, "OUT")},
  .DAC_Channel_OCPC      = LL_DAC_CHANNEL_${_last_word(MC.M2_OCP_DAC_CHANNEL_W, "OUT")},   
<#else>
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t) 0,
  .DAC_Channel_OCPB      = (uint32_t) 0,
  .DAC_Channel_OCPC      = (uint32_t) 0,  
</#if>                                                                           
<#else>  
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,                          
  .CompOCPBSelection     = MC_NULL,      
  .CompOCPBInvInput_MODE = NONE,              
  .CompOCPCSelection     = MC_NULL,        
  .CompOCPCInvInput_MODE = NONE,   
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t) 0,
  .DAC_Channel_OCPB      = (uint32_t) 0,
  .DAC_Channel_OCPC      = (uint32_t) 0,                                        
</#if>

<#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION2,                  
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE2,  
<#if MC.DAC_OVP_M2 != "NOT_USED">    
  .DAC_OVP_Selection     = ${_first_word(MC.DAC_OVP_M2)},
  .DAC_Channel_OVP       = LL_DAC_CHANNEL_${_last_word(MC.DAC_OVP_M2, "CH")},
<#else>
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t) 0,
</#if>  
<#else>
  .CompOVPSelection      = MC_NULL,       
  .CompOVPInvInput_MODE  = NONE,
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t) 0,   
</#if>   
                                                         
/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold =  ${MC.M2_DAC_CURRENT_THRESHOLD},                                        
  .DAC_OVP_Threshold =  ${MC.M2_OVPREF},                                                                       
                                                    
};
      </#if><#-- MC.MC.M2_CS_ADC_NUM=='1' -->
    </#if> <#-- MC.M2_CURRENT_SENSING_TOPO=='THREE_SHUNT' --> <#-- Inside CondFamily_STM32G4 -->

    <#if MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
const ICS_Params_t ICS_ParamsM2 = 
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio = FREQ_RATIO,
  .IsHigherFreqTim = FREQ_RELATION2,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx_1 = ${MC.M2_CS_ADC_U},
  .ADCx_2 = ${MC.M2_CS_ADC_V},
      <#if MC.M2_PWM_TIMER_SELECTION == "PWM_TIM1">
  .ADCConfig1        = (uint32_t)(MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_U} << ADC_JSQR_JSQ1_Pos) | LL_ADC_INJ_TRIG_EXT_RISING
                     | LL_ADC_INJ_TRIG_EXT_TIM1_TRGO,
  .ADCConfig2        = (uint32_t)(MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_V} << ADC_JSQR_JSQ1_Pos) | LL_ADC_INJ_TRIG_EXT_RISING
                     | LL_ADC_INJ_TRIG_EXT_TIM1_TRGO,
      <#elseif MC.M2_PWM_TIMER_SELECTION == "PWM_TIM8">
  .ADCConfig1        = (uint32_t)(MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_U} << ADC_JSQR_JSQ1_Pos) | LL_ADC_INJ_TRIG_EXT_RISING
                     | LL_ADC_INJ_TRIG_EXT_TIM8_TRGO,
  .ADCConfig2        = (uint32_t)(MC_ADC_CHANNEL_${MC.M2_CS_CHANNEL_V} << ADC_JSQR_JSQ1_Pos) | LL_ADC_INJ_TRIG_EXT_RISING
                     | LL_ADC_INJ_TRIG_EXT_TIM8_TRGO,
      </#if><#-- MC.M2_PWM_TIMER_SELECTION == "PWM_TIM1" -->

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER2,
  .TIMx              = ${_last_word(MC.M2_PWM_TIMER_SELECTION)}

};
    </#if><#-- MC.M2_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->

    <#if MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS'>
  const ICS_Params_t ICS_ParamsM1 = 
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio         = FREQ_RATIO,
  .IsHigherFreqTim   = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx_1 = ${MC.M1_CS_ADC_U},
  .ADCx_2 = ${MC.M1_CS_ADC_V},
      <#if MC.M1_PWM_TIMER_SELECTION == "PWM_TIM1">
  .ADCConfig1        = (uint32_t)(MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U} << ADC_JSQR_JSQ1_Pos) | LL_ADC_INJ_TRIG_EXT_RISING
                     | LL_ADC_INJ_TRIG_EXT_TIM1_TRGO,
  .ADCConfig2        = (uint32_t)(MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V} << ADC_JSQR_JSQ1_Pos) | LL_ADC_INJ_TRIG_EXT_RISING
                     | LL_ADC_INJ_TRIG_EXT_TIM1_TRGO,
      <#elseif MC.M1_PWM_TIMER_SELECTION == "PWM_TIM8">
  .ADCConfig1        = (uint32_t)(MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_U} << ADC_JSQR_JSQ1_Pos) | LL_ADC_INJ_TRIG_EXT_RISING
                     | LL_ADC_INJ_TRIG_EXT_TIM8_TRGO,
  .ADCConfig2        = (uint32_t)(MC_ADC_CHANNEL_${MC.M1_CS_CHANNEL_V} << ADC_JSQR_JSQ1_Pos) | LL_ADC_INJ_TRIG_EXT_RISING
                     | LL_ADC_INJ_TRIG_EXT_TIM8_TRGO,
      </#if><#-- MC.M1_PWM_TIMER_SELECTION == "PWM_TIM1" -->

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)}

};
    </#if><#-- MC.M1_CURRENT_SENSING_TOPO == 'ICS_SENSORS' -->
  
    <#if MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT'><#-- Inside CondFamily_STM32G4 --->
  
<#if MC.M1_SPEED_SENSOR == "HSO" ||  MC.M1_SPEED_SENSOR == "ZEST" >  
        <#if MC.M1_USE_INTERNAL_OPAMP == true>
/**
  * @brief  Internal OPAMP parameters Motor 1 - three shunt - G4xx 
  */
R3_3_OPAMPParams_t R3_3_OPAMPParamsM1 =
{
   .OPAMPx_1 = ${MC.M1_CS_OPAMP_U},
   .OPAMPx_2 = ${MC.M1_CS_OPAMP_V},
   .OPAMPx_3 = ${MC.M1_CS_OPAMP_W},              
};
        </#if>

/**
  * @brief  Current sensor parameters Motor 1 - three shunt - G4 HSO
  */
const R3_Params_t R3_ParamsM1 =
{
                                                          
        <#if MC.M1_CS_ADC_NUM == "3">  <#-- Inside G4 Family HSO Speed Sensor -->
/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx_1           = ${MC.M1_CS_ADC_U},                   
  .ADCx_2           = ${MC.M1_CS_ADC_V},
  .ADCx_3           = ${MC.M1_CS_ADC_W},
        <#else>
/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx_1           = ${MC.M1_CS_ADC_U},                   
  .ADCx_2           = <#if MC.M1_CS_ADC_U != MC.M1_CS_ADC_V>${MC.M1_CS_ADC_V}<#else>${MC.M1_CS_ADC_W}</#if>,

        </#if>
  /* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,  
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)}, 
  .TIMx_Oversample   = TIM3,

/* Internal OPAMP common settings --------------------------------------------*/
        <#if MC.M1_USE_INTERNAL_OPAMP == true>
  .OPAMPParams     = &R3_3_OPAMPParamsM1,
        <#else>
  .OPAMPParams     = MC_NULL,
        </#if>

/* Internal COMP settings ----------------------------------------------------*/
        <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M1_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE},
  .CompOCPBSelection     = ${MC.M1_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE},
  .CompOCPCSelection     = ${MC.M1_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE},
          <#if MC.M1_OCP_COMP_SRC == "DAC">
  .DAC_OCP_ASelection    = ${MC.M1_OCP_DAC_U},
  .DAC_OCP_BSelection    = ${MC.M1_OCP_DAC_V},
  .DAC_OCP_CSelection    = ${MC.M1_OCP_DAC_W},
  .DAC_Channel_OCPA      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_U, "OUT")},
  .DAC_Channel_OCPB      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_V, "OUT")},
  .DAC_Channel_OCPC      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_W, "OUT")},
          <#else><#-- MC.M1_OCP_COMP_SRC != "DAC" -->
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t)0,
  .DAC_Channel_OCPB      = (uint32_t)0,
  .DAC_Channel_OCPC      = (uint32_t)0,
          </#if><#-- MC.M1_OCP_COMP_SRC == "DAC" -->
        <#else><#-- MC.M1_OCP_TOPOLOGY != "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t)0,
  .DAC_Channel_OCPB      = (uint32_t)0,
  .DAC_Channel_OCPC      = (uint32_t)0,
        </#if><#-- MC.M1_OCP_TOPOLOGY == "EMBEDDED" -->
        <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
          <#if MC.DAC_OVP_M1 != "NOT_USED">
  .DAC_OVP_Selection     = ${_first_word(MC.DAC_OVP_M1)},
  .DAC_Channel_OVP       = LL_DAC_CHANNEL_${_last_word(MC.DAC_OVP_M1, "CH")},
          <#else><#-- MC.DAC_OVP_M1 == "NOT_USED" -->
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t)0,
          </#if><#-- MC.DAC_OVP_M1 != "NOT_USED" -->
        <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION == false -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t)0,
        </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M1_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M1_OVPREF},                                                                     

        <#if MC.M1_CS_ADC_NUM == "2">
          <#assign DMA_CHX=MC.M1_HSO_DMACH_ADC_U>
          <#if MC.M1_HSO_DMACH_ADC_U == MC.M1_HSO_DMACH_ADC_V>
            <#assign DMA_CHY=MC.M1_HSO_DMACH_ADC_W>
          <#else>
            <#assign DMA_CHY=MC.M1_HSO_DMACH_ADC_V>
          </#if>
        <#else> <#-- MC.M1_CS_ADC_NUM == "2" -->
          <#assign DMA_CHX=MC.M1_HSO_DMACH_ADC_U>
          <#assign DMA_CHY=MC.M1_HSO_DMACH_ADC_V>
        </#if> <#-- MC.M1_CS_ADC_NUM == "2" -->
<#assign DMA_CHZ=MC.M1_HSO_DMACH_ADC_W>
  .DMA_ADCx_1 = ${MC.M1_HSO_DMA_ADC_U}_Channel${DMA_CHX},
  .DMA_ADCx_2 = ${MC.M1_HSO_DMA_ADC_V}_Channel${DMA_CHY},
  .DMA_ADCx_3 = ${MC.M1_HSO_DMA_ADC_W}_Channel${DMA_CHZ},

};

      <#else> <#-- MC.M1_SPEED_SENSOR != "HSO" -->  
        <#if MC.M1_USE_INTERNAL_OPAMP> <#-- Inside G4 Family 3 shunts -->
/**
  * @brief  Internal OPAMP parameters Motor 1 - three shunt - G4xx - Shared Resources
  * 
  */
R3_3_OPAMPParams_t R3_3_OPAMPParamsM1 =
{

  .OPAMPSelect_1 = {
                     ${getOPAMP("IP",1,"PHASE_1",1)}
                    ,${getOPAMP("IP",2,"PHASE_1",1)}
                    ,${getOPAMP("IP",3,"PHASE_1",1)}
                    ,${getOPAMP("IP",4,"PHASE_1",1)}
                    ,${getOPAMP("IP",5,"PHASE_1",1)}
                    ,${getOPAMP("IP",6,"PHASE_1",1)}
                   },
  .OPAMPSelect_2 = {
                     ${getOPAMP("IP",1,"PHASE_2",1)}
                    ,${getOPAMP("IP",2,"PHASE_2",1)}
                    ,${getOPAMP("IP",3,"PHASE_2",1)}
                    ,${getOPAMP("IP",4,"PHASE_2",1)}
                    ,${getOPAMP("IP",5,"PHASE_2",1)}
                    ,${getOPAMP("IP",6,"PHASE_2",1)}
                   },

  .OPAMPConfig1 = {
                    ${getOPAMP("CFG",1,"PHASE_1",1)}
                   ,${getOPAMP("CFG",2,"PHASE_1",1)}
                   ,${getOPAMP("CFG",3,"PHASE_1",1)}
                   ,${getOPAMP("CFG",4,"PHASE_1",1)}
                   ,${getOPAMP("CFG",5,"PHASE_1",1)}
                   ,${getOPAMP("CFG",6,"PHASE_1",1)}
  },
  .OPAMPConfig2 = {${getOPAMP("CFG",1,"PHASE_2",1)}
                  ,${getOPAMP("CFG",2,"PHASE_2",1)}
                  ,${getOPAMP("CFG",3,"PHASE_2",1)}
                  ,${getOPAMP("CFG",4,"PHASE_2",1)}
                  ,${getOPAMP("CFG",5,"PHASE_2",1)}
                  ,${getOPAMP("CFG",6,"PHASE_2",1)}
                 },
};
      </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->
	  <#if MC.M1_CS_ADC_NUM=='2'>
/**
  * @brief  Current sensor parameters Motor 1 - three shunt - G4 
  */
//cstat !MISRAC2012-Rule-8.4
const R3_2_Params_t R3_2_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCConfig1 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                },
  .ADCConfig2 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                 ,(uint32_t)(${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                },
 
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",1)}
                  ,${getADC("IP", 2,"PHASE_1",1)}
                  ,${getADC("IP", 3,"PHASE_1",1)}
                  ,${getADC("IP", 4,"PHASE_1",1)}
                  ,${getADC("IP", 5,"PHASE_1",1)}
                  ,${getADC("IP", 6,"PHASE_1",1)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",1)}
                  ,${getADC("IP", 2,"PHASE_2",1)}
                  ,${getADC("IP", 3,"PHASE_2",1)}
                  ,${getADC("IP", 4,"PHASE_2",1)}
                  ,${getADC("IP", 5,"PHASE_2",1)}
                  ,${getADC("IP", 6,"PHASE_2",1)}
                  },
 //cstat +MISRAC2012-Rule-12.1 +MISRAC2012-Rule-10.1_R6

  /* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter     = REP_COUNTER,
  .Tafter                = TW_AFTER,
  .Tbefore               = TW_BEFORE,
  .Tsampling             = (uint16_t)SAMPLING_TIME,
  .Tcase2                = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3                = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,
  .TIMx                  = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},

/* Internal OPAMP common settings --------------------------------------------*/
          <#if MC.M1_USE_INTERNAL_OPAMP>
  .OPAMPParams           = &R3_3_OPAMPParamsM1,
          <#else><#-- MC.M1_USE_INTERNAL_OPAMP == false -->
  .OPAMPParams           = MC_NULL,
          </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->

/* Internal COMP settings ----------------------------------------------------*/
          <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M1_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE},
  .CompOCPBSelection     = ${MC.M1_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE},
  .CompOCPCSelection     = ${MC.M1_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE},
            <#if MC.M1_OCP_COMP_SRC == "DAC">
  .DAC_OCP_ASelection    = ${MC.M1_OCP_DAC_U},
  .DAC_OCP_BSelection    = ${MC.M1_OCP_DAC_V},
  .DAC_OCP_CSelection    = ${MC.M1_OCP_DAC_W},
  .DAC_Channel_OCPA      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_U, "OUT")},
  .DAC_Channel_OCPB      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_V, "OUT")},
  .DAC_Channel_OCPC      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_W, "OUT")},
            <#else><#-- MC.M1_OCP_COMP_SRC != "DAC" -->
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t)0,
  .DAC_Channel_OCPB      = (uint32_t)0,
  .DAC_Channel_OCPC      = (uint32_t)0,
            </#if><#-- MC.M1_OCP_COMP_SRC == "DAC" -->
          <#else><#-- MC.M1_OCP_TOPOLOGY != "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t)0,
  .DAC_Channel_OCPB      = (uint32_t)0,
  .DAC_Channel_OCPC      = (uint32_t)0,
          </#if><#-- MC.M1_OCP_TOPOLOGY == "EMBEDDED" -->
          <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
            <#if MC.DAC_OVP_M1 != "NOT_USED">
  .DAC_OVP_Selection     = ${_first_word(MC.DAC_OVP_M1)},
  .DAC_Channel_OVP       = LL_DAC_CHANNEL_${_last_word(MC.DAC_OVP_M1, "CH")},
            <#else><#-- MC.DAC_OVP_M1 == "NOT_USED" -->
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t)0,
            </#if><#-- MC.DAC_OVP_M1 != "NOT_USED" -->
          <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION == false -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t)0,
          </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M1_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M1_OVPREF},

};
        </#if> <#-- <#if MC.M1_CS_ADC_NUM=='2' -->
        <#if MC.M1_CS_ADC_NUM=='1'> <#-- Inside G4 Family HSO excluded -->
/**
  * @brief  Current sensor parameters Motor 1 - three shunt - G4 
  */
const R3_1_Params_t R3_1_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio       = FREQ_RATIO,                         
  .IsHigherFreqTim = FREQ_RELATION,                      
                                                          
/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx            = ${MC.M1_CS_ADC_U},

  .ADCConfig = {
                 (${getADC("CFG",1,"PHASE_1")}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",1,"PHASE_2")}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",2,"PHASE_1")}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",2,"PHASE_2")}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",3,"PHASE_1")}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",3,"PHASE_2")}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",4,"PHASE_1")}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",4,"PHASE_2")}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",5,"PHASE_1")}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",5,"PHASE_2")}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                ,(${getADC("CFG",6,"PHASE_1")}U << ADC_JSQR_JSQ1_Pos)
               | (${getADC("CFG",6,"PHASE_2")}U << ADC_JSQR_JSQ2_Pos) | 1<< ADC_JSQR_JL_Pos
               | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)                       
               },

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = REP_COUNTER,                    
  .Tafter            = TW_AFTER,                       
  .Tbefore           = TW_BEFORE_R3_1,     
  .Tsampling         = (uint16_t)SAMPLING_TIME,         
  .Tcase2            = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3            = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME)/2u,                     
  .TIMx               = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},            
                                                        
/* Internal OPAMP common settings --------------------------------------------*/
          <#if MC.M1_USE_INTERNAL_OPAMP>
  .OPAMPParams     = &R3_3_OPAMPParamsM1,
          <#else>  
  .OPAMPParams     = MC_NULL,
          </#if>  
/* Internal COMP settings ----------------------------------------------------*/
          <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M1_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE},                          
  .CompOCPBSelection     = ${MC.M1_OCP_COMP_V},                
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE},              
  .CompOCPCSelection     = ${MC.M1_OCP_COMP_W},  
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE},
            <#if MC.M1_OCP_COMP_SRC == "DAC">  
  .DAC_OCP_ASelection    = ${MC.M1_OCP_DAC_U},
  .DAC_OCP_BSelection    = ${MC.M1_OCP_DAC_V},
  .DAC_OCP_CSelection    = ${MC.M1_OCP_DAC_W},
  .DAC_Channel_OCPA      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_U, "OUT")},
  .DAC_Channel_OCPB      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_V, "OUT")},
  .DAC_Channel_OCPC      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_W, "OUT")},   
            <#else>
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t) 0,
  .DAC_Channel_OCPB      = (uint32_t) 0,
  .DAC_Channel_OCPC      = (uint32_t) 0,  
            </#if> <#-- M1_OCP_COMP_SRC == DAC -->
          <#else> <#-- M1_OCP_TOPLOGY != EMBEDDED -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,                          
  .CompOCPBSelection     = MC_NULL,      
  .CompOCPBInvInput_MODE = NONE,              
  .CompOCPCSelection     = MC_NULL,        
  .CompOCPCInvInput_MODE = NONE,   
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t) 0,
  .DAC_Channel_OCPB      = (uint32_t) 0,
  .DAC_Channel_OCPC      = (uint32_t) 0,                                        
          </#if> <#-- MC.M1_OCP_TOPOLOGY == "EMBEDDED"-->
          <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,                  
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,  
            <#if MC.DAC_OVP_M1 != "NOT_USED">    
  .DAC_OVP_Selection     = ${_first_word(MC.DAC_OVP_M1)},
  .DAC_Channel_OVP       = LL_DAC_CHANNEL_${_last_word(MC.DAC_OVP_M1, "CH")},
            <#else>
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t) 0,
            </#if>  
          <#else> <#-- ! MC.INTERNAL_OVERVOLTAGEPROTECTION -->
  .CompOVPSelection      = MC_NULL,       
  .CompOVPInvInput_MODE  = NONE,
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t) 0,   
          </#if> <#-- #if MC.INTERNAL_OVERVOLTAGEPROTECTION -->
                                                         
/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold =  ${MC.M1_DAC_CURRENT_THRESHOLD},                                        
  .DAC_OVP_Threshold =  ${MC.M1_OVPREF},                                                                       
                                                    
};
        </#if> <#-- <#if MC.M1_CS_ADC_NUM=='1' --> <#-- Inside CondFamily_STM32G4 -->
      </#if> <#-- <#else of if MC.M1_SPEED_SENSOR == "HSO">  -->
	</#if><#-- MC.M1_CURRENT_SENSING_TOPO=='THREE_SHUNT' --><#-- Inside CondFamily_STM32G4 -->
  </#if><#-- CondFamily_STM32G4 -->
  <#if CondFamily_STM32C0 || CondFamily_STM32G0 || CondFamily_STM32F3 || CondFamily_STM32L4 || CondFamily_STM32F0 || CondFamily_STM32G4 || CondFamily_STM32H5>
<#assign currentFactor = "">
  <#elseif CondFamily_STM32F4 || CondFamily_STM32F7>
<#assign currentFactor = "*2">
  </#if><#-- CondFamily_STM32C0 || CondFamily_STM32G0 || CondFamily_STM32F3 || CondFamily_STM32L4 || CondFamily_STM32F0
          || CondFamily_STM32G4|| CondFamily_STM32H5 -->
  <#if CondFamily_STM32F3 || CondFamily_STM32L4 || CondFamily_STM32F4 ||  CondFamily_STM32F7 || CondFamily_STM32G4 || CondFamily_STM32H5>
<#assign HAS_ADC_INJ = true>
  <#else><#-- CondFamily_STM32F3 == false && CondFamily_STM32L4 == false && CondFamily_STM32F4 == false
           && CondFamily_STM32F7 == false && CondFamily_STM32G4 == false -->
<#assign HAS_ADC_INJ = false>
  </#if><#-- CondFamily_STM32F3 || CondFamily_STM32L4 || CondFamily_STM32F4 ||  CondFamily_STM32F7
          || CondFamily_STM32G4 -->
  <#if CondFamily_STM32F3 || CondFamily_STM32L4 || CondFamily_STM32G0 || CondFamily_STM32F7 || CondFamily_STM32G4 || CondFamily_STM32H5>
<#assign HAS_TIM_6_CH = true>
  <#else><#-- CondFamily_STM32F3 == false && CondFamily_STM32L4 == false && CondFamily_STM32G0 == false
           &&| CondFamily_STM32F7 == false && CondFamily_STM32G4 && CondFamily_STM32H5 == false -->
<#assign HAS_TIM_6_CH = false>
  </#if><#-- CondFamily_STM32F3 || CondFamily_STM32L4 || CondFamily_STM32G0 || CondFamily_STM32F7 
          || CondFamily_STM32G4  || CondFamily_STM32H5 -->
  <#if CondFamily_STM32F4 || CondFamily_STM32F7>
<#assign DMA_TYPE = "Stream">
  <#else><#-- CondFamily_STM32F4 == false && CondFamily_STM32F7 == false -->
<#assign DMA_TYPE = "Channel">
  </#if><#-- CondFamily_STM32F4 || CondFamily_STM32F7 -->
  <#if CondFamily_STM32F3 || CondFamily_STM32L4 || CondFamily_STM32G4 || CondFamily_STM32F7 || CondFamily_STM32H5>
<#assign HAS_BRK2 = true>
  <#else><#-- CondFamily_STM32F3 == false && CondFamily_STM32L4 == false && CondFamily_STM32G4 == false
           && CondFamily_STM32F7 == false -->
<#assign HAS_BRK2 = false>
  </#if><#-- CondFamily_STM32F3 || CondFamily_STM32L4 || CondFamily_STM32G4 || CondFamily_STM32F7 -->

  <#if CondFamily_STM32F0 == true>
<#assign R1_DMA_M1 = "DMA1">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_5">
<#assign R1_DMA_CH_AUX_M1 = "LL_DMA_CHANNEL_4">
<#assign R1_DMA_CH_ADC_M1 = "LL_DMA_CHANNEL_1">
  </#if><#-- CondFamily_STM32F0 == true -->
  <#if CondFamily_STM32G0 == true>
<#assign R1_DMA_M1 = "DMA1">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_4">
<#assign R1_DMA_CH_ADC_M1 = "LL_DMA_CHANNEL_1">
  </#if><#-- CondFamily_STM32G0 == true -->
  <#if CondFamily_STM32C0 == true>
<#assign R1_DMA_M1 = "DMA1">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_3">
<#assign R1_DMA_CH_AUX_M1 = "LL_DMA_CHANNEL_2">  
<#assign R1_DMA_CH_ADC_M1 = "LL_DMA_CHANNEL_1">
  </#if><#-- CondFamily_STM32C0 == true -->
  <#if CondFamily_STM32F3 == true>
    <#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M1 = "DMA1">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_4">
    <#elseif MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M1_PWM_TIMER_SELECTION == 'TIM8'>
<#assign R1_DMA_M1 = "DMA2">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_2">
    </#if><#-- MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1' -->
  </#if><#-- CondFamily_STM32F3 == true -->
  <#if CondFamily_STM32G4 == true>
    <#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M1 = "DMA1">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_1">
    <#elseif MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M1_PWM_TIMER_SELECTION == 'TIM8'>
<#assign R1_DMA_M1 = "DMA2">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_1">
    </#if>
  </#if> 
  <#if CondFamily_STM32H5 == true> 
    <#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M1 = "GPDMA1">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_0"> 
    <#elseif MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M1_PWM_TIMER_SELECTION == 'TIM8'>
#error To Be Done
    </#if><#-- MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1' -->
  </#if><#-- CondFamily_STM32H5 == true -->
  <#if CondFamily_STM32L4 == true>
<#assign R1_DMA_STATUS_REG_M1 = "ISR">
<#assign R1_DMA_CLEAR_REG_M1 = "IFCR">
    <#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M1 = "DMA1">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_4">
    <#elseif MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M1_PWM_TIMER_SELECTION == 'TIM8'>
<#assign R1_DMA_M1 = "DMA2">
<#assign R1_DMA_CH_M1 = "LL_DMA_CHANNEL_2">
    </#if><#-- MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1' -->
  </#if><#-- CondFamily_STM32L4 == true -->
  <#if CondFamily_STM32F4 == true>
    <#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M1 = "DMA2">
<#assign R1_DMA_CH_M1 = "LL_DMA_STREAM_5">
<#assign R1_DMA_CH_AUX_M1 = "LL_DMA_STREAM_4">
    <#elseif MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M1_PWM_TIMER_SELECTION == 'TIM8'>
<#assign R1_DMA_M1 = "DMA2">
<#assign R1_DMA_CH_M1 = "LL_DMA_STREAM_1">
<#assign R1_DMA_CH_AUX_M1 = "LL_DMA_STREAM_7">
    </#if><#-- MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1' -->
  </#if><#-- CondFamily_STM32F4 == true -->
  <#if CondFamily_STM32F7 == true>
    <#if MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M1 = "DMA2">
<#assign R1_DMA_CH_M1 = "LL_DMA_STREAM_4">
    <#elseif MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M1_PWM_TIMER_SELECTION == 'TIM8'>
<#assign R1_DMA_M1 = "DMA2">
<#assign R1_DMA_CH_M1 = "LL_DMA_STREAM_7">
    </#if><#-- MC.M1_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M1_PWM_TIMER_SELECTION == 'TIM1' -->
  </#if><#-- CondFamily_STM32F7 == true -->
  <#if MC.DRIVE_NUMBER != "1">
    <#if CondFamily_STM32F3 == true>
      <#if MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M2_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M2 = "DMA1">
<#assign R1_DMA_CH_M2 = "LL_DMA_CHANNEL_5">
      <#elseif MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M2_PWM_TIMER_SELECTION == 'TIM8'>
<#assign R1_DMA_M2 = "DMA2">
<#assign R1_DMA_CH_M2 = "LL_DMA_CHANNEL_2">
      </#if><#-- MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M2_PWM_TIMER_SELECTION == 'TIM1' -->
    </#if><#-- CondFamily_STM32F3 == true -->
    <#if CondFamily_STM32G4 == true>
      <#if MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M2_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M2 = "DMA1">
<#assign R1_DMA_CH_M2 = "LL_DMA_CHANNEL_1">
      <#elseif MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M2_PWM_TIMER_SELECTION == 'TIM8'>
<#assign R1_DMA_M2 = "DMA2">
<#assign R1_DMA_CH_M2 = "LL_DMA_CHANNEL_1">
      </#if><#-- MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M2_PWM_TIMER_SELECTION == 'TIM1' -->
    </#if><#--CondFamily_STM32G4 == true -->
    <#if CondFamily_STM32F4 == true>
      <#if MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M2_PWM_TIMER_SELECTION == 'TIM1'>
<#assign R1_DMA_M2 = "DMA2">
<#assign R1_DMA_CH_M2 = "LL_DMA_STREAM_5">
<#assign R1_DMA_CH_AUX_M2 = "LL_DMA_STREAM_4">
      <#elseif MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM8' || MC.M2_PWM_TIMER_SELECTION == 'TIM8'>
<#assign R1_DMA_M2 = "DMA2">
<#assign R1_DMA_CH_M2 = "LL_DMA_STREAM_1">
<#assign R1_DMA_CH_AUX_M2 = "LL_DMA_STREAM_7">
      </#if><#-- MC.M2_PWM_TIMER_SELECTION == 'PWM_TIM1' || MC.M2_PWM_TIMER_SELECTION == 'TIM1' -->
    </#if><#-- CondFamily_STM32F4 == true -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->
<#-- one shunt pahse shift********************************* -->
  <#if ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
/**
  * @brief  Current sensor parameters Motor 1 - single shunt phase shift
  */
//cstat !MISRAC2012-Rule-8.4
const R1_Params_t R1_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx                  = ${MC.M1_CS_ADC_U},
  .IChannel              = ${getADCChannel(MC.M1_CS_ADC_U,MC.M1_CS_CHANNEL_U)},
    <#if CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0>
  .ISamplingTime = LL_ADC_SAMPLINGTIME_${MC.M1_CURR_SAMPLING_TIME}<#if MC.M1_CURR_SAMPLING_TIME != "1">CYCLES_5<#else>CYCLE_5</#if>,
    </#if><#-- CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0 -->

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter     = ${MC.M1_REGULATION_EXECUTION_RATE},
  .TMin                  = TMIN,
  .TSample               = (uint16_t)(TBEFORE),
  .TIMx                  = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},
  .DMAx                  = ${R1_DMA_M1},
  .DMAChannelX           = ${R1_DMA_CH_M1},
    <#if HAS_TIM_6_CH == false>
  .DMASamplingPtChannelX = ${R1_DMA_CH_AUX_M1},
    </#if><#-- HAS_TIM_6_CH == false -->
    <#if HAS_ADC_INJ == false>
  .DMA_ADC_DR_ChannelX   = ${R1_DMA_CH_ADC_M1},
    </#if><#-- HAS_ADC_INJ == false -->
  .hTADConv              = (uint16_t)((ADC_SAR_CYCLES+ADC_TRIG_CONV_LATENCY_CYCLES) * (ADV_TIM_CLK_MHz/ADC_CLK_MHz)),
<#-- ADC_TRIG_CONV_LATENCY_CYCLES = 40 for G4 TODO -->

/* Internal OPAMP common settings --------------------------------------------*/
    <#if MC.M1_USE_INTERNAL_OPAMP>
  .OPAMP_Selection       = ${_filter_opamp (MC.M1_CS_OPAMP_U)},
    </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->

  <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
/* Internal COMP settings ----------------------------------------------------*/   
  .CompOCPSelection      = ${MC.M1_OCP_COMP_U},
  .CompOCPInvInput_MODE  = <#if MC.M1_OCP_COMP_SRC=="DAC">DAC_MODE<#else>NONE</#if>,
  </#if><#-- M1_OCP_TOPOLOGY == "EMBEDDED" -->
  <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
  </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

  <#if MC.M1_OCP_COMP_SRC == "DAC">
/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M1_DAC_CURRENT_THRESHOLD},
    </#if><#-- MC.M1_OCP_COMP_SRC -->
    <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .DAC_OVP_Threshold     = ${MC.M1_OVPREF},
    </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

};  
  </#if><#-- ((MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M1_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
  <#if ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN'))>
/**
  * @brief  Current sensor parameters Motor 2 - single shunt phase shift
  */
//cstat !MISRAC2012-Rule-8.4
const R1_Params_t R1_ParamsM2 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCx                  = ${MC.M2_CS_ADC_U},
  .IChannel              = ${getADCChannel(MC.M2_CS_ADC_U,MC.M2_CS_CHANNEL_U)},
    <#if CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0>
  .ISamplingTime = LL_ADC_SAMPLINGTIME_${MC.M2_CURR_SAMPLING_TIME}<#if MC.M2_CURR_SAMPLING_TIME != "1">CYCLES_5<#else>CYCLE_5</#if>,
    </#if><#-- CondFamily_STM32F0 || CondFamily_STM32G0 || CondFamily_STM32C0 -->

/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter     = ${MC.M2_REGULATION_EXECUTION_RATE},
  .TMin                  = TMIN2,
  .TSample               = (uint16_t)(TBEFORE2),
  .TIMx                  = ${_last_word(MC.M2_PWM_TIMER_SELECTION)},
  .DMAx                  = ${R1_DMA_M2},
  .DMAChannelX           = ${R1_DMA_CH_M2},
    <#if HAS_TIM_6_CH == false>
  .DMASamplingPtChannelX = ${R1_DMA_CH_AUX_M2},
    </#if><#-- HAS_TIM_6_CH == false -->
    <#if HAS_ADC_INJ == false>
  .DMA_ADC_DR_ChannelX   = ${R1_DMA_CH_ADC_M2},
    </#if><#-- HAS_ADC_INJ == false -->
  .hTADConv              = (uint16_t)((ADC_SAR_CYCLES+ADC_TRIG_CONV_LATENCY_CYCLES) * (ADV_TIM_CLK_MHz/ADC_CLK_MHz)),
<#-- ADC_TRIG_CONV_LATENCY_CYCLES = 40 for G4 TODO -->

/* Internal OPAMP common settings --------------------------------------------*/
    <#if MC.M2_USE_INTERNAL_OPAMP>
  .OPAMP_Selection       = ${_filter_opamp (MC.M2_CS_OPAMP_U)},
    </#if><#-- MC.M2_USE_INTERNAL_OPAMP -->

<#if MC.M2_OCP_TOPOLOGY == "EMBEDDED">
/* Internal COMP settings ----------------------------------------------------*/
  .CompOCPSelection      = ${MC.M2_OCP_COMP_U},
  .CompOCPInvInput_MODE  = <#if MC.M2_OCP_COMP_SRC=="DAC">DAC_MODE<#else>NONE</#if>,
    </#if><#-- MC.M2_OCP_TOPOLOGY -->
    <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
    </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

/* DAC settings --------------------------------------------------------------*/
    <#if MC.M2_OCP_COMP_SRC=="DAC">
  .DAC_OCP_Threshold     = ${MC.M2_DAC_CURRENT_THRESHOLD},
    </#if><#-- MC.M2_OCP_COMP_SRC=="DAC" -->
    <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .DAC_OVP_Threshold     = ${MC.M2_OVPREF},
    </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

};  
  </#if><#-- ((MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_PHASE_SHIFT') || (MC.M2_CURRENT_SENSING_TOPO == 'SINGLE_SHUNT_ACTIVE_WIN')) -->
<#-- one shunt pahse shift********************************* -->

  <#if MC.PFC_ENABLED == true>
const PFC_Parameters_t PFC_Params = 
{
    <#if CondFamily_STM32F3>
  .ADCx                     = ADC4,
  .TIMx                     = TIM16,
  .TIMx_2                   = TIM4,
    </#if><#-- CondFamily_STM32F3 -->
  .wCPUFreq                 = SYSCLK_FREQ,
  .wPWMFreq                 = PFC_PWMFREQ,
  .hPWMARR                  = (SYSCLK_FREQ / PFC_PWMFREQ),
  .bFaultPolarity           = PFC_FAULTPOLARITY,
  .hFaultPort               = PFC_FAULTPORT,
  .hFaultPin                = PFC_FAULTPIN,
  .bCurrentLoopRate         = (uint8_t)(PFC_PWMFREQ / PFC_CURRCTRLFREQUENCY),
  .bVoltageLoopRate         = (uint8_t)(SYS_TICK_FREQUENCY / PFC_VOLTCTRLFREQUENCY),
  .hNominalCurrent          = (uint16_t)PFC_NOMINALCURRENTS16A,
  .hMainsFreqLowTh          = PFC_MAINSFREQLOWTHR,
  .hMainsFreqHiTh           = PFC_MAINSFREQHITHR,
  .OutputPowerActivation    = PFC_OUTPUTPOWERACTIVATION,
  .OutputPowerDeActivation  = PFC_OUTPUTPOWERDEACTIVATION,
  .SWOverVoltageTh          = PFC_SWOVERVOLTAGETH,
  .hPropDelayOnTimCk        = (uint16_t)(PFC_PROPDELAYON / PFC_TIMCK_NS),
  .hPropDelayOffTimCk       = (uint16_t)(PFC_PROPDELAYOFF / PFC_TIMCK_NS),
  .hADCSamplingTimeTimCk    = (uint16_t)(SYSCLK_FREQ / (ADC_CLK_MHz * 1000000.0) * PFC_ISAMPLINGTIMEREAL),
  .hADCConversionTimeTimCk  = (uint16_t)(SYSCLK_FREQ / (ADC_CLK_MHz * 1000000.0) * 12),
  .hADCLatencyTimeTimCk     = (uint16_t)(SYSCLK_FREQ / (ADC_CLK_MHz * 1000000.0) * 3),
  .hMainsConversionFactor   = (uint16_t)(65535.0 / (3.3 * PFC_VMAINS_PARTITIONING_FACTOR)),
  .hCurrentConversionFactor = (uint16_t)((PFC_SHUNTRESISTOR * PFC_AMPLGAIN * 65536.0) / 3.3),
    <#if CondFamily_STM32F3>
  .hDAC_OCP_Threshold       = (uint16_t)PFC_OCP_REF,
    </#if><#-- CondFamily_STM32F3 -->
};
  </#if><#-- MC.PFC_ENABLED == true -->

  <#if (MC.MOTOR_PROFILER == true) && ( MC.M1_SPEED_SENSOR != "HSO" && MC.M1_SPEED_SENSOR != "ZEST")>

/*** Motor Profiler ***/

SCC_Params_t SCC_Params = 
{
  {
    .FrequencyHz = TF_REGULATION_RATE,
  },
  .fRshunt                 = RSHUNT,
  .fAmplificationGain      = AMPLIFICATION_GAIN,
  .fVbusConvFactor         = BUS_VOLTAGE_CONVERSION_FACTOR,
  .fVbusPartitioningFactor = VBUS_PARTITIONING_FACTOR,

  .fRVNK                   = (float)(RESISTOR_OFFSET),

  .fRSMeasCurrLevelMax     = (float)(DC_CURRENT_RS_MEAS),

  .hDutyRampDuration       = (uint16_t)8000,
  
  .hAlignmentDuration      = (uint16_t)(1000),
  .hRSDetectionDuration    = (uint16_t)(500),
  .fLdLqRatio              = (float)(LDLQ_RATIO),
  .fCurrentBW              = (float)(CURRENT_REGULATOR_BANDWIDTH),
  .bPBCharacterization     = false,

  .wNominalSpeed           = MOTOR_MAX_SPEED_RPM,
  .hPWMFreqHz              = (uint16_t)(PWM_FREQUENCY),
  .bFOCRepRate             = (uint8_t)(REGULATION_EXECUTION_RATE),
  .fMCUPowerSupply         = (float)ADC_REFERENCE_VOLTAGE,
  .IThreshold              = I_THRESHOLD

};

OTT_Params_t OTT_Params =
{
  {
    .FrequencyHz        = MEDIUM_FREQUENCY_TASK_RATE,         /*!< Frequency expressed in Hz at which the user
                                                                  clocks the OTT calling OTT_MF method */
  },
  .fBWdef               = (float)(SPEED_REGULATOR_BANDWIDTH), /*!< Default bandwidth of speed regulator.*/
  .fMeasWin             = 1.0f,                               /*!< Duration of measurement window for speed and
                                                                  current Iq, expressed in seconds.*/
  .bPolesPairs          = POLE_PAIR_NUM,                      /*!< Number of motor poles pairs.*/
  .hMaxPositiveTorque   = (int16_t)NOMINAL_CURRENT,           /*!< Maximum positive value of motor
                                                                   torque. This value represents
                                                                   actually the maximum Iq current
                                                                   expressed in digit.*/
  .fCurrtRegStabTimeSec = 10.0f,                              /*!< Current regulation stabilization time in seconds.*/
  .fOttLowSpeedPerc     = 0.6f,                               /*!< OTT lower speed percentage.*/
  .fOttHighSpeedPerc    = 0.8f,                               /*!< OTT higher speed percentage.*/
  .fSpeedStabTimeSec    = 20.0f,                              /*!< Speed stabilization time in seconds.*/
  .fTimeOutSec          = 10.0f,                              /*!< Timeout for speed stabilization.*/
  .fSpeedMargin         = 0.90f,                              /*!< Speed margin percentage to validate speed ctrl.*/
  .wNominalSpeed        = MOTOR_MAX_SPEED_RPM,                /*!< Nominal speed set by the user expressed in RPM.*/
  .spdKp                = MP_KP,                              /*!< Initial KP factor of the speed regulator to be
                                                                   tuned.*/
  .spdKi                = MP_KI,                              /*!< Initial KI factor of the speed regulator to be 
                                                                   tuned.*/
  .spdKs                = 0.1f,                               /*!< Initial antiwindup factor of the speed regulator to
                                                                   be tuned.*/
  .fRshunt              = (float)(RSHUNT),                    /*!< Value of shunt resistor.*/
  .fAmplificationGain   = (float)(AMPLIFICATION_GAIN)         /*!< Current sensing amplification gain.*/

};
  </#if><#-- MC.MOTOR_PROFILER == true -->

  <#if MC.M1_SPEED_SENSOR == "HSO" || MC.M1_SPEED_SENSOR == "ZEST">
#if defined (__ICCARM__)
#pragma location=0x801f800
#elif defined (__CC_ARM) || defined(__GNUC__)
__attribute__((section (".params")))
#endif
const FLASH_Params_t  flashParams =
{
  .motor = {
    .polePairs = POLE_PAIR_NUM,
    .ratedFlux = MOTOR_RATED_FLUX,
    .rs = RS,
    .rsSkinFactor = MOTOR_RS_SKINFACTOR,
    .ls = LS,
    .maxCurrent = NOMINAL_CURRENT_A,
    .mass_copper_kg = MOTOR_MASS_COPPER_KG,
    .cooling_tau_s = MOTOR_COOLINGTAU_S, 
    .name = MOTOR_NAME,
  },
  .polPulse =
  {
    .N  = NB_PULSE_PERIODS,
    .Nd = NB_DECAY_PERIODS,         
  },
<#if MC.M1_SPEED_SENSOR == "ZEST">  
  .zest = 
  {
    .zestThresholdFreqHz = MOTOR_ZEST_THRESHOLD_FREQ_HZ,
	.zestInjectFreq = MOTOR_ZEST_INJECT_FREQ,
	.zestInjectD = MOTOR_ZEST_INJECT_D,
#if defined(MOTOR_ZEST_INJECT_Q)
    .zestInjectQ = MOTOR_ZEST_INJECT_Q,
#endif
    .zestGainD = MOTOR_ZEST_GAIN_D,
    .zestGainQ = MOTOR_ZEST_GAIN_Q,          
  },
</#if>   
  .PIDSpeed = 
  {
    .pidSpdKp = PID_SPD_KP,
	.pidSpdKi = PID_SPD_KI,        
  },
  .board = 
  {
   .limitOverVoltage = BOARD_LIMIT_OVERVOLTAGE,
   .limitRegenHigh = BOARD_LIMIT_REGEN_HIGH,
   .limitRegenLow = BOARD_LIMIT_REGEN_LOW,
   .limitAccelHigh = BOARD_LIMIT_ACCEL_HIGH,
   .limitAccelLow = BOARD_LIMIT_ACCEL_LOW,
   .limitUnderVoltage = BOARD_LIMIT_UNDERVOLTAGE,
   .maxModulationIndex = BOARD_MAX_MODULATION,
   .softOverCurrentTrip = BOARD_SOFT_OVERCURRENT_TRIP,
},
  .KSampleDelay = KSAMPLE_DELAY,
  .throttle = 
  {
    .offset = THROTTLE_OFFSET, /*TODO: Tobe defined */
	.gain = THROTTLE_GAIN, /*TODO: Tobe defined */
	.speedMaxRPM = THROTTLE_SPEED_MAX_RPM,
    .direction = 1,
  },
  .scale = 
  {
    .voltage = VOLTAGE_SCALE,
    .current = CURRENT_SCALE,
    .frequency = FREQUENCY_SCALE
  },
};

<#if MC.M1_SPEED_SENSOR == "ZEST">
ZEST_Params ZeST_params_M1 =
{
  .backgroundFreq_Hz = SPEED_LOOP_FREQUENCY_HZ,   /* Frequency for ZEST_runBackground() */
  .isrFreq_Hz = (PWM_FREQUENCY / REGULATION_EXECUTION_RATE), /* Frequency of ZEST_run() calls */
  .speedPole_rps = SPEED_POLE_RPS,
};
</#if>

const MotorConfig_reg_t *motorParams = &flashParams.motor;
const zestFlashParams_t *zestParams = &flashParams.zest; 
const boardFlashParams_t *boardParams = &flashParams.board;
const scaleFlashParams_t *scaleParams = &flashParams.scale;
const throttleParams_t *throttleParams = &flashParams.throttle;
const float *KSampleDelayParams = &flashParams.KSampleDelay;
const PIDSpeedFlashParams_t *PIDSpeedParams = &flashParams.PIDSpeed;

  </#if>
  
</#if><#-- FOC -->


<#if (FOC || ACIM) >
  <#if MC.M1_SPEED_SENSOR != "HSO">
ScaleParams_t scaleParams_M1 =
{
 .voltage = NOMINAL_BUS_VOLTAGE_V/(1.73205 * 32767), /* sqrt(3) = 1.73205 */
 .current = CURRENT_CONV_FACTOR_INV,
 .frequency = (1.15 * MAX_APPLICATION_SPEED_UNIT * U_RPM)/(32768* SPEED_UNIT)
};
  </#if> <#-- MC.M1_SPEED_SENSOR != "HSO" -->
  <#if MC.DRIVE_NUMBER != "1">
      <#if MC.M2_SPEED_SENSOR != "HSO">
ScaleParams_t scaleParams_M2 =
{
 .voltage = NOMINAL_BUS_VOLTAGE_V2/(1.73205 * 32767), /* sqrt(3) = 1.73205 */
 .current = CURRENT_CONV_FACTOR_INV2,
 .frequency = (1.15 * MAX_APPLICATION_SPEED_UNIT2 * U_RPM)/(32768* SPEED_UNIT)
};
    </#if> <#-- MC.M2_SPEED_SENSOR != "HSO" -->
  </#if><#-- MC.DRIVE_NUMBER > 1 -->

</#if><#-- FOC -->


<#if SIX_STEP>
  <#if MC.STSPIN32G4 == false>
    <#if configs[0].peripheralParams.get(_last_word(MC.M1_PWM_TIMER_SELECTION))??>
<#assign PWMTIM = configs[0].peripheralParams.get(_last_word(MC.M1_PWM_TIMER_SELECTION))>
    </#if><#-- configs[0].peripheralParams.get(_last_word(MC.M1_PWM_TIMER_SELECTION))?? -->
    <#if !PWMTIM??>
#error SORRY, it didn't work
    </#if><#-- !PWMTIM?? -->
  </#if><#-- MC.STSPIN32G4 == false -->
  <#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
#include "pwmc_6pwm.h"
  <#else><#--MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
#include "pwmc_3pwm.h"
  </#if>
  <#if  MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
    <#if CondFamily_STM32F0>
      <#if  MC.BEMF_OVERSAMPLING> 
#include "f0xx_bemf_ADC_OS_fdbk.h"
      <#else>
#include "f0xx_bemf_ADC_fdbk.h"
      </#if>
    </#if>
    <#if CondFamily_STM32G0>
      <#if  MC.BEMF_OVERSAMPLING> 
#include "g0xx_bemf_ADC_OS_fdbk.h"
      <#else>
#include "g0xx_bemf_ADC_fdbk.h"
      </#if>
    </#if>
    <#if CondFamily_STM32C0>
      <#if  MC.BEMF_OVERSAMPLING> 
#include "c0xx_bemf_ADC_OS_fdbk.h"
      <#else>
#include "c0xx_bemf_ADC_fdbk.h"
      </#if>
    </#if>
    <#if CondFamily_STM32G4>
#include "g4xx_bemf_ADC_fdbk.h"
    </#if><#-- CondFamily_STM32G4 -->
  </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->
  <#if MC.DRIVE_MODE == "CM">
#include "current_ref_ctrl.h"
  </#if><#-- MC.DRIVE_MODE == "CM" -->
/* USER CODE BEGIN Additional include */

/* USER CODE END Additional include */

/**
  * @brief  Current sensor parameters Motor 1 - single shunt phase shift
  */
  <#if MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER">
const SixPwm_Params_t SixPwm_ParamsM1 =
{
/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = ${MC.M1_REGULATION_EXECUTION_RATE},
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},

/* PWM Driving signals initialization ----------------------------------------*/
    <#if MC.STSPIN32G4 == false>
  .OCPolarity = LL_TIM_OCPOLARITY_${_last_word(PWMTIM.get("OCPolarity_1"))},
  .OCNPolarity = LL_TIM_OCPOLARITY_${_last_word(PWMTIM.get("OCNPolarity_1"))},
    <#else><#-- MC.STSPIN32G4 == true -->
  .OCPolarity = LL_TIM_OCPOLARITY_HIGH,
  .OCNPolarity = LL_TIM_OCPOLARITY_HIGH,
    </#if><#-- MC.STSPIN32G4 == false -->
};
  <#else><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING != "LS_PWM_TIMER" -->
const ThreePwm_Params_t ThreePwm_ParamsM1 =
{
/* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter = ${MC.M1_REGULATION_EXECUTION_RATE},
  .TIMx              = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},

/* PWM Driving signals initialization ----------------------------------------*/
  .OCPolarity        = LL_TIM_OCPOLARITY_${_last_word(PWMTIM.get("OCPolarity_1"))},
  .pwm_en_u_port     = M1_PWM_EN_U_GPIO_Port,
  .pwm_en_u_pin      = M1_PWM_EN_U_Pin,
  .pwm_en_v_port     = M1_PWM_EN_V_GPIO_Port,
  .pwm_en_v_pin      = M1_PWM_EN_V_Pin,
  .pwm_en_w_port     = M1_PWM_EN_W_GPIO_Port,
  .pwm_en_w_pin      = M1_PWM_EN_W_Pin,
};
  </#if><#-- MC.M1_LOW_SIDE_SIGNALS_ENABLING == "LS_PWM_TIMER" -->

  <#if MC.M1_SPEED_SENSOR == "SENSORLESS_ADC">
/**
  * @brief  Current sensor parameters Motor 1 - single shunt phase shift
  */
const Bemf_ADC_Params_t Bemf_ADC_ParamsM1 =
{
  .LfTim                  = ${_last_word(MC.LF_TIMER_SELECTION)},
  .LfTimerChannel         = LL_TIM_CHANNEL_CH1,        /*!< Channel of the LF timer used for speed measurement */ 
    <#if MC.BEMF_DIVIDER_AVAILABLE>
  .gpio_divider_available = true,               /*!< Availability of the GPIO port enabling the bemf resistor divider */
  .bemf_divider_port      = M1_BEMF_DIVIDER_GPIO_Port, /*!< GPIO port of OnSensing divider enabler */
  .bemf_divider_pin       = M1_BEMF_DIVIDER_Pin,
    <#else><#-- MC.BEMF_DIVIDER_AVAILABLE == false -->
  .gpio_divider_available = false,              /*!< Availability of the GPIO port enabling the bemf resistor divider */
    </#if><#-- MC.BEMF_DIVIDER_AVAILABLE -->
  /*!< Pointer to the ADC */
  .pAdc                   = {${MC.PHASE_U_BEMF_ADC}, ${MC.PHASE_V_BEMF_ADC}, ${MC.PHASE_W_BEMF_ADC}},
  .AdcChannel             = {MC_${MC.PHASE_U_BEMF_CHANNEL}, MC_${MC.PHASE_V_BEMF_CHANNEL}, MC_${MC.PHASE_W_BEMF_CHANNEL}},
    <#if  MC.BEMF_OVERSAMPLING && !CondFamily_STM32G4>
  .TIM_Trigger               = ${_last_word(MC.BEMF_TIMER_SELECTION)},
  .TIM_ADC_Trigger_Channel = LL_TIM_CHANNEL_CH1,
    </#if>  
};
  </#if><#-- MC.M1_SPEED_SENSOR == "SENSORLESS_ADC" -->

  <#if MC.DRIVE_MODE == "CM">
const CurrentRef_Params_t CurrentRef_ParamsM1 =
{
  .TIMx            = ${_last_word(MC.CURR_REF_TIMER_SELECTION)}, /*!< It contains the pointer to the timer
                                                                      used for current reference PWM generation. */
  .RefTimerChannel = LL_TIM_CHANNEL_CH1,
};
  </#if><#-- MC.DRIVE_MODE == "CM" -->
</#if><#-- SIX_STEP -->


<#if MC.ESC_ENABLE>
const ESC_Params_t ESC_ParamsM1 =
{
  .Command_TIM        = TIM2,
  .Motor_TIM          = TIM1,
  .ARMING_TIME        = 200,
  .PWM_TURNOFF_MAX    = 500,
  .TURNOFF_TIME_MAX   = 500,
  .Ton_max            = ESC_TON_MAX,               /*!<  Maximum ton value for PWM (by default is 1800 us) */
  .Ton_min            = ESC_TON_MIN,               /*!<  Minimum ton value for PWM (by default is 1080 us) */ 
  .Ton_arming         = ESC_TON_ARMING,            /*!<  Minimum value to start the arming of PWM */ 
  .delta_Ton_max      = ESC_TON_MAX - ESC_TON_MIN,
  .speed_max_valueRPM = MOTOR_MAX_SPEED_RPM,       /*!< Maximum value for speed reference from Workbench */
  .speed_min_valueRPM = 1000,                      /*!< Set the minimum value for speed reference */
  .motor              = M1,
};
</#if><#-- MC.ESC_ENABLE -->


<#if ACIM>

#define FREQ_RATIO 1                /* Dummy value for single drive */
#define FREQ_RELATION HIGHEST_FREQ  /* Dummy value for single drive */

/**
  * @brief  Current sensor parameters Motor 1 - three shunt - G4 
  */
//cstat !MISRAC2012-Rule-8.4
R3_2_Params_t R3_2_ParamsM1 =
{
/* Dual MC parameters --------------------------------------------------------*/
  .FreqRatio             = FREQ_RATIO,
  .IsHigherFreqTim       = FREQ_RELATION,

/* Current reading A/D Conversions initialization -----------------------------*/
  .ADCConfig1 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",2,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",3,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",4,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",5,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",6,"PHASE_1",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                },
  .ADCConfig2 = {
                  (uint32_t)(${getADC("CFG",1,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",2,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",3,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",4,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",5,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                , (uint32_t)(${getADC("CFG",6,"PHASE_2",1)}U << ADC_JSQR_JSQ1_Pos)
                | (LL_ADC_INJ_TRIG_EXT_${PWM_Timer_M1}_TRGO & ~ADC_INJ_TRIG_EXT_EDGE_DEFAULT)
                },
 
  .ADCDataReg1 = {
                   ${getADC("IP", 1,"PHASE_1",1)}
                 , ${getADC("IP", 2,"PHASE_1",1)}
                 , ${getADC("IP", 3,"PHASE_1",1)}
                 , ${getADC("IP", 4,"PHASE_1",1)}
                 , ${getADC("IP", 5,"PHASE_1",1)}
                 , ${getADC("IP", 6,"PHASE_1",1)}
                 },
  .ADCDataReg2 = {
                   ${getADC("IP", 1,"PHASE_2",1)}
                 , ${getADC("IP", 2,"PHASE_2",1)}
                 , ${getADC("IP", 3,"PHASE_2",1)}
                 , ${getADC("IP", 4,"PHASE_2",1)}
                 , ${getADC("IP", 5,"PHASE_2",1)}
                 , ${getADC("IP", 6,"PHASE_2",1)}
                 },
 //cstat +MISRAC2012-Rule-12.1 +MISRAC2012-Rule-10.1_R6

  /* PWM generation parameters --------------------------------------------------*/
  .RepetitionCounter     = REP_COUNTER,
  .Tafter                = TW_AFTER,
  .Tbefore               = TW_BEFORE,
  .Tsampling             = (uint16_t)SAMPLING_TIME,
  .Tcase2                = (uint16_t)SAMPLING_TIME + (uint16_t)TDEAD + (uint16_t)TRISE,
  .Tcase3                = ((uint16_t)TDEAD + (uint16_t)TNOISE + (uint16_t)SAMPLING_TIME) / 2u,
  .TIMx                  = ${_last_word(MC.M1_PWM_TIMER_SELECTION)},

/* Internal OPAMP common settings --------------------------------------------*/
  <#if MC.M1_USE_INTERNAL_OPAMP>
  .OPAMPParams           = &R3_3_OPAMPParamsM1,
  <#else><#-- MC.M1_USE_INTERNAL_OPAMP == false -->
  .OPAMPParams           = MC_NULL,
  </#if><#-- MC.M1_USE_INTERNAL_OPAMP -->

/* Internal COMP settings ----------------------------------------------------*/
  <#if MC.M1_OCP_TOPOLOGY == "EMBEDDED">
  .CompOCPASelection     = ${MC.M1_OCP_COMP_U},
  .CompOCPAInvInput_MODE = ${MC.OCPA_INVERTINGINPUT_MODE},
  .CompOCPBSelection     = ${MC.M1_OCP_COMP_V},
  .CompOCPBInvInput_MODE = ${MC.OCPB_INVERTINGINPUT_MODE},
  .CompOCPCSelection     = ${MC.M1_OCP_COMP_W},
  .CompOCPCInvInput_MODE = ${MC.OCPC_INVERTINGINPUT_MODE},
    <#if MC.M1_OCP_COMP_SRC == "DAC">
  .DAC_OCP_ASelection    = ${MC.M1_OCP_DAC_U},
  .DAC_OCP_BSelection    = ${MC.M1_OCP_DAC_V},
  .DAC_OCP_CSelection    = ${MC.M1_OCP_DAC_W},
  .DAC_Channel_OCPA      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_U, "OUT")},
  .DAC_Channel_OCPB      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_V, "OUT")},
  .DAC_Channel_OCPC      = LL_DAC_CHANNEL_${_last_word(MC.M1_OCP_DAC_CHANNEL_W, "OUT")},
    <#else><#-- MC.M1_OCP_COMP_SRC != "DAC" -->
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t)0,
  .DAC_Channel_OCPB      = (uint32_t)0,
  .DAC_Channel_OCPC      = (uint32_t)0,
    </#if><#-- MC.M1_OCP_COMP_SRC == "DAC" -->
  <#else><#--  MC.M1_OCP_TOPOLOGY != "EMBEDDED" -->
  .CompOCPASelection     = MC_NULL,
  .CompOCPAInvInput_MODE = NONE,
  .CompOCPBSelection     = MC_NULL,
  .CompOCPBInvInput_MODE = NONE,
  .CompOCPCSelection     = MC_NULL,
  .CompOCPCInvInput_MODE = NONE,
  .DAC_OCP_ASelection    = MC_NULL,
  .DAC_OCP_BSelection    = MC_NULL,
  .DAC_OCP_CSelection    = MC_NULL,
  .DAC_Channel_OCPA      = (uint32_t)0,
  .DAC_Channel_OCPB      = (uint32_t)0,
  .DAC_Channel_OCPC      = (uint32_t)0,
  </#if><#-- MC.M1_OCP_TOPOLOGY == "EMBEDDED" -->

  <#if MC.INTERNAL_OVERVOLTAGEPROTECTION>
  .CompOVPSelection      = OVP_SELECTION,
  .CompOVPInvInput_MODE  = OVP_INVERTINGINPUT_MODE,
    <#if MC.DAC_OVP_M1 != "NOT_USED">
  .DAC_OVP_Selection     = ${_first_word(MC.DAC_OVP_M1)},
  .DAC_Channel_OVP       = LL_DAC_CHANNEL_${_last_word(MC.DAC_OVP_M1,"CH")},
    <#else><#-- MC.DAC_OVP_M1 == "NOT_USED" -->
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t)0,
    </#if><#-- MC.DAC_OVP_M1 != "NOT_USED" -->
  <#else><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION == false -->
  .CompOVPSelection      = MC_NULL,
  .CompOVPInvInput_MODE  = NONE,
  .DAC_OVP_Selection     = MC_NULL,
  .DAC_Channel_OVP       = (uint32_t)0,
  </#if><#-- MC.INTERNAL_OVERVOLTAGEPROTECTION -->

/* DAC settings --------------------------------------------------------------*/
  .DAC_OCP_Threshold     = ${MC.M1_DAC_CURRENT_THRESHOLD},
  .DAC_OVP_Threshold     = ${MC.M1_OVPREF},

};
</#if><#-- ACIM --><#-- Inside CondFamily_STM32G4 -->





/* USER CODE BEGIN Additional parameters */


/* USER CODE END Additional parameters */  

/******************* (C) COPYRIGHT 2023 STMicroelectronics *****END OF FILE****/

