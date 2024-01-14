/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sun Jan 14 16:23:07 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "f2c7b94f7e2c2f4a5fbcd95ed24b7d73"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Sun Jan 14 16:23:07 2024"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  obs_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _model_running_mean_std_Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _model_running_mean_std_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _model_running_mean_std_Clip_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_0_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_0_Elu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 131, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_1_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_1_Elu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_Concat_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 67, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_2_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_2_Elu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  value_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  mu_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  log_std_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  model__model_a2c_network_sigma_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_Constant_output_0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  onnxDiv_43_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  onnxSub_40_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_0_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_0_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_1_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8384, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_1_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_2_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2144, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_2_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  value_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  value_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  mu_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  mu_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  obs_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &obs_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _model_running_mean_std_Sub_output_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_running_mean_std_Sub_output_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _model_running_mean_std_Div_output_0_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_running_mean_std_Div_output_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _model_running_mean_std_Clip_output_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_model_running_mean_std_Clip_output_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_0_Gemm_output_0_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_a2c_network_actor_mlp_0_Gemm_output_0_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_0_Elu_output_0_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_a2c_network_actor_mlp_activations_0_Elu_output_0_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_Concat_output_0_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 131, 1, 1), AI_STRIDE_INIT(4, 4, 4, 524, 524),
  1, &_model_a2c_network_actor_mlp_Concat_output_0_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_1_Gemm_output_0_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_a2c_network_actor_mlp_1_Gemm_output_0_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_1_Elu_output_0_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_a2c_network_actor_mlp_activations_1_Elu_output_0_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_Concat_1_output_0_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 67, 1, 1), AI_STRIDE_INIT(4, 4, 4, 268, 268),
  1, &_model_a2c_network_actor_mlp_Concat_1_output_0_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_2_Gemm_output_0_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_a2c_network_actor_mlp_2_Gemm_output_0_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_2_Elu_output_0_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_a2c_network_actor_mlp_activations_2_Elu_output_0_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  value_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &value_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  mu_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &mu_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_Mul_output_0_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_a2c_network_Mul_output_0_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  log_std_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &log_std_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  model__model_a2c_network_sigma, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &model__model_a2c_network_sigma_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_Constant_output_0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_model_a2c_network_Constant_output_0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  onnxDiv_43, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &onnxDiv_43_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  onnxSub_40, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &onnxSub_40_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_0_Gemm_output_0_weights, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 3, 128, 1, 1), AI_STRIDE_INIT(4, 4, 12, 1536, 1536),
  1, &_model_a2c_network_actor_mlp_0_Gemm_output_0_weights_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_0_Gemm_output_0_bias, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_model_a2c_network_actor_mlp_0_Gemm_output_0_bias_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_1_Gemm_output_0_weights, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 131, 64, 1, 1), AI_STRIDE_INIT(4, 4, 524, 33536, 33536),
  1, &_model_a2c_network_actor_mlp_1_Gemm_output_0_weights_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_1_Gemm_output_0_bias, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_a2c_network_actor_mlp_1_Gemm_output_0_bias_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_2_Gemm_output_0_weights, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 67, 32, 1, 1), AI_STRIDE_INIT(4, 4, 268, 8576, 8576),
  1, &_model_a2c_network_actor_mlp_2_Gemm_output_0_weights_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_2_Gemm_output_0_bias, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_a2c_network_actor_mlp_2_Gemm_output_0_bias_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  value_weights, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 1), AI_STRIDE_INIT(4, 4, 128, 128, 128),
  1, &value_weights_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  value_bias, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &value_bias_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  mu_weights, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 1), AI_STRIDE_INIT(4, 4, 128, 128, 128),
  1, &mu_weights_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  mu_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &mu_bias_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  log_std_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_a2c_network_Mul_output_0_output, &model__model_a2c_network_sigma),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &log_std_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  log_std_layer, 19,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &log_std_chain,
  NULL, &log_std_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mu_output, &_model_a2c_network_Constant_output_0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_Mul_output_0_layer, 18,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_a2c_network_Mul_output_0_chain,
  NULL, &log_std_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_activations_2_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mu_weights, &mu_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mu_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &mu_chain,
  NULL, &_model_a2c_network_Mul_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  value_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_activations_2_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &value_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &value_weights, &value_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  value_layer, 15,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &value_chain,
  NULL, &mu_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _model_a2c_network_actor_mlp_activations_2_Elu_output_0_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _model_a2c_network_actor_mlp_activations_2_Elu_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _model_a2c_network_actor_mlp_activations_2_Elu_output_0_nl_params_data, _model_a2c_network_actor_mlp_activations_2_Elu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_2_Elu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_activations_2_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_2_Elu_output_0_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_elu,
  &_model_a2c_network_actor_mlp_activations_2_Elu_output_0_chain,
  NULL, &value_layer, AI_STATIC, 
  .nl_params = &_model_a2c_network_actor_mlp_activations_2_Elu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_2_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_Concat_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_a2c_network_actor_mlp_2_Gemm_output_0_weights, &_model_a2c_network_actor_mlp_2_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_2_Gemm_output_0_layer, 13,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_model_a2c_network_actor_mlp_2_Gemm_output_0_chain,
  NULL, &_model_a2c_network_actor_mlp_activations_2_Elu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_Concat_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_a2c_network_actor_mlp_activations_1_Elu_output_0_output, &_model_running_mean_std_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_Concat_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_Concat_1_output_0_layer, 12,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_a2c_network_actor_mlp_Concat_1_output_0_chain,
  NULL, &_model_a2c_network_actor_mlp_2_Gemm_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float _model_a2c_network_actor_mlp_activations_1_Elu_output_0_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _model_a2c_network_actor_mlp_activations_1_Elu_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _model_a2c_network_actor_mlp_activations_1_Elu_output_0_nl_params_data, _model_a2c_network_actor_mlp_activations_1_Elu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_1_Elu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_1_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_activations_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_1_Elu_output_0_layer, 11,
  NL_TYPE, 0x0, NULL,
  nl, forward_elu,
  &_model_a2c_network_actor_mlp_activations_1_Elu_output_0_chain,
  NULL, &_model_a2c_network_actor_mlp_Concat_1_output_0_layer, AI_STATIC, 
  .nl_params = &_model_a2c_network_actor_mlp_activations_1_Elu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_1_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_1_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_a2c_network_actor_mlp_1_Gemm_output_0_weights, &_model_a2c_network_actor_mlp_1_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_1_Gemm_output_0_layer, 10,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_model_a2c_network_actor_mlp_1_Gemm_output_0_chain,
  NULL, &_model_a2c_network_actor_mlp_activations_1_Elu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_a2c_network_actor_mlp_activations_0_Elu_output_0_output, &_model_running_mean_std_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_Concat_output_0_layer, 9,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_model_a2c_network_actor_mlp_Concat_output_0_chain,
  NULL, &_model_a2c_network_actor_mlp_1_Gemm_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float _model_a2c_network_actor_mlp_activations_0_Elu_output_0_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _model_a2c_network_actor_mlp_activations_0_Elu_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _model_a2c_network_actor_mlp_activations_0_Elu_output_0_nl_params_data, _model_a2c_network_actor_mlp_activations_0_Elu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_0_Elu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_activations_0_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_activations_0_Elu_output_0_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_elu,
  &_model_a2c_network_actor_mlp_activations_0_Elu_output_0_chain,
  NULL, &_model_a2c_network_actor_mlp_Concat_output_0_layer, AI_STATIC, 
  .nl_params = &_model_a2c_network_actor_mlp_activations_0_Elu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_0_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_running_mean_std_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_a2c_network_actor_mlp_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_a2c_network_actor_mlp_0_Gemm_output_0_weights, &_model_a2c_network_actor_mlp_0_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_a2c_network_actor_mlp_0_Gemm_output_0_layer, 7,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_model_a2c_network_actor_mlp_0_Gemm_output_0_chain,
  NULL, &_model_a2c_network_actor_mlp_activations_0_Elu_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _model_running_mean_std_Clip_output_0_nl_params_data[] = { -5.0, 5.0 };
AI_ARRAY_OBJ_DECLARE(
    _model_running_mean_std_Clip_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _model_running_mean_std_Clip_output_0_nl_params_data, _model_running_mean_std_Clip_output_0_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_running_mean_std_Clip_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_running_mean_std_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_running_mean_std_Clip_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_running_mean_std_Clip_output_0_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &_model_running_mean_std_Clip_output_0_chain,
  NULL, &_model_a2c_network_actor_mlp_0_Gemm_output_0_layer, AI_STATIC, 
  .nl_params = &_model_running_mean_std_Clip_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_running_mean_std_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_running_mean_std_Sub_output_0_output, &onnxDiv_43),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_running_mean_std_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_running_mean_std_Div_output_0_layer, 2,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_running_mean_std_Div_output_0_chain,
  NULL, &_model_running_mean_std_Clip_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_running_mean_std_Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &obs_output, &onnxSub_40),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_running_mean_std_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_running_mean_std_Sub_output_0_layer, 1,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_model_running_mean_std_Sub_output_0_chain,
  NULL, &_model_running_mean_std_Div_output_0_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 44840, 1, 1),
    44840, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1048, 1, 1),
    1048, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &obs_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &mu_output, &log_std_output, &value_output),
  &_model_running_mean_std_Sub_output_0_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 44840, 1, 1),
      44840, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1048, 1, 1),
      1048, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &obs_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &mu_output, &log_std_output, &value_output),
  &_model_running_mean_std_Sub_output_0_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    obs_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    obs_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    _model_running_mean_std_Sub_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _model_running_mean_std_Sub_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    _model_running_mean_std_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _model_running_mean_std_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    _model_running_mean_std_Clip_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _model_running_mean_std_Clip_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    _model_a2c_network_actor_mlp_0_Gemm_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 12);
    _model_a2c_network_actor_mlp_0_Gemm_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12);
    
    _model_a2c_network_actor_mlp_activations_0_Elu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 12);
    _model_a2c_network_actor_mlp_activations_0_Elu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12);
    
    _model_a2c_network_actor_mlp_Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 524);
    _model_a2c_network_actor_mlp_Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 524);
    
    _model_a2c_network_actor_mlp_1_Gemm_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 12);
    _model_a2c_network_actor_mlp_1_Gemm_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12);
    
    _model_a2c_network_actor_mlp_activations_1_Elu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 268);
    _model_a2c_network_actor_mlp_activations_1_Elu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 268);
    
    _model_a2c_network_actor_mlp_Concat_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 524);
    _model_a2c_network_actor_mlp_Concat_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 524);
    
    _model_a2c_network_actor_mlp_2_Gemm_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _model_a2c_network_actor_mlp_2_Gemm_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    _model_a2c_network_actor_mlp_activations_2_Elu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 128);
    _model_a2c_network_actor_mlp_activations_2_Elu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 128);
    
    value_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    value_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    mu_output_array.data = AI_PTR(g_network_activations_map[0] + 4);
    mu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4);
    
    _model_a2c_network_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8);
    _model_a2c_network_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8);
    
    log_std_output_array.data = AI_PTR(g_network_activations_map[0] + 12);
    log_std_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    model__model_a2c_network_sigma_array.format |= AI_FMT_FLAG_CONST;
    model__model_a2c_network_sigma_array.data = AI_PTR(g_network_weights_map[0] + 0);
    model__model_a2c_network_sigma_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    
    _model_a2c_network_Constant_output_0_array.format |= AI_FMT_FLAG_CONST;
    _model_a2c_network_Constant_output_0_array.data = AI_PTR(g_network_weights_map[0] + 4);
    _model_a2c_network_Constant_output_0_array.data_start = AI_PTR(g_network_weights_map[0] + 4);
    
    onnxDiv_43_array.format |= AI_FMT_FLAG_CONST;
    onnxDiv_43_array.data = AI_PTR(g_network_weights_map[0] + 8);
    onnxDiv_43_array.data_start = AI_PTR(g_network_weights_map[0] + 8);
    
    onnxSub_40_array.format |= AI_FMT_FLAG_CONST;
    onnxSub_40_array.data = AI_PTR(g_network_weights_map[0] + 20);
    onnxSub_40_array.data_start = AI_PTR(g_network_weights_map[0] + 20);
    
    _model_a2c_network_actor_mlp_0_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_a2c_network_actor_mlp_0_Gemm_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 32);
    _model_a2c_network_actor_mlp_0_Gemm_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 32);
    
    _model_a2c_network_actor_mlp_0_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_a2c_network_actor_mlp_0_Gemm_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1568);
    _model_a2c_network_actor_mlp_0_Gemm_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1568);
    
    _model_a2c_network_actor_mlp_1_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_a2c_network_actor_mlp_1_Gemm_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 2080);
    _model_a2c_network_actor_mlp_1_Gemm_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2080);
    
    _model_a2c_network_actor_mlp_1_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_a2c_network_actor_mlp_1_Gemm_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 35616);
    _model_a2c_network_actor_mlp_1_Gemm_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 35616);
    
    _model_a2c_network_actor_mlp_2_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_a2c_network_actor_mlp_2_Gemm_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 35872);
    _model_a2c_network_actor_mlp_2_Gemm_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 35872);
    
    _model_a2c_network_actor_mlp_2_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_a2c_network_actor_mlp_2_Gemm_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 44448);
    _model_a2c_network_actor_mlp_2_Gemm_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 44448);
    
    value_weights_array.format |= AI_FMT_FLAG_CONST;
    value_weights_array.data = AI_PTR(g_network_weights_map[0] + 44576);
    value_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 44576);
    
    value_bias_array.format |= AI_FMT_FLAG_CONST;
    value_bias_array.data = AI_PTR(g_network_weights_map[0] + 44704);
    value_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 44704);
    
    mu_weights_array.format |= AI_FMT_FLAG_CONST;
    mu_weights_array.data = AI_PTR(g_network_weights_map[0] + 44708);
    mu_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 44708);
    
    mu_bias_array.format |= AI_FMT_FLAG_CONST;
    mu_bias_array.data = AI_PTR(g_network_weights_map[0] + 44836);
    mu_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 44836);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 13692,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 13692,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_network_data_params_get(&params) != true) {
        err = ai_network_get_error(*network);
        return err;
    }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_network_init(*network, &params) != true) {
        err = ai_network_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

