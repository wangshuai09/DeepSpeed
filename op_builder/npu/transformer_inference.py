# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Remove me, only add for test 
import sys
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import torch 

# del . for test
from .builder import NPUOpBuilder

try:
    import torch_npu
except ImportError as e:
    pass


def swap_two_rows(x):
    # [..., [x1, x2, x3, x4, ...]] --> [..., [-x2, x1, -x4, x3, ...]]
    x1 = x[..., ::2].clone()
    x2 = x[..., 1::2]
    
    x[..., ::2] = -x2
    x[..., 1::2] = x1
    return x


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class InferenceContext:
    _workspace = None

    _seed = 42
    _curr_offset = 0
    _stream = 0
    _free_memory_size = 0
    _num_tokens = 1
    _attention_unfused_workspace_offset = 0
    _workSpaceSize = 0 
    _workSpaceSize = 0
    _workspace = 0

    workSpaceSize = 0

    kv_caches = None

    @staticmethod
    def reset_tokens(initial_tokens=1):
        InferenceContext._num_tokens = initial_tokens

    @staticmethod
    def current_tokens():
        return InferenceContext._num_tokens
    
    @staticmethod
    def GetWorkSpace():
        return InferenceContext._workspace
    
    @staticmethod
    def GetMaxTokenLength():
        return InferenceContext._max_seq_len
    
    @staticmethod
    def retake_workspace():
        InferenceContext.kv_cache = None
        return True
    

class NPUInference():

    # def _qkv_gemm(cls, input, weight, q_scale, bias, gamma, beta, epsilon,
    #                   add_bias, q_int8, transposed_mode):
    #     inp_norm = torch.nn.functional.layer_norm(input, (input.shape[2],),
    #                                               gamma, beta, epsilon)
    #     tmp = torch.matmul(inp_norm, weight.t())
    #     if add_bias:
    #         tmp += bias
        
    #     output = [tmp, inp_norm]
    #     return output
    @staticmethod
    def _qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        inp_norm = torch.nn.functional.layer_norm(inputs, (inputs.shape[2],), gamma, beta, eps)
        tmp = torch.matmul(inp_norm, weight.t())
        if add_bias:
            tmp += bias
        output = [tmp, inp_norm]
        return output

    @staticmethod
    def qkv_gemm_fp32(input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode):
        
        return NPUInference._qkv_gemm(input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode)
    
    @staticmethod
    def qkv_gemm_fp16(input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode):
        return NPUInference._qkv_gemm(input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode)

    @staticmethod
    def qkv_gemm_bf16(input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode):
        return NPUInference._qkv_gemm(input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode)
    
    @classmethod
    def _bias_add_transform_0213(cls, vals, bias, 
                                 hidden_dim, seq_length, seq_offset, heads,
                                 num_kv, # num_kv > 0 ? num_kv : heads,
                                 rotary_dim,
                                 rotate_half,
                                 rotate_every_two,
                                 rope_theta):
        # vals: [bsz, seq_len, 3*heads*d]
        bsz, _, _ = vals.shape
        q = vals[..., :hidden_dim].reshape(bsz, seq_length, heads, -1)
        k = vals[..., hidden_dim: hidden_dim + num_kv * (hidden_dim // heads)].reshape(bsz, seq_length, num_kv, -1)
        v = vals[..., hidden_dim + num_kv * (hidden_dim // heads):]
        print("q", q.shape, "k", k.shape, "v", v.shape)

        # rope 位置编码, npu 
        if rotary_dim > 0 and rotate_every_two:
            # sin, cos may use cache
            seq_id = torch.arange(0, seq_length).to("npu")
            inv_freq = torch.arange(0, rotary_dim , 2) / rotary_dim
            inv_freq = inv_freq.to("npu")
            inv_freq = 1.0 / torch.pow(rope_theta, inv_freq)
            inv_freq = torch.outer(seq_id, inv_freq)
            sin = inv_freq.sin()
            cos = inv_freq.cos()
            # shape: [bsz=1, seq_len, heads=1, rotary_dim], 相邻两行相同
            sin = sin.view(-1, seq_length, 1, rotary_dim//2).repeat_interleave(2, dim=-1)
            cos = cos.view(-1, seq_length, 1, rotary_dim//2).repeat_interleave(2, dim=-1)

            # 只在 rotary_dim 范围内计算
            q_pos, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_pos, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        
            q_pos = q_pos * cos + swap_two_rows(q_pos) * sin
            q = torch.cat([q_pos, q_pass], dim=-1)
            k_pos = k_pos * cos + swap_two_rows(k_pos) * sin
            k = torch.cat([k_pos, k_pass], dim=-1)
    
        # return 
        # ouput: [bsz, seq_len, heads*d]
        # k_cache: [bsz, seq_len, heads*d]
        # v_cache: [bsz, seq_len, heads*d]
        output = q.reshape(bsz, seq_length, -1).contiguous()
        k_cache= k.reshape(bsz, seq_length, -1).contiguous()
        v_cache = v.contiguous()
       
        print("result:", output.shape, k_cache.shape, v_cache.shape)
        return output, k_cache, v_cache
        
    @staticmethod
    def _softmax_context(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta):
        bsz, seq_len, k = query_key_value.size()
        k = k // (heads + 2 * (num_kv if num_kv > 0 else heads))
        hidden_dim = heads * k 
       
        is_promt = seq_len > 1

        if not InferenceContext.kv_caches:
            InferenceContext.kv_caches = [[None, None] for _ in range(num_layers)]
        if is_promt:
            InferenceContext.reset_tokens(seq_len)
            InferenceContext.kv_caches[layer_id] = [None, None]

        soft_len = InferenceContext.current_tokens()  
        workspace = InferenceContext.GetWorkSpace()  
        seq_offset = 0 if is_promt else soft_len - 1
        
        # 
        print("soft_len", soft_len, "layer_id", layer_id, "num_layers", num_layers)
        print("alibi", alibi)

        q, k, v = NPUInference._bias_add_transform_0213(vals=query_key_value, 
                                              bias=None, 
                                              hidden_dim=hidden_dim, 
                                              seq_length=seq_len, 
                                              seq_offset=seq_offset, 
                                              heads=heads,
                                              num_kv=num_kv if num_kv > 0 else heads, # num_kv > 0 ? num_kv : heads,
                                              rotary_dim=rotary_dim,
                                              rotate_half=rotate_half,
                                              rotate_every_two=rotate_every_two,
                                              rope_theta=rope_theta)

        print("qkv shape after bias_add_transform_0213:", q.shape, k.shape, v.shape)
        torch.npu.synchronize()

        if not is_promt:
            k_cache, v_cache = InferenceContext.kv_caches[layer_id]
            if k_cache is not None:
                k = torch.cat([k_cache, k], dim=1)
                v = torch.cat([v_cache, v], dim=1)
            print("k,v", [k.shape, v.shape]) 
        InferenceContext.kv_caches[layer_id] = [k, v]
        pre_seq_len = k.shape[1]

        q = q.reshape(bsz, seq_len, heads, -1).transpose(1, 2).reshape(bsz*heads, seq_len, -1).contiguous()  # [b * n, s, d]
        k = k.reshape(bsz, seq_len, (num_kv if num_kv > 0 else heads), -1).transpose(1, 2).reshape(bsz*(num_kv if num_kv > 0 else heads), pre_seq_len, -1).contiguous()  # [b * n, s, d]
        v = v.reshape(bsz, seq_len, (num_kv if num_kv > 0 else heads), -1).transpose(1, 2).reshape(bsz*(num_kv if num_kv > 0 else heads), pre_seq_len, -1).contiguous()  # [b * n, s, d]
        
        output, attn_weight_reshaped = NPUInference.replace_npu_fusion_attention(q, k, v, norm_factor, attn_mask, heads, bsz, pre_seq_len)

        # [b * n, s, d] --> [b, n, s, d] --> [b, s, n, d]
        k_layer = k.reshape(bsz, heads, pre_seq_len, -1).contiguous()
        v_layer = v.reshape(bsz, heads, pre_seq_len, -1).contiguous()

        return output, k_layer, v_layer

    @staticmethod
    def replace_npu_fusion_attention(query_layer, key_layer, value_layer, norm_factor, 
                                     attn_mask, heads, bsz, seq_len):
        """
            query_layer: [bsz*heads, seq_len, -1]
            key_layer,value_layer: [bsz*heads, soft_len, -1]
        """
        # [bsz*heads, seq_len, soft_len]
        attn_score = torch.bmm(query_layer, key_layer.transpose(1, 2))
        attn_score *= norm_factor * norm_factor
        print('attn_score', attn_score.shape, attn_mask.shape)
        if attn_mask is not None:
            attn_score = attn_score.view(bsz, heads, seq_len, -1).contiguous()
            attn_score = attn_score + attn_mask
            attn_score = torch.max(attn_score, torch.tensor(torch.finfo(attn_score.dtype).min, device=attn_score.device))
            attn_score = attn_score.view(bsz*heads, seq_len, -1).contiguous()
        print('attn_score', attn_score.shape)

        dtype = attn_score.dtype
        attn_score = torch.nn.Softmax(dim=-1)(attn_score.float()).to(dtype)
        attn_weight_reshaped = attn_score.view(bsz, heads, seq_len, -1).contiguous()

        # [b * n, s, soft_len] * [b * n, soft_len, d] --> [b * n, s, d]
        print(attn_score.shape, value_layer.shape)
        out = torch.bmm(attn_score, value_layer)

        # [b * n, s, d] --> [b, n, s, d] --> [b, s, n, d] --> [b, s, H]
        output = out.reshape(bsz, heads, seq_len, -1).permute(0, 2, 1, 3).reshape(bsz, seq_len, -1).contiguous()
        
        return output, attn_weight_reshaped
    
    @staticmethod
    def softmax_context_fp32(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta):
        
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta)
    
    @staticmethod
    def softmax_context_fp16(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta)

    @staticmethod
    def softmax_context_bf16(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta)

    @staticmethod
    def _vector_matmul_(input,
                        weight,
                        transposed_mode):
        return torch.matmul(input, weight.t())
        # if transposed_mode:
        #     output = torch.matmul(input, weight.t())
        # else:
        #     output = torch.matmul(input, weight)
        # print("vector matmul", output.shape)
        # return output 
        
    
    @staticmethod
    def vector_matmul_fp32(input,
                            weight,
                            async_op,
                            q_scale,
                            q_int8,
                            transposed_mode):
        return NPUInference._vector_matmul_(input,
                        weight,
                        transposed_mode)
                                
    @staticmethod
    def vector_matmul_fp16(input,
                            weight,
                            async_op,
                            q_scale,
                            q_int8,
                            transposed_mode):
        return NPUInference._vector_matmul_(input,
                        weight,
                        transposed_mode)
    
    @staticmethod
    def vector_matmul_bf16(input,
                            weight,
                            async_op,
                            q_scale,
                            q_int8,
                            transposed_mode):
        return NPUInference._vector_matmul_(input,
                        weight,
                        transposed_mode)
    
    @staticmethod
    def _mlp_gemm(input,
                   residual,
                   input_bias,
                   weight_interm,
                   weight_out,
                   bias,
                   gamma,
                   beta,
                   epsilon,
                   preLayerNorm,
                   mlp_after_attn,
                   q_scale,
                   q_scale1,
                   q_int8,
                   activation_type,
                   transposed_mode):
        # print(type(input), type(residual), type(input_bias))
        # if mlp_after_attn:
        #     residual_add = torch.nn.functional.layer_norm(input + residual + input_bias, (input.shape[2], ), gamma, beta, epsilon)
        # else:
        #     residual_add = torch.nn.functional.layer_norm(input, (input.shape[2], ), gamma, beta,
        #                                 epsilon)
        # tmp = torch.matmul(residual_add, weight_interm.t())
        # from deepspeed.utils.types import ActivationFuncType
        # if activation_type == ActivationFuncType.GELU:
        #     tmp = torch.nn.functional.gelu(tmp + bias)
        # elif activation_type == ActivationFuncType.ReLU:
        #     tmp = torch.nn.functional.relu(tmp + bias)
        # output = torch.matmul(tmp, weight_out.t())
        # return output, residual_add    
        residual_add = torch.nn.functional.layer_norm(input + residual + input_bias, (input.shape[2],), gamma, beta,
                                                      epsilon)
        tmp = torch.matmul(residual_add, weight_interm.t())
        tmp = torch.nn.functional.gelu(tmp + bias)
        output = torch.matmul(tmp, weight_out.t())
        return output, residual_add
    
    @staticmethod
    def mlp_gemm_fp32(input,
                   residual,
                   input_bias,
                   weight_interm,
                   weight_out,
                   bias,
                   gamma,
                   beta,
                   epsilon,
                   preLayerNorm,
                   mlp_after_attn,
                   q_scale,
                   q_scale1,
                   q_int8,
                   activation_type,
                   transposed_mode):
        return NPUInference._mlp_gemm(input,
                   residual,
                   input_bias,
                   weight_interm,
                   weight_out,
                   bias,
                   gamma,
                   beta,
                   epsilon,
                   preLayerNorm,
                   mlp_after_attn,
                   q_scale,
                   q_scale1,
                   q_int8,
                   activation_type,
                   transposed_mode)

    @staticmethod
    def mlp_gemm_fp16(input,
                   residual,
                   input_bias,
                   weight_interm,
                   weight_out,
                   bias,
                   gamma,
                   beta,
                   epsilon,
                   preLayerNorm,
                   mlp_after_attn,
                   q_scale,
                   q_scale1,
                   q_int8,
                   activation_type,
                   transposed_mode):
        return NPUInference._mlp_gemm(input,
                   residual,
                   input_bias,
                   weight_interm,
                   weight_out,
                   bias,
                   gamma,
                   beta,
                   epsilon,
                   preLayerNorm,
                   mlp_after_attn,
                   q_scale,
                   q_scale1,
                   q_int8,
                   activation_type,
                   transposed_mode)

    @staticmethod
    def mlp_gemm_bf16(input,
                   residual,
                   input_bias,
                   weight_interm,
                   weight_out,
                   bias,
                   gamma,
                   beta,
                   epsilon,
                   preLayerNorm,
                   mlp_after_attn,
                   q_scale,
                   q_scale1,
                   q_int8,
                   activation_type,
                   transposed_mode):
        return NPUInference._mlp_gemm(input,
                   residual,
                   input_bias,
                   weight_interm,
                   weight_out,
                   bias,
                   gamma,
                   beta,
                   epsilon,
                   preLayerNorm,
                   mlp_after_attn,
                   q_scale,
                   q_scale1,
                   q_int8,
                   activation_type,
                   transposed_mode)

    @staticmethod
    def _residual_add_bias(hidden_state,
                           residual,
                           attention_output,
                           attention_bias,
                           final_bias,
                           mp_size,
                           mlp_after_attn,
                           add_bias,
                           preln):
        # if mlp_after_attn:
        #     if preln:
        #         # residual = (residual + attention + bias + attention_bias) *
        #         # mp_scale + hidden_state
        #         tmp = (residual + attention_output + attention_bias + final_bias) / mp_size + hidden_state
        #     else:
        #         # residual += hidden_state + bias
        #         tmp = residual + hidden_state + final_bias
        # else:
        #     if add_bias:
        #         residual += attention_bias   
        #     tmp = hidden_state + attention_output + (residual + final_bias) * mp_size

        # input_dtype = hidden_state.dtype
        # residual = tmp.to(input_dtype)
        # print("residual output", residual.shape)
        # return residual
        if preln:
            tmp = (residual.float() + attention_output.float() + attention_bias.float() +
                   final_bias.float()) / mp_size + hidden_state.float()
        else:
            tmp = residual.float() + hidden_state.float() + final_bias.float()

        input_dtype = hidden_state.dtype
        residual = tmp.to(input_dtype)
        return residual

    @staticmethod
    def residual_add_bias_fp32(hidden_state,
                           residual,
                           attention_output,
                           attention_bias,
                           final_bias,
                           mp_size,
                           mlp_after_attn,
                           add_bias,
                           preln):
        return NPUInference._residual_add_bias(hidden_state,
                           residual,
                           attention_output,
                           attention_bias,
                           final_bias,
                           mp_size,
                           mlp_after_attn,
                           add_bias,
                           preln)

    @staticmethod
    def residual_add_bias_fp16(hidden_state,
                           residual,
                           attention_output,
                           attention_bias,
                           final_bias,
                           mp_size,
                           mlp_after_attn,
                           add_bias,
                           preln):
        return NPUInference._residual_add_bias(hidden_state,
                           residual,
                           attention_output,
                           attention_bias,
                           final_bias,
                           mp_size,
                           mlp_after_attn,
                           add_bias,
                           preln)

    @staticmethod
    def residual_add_bias_bf16(hidden_state,
                           residual,
                           attention_output,
                           attention_bias,
                           final_bias,
                           mp_size,
                           mlp_after_attn,
                           add_bias,
                           preln):
        return NPUInference._residual_add_bias(hidden_state,
                           residual,
                           attention_output,
                           attention_bias,
                           final_bias,
                           mp_size,
                           mlp_after_attn,
                           add_bias,
                           preln)

class InferenceBuilder(NPUOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def sources(self):
        return []

    def include_paths(self):
        return []
    
    def load(self):
        return NPUInference


def test_bias_add_transform_0213():

    BATCH = 4
    SEQ_LENGTH = 128
    HEADS = 32
    HEAD_DIM = 256
    NUM_KV = HEADS//2
    rotary_dim = 100
    rotate_every_two=True
    rope_theta = 0.1

    output = torch.empty((BATCH, SEQ_LENGTH, HEAD_DIM * HEADS), dtype=torch.float16, device="npu").normal_(mean=0, std=.5)
    k_cache = torch.empty((BATCH, SEQ_LENGTH, HEAD_DIM * NUM_KV), dtype=torch.float16, device="npu").normal_(mean=0, std=.5) 
    v_cache = torch.empty((BATCH, SEQ_LENGTH, HEAD_DIM * NUM_KV), dtype=torch.float16, device="npu").normal_(mean=0, std=.5)

    vals = torch.empty((BATCH, SEQ_LENGTH, HEAD_DIM * HEADS + 2 * NUM_KV * HEAD_DIM), dtype=torch.float16, device="npu").normal_(mean=0, std=.5)

    result = NPUInference._bias_add_transform_0213(output, k_cache, v_cache, vals, None, 
                                                    hidden_dim=HEAD_DIM*HEADS, 
                                                    seq_length=SEQ_LENGTH, 
                                                    seq_offset=0, 
                                                    heads=HEADS,
                                                    num_kv=NUM_KV, # num_kv > 0 ? num_kv : heads,
                                                    rotary_dim=rotary_dim,
                                                    rotate_half=None,
                                                    rotate_every_two=rotate_every_two,
                                                    rope_theta=rope_theta)

def test_softmax_context():
    BATCH = 4
    SEQ_LENGTH = 128
    HEADS = 32
    HEAD_DIM = 256
    NUM_KV = -1
    rotary_dim = -1
    rotate_every_two=True
    rope_theta = 0.1

    query_key_value = torch.empty((BATCH, SEQ_LENGTH, HEAD_DIM * HEADS + 2 * NUM_KV * HEAD_DIM), dtype=torch.float16, device="npu").normal_(mean=0, std=.5)
    attn_mask = None 
    rotate_half = None 
    norm_factor = None 
    triangular = None 
    local_attention = None 
    window_size = None 
    no_masking = None 
    layer_id = None 
    num_layers = None 
    alibi = None 

    result = NPUInference._softmax_context(query_key_value, attn_mask, 
                                           rotary_dim=rotary_dim, 
                                           rotate_half=rotate_half, 
                                           rotate_every_two=rotate_every_two, 
                                           heads=HEADS, 
                                           num_kv=NUM_KV, 
                                           norm_factor=norm_factor, 
                                           triangular=triangular, 
                                           local_attention=local_attention, 
                                           window_size=window_size,
                                           no_masking=no_masking, 
                                           layer_id=layer_id, 
                                           num_layers=num_layers, 
                                           alibi=alibi, 
                                           rope_theta=rope_theta)

if __name__ == '__main__':
    test_bias_add_transform_0213()
    test_softmax_context()