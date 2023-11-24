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

    kv_cache = None

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

    @staticmethod
    def softmax_context_int8(query, prev_key, new_key, attn_mask, prev_value, 
                             new_value, heads, norm_factor, merging, triangular, 
                             local_attention, window_size, no_masking):
        pass
                              
    @staticmethod
    def gated_activation(activation, bias, actFun):
        pass
    
    @staticmethod
    def layer_norm(input, gamma, beta, epsilon):
        norm, _, _ = torch.native_layer_norm(input, [input.shape[-1]], gamma,
                                             beta, eps=epsilon)
        return norm
    
    @staticmethod
    def _layer_norm_residual(input, bias, residual, gamma, beta, epsilon):
        pass
    
    @staticmethod
    def layer_norm_residual_store_pre_ln_res(input, bias, residual, gamma, beta,
                                              epsilon):
        pass
    
    @staticmethod
    def layer_norm_residual_store_pre_ln_res(input, bias, residual, gamma, beta, 
                                             epsilon):
        pass
    
    @staticmethod
    def rms_norm(input, gamma, epsilon):
        pass

    @staticmethod
    def pre_rms_norm(input, residual, gamma, epsilon):
        pass

    @staticmethod
    def _vector_add(a, b, gamma):
        pass

    @staticmethod
    def apply_rotary_pos_emb(mixed_query, key_layer, rotary_dim,  offset,
                             num_heads, rotate_half, rope_theta):
        pass
    
    @staticmethod
    def moe_res_matmul(moe_res, coef, output):
        pass

    @staticmethod
    def reset_cache():
        pass

    @staticmethod
    def release_workspace():
        pass
    
    @staticmethod
    def retake_workspace():
        pass
    
    @classmethod
    def _softmax(cls, attn_scores, attn_mask, alibi, triangular, recompute,
                 local_attention, window_size, async_op, layer_scale,
                 head_offset, mp_size):
        pass

    @staticmethod
    def softmax_fp32(cls, attn_scores, attn_mask, alibi, triangular, recompute,
                 local_attention, window_size, async_op, layer_scale,
                 head_offset, mp_size):
        return NPUInference._softmax(cls, attn_scores, attn_mask, alibi, triangular, recompute,
                 local_attention, window_size, async_op, layer_scale,
                 head_offset, mp_size)

    @staticmethod
    def softmax_fp16():
        pass
    
    @staticmethod
    def softmax_bf16():
        pass

    @classmethod
    def _qkv_gemm(cls, input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode):
        inp_norm = torch.nn.functional.layer_norm(input, (input.shape[2],),
                                                  gamma, beta, epsilon)
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
    def _bias_add_transform_0213(cls, output, k_cache, v_cache, vals, bias, 
                                 hidden_dim, seq_length, seq_offset, heads,
                                 num_kv, # num_kv > 0 ? num_kv : heads,
                                 rotary_dim,
                                 rotate_half,
                                 rotate_every_two,
                                 rope_theta):
        # q,k,v
        # q shape: [bsz, seq, heads, head_dim]
        # k shape: [bsz, seq, num_kv, head_dim]
        # v shape: [bsz, seq, num_kv * head_dim]
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
    
        # 结果，v 不变, shape: [bsz, heads, seq, head_dim]
        output = q.reshape(bsz, seq_length, -1).contiguous()
        k_cache= k.reshape(bsz, seq_length, -1).contiguous()
        v_cache = v.contiguous()
        # output = q.permute(0,2,1,3).contiguous()
        # k_cache= k.permute(0,2,1,3).contiguous()
        # v_cache = v.permute(0,2,1,3).contiguous()
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
        if is_promt:
            InferenceContext.reset_tokens(seq_len)
        
        soft_len = InferenceContext.current_tokens()  
        workspace = InferenceContext.GetWorkSpace()  
        seq_offset = 0 if is_promt else soft_len - 1
        
        # 
        print("soft_len", soft_len, "layer_id", layer_id, "num_layers", num_layers)

        output = torch.empty((bsz, seq_len, heads * k), dtype=torch.float16, device="npu")
        k_cache = torch.empty((bsz, seq_len, (num_kv if num_kv > 0 else heads) * k), dtype=torch.float16, device="npu")
        v_cache = torch.empty((bsz, seq_len, (num_kv if num_kv > 0 else heads) * k), dtype=torch.float16, device="npu")
        q, k, v = NPUInference._bias_add_transform_0213(output=output, 
                                              k_cache=k_cache, 
                                              v_cache=v_cache, 
                                              vals=query_key_value, 
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
        
        try:
            if not is_promt: # use_cache
                if not InferenceContext.kv_cache:
                    InferenceContext.kv_cache = [[None, None] for _ in range(num_layers)]
                print(InferenceContext.kv_cache[layer_id])
                k_cache, v_cache = InferenceContext.kv_cache[layer_id]
                if k_cache:
                    k = torch.cat((k_cache, k), 1)
                    v = torch.cat((v_cache, v), 1)
                print("k,v", [k.shape, v.shape], InferenceContext.kv_cache[layer_id]) 
                InferenceContext.kv_cache[layer_id] = [k, v]
            else:
                InferenceContext.kv_cache = None
        except:
            import traceback
            print("traceback", traceback.format_exc())

        print(q.shape, k.shape, v.shape, heads, attn_mask.shape)
                    
        output = torch_npu.npu_fusion_attention(
            q, k, v, heads, "BSH",
            pse=None,
            padding_mask=None,
            atten_mask=attn_mask.bool(),
            scale=norm_factor,
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1,
            inner_precise=0
        )[0]

        # [b, s, H] --> [b, s, h, k] --> [b, h, s, k]
        with torch.no_grad():
            k = k.view(k.size(0), k.size(1), heads, -1).transpose(1, 2).contiguous()
            v = v.view(v.size(0), v.size(1), heads, -1).transpose(1, 2).contiguous()
        
        return output, k, v
    
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
        if transposed_mode:
            output = torch.matmul(input, weight.t())
        else:
            output = torch.matmul(input, weight)
        print("vector matmul", output.shape)
        return output 
        
    
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
        print(type(input), type(residual), type(input_bias))
        if mlp_after_attn:
            residual_add = torch.nn.functional.layer_norm(input + residual + input_bias, (input.shape[2], ), gamma, beta, epsilon)
        else:
            residual_add = torch.nn.functional.layer_norm(input, (input.shape[2], ), gamma, beta,
                                        epsilon)
        tmp = torch.matmul(residual_add, weight_interm.t())
        from deepspeed.utils.types import ActivationFuncType
        if activation_type == ActivationFuncType.GELU:
            tmp = torch.nn.functional.gelu(tmp + bias)
        elif activation_type == ActivationFuncType.ReLU:
            tmp = torch.nn.functional.relu(tmp + bias)
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
        if mlp_after_attn:
            if preln:
                # residual = (residual + attention + bias + attention_bias) *
                # mp_scale + hidden_state
                tmp = (residual + attention_output + attention_bias + final_bias) / mp_size + hidden_state
            else:
                # residual += hidden_state + bias
                tmp = residual + hidden_state + final_bias
        else:
            if add_bias:
                residual += attention_bias   
            tmp = hidden_state + attention_output + (residual + final_bias) * mp_size

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