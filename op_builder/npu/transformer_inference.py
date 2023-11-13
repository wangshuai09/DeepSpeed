# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch 
from .builder import NPUOpBuilder

try:
    import torch_npu
except ImportError as e:
    pass


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
    

# def bias_add_transform_0213(float* output,
#                                         float* k_cache,
#                                         float* v_cache,
#                                         const float* vals,
#                                         const float* bias,
#                                         int hidden_dim,
#                                         int seq_length,
#                                         unsigned seq_offset,
#                                         int heads,
#                                         int head_stride,
#                                         int num_kv,
#                                         int rotary_dim,
#                                         bool rotate_half,
#                                         bool rotate_every_two,
#                                         int head_ext,
#                                         int max_out_tokens,
#                                         float rope_theta):
    
    
    



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
    def softmax_bp16():
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
    def qkv_gemm_bp16(input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode):
        return NPUInference._qkv_gemm(input, weight, q_scale, bias, gamma, beta, epsilon,
                      add_bias, q_int8, transposed_mode)

    def launch_bias_add_transform_0213(float* output,
                                           float* k_cache,
                                           float* v_cache,
                                           const float* vals,
                                           const float* bias,
                                           int batch_size,
                                           int seq_length,
                                           unsigned seq_offset,
                                           int all_tokens,
                                           int hidden_dim,
                                           int heads,
                                           int num_kv,
                                           int rotary_dim,
                                           bool rotate_half,
                                           bool rotate_every_two,
                                           cudaStream_t stream,
                                           int trans_count,
                                           int max_out_tokens,
                                           float rope_theta):
        #MAX_HTHREADS = 2048
        ## hidden_dim /= 4
        #head_ext = int(hidden_dim - 1) / MAX_HTHREADS + 1
        def _bias_add_transform_0213(output, k_cache, v_cache, vals, bias, 
                                    hidden_dim, seq_length, seq_offset, heads,
                                    num_kv > 0 ? (heads / num_kv) : 1,
                                    num_kv > 0 ? num_kv : heads,
                                    rotary_dim >> 2,
                                    rotate_half,
                                    rotate_every_two,
                                    head_ext,
                                    max_out_tokens,
                                    rope_theta):

            bsz, _, _ = vals.shape
            # q,k,v 的值, hidden_dim 待修改
            # q shape: [bsz, seq, heads, head_dim]
            q = vals[..., :hidden_dim].reshape([bsz, seq_len, heads, hidden_dim/heads])
            k = vals[..., hidden_dim: 2 * hidden_dim].reshape([bsz, seq_len, heads, hidden_dim/heads])
            v = vals[..., 2 * hidden_dim:].reshape([bsz, seq_len, heads, hidden_dim/heads])
            
            # q，k 计算 rope 位置编码
            # hidden_dim = 3 * heads * head_dim
            seq_id = torch.arange(0, seq_length)
            inv_freq = torch.arange(0, rotary_dim , 2) / rotary_dim << 2

            inv_freq = 1.0 / torch.pow(rope_theta, inv_freq)
            inv_freq = torch.outer(seq_id, inv_freq)
            sin = inv_freq.sin() 
            cos = inv_freq.cos()
            # shape: [bsz=1, seq_len, heads=1, rotary_dim//2]
            sin = sin.view(-1, seq_length, 1, rotary_dim//2)
            cos = cos.view(-1, seq_length, 1, rotary_dim//2)

            if rotary_dim > 0 and rotate_every_two:
                q, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
                k, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
                
                qa = q[..., ::2] 
                qb = -q[..., 1::2]
                ka = k[..., ::2]
                kb = -k[..., 1::2]

                q[..., ::2] = qb * sin + qa * cos
                q[..., 1::2] = qa * sin - qb * cos

                k[..., ::2] = kb * sin + ka * cos
                k[..., 1::2] = ka * sin - kb * cos
                
            # 得到结果，v 不变
            output[..., :rotary_dim] = q
            output[..., rotary_dim:] = q_pass
            
            k_cache[..., :rotary_dim] = q
            k_cache[..., rotary_dim:] = q_pass

            v_cache[..., :rotary_dim] = v      
            return output, k_cache, v_cache 

    @classmethod
    def _softmax_context(cls, query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta):
        print("!!!", query_key_value.size())
        bsz, seq_len, k = query_key_value.size()
        k = k / (heads + 2 * (num_kv if num_kv > 0 else heads))
        hidden_dim = heads * k 

        is_promt = seq_len > 1
        if is_promt:
            InferenceContext.reset_tokens(seq_len)
        
        soft_len = InferenceContext.current_tokens()  
        workspace = InferenceContext.GetWorkSpace()  

        buf_size = bsz * seq_len * hidden_dim 
        
        #
        #auto output = torch::from_blob(workspace + 4 * buf_size, {bsz, seq_len, hidden_dim}, options);
        query_cont = workspace + 5 * buf_size
        offset = 10 * (hidden_dim * bsz * InferenceContext.GetMaxTokenLength()) 
        +  layer_id * 2 * bsz * InferenceContext.GetMaxTokenLength() * hidden_dim

        all_tokens = soft_len
        kv_cache = workspace + offset + (hidden_dim / heads) * (0 if is_promt 
                                                                else soft_len - 
                                                                1)

        value_offset = bsz * InferenceContext.GetMaxTokenLength() * hidden_dim

        # temp_buf = (T*)output.data_ptr() + at::numel(output); #output 的元素个数

        # launch_bias_add_transform_0213((T*)query_cont,
        #                               kv_cache,
        #                               kv_cache + value_offset,
        #                               (T*)query_key_value.data_ptr(),
        #                               nullptr,
        #                               bsz,
        #                               seq_len,
        #                               (is_prompt ? 0 : soft_len - 1),
        #                               soft_len,
        #                               hidden_dim,
        #                               heads,
        #                               (num_kv > 0 ? num_kv : heads),
        #                               rotary_dim,
        #                               rotate_half,
        #                               rotate_every_two,
        #                               InferenceContext::Instance().GetCurrentStream(),
        #                               3,
        #                               InferenceContext::Instance().GetMaxTokenLength(),
        #                               rope_theta);
    
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
    def softmax_context_bp32(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, 
                         rotate_half, rotate_every_two, heads, num_kv, 
                         norm_factor, triangular, local_attention, window_size,
                         no_masking, layer_id, num_layers, alibi, rope_theta)


        
        
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
