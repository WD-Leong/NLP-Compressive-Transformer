import tensorflow as tf

def split_heads(x, num_heads):
    batch_size = tf.shape(x)[0]
    input_len  = tf.shape(x)[1]
    depth_size = tf.cast(
        tf.shape(x)[2] / num_heads, tf.int32)
    
    split_outputs = tf.reshape(
        x, [batch_size, input_len, num_heads, depth_size])
    return tf.transpose(split_outputs, [0, 2, 1, 3])

def combine_heads(x):
    batch_size = tf.shape(x)[0]
    input_len  = tf.shape(x)[2]
    num_heads  = tf.shape(x)[1]
    depth_size = tf.shape(x)[3]
    hidden_size = num_heads*depth_size
    
    combined_outputs = tf.reshape(tf.transpose(
        x, [0, 2, 1, 3]), [batch_size, input_len, hidden_size])
    return combined_outputs

def layer_normalisation(x, bias, scale, eps=1.0e-6):
    x_mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    x_var  = tf.reduce_mean(
        tf.square(x - x_mean), axis=[-1], keepdims=True)
    x_isig = tf.math.rsqrt(x_var + tf.constant(eps))
    x_norm = (x - x_mean) * x_isig
    return (x_norm * scale) + bias

class GPT_Compressive(tf.keras.Model):
    def __init__(
    self, n_layers, n_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, embed_size=128, 
    p_keep=0.9, p_reg=1.0, var_type="norm_add", **kwargs):
        super(GPT_Compressive, self).__init__(**kwargs)
        self.p_keep = p_keep
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.var_type = var_type
        
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.ffwd_size = ffwd_size
        self.head_size = int(hidden_size / n_heads)
        self.hidden_size = hidden_size
        
        # Embedding matrices. #
        emb_shape = [self.vocab_size, self.embed_size]
        lin_shape = [self.embed_size, self.hidden_size]
        
        self.W_dec_lin = tf.Variable(tf.random.normal(
            lin_shape, stddev=0.1), name="dec_linear")
        self.W_emb_dec = tf.Variable(tf.random.normal(
            emb_shape, stddev=0.1), name="dec_embedding")
        
        # Output projection. #
        logits_shape = [self.hidden_size, self.vocab_size]
        self.p_decoder = tf.Variable(tf.random.normal(
            logits_shape, stddev=0.1), name="p_decoder")
        
        # GPT Variables. #
        pos_seq_shape  = [
            self.n_layers, self.seq_length, self.hidden_size]
        norm_param_shp = [self.n_layers, self.hidden_size]
        attn_wgt_shape = [
            self.n_layers, self.hidden_size, self.hidden_size]
        attn_comb_shape = [
            self.n_layers, self.hidden_size, self.hidden_size]
        attn_ffw1_shape = [
            self.n_layers, self.hidden_size, self.ffwd_size]
        attn_ffw2_shape = [
            self.n_layers, self.ffwd_size, self.hidden_size]
        
        self.p_d_q = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_q")
        self.p_d_k = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_k")
        self.p_d_v = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_v")
        self.p_d_c = tf.Variable(tf.random.normal(
            attn_comb_shape, stddev=0.1), name="p_d_c")
        
        self.p_d_ff1 = tf.Variable(tf.random.normal(
            attn_ffw1_shape, stddev=0.1), name="p_d_ff1")
        self.p_d_ff2 = tf.Variable(tf.random.normal(
            attn_ffw2_shape, stddev=0.1), name="p_d_ff2")
        self.b_d_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_d_ff1")
        self.b_d_ff2 = tf.Variable(tf.zeros([
            self.n_layers, self.hidden_size]), name="b_d_ff2")
        
        self.d_o_bias  = tf.Variable(tf.zeros(
            [self.hidden_size]), name="d_o_bias")
        self.d_o_scale = tf.Variable(tf.ones(
            [self.hidden_size]), name="d_o_scale")
        
        self.b_d_bias_1  = tf.Variable(
            tf.zeros(norm_param_shp), name="b_d_bias_1")
        self.b_d_bias_2  = tf.Variable(
            tf.zeros(norm_param_shp), name="b_d_bias_2")
        self.b_d_scale_1 = tf.Variable(
            tf.ones(norm_param_shp), name="b_d_scale_1")
        self.b_d_scale_2 = tf.Variable(
            tf.ones(norm_param_shp), name="b_d_scale_2")
        
        self.b_i_bias  = tf.Variable(
            tf.zeros(norm_param_shp), name="b_i_bias")
        self.b_i_scale = tf.Variable(
            tf.ones(norm_param_shp), name="b_i_scale")
        
        # Position Embeddings. #
        self.x_emb_pos_dec = tf.Variable(tf.random.normal(
            pos_seq_shape, stddev=0.1), name="pos_seq_embed")
    
    def gpt_decode(
        self, step, curr_state, dec_inputs, training=False):
        head_size = tf.cast(self.head_size, tf.float32)
        
        p_reg = 1.0
        n_heads = self.n_heads
        if training:
            p_keep = self.p_keep
        else:
            p_keep = 1.0
        
        neg_infty = -1.0e9
        ones_mask = tf.linalg.band_part(
            tf.ones([step, step]), -1, 0)
        attn_mask = neg_infty * (1.0 - ones_mask)
        attn_mask = tf.expand_dims(attn_mask, axis=0)
        
        avg_embed  = tf.reduce_mean(
            dec_inputs, axis=1, keepdims=True)
        next_state = [tf.expand_dims(avg_embed, axis=0)]
        
        layer_input = dec_inputs
        for m in range(self.n_layers):
            tmp_pos = tf.add(
                curr_state[m], 
                self.x_emb_pos_dec[m, :step, :])
            
            tmp_i_bias  = self.b_i_bias[m]
            tmp_i_scale = self.b_i_scale[m]
            
            if self.var_type == "norm_add":
                layer_in =  tf.add(
                    tmp_pos, layer_normalisation(
                        layer_input, tmp_i_bias, tmp_i_scale))
            elif self.var_type == "add_norm":
                layer_in = layer_normalisation(tf.add(
                    tmp_pos, layer_input), tmp_i_bias, tmp_i_scale)
            
            # Self Attention Layer. #
            x_sq = split_heads(tf.tensordot(
                layer_in, self.p_d_q[m], [[2], [0]]), n_heads)
            x_sq = x_sq * tf.math.rsqrt(head_size)
            
            x_sk = split_heads(tf.tensordot(
                layer_in, self.p_d_k[m], [[2], [0]]), n_heads)
            x_sv = split_heads(tf.tensordot(
                layer_in, self.p_d_v[m], [[2], [0]]), n_heads)
            
            x_s_scores = tf.matmul(
                x_sq, x_sk, transpose_b=True)
            x_s_alphas = tf.nn.softmax(
                x_s_scores + attn_mask)
            x_self_mha = tf.matmul(x_s_alphas, x_sv)
            
            x_multi_self = tf.tensordot(combine_heads(
                x_self_mha), self.p_d_c[m], [[2], [0]])
            x_multi_self = tf.nn.dropout(
                x_multi_self, rate=1.0-p_reg)
            
            tmp_bias1  = self.b_d_bias_1[m]
            tmp_scale1 = self.b_d_scale_1[m]
            if self.var_type == "norm_add":
                x_self_norm = tf.add(
                    layer_in, layer_normalisation(
                        x_multi_self, tmp_bias1, tmp_scale1))
            elif self.var_type == "add_norm":
                x_self_norm = layer_normalisation(tf.add(
                    layer_in, x_multi_self), tmp_bias1, tmp_scale1)
            
            # Feed forward layer. #
            x_self_ffw1 = tf.nn.relu(tf.add(
                self.b_d_ff1[m], tf.tensordot(
                    x_self_norm, self.p_d_ff1[m], [[2], [0]])))
            x_self_ffw2 = tf.add(
                self.b_d_ff2[m], tf.tensordot(
                    x_self_ffw1, self.p_d_ff2[m], [[2], [0]]))
            x_self_ffwd = tf.nn.dropout(x_self_ffw2, rate=1.0-p_keep)
            
            tmp_bias2  = self.b_d_bias_2[m]
            tmp_scale2 = self.b_d_scale_2[m]
            if self.var_type == "norm_add":
                x_self_out = tf.add(
                    x_self_norm, layer_normalisation(
                        x_self_ffwd, tmp_bias2, tmp_scale2))
            elif self.var_type == "add_norm":
                x_self_out = layer_normalisation(tf.add(
                    x_self_norm, x_self_ffwd), tmp_bias2, tmp_scale2)
            
            # Set the input to the next layer. #
            layer_input  = x_self_out
            avg_self_out = tf.reduce_mean(
                x_self_out, axis=1, keepdims=True)
            next_state.append(
                tf.expand_dims(avg_self_out, axis=0))
        
        # Set the decoder output. #
        curr_output = x_self_out
        
        next_state = tf.concat(next_state, axis=0)
        next_state = next_state[:self.n_layers, :, :, :]
        
        # Residual Connection. #
        tmp_o_bias  = self.d_o_bias
        tmp_o_scale = self.d_o_scale
        if self.var_type == "norm_add":
            dec_outputs = tf.add(
                dec_inputs, layer_normalisation(
                    curr_output, tmp_o_bias, tmp_o_scale))
        elif self.var_type == "add_norm":
            dec_res_out = tf.add(dec_inputs, curr_output)
            dec_outputs = layer_normalisation(
                dec_res_out, tmp_o_bias, tmp_o_scale)
        return dec_outputs, next_state
    
    def call(self, x_input, curr_state=None, training=False):
        batch_size = tf.shape(x_input)[0]
        if curr_state is None:
            curr_state = tf.zeros(
                [self.n_layers, batch_size, 
                 1, self.hidden_size], dtype=tf.float32)
        
        # Character or Word or Sub-word embeddings. #
        x_dec_token = tf.nn.embedding_lookup(
            self.W_emb_dec, x_input)
        x_dec_embed = tf.tensordot(
            x_dec_token, self.W_dec_lin, [[2], [0]])
        
        # Transformer Decoder for current block. #
        out_tuple = self.gpt_decode(
            self.seq_length, curr_state, 
            x_dec_embed, training=training)
        
        dec_logits = tf.tensordot(
            out_tuple[0], self.p_decoder, [[2], [0]])
        dec_state  = out_tuple[1]
        return dec_logits, dec_state
    
    def infer(self, x_infer, dec_steps):
        # Inference. #
        batch_sz  = tf.shape(x_infer)[0]
        infer_len = tf.shape(x_infer)[1]
        
        infer_ids = [
            tf.expand_dims(x_infer[:, 0], axis=1)]
        curr_state = tf.zeros(
            [self.n_layers, batch_sz, 1, self.hidden_size])
        
        for step in range(dec_steps):
            local_step = step % self.seq_length
            
            n_blk  = int(step / self.seq_length)
            idx_st = n_blk * self.seq_length
            idx_en = step + 1
            
            x_infer_ids = tf.concat(infer_ids, axis=1)
            x_blk_input = x_infer_ids[:, idx_st:idx_en]
            x_tok_input = tf.nn.embedding_lookup(
                self.W_emb_dec, x_blk_input)
            
            x_emb_input = tf.tensordot(
                x_tok_input, self.W_dec_lin, [[2], [0]])
            
            output_tuple = self.gpt_decode(
                local_step+1, curr_state, 
                x_emb_input, training=False)
            tmp_outputs  = output_tuple[0]
            
            if (step+1) % self.seq_length == 0:
                curr_state = output_tuple[1]
            
            tmp_logit  = tf.matmul(
                tmp_outputs[:, -1, :], self.p_decoder)
            tmp_argmax = tf.cond(
                step < (infer_len-1), 
                lambda: x_infer[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_argmax, axis=1))
        
        infer_indices = tf.concat(infer_ids, axis=1)
        return infer_indices
