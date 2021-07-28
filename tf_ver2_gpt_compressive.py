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
    x_std  = tf.math.sqrt(x_var + tf.constant(eps))
    x_norm = (x - x_mean) / x_std
    return (x_norm * scale) + bias

class GPT_Compressive(tf.keras.Model):
    def __init__(
    self, n_layers, n_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    ns, nm, ncm, c_factor=3, embed_size=128, 
    p_keep=0.9, p_reg=1.0, var_type="norm_add", **kwargs):
        super(GPT_Compressive, self).__init__(**kwargs)
        self.ns  = ns
        self.nm  = nm
        self.ncm = ncm
        self.nfc = int(ns / c_factor)
        
        self.n_mem  = nm + ncm
        self.p_keep = p_keep
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.var_type  = var_type
        self.c_factor  = c_factor
        
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.ffwd_size = ffwd_size
        self.head_size = int(hidden_size / n_heads)
        self.hidden_size = hidden_size
        
        if seq_length % self.ns == 0:
            self.n_blocks = int(seq_length / self.ns)
        else:
            self.n_blocks = int(seq_length / self.ns) + 1
        
        # Embedding matrices. #
        emb_shape = [self.vocab_size, self.embed_size]
        lin_shape = [self.embed_size, self.hidden_size]
        
        self.W_dec_lin = tf.Variable(tf.random.normal(
            lin_shape, stddev=0.1), name="dec_linear")
        self.W_emb_dec = tf.Variable(tf.random.normal(
            emb_shape, stddev=0.1), name="dec_embedding")
        
        # Compression projection. #
        compress_shape = [
            self.n_layers, self.hidden_size, self.nfc]
        c_linear_shape = [
            self.n_layers, self.hidden_size, self.hidden_size]
        self.p_compress = tf.Variable(tf.random.normal(
            compress_shape, stddev=0.1), name="p_compress")
        self.p_c_linear = tf.Variable(tf.random.normal(
            c_linear_shape, stddev=0.1), name="p_c_linear")
        
        # Output projection. #
        logits_shape = [self.hidden_size, self.vocab_size]
        self.p_decoder = tf.Variable(tf.random.normal(
            logits_shape, stddev=0.1), name="p_decoder")
        
        # GPT Variables. #
        pos_mem_shape = [
            self.n_blocks, self.n_layers, 
            1, self.n_mem, self.hidden_size]
        pos_seq_shape = [
            self.n_layers, self.ns, self.hidden_size]
        
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
        
        self.p_m_q = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_m_q")
        self.p_m_k = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_m_k")
        self.p_m_v = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_m_v")
        self.p_m_c = tf.Variable(tf.random.normal(
            attn_comb_shape, stddev=0.1), name="p_m_c")
        
        self.p_d_ff1 = tf.Variable(tf.random.normal(
            attn_ffw1_shape, stddev=0.1), name="p_d_ff1")
        self.p_d_ff2 = tf.Variable(tf.random.normal(
            attn_ffw2_shape, stddev=0.1), name="p_d_ff2")
        self.b_d_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_d_ff1")
        self.b_d_ff2 = tf.Variable(tf.zeros([
            self.n_layers, self.hidden_size]), name="b_d_ff2")
        
        self.p_m_ff1 = tf.Variable(tf.random.normal(
            attn_ffw1_shape, stddev=0.1), name="p_m_ff1")
        self.p_m_ff2 = tf.Variable(tf.random.normal(
            attn_ffw2_shape, stddev=0.1), name="p_m_ff2")
        self.b_m_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_m_ff1")
        self.b_m_ff2 = tf.Variable(tf.zeros([
            self.n_layers, self.hidden_size]), name="b_m_ff2")
        
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
        
        self.b_m_bias_1  = tf.Variable(
            tf.zeros(norm_param_shp), name="b_m_bias_1")
        self.b_m_bias_2  = tf.Variable(
            tf.zeros(norm_param_shp), name="b_m_bias_2")
        self.b_m_scale_1 = tf.Variable(
            tf.ones(norm_param_shp), name="b_m_scale_1")
        self.b_m_scale_2 = tf.Variable(
            tf.ones(norm_param_shp), name="b_m_scale_2")
        
        # Position Embeddings. #
        self.x_emb_pos_dec = tf.Variable(tf.random.normal(
            pos_seq_shape, stddev=0.1), name="pos_seq_embed")
        self.x_emb_memory  = tf.Variable(tf.random.normal(
            pos_mem_shape, stddev=0.1), name="pos_mem_embed")
    
    def transformer_decode(
        self, step, dec_inputs, mem_inputs, training=False):
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
        
        layer_input  = dec_inputs
        list_outputs = []
        for m in range(self.n_layers):
            tmp_pos = self.x_emb_pos_dec[m, :step, :]
            tmp_i_bias  = self.b_i_bias[m]
            tmp_i_scale = self.b_i_scale[m]
            
            if self.var_type == "norm_add":
                layer_in = tf.add(
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
            x_s_alphas = \
                tf.nn.softmax(x_s_scores + attn_mask)
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
            
            # Encoder-Decoder Layer. #
            x_mq = split_heads(tf.tensordot(
                x_self_out, self.p_m_q[m], [[2], [0]]), n_heads)
            x_mq = x_mq * tf.math.rsqrt(head_size)
            
            x_mk = split_heads(tf.tensordot(
                mem_inputs[m], self.p_m_k[m], [[2], [0]]), n_heads)
            x_mv = split_heads(tf.tensordot(
                mem_inputs[m], self.p_m_v[m], [[2], [0]]), n_heads)
            
            x_m_scores = tf.matmul(
                x_mq, x_mk, transpose_b=True)
            x_m_alphas = tf.nn.softmax(x_m_scores)
            x_mem_mha  = tf.matmul(x_m_alphas, x_mv)
            
            x_multi_mem = tf.tensordot(combine_heads(
                x_mem_mha), self.p_m_c[m], [[2], [0]])
            x_multi_mem = tf.nn.dropout(
                x_multi_mem, rate=1.0-p_reg)
            
            tmp_bias1  = self.b_m_bias_1[m]
            tmp_scale1 = self.b_m_scale_1[m]
            if self.var_type == "norm_add":
                x_mem_norm = tf.add(
                    layer_in, layer_normalisation(
                        x_multi_mem, tmp_bias1, tmp_scale1))
            elif self.var_type == "add_norm":
                x_mem_norm = layer_normalisation(tf.add(
                    layer_in, x_multi_mem), tmp_bias1, tmp_scale1)
            
            # Feed forward layer. #
            x_mem_ffw1 = tf.nn.relu(tf.add(
                self.b_m_ff1[m], tf.tensordot(
                    x_mem_norm, self.p_m_ff1[m], [[2], [0]])))
            x_mem_ffw2 = tf.add(
                self.b_m_ff2[m], tf.tensordot(
                    x_mem_ffw1, self.p_m_ff2[m], [[2], [0]]))
            x_mem_ffwd = tf.nn.dropout(x_mem_ffw2, rate=1.0-p_keep)
            
            tmp_bias2  = self.b_m_bias_2[m]
            tmp_scale2 = self.b_m_scale_2[m]
            if self.var_type == "norm_add":
                x_mem_out = tf.add(
                    x_mem_norm, layer_normalisation(
                        x_mem_ffwd, tmp_bias2, tmp_scale2))
            elif self.var_type == "add_norm":
                x_mem_out = layer_normalisation(tf.add(
                    x_mem_norm, x_mem_ffwd), tmp_bias2, tmp_scale2)
            
            # Set the input to the next layer. #
            layer_input = x_mem_out
            list_outputs.append(
                tf.expand_dims(x_mem_out, axis=0))
        
        # Concatenate the outputs of all layers. #
        layer_outputs = tf.concat(list_outputs, axis=0)
        del list_outputs
        
        # Residual Connection. #
        if self.var_type == "norm_add":
            dec_outputs = tf.add(
                dec_inputs, layer_normalisation(
                    x_mem_out, self.d_o_bias, self.d_o_scale))
        elif self.var_type == "add_norm":
            dec_outputs = layer_normalisation(
                tf.add(dec_inputs, x_mem_out), 
                self.d_o_bias, self.d_o_scale)
        return dec_outputs, layer_outputs
    
    def call(self, x_input, training=False):
        batch_size = tf.shape(x_input)[0]
        
        curr_memory = tf.zeros(
            [self.n_layers, batch_size, self.nm, self.hidden_size])
        curr_compress = tf.zeros(
            [self.n_layers, batch_size, self.ncm, self.hidden_size])
        
        dec_logits = []
        for n_block in range(self.n_blocks):
            idx_st = n_block * self.ns
            if n_block == (self.n_blocks-1):
                idx_en = self.seq_length
            else:
                idx_en = (n_block+1) * self.ns
            seq_step = idx_en - idx_st
            
            # Concatenate the memory inputs. #
            x_mem_inputs = tf.add(
                self.x_emb_memory[n_block], tf.concat([
                    curr_compress, curr_memory], axis=2))
            
            # Word or Sub-word embeddings. #
            x_blk_input = x_input[:, idx_st:idx_en]
            x_dec_token = tf.nn.embedding_lookup(
                self.W_emb_dec, x_blk_input)
            x_dec_embed = tf.tensordot(
                x_dec_token, self.W_dec_lin, [[2], [0]])
            
            # Transformer Decoder for current block. #
            tmp_out_tuple = self.transformer_decode(
                seq_step, x_dec_embed, 
                x_mem_inputs, training=training)
            
            blk_logits = tf.tensordot(
                tmp_out_tuple[0], self.p_decoder, [[2], [0]])
            dec_logits.append(blk_logits)
            
            # Update the memory inputs. #
            next_memory = []
            next_compress = []
            for m in range(self.n_layers):
                old_memory = curr_memory[m, :, :self.ns, :]
                tmp_compress = tf.tensordot(
                    old_memory, self.p_compress[m], [[2], [0]])
                tmp_compress = tf.transpose(tmp_compress, [0, 2, 1])
                x_old_memory = tf.tensordot(
                    old_memory, self.p_c_linear[m], [[2], [0]])
                
                prev_compress  = curr_compress[m]
                x_new_compress = tf.matmul(
                    tf.nn.softmax(tmp_compress), x_old_memory)
                
                conc_compress = tf.concat(
                    [prev_compress, x_new_compress], axis=1)
                conc_compress = conc_compress[:, -self.ncm:, :]
                
                prev_memory = curr_memory[m]
                conc_memory = tf.concat(
                    [prev_memory, tmp_out_tuple[1][m]], axis=1)
                conc_memory = conc_memory[:, -self.nm:, :]
                
                next_memory.append(
                    tf.expand_dims(conc_memory, axis=0))
                next_compress.append(
                    tf.expand_dims(conc_compress, axis=0))
            
            curr_memory = tf.concat(next_memory, axis=0)
            curr_compress = tf.concat(next_compress, axis=0)
            del next_memory, next_compress
        
        dec_logits = tf.concat(dec_logits, axis=1)
        return dec_logits
    
    def compute_loss(self, x_input, x_output):
        batch_size = tf.shape(x_input)[0]
        
        curr_memory = tf.zeros(
            [self.n_layers, batch_size, self.nm, self.hidden_size])
        curr_compress = tf.zeros(
            [self.n_layers, batch_size, self.ncm, self.hidden_size])
        
        acc_losses = 0.0
        for n_block in range(self.n_blocks):
            idx_st = n_block * self.ns
            if n_block == (self.n_blocks-1):
                idx_en = self.seq_length
            else:
                idx_en = (n_block+1) * self.ns
            seq_step = idx_en - idx_st
            
            # Get the block outputs. #
            x_blk_output = x_output[:, idx_st:idx_en]
            
            # Concatenate the memory inputs. #
            x_mem_inputs = tf.add(
                self.x_emb_memory[n_block], tf.concat([
                    curr_compress, curr_memory], axis=2))
            
            # Word or Sub-word embeddings. #
            x_blk_input = x_input[:, idx_st:idx_en]
            x_dec_token = tf.nn.embedding_lookup(
                self.W_emb_dec, x_blk_input)
            x_dec_embed = tf.tensordot(
                x_dec_token, self.W_dec_lin, [[2], [0]])
            
            # Transformer Decoder for current block. #
            tmp_out_tuple = self.transformer_decode(
                seq_step, x_dec_embed, 
                x_mem_inputs, training=True)
            
            # Compute the loss for the current block. #
            blk_logits = tf.tensordot(
                tmp_out_tuple[0], self.p_decoder, [[2], [0]])
            acc_losses += tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=x_blk_output, logits=blk_logits), axis=1))
            
            # Update the memory inputs. #
            next_memory = []
            next_compress = []
            for m in range(self.n_layers):
                old_memory = curr_memory[m, :, :self.ns, :]
                tmp_compress = tf.tensordot(
                    old_memory, self.p_compress[m], [[2], [0]])
                tmp_compress = tf.transpose(tmp_compress, [0, 2, 1])
                x_old_memory = tf.tensordot(
                    old_memory, self.p_c_linear[m], [[2], [0]])
                
                prev_compress  = curr_compress[m]
                x_new_compress = tf.matmul(
                    tf.nn.softmax(tmp_compress), x_old_memory)
                
                conc_compress = tf.concat(
                    [prev_compress, x_new_compress], axis=1)
                conc_compress = conc_compress[:, -self.ncm:, :]
                
                prev_memory = curr_memory[m]
                conc_memory = tf.concat(
                    [prev_memory, tmp_out_tuple[1][m]], axis=1)
                conc_memory = conc_memory[:, -self.nm:, :]
                
                next_memory.append(
                    tf.expand_dims(conc_memory, axis=0))
                next_compress.append(
                    tf.expand_dims(conc_compress, axis=0))
            
            curr_memory = tf.concat(next_memory, axis=0)
            curr_compress = tf.concat(next_compress, axis=0)
            del next_memory, next_compress
        return acc_losses
    
    def infer(self, x_infer):
        # Inference. #
        batch_sz  = tf.shape(x_infer)[0]
        infer_len = tf.shape(x_infer)[1]
        
        x_inf_emb = tf.nn.embedding_lookup(
            self.W_emb_dec, x_infer)
        infer_emb = [tf.expand_dims(
            x_inf_emb[:, 0, :], axis=1)]
        infer_ids = [
            tf.expand_dims(x_infer[:, 0], axis=1)]
        
        curr_memory = tf.zeros(
            [self.n_layers, batch_sz, self.nm, self.hidden_size])
        curr_compress = tf.zeros(
            [self.n_layers, batch_sz, self.ncm, self.hidden_size])
        
        x_mem_inputs = tf.add(
            self.x_emb_memory[0], tf.concat([
                curr_compress, curr_memory], axis=2))
        
        blk_outputs_list = []
        for step in range(self.seq_length):
            local_step = step % self.ns
            x_inf_inputs = tf.concat(infer_emb, axis=1)
            
            n_blk  = int(step / self.ns)
            idx_st = n_blk * self.ns
            idx_en = step + 1
            
            tmp_inputs = x_inf_inputs[:, idx_st:idx_en, :]
            tmp_inputs = tf.tensordot(
                tmp_inputs, self.W_dec_lin, [[2], [0]])
            
            tmp_out_tuple = self.transformer_decode(
                local_step+1, tmp_inputs, 
                x_mem_inputs, training=False)
            blk_outputs_list.append(tf.expand_dims(
                tmp_out_tuple[1][:, :, -1, :], axis=2))
            
            if (step+1) % self.ns == 0:
                blk_outputs = tf.concat(blk_outputs_list, axis=2)
                blk_outputs_list = []
                
                # Update the memory inputs. #
                next_memory = []
                next_compress = []
                for m in range(self.n_layers):
                    old_memory = curr_memory[m, :, :self.ns, :]
                    tmp_compress = tf.tensordot(
                        old_memory, self.p_compress[m], [[2], [0]])
                    tmp_compress = tf.transpose(tmp_compress, [0, 2, 1])
                    x_old_memory = tf.tensordot(
                        old_memory, self.p_c_linear[m], [[2], [0]])
                    
                    prev_compress  = curr_compress[m]
                    x_new_compress = tf.matmul(
                        tf.nn.softmax(tmp_compress), x_old_memory)
                    
                    conc_compress = tf.concat(
                        [prev_compress, x_new_compress], axis=1)
                    conc_compress = conc_compress[:, -self.ncm:, :]
                    
                    prev_memory = curr_memory[m]
                    conc_memory = tf.concat(
                        [prev_memory, blk_outputs[m]], axis=1)
                    conc_memory = conc_memory[:, -self.nm:, :]
                    
                    next_memory.append(
                        tf.expand_dims(conc_memory, axis=0))
                    next_compress.append(
                        tf.expand_dims(conc_compress, axis=0))
                
                curr_memory = tf.concat(next_memory, axis=0)
                curr_compress = tf.concat(next_compress, axis=0)
                del next_memory, next_compress
                
                # Concatenate the memory inputs. #
                x_mem_inputs = tf.add(
                    self.x_emb_memory[n_blk], tf.concat([
                        curr_compress, curr_memory], axis=2))
            
            tmp_logit  = tf.matmul(
                tmp_out_tuple[0][:, -1, :], self.p_decoder)
            tmp_argmax = tf.cond(
                step < (infer_len-1), 
                lambda: x_infer[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            next_embed = tf.cond(
                step < (infer_len-1), 
                lambda: x_inf_emb[:, step+1, :], 
                lambda: tf.matmul(
                    tf.nn.softmax(tmp_logit), self.W_emb_dec))
            
            infer_ids.append(tf.expand_dims(tmp_argmax, axis=1))
            infer_emb.append(tf.expand_dims(next_embed, axis=1))
        
        infer_indices = tf.concat(infer_ids, axis=1)
        return infer_indices
