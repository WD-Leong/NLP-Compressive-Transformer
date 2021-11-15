import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
#import tensorflow_addons as tfa
import tf_ver2_compressive_gpt_v2 as gpt

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, seq_len, 
    x_encode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0, foc_loss=False):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    input_len = x_encode.shape[1]
    if input_len <= seq_len:
        num_block = 1
    elif input_len % seq_len == 0:
        num_block = int(input_len / seq_len)
    else:
        num_block = int(input_len / seq_len) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_state = None
        for n_blk in range(num_block):
            seq_st = n_blk * seq_len
            seq_en = (n_blk+1) * seq_len
            
            tmp_encode = x_encode[id_st:id_en, seq_st:seq_en]
            tmp_output = x_output[id_st:id_en, seq_st:seq_en]
            
            with tf.GradientTape() as grad_tape:
                tmp_output_tuple = model(
                    tmp_encode, curr_state=tmp_state, training=True)
                
                tmp_state  = tmp_output_tuple[1]
                tmp_logits = tmp_output_tuple[0]
                tmp_losses = tf.reduce_sum(tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tmp_output, logits=tmp_logits), axis=1))
            
            # Accumulate the gradients. #
            tot_losses += tmp_losses
            tmp_gradients = grad_tape.gradient(
                tmp_losses, model_params)
            acc_gradients = [tf.add(
                acc_grad, grad) for acc_grad, grad \
                    in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_losses

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 50
input_len  = 100
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 10000
restore_flag  = False
save_step     = 250
warmup_steps  = 5000
display_step  = 10
anneal_step   = 2500
anneal_rate   = 0.75

prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 250

model_ckpt_dir  = "TF_Models/reddit_jokes_sw_gpt_compressive_v2"
train_loss_file = "train_loss_gpt_compressive_reddit_jokes_sw_v2.csv"

# Load the data. #
tmp_pkl_file = "/home/Data/reddit_jokes/"
tmp_pkl_file += "reddit_jokes_subword_v1.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size) + ".")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

tmp_data = []
for tmp_row in full_data:
    if len(tmp_row) > 1 and \
        len(tmp_row) <= input_len:
        tmp_data.append(tmp_row)

num_data  = len(tmp_data)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]
print("Total of", str(len(tmp_data)), "rows loaded.")

# Build the GPT Compressive. #
print("Building the GPT Compressive Model.")
start_time = time.time()

gpt_model = gpt.GPT_Compressive(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    embed_size=hidden_size, p_keep=prob_keep)
gpt_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time()-start_time) / 60
print("GPT Compressive Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Print the model summary. #
tmp_zero = np.zeros(
    [1, seq_length], dtype=np.int32)
tmp_pred = gpt_model(tmp_zero)

print(gpt_model.summary())
del tmp_zero, tmp_pred

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(
            manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Train the Transformer model. #
tmp_out_seq = np.zeros(
    [batch_size, input_len+1], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    anneal_pow = int(n_iter / anneal_step)
    learning_rate = max(np.power(
        anneal_rate, anneal_pow)*initial_lr, 1.0e-5)

print("-" * 50)
print("Training the GPT Compressive Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            anneal_pow = int(n_iter / anneal_step)
            learning_rate = max(np.power(
                anneal_rate, anneal_pow)*initial_lr, 1.0e-6)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_p_idx = tmp_data[tmp_index] + [EOS_token]
        
        n_input = len(tmp_p_idx)
        tmp_out_seq[n_index, :n_input] = tmp_p_idx
        del tmp_p_idx
    
    # Set the training data. #
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    tmp_loss = sub_batch_train_step(
        gpt_model, sub_batch, 
        seq_length, tmp_input, tmp_output, 
        gpt_optimizer, learning_rate=learning_rate)

    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        sample_test = np.random.choice(num_data, size=1)
        tmp_p_index = tmp_data[sample_test[0]]
        
        in_phrase = bpe.bp_decode(
            tmp_p_index, idx_2_subword)
        in_phrase = " ".join(
            in_phrase).replace("<", "").replace(">", "")
        
        n_tokens = len(tmp_p_index)
        n_sample = np.random.randint(1, high=seq_length)
        tmp_test = np.array(
            [tmp_p_index[:n_sample]], dtype=np.int32)
        tmp_test = tmp_test.reshape(1, -1)
        
        tmp_infer = gpt_model.infer(
            tmp_test, input_len)
        del sample_test, n_tokens
        
        gen_phrase = bpe.bp_decode(
            tmp_infer[0].numpy(), idx_2_subword)
        gen_phrase = " ".join(gen_phrase).replace(
            "<", "").replace(">", "")
        
        test_phrase = bpe.bp_decode(
            tmp_p_index[:n_sample], idx_2_subword)
        test_phrase = " ".join(test_phrase).replace(
            "<", "").replace(">", "")
        del tmp_p_index
        
        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Loss:", str(avg_loss) + ".")
        
        print("")
        print("Input Phrase:")
        print(test_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Phrase:")
        print(in_phrase)
        del n_sample
        
        train_loss_list.append((n_iter, avg_loss))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

