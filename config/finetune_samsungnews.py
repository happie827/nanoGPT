import time

out_dir = 'out-samsungnews'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'samsungnews'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'samsungnews'
init_from = 'gpt2-medium' # 355M
# 'gpt2' # 124M   
# 'gpt2-medium' # 355M
# 'gpt2-large' # 762M
# 'gpt2-xl' # 1558M # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# samsungnews has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False


