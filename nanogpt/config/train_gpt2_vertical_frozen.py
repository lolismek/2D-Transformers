# Train the vertical (over-layers) transformer on top of a FROZEN pretrained GPT-2 124M.
# Baseline to beat: vanilla GPT-2 124M val loss on OpenWebText (config/eval_gpt2.py).
#
#   python train.py config/train_gpt2_vertical_frozen.py

# I/O
out_dir = 'out-gpt2-vertical-frozen'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = True

# logging
wandb_log = False
wandb_project = '2d-transformer'
wandb_run_name = 'gpt2-frozen-vertical'

# start from pretrained GPT-2 and freeze the base; train only the vertical transformer
init_from = 'gpt2'
freeze_base = True

# vertical transformer (full bidirectional self-attention over the 13-layer stack, top-layer readout)
vertical = True
n_vertical_layer = 1
n_vertical_head = 12
vertical_mlp_ratio = 2

# data
dataset = 'openwebtext'
block_size = 1024
batch_size = 12
gradient_accumulation_steps = 8   # ~98k tokens/iter; base is frozen+forward-only so this is cheap

# optimizer: only ~4.7M params train, so a short run with a modest LR shows signal quickly
learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 100
min_lr = 6e-5
weight_decay = 1e-1
beta2 = 0.95
grad_clip = 1.0

# system: V100 has NO native bfloat16 -> use fp16 (auto-enables GradScaler)
dtype = 'float16'
compile = False   # start without compile for stability; can enable later
