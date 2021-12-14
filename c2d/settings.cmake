# Explore steps
set(CR_EXPLORE_KSTEPS "int(250e3)")

# Starting epsilon greedy value 
set(CR_START_EPS 1.0)

# Final epsilon greedy value
set(CR_FINAL_EPSILON 0.01)

# Eval epsilon
set(CR_EVAL_EPS 0.001)

# Replay Buffer Size 
set(CR_MEM_SIZE "int(1e6)")

# History Prefill Size
set(CR_MEM_PREFILL "int(20e3)")

# Probability Warmup Iterations
set(CR_WARMUP 0)

# Target network update period
set(CR_TUP "int(8e3)")

# Gamma value
set(CR_GAMMA 0.99)

# Maximum number of steps/frames in an episode
set(CR_MAX_STEPS 27000)

# Number of frames to skip per action
set(CR_FRAME_SKIP 4)

# Number of stacked observations = observation window
set(CR_OBS_STACK 4)

# Abstraction frame = gray scale [OBS_WIDTH x OBS_WIDTH]
set(CR_OBS_WIDTH 84)

# Repeat action probability (sticky actions 0.25)
set(CR_REPEAT_ACTION_PROBABILITY 0.25)

# Scaling function epsilon value
set(CR_HEPS 0.001)

# Training intensity = trained on samples per experienced
set(CR_TRAIN_INTENSITY 8)

# Single environment Adam Learning Rate
set(CR_LR 0.5e-4)

# Single environment Adam Epsilon value 
set(CR_ADAM_EPS 0.01/32)

# Single environment batchsize
set(CR_BATCH_SIZE_ONE 32)

# Single prefetch batch items (divisor of CR_BATCH_SIZE_ONE)
set(CR_PREFETCH 8)

# Training phase steps 
set(CR_TRAIN_PHASE_STEPS 250000)

# Eval phase steps 
set(CR_EVAL_PHASE_STEPS 0)

# Single Environment Atoms
set(CR_ATOMS 32)
