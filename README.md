# GRPO Project

GRPO training on math tasks using a small causal language model. Based on the original deepseek paper. 
 
## Files

- `GRPO.py`: trainer + reward models, this is the main file
    - rewardmodelv1 is a very simple reward model,
    - rewardmodelv2 is the one I was using on my computer as I have a 4GB gpu so I needed smaller rewards
    - rewardmodelv3 is the model that should be used for standard grpo
- `train.py`: This is a file to try the model
- `simple_dataset.py`: generated arithmetic dataset used by `--simple-dataset`, 
- `test/test_grpo.py`: unit tests

## Requirements

Python 3.10+ recommended.

Install these:
```bash
python -m pip install torch transformers datasets matplotlib pytest
```

## Current Training Flow

1. Load model/tokenizer.
2. Load either:
   - Hugging Face dataset (`--dataset`), or
   - local generated simple dataset (`--simple-dataset`).
3. Split dataset into train/test.
4. Train with GRPO rollouts.
5. Evaluate periodically and print reward
6. Save reward graph to `train_test_reward_curve.png`.
7. Save checkpoints every 500 training steps and at the end

## CLI Options (train.py)

- `--model`: Hugging Face model id
- `--dataset`: Hugging Face dataset id (ignored when `--simple-dataset` is set)
- `--max-steps`: max train iterations
- `--eval-every`: evaluation interval
- `--eval-sample-size`: number of examples per split per eval
- `--eval-num-return-seq`: eval rollouts per prompt
- `--eval-max-new-tokens`: eval generation length
- `--simple-dataset`: use generated local arithmetic dataset
- `--simple-size`: number of generated examples
- `--train-num-return-seq`: training rollouts per prompt
- `--train-max-new-tokens`: training generation length
- `--train-epochs`: PPO update epochs per step
- `--accumulation-steps`: optimizer step interval via gradient accumulation
- `--no-memory-saving`: disable memory-saving mode (default mode is memory-saving enabled)

Show all arguments:

```bash
python train.py --help
```

## Running it:

### Quick smoke test

```bash
python -u train.py --simple-dataset --simple-size 120 --max-steps 20 --eval-every 5
```
This will probably give you zero rewards unless you choose a better model and fix the system prompt.
### Stable low-memory profile

```bash
python -u train.py --simple-dataset --simple-size 300 --max-steps 200 --eval-every 20 --eval-sample-size 4 --eval-num-return-seq 1 --eval-max-new-tokens 16 --train-num-return-seq 2 --train-max-new-tokens 16 --train-epochs 1 --accumulation-steps 8
```

### Higher-signal profile (more expensive)

```bash
python -u train.py --simple-dataset --simple-size 500 --max-steps 400 --eval-every 25 --eval-sample-size 6 --eval-num-return-seq 2 --eval-max-new-tokens 24 --train-num-return-seq 4 --train-max-new-tokens 24 --train-epochs 1 --accumulation-steps 8
```

## Notes and Troubleshooting

- If you see no output after `Using device: ...`, check `--eval-every` versus dataset size:
  - only eval logs are printed by default.
  - if eval interval is larger than actual train steps, no eval lines appear.
- If rewards are all zero, increase generation length and/or eval sample size.
- If CUDA crashes, restart the Python process before rerunning.
- `--simple-dataset` uses generated multi-step arithmetic tasks that are harder than basic one-step math.

## Output Files

- `train_test_reward_curve.png`: train/test reward curve
- `grpo_step_<N>/`: periodic checkpoints

## Tests

```bash
python -m pytest -q test
```

## Future improvements

- The reward model right now is very bad I need to imporve it (Should accept latex, can't differentiate between two correct answers if structured differently: 2x+2 and 2(x+1)),it could also reward showing steps and other things
- Would be nice to reproduce the first part of the paper that talks about creating the dataset
- Right now i just ran it on my local GPU, it's an RTX 3050 meaning that it is very weak, so my only tests have been very small, and I can't really prove it works well or make a model that Improved, to do this I'm trying to run it on a GPU cluster that is much more powerful. And will allow me to prove it works.
- making it distributed so that it can run in more than just one gpu in a cluster

