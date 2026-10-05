# Transformer visualisation

An interactive, step-by-step dry run of a GPT-style transformer. Every number on screen is computed live in the browser from the weights of a real (tiny) GPT-2-architecture model, and hovering any cell shows the exact arithmetic that produced it.

It has three chapters, listed at the top of the sidebar:

- **Inference: generate one token.** Tokenisation, embeddings, every attention and MLP computation in both blocks, the logits, temperature, softmax and sampling (greedy, top-k, top-p).
- **Pretraining: one training step.** Inputs and targets, random initialisation, the loss at every position, backpropagation through every layer in reverse, a numerical gradient check, and the Adam update. The model trains live in the browser, so you can watch the loss fall, then inspect the exact gradients of any later step.
- **Post-training: one PPO iteration (RLHF).** The InstructGPT recipe (Ouyang et al., 2022, following Ziegler et al., 2019 and Stiennon et al., 2020) applied to the pretrained model, which plays the SFT model's role. A reward model is trained with a Bradley–Terry loss on comparisons from a simulated labeler who dislikes chasing and nonsense. Then one PPO iteration (Schulman et al., 2017): sample responses, score them, add a per-token KL penalty to the frozen reference, estimate advantages with a value model initialised from the reward model and GAE, whiten them, and take four epochs of clipped policy and value updates with the pretraining mix of PPO-ptx. Implementation details follow Huang et al. (2024). PPO runs live in the browser; the last page compares it with the exact KL-regularised optimum and shows reward hacking when the regularisers are turned off.

The model has 2 blocks, width 8, 2 attention heads, a 32-unit MLP and an 18-word vocabulary. It was trained on a toy corpus ("the cat sat on the mat .") to the optimal loss.

## Running it

Requires Node.js 20.19+ or 22.12+.

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173). Use the arrow keys or the buttons at the bottom to step through; at the end of a chapter they carry on into the next one.

## Other commands

| Command | What it does |
| --- | --- |
| `npm test` | Checks the TypeScript forward and backward passes and the reward model against the NumPy reference, that in-browser training converges, and the PPO gradients (by finite differences) and convergence |
| `npm run build` | Type-checks and builds a static site into `dist/` (serve it with `npm run preview`) |
| `npm run train` | Retrains the model and rewrites `src/model/tiny-gpt-weights.json` (needs Python 3 with NumPy) |
| `npm run train-rm` | Regenerates the labeler's comparisons, retrains the reward model, and rewrites `src/model/reward-model-weights.json` |

## Layout

- `training/train-tiny-gpt.py`: NumPy training script with a hand-written, gradient-checked backward pass.
- `training/train-reward-model.py`: the simulated labeler, its comparisons, and the Bradley–Terry reward model (same NumPy backward pass, starting from the final hidden state).
- `src/model/`: the instrumented forward and backward passes, sampling, the reward model, and the trained weights.
- `src/training/`: the in-browser pretraining trainer (initialisation, minibatches, Adam).
- `src/post-training/`: PPO (rollouts, rewards, GAE, clipped losses and their gradients, PPO-ptx, exact evaluation by enumerating responses) and the live PPO trainer.
- `src/steps/step-defs.ts`: the chapters (`buildChapters`) and the steps in each. A new chapter is an entry there, plus a component per step in `STEP_COMPONENTS` and optional top-bar controls in `chapterBar` (both in `src/app.tsx`).
- `src/steps/`: one component per step of the inference walkthrough; `src/steps/training/` and `src/steps/post-training/` for the other two chapters.
- `src/components/`: matrix heatmaps, charts, the calculation panel and layout pieces.
