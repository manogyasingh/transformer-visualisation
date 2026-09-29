# Transformer visualisation

An interactive, step-by-step dry run of a GPT-style transformer. Every number on screen is computed live in the browser from the weights of a real (tiny) GPT-2-architecture model, and hovering any cell shows the exact arithmetic that produced it.

It has two walkthroughs, switched with the tabs at the top:

- **Inference: generate one token.** Tokenisation, embeddings, every attention and MLP computation in both blocks, the logits, temperature, softmax and sampling (greedy, top-k, top-p).
- **Pretraining: one training step.** Inputs and targets, random initialisation, the loss at every position, backpropagation through every layer in reverse, a numerical gradient check, and the Adam update. The model trains live in the browser, so you can watch the loss fall, then inspect the exact gradients of any later step.

The model has 2 blocks, width 8, 2 attention heads, a 32-unit MLP and an 18-word vocabulary. It was trained on a toy corpus ("the cat sat on the mat .") to the optimal loss.

## Running it

Requires Node.js 20.19+ or 22.12+.

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173). Use the arrow keys or the buttons at the bottom to step through.

## Other commands

| Command | What it does |
| --- | --- |
| `npm test` | Checks the TypeScript forward and backward passes against the NumPy reference, and that in-browser training converges |
| `npm run build` | Type-checks and builds a static site into `dist/` (serve it with `npm run preview`) |
| `npm run train` | Retrains the model and rewrites `src/model/tiny-gpt-weights.json` (needs Python 3 with NumPy) |

## Layout

- `training/train-tiny-gpt.py`: NumPy training script with a hand-written, gradient-checked backward pass.
- `src/model/`: the instrumented forward and backward passes, sampling, and the trained weights.
- `src/training/`: the in-browser trainer (initialisation, minibatches, Adam).
- `src/steps/`: one component per step of the inference walkthrough; `src/steps/training/` for the pretraining walkthrough.
- `src/components/`: matrix heatmaps, charts, the calculation panel and layout pieces.
