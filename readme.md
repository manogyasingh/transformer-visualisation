# Transformer visualisation

An interactive, step-by-step dry run of a GPT-style transformer generating a single token. Every number on screen is computed live in the browser from the weights of a real (tiny) GPT-2-architecture model, and hovering any cell shows the exact arithmetic that produced it.

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
| `npm test` | Checks the TypeScript forward pass against the NumPy reference and tests decoding |
| `npm run build` | Type-checks and builds a static site into `dist/` (serve it with `npm run preview`) |
| `npm run train` | Retrains the model and rewrites `src/model/tiny-gpt-weights.json` (needs Python 3 with NumPy) |

## Layout

- `training/train-tiny-gpt.py`: NumPy training script with a hand-written, gradient-checked backward pass.
- `src/model/`: the instrumented forward pass, sampling, and the trained weights.
- `src/steps/`: one component per step of the dry run.
- `src/components/`: matrix heatmaps, charts, the calculation panel and layout pieces.
