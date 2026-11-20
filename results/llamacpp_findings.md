# Overnight: packs vs Gemma 4 12B on stock llama.cpp (2026-11-19/20)

Write-up of the size-matched llama.cpp comparison with Gemma 4 12B. Summary tables are also in docs/EXPERIMENTS.md.

## Bottom line

The prototype headline ("from 7.8 GB a pack beats the 12B") **does not survive llama.cpp at equal file size**. The prototype's plain 4-bit rounding hurt the dense 12B more than the MoE; with llama.cpp's K-quants the 12B improves more than the pack does.

What does survive, and is new:

1. **Speed.** At every size a pack writes 2.3–3.5× faster than the 12B, and up to 4.5× on a CPU without a GPU.
2. **Below ~12 GB a pack is the only way to run the 26B at all.** The full 26B at Q2_K (10.6 GB) collapses into repeated fragments on all 300 questions; a 64-per-layer pack at 7.4 or 9.2 GB answers coherently.
3. **At ~13 GB a 26B beats the 12B at its best quantisation, clearly** (pack 83–29, full model 112–51), and is about 5× faster on CPU. There, the full model at Q3_K_M is as good as a pack and slightly faster.

## Quality (Gemini 3.1 Pro, blind, both orders; win only when both agree; 300 questions unless noted)

| Pair (llama.cpp b12549, unmodified) | File sizes | First wins | Ties | Second wins |
|---|---|---|---|---|
| Pack 64/layer Q3_K_M vs 12B Q4_K_M | 7.37 vs 7.38 GB | 69 | 117 | **114** |
| Pack 64/layer Q4_K_M vs 12B Q6_K | 9.21 vs 9.79 GB | 71 | 124 | **105** |
| **Pack 96/layer Q4_K_M vs 12B Q8_0** (7 of 10 subjects, 210 q) | 13.0 vs 12.67 GB | **83** | 98 | 29 |
| **Full 26B Q3_K_M vs 12B Q8_0** | 13.29 vs 12.67 GB | **112** | 137 | 51 |
| same, on the same 7 subjects as the pack row | | 76 | 99 | 35 |

Diagnostics (is the export broken?):

| GGUF vs its own prototype version | GGUF wins | Ties | Prototype wins |
|---|---|---|---|
| Pack 64/layer Q4_K_M vs prototype pack (RTN 4-bit) | 89 | 151 | 60 |
| 12B Q4_K_M vs prototype 12B (RTN 4-bit) | 112 | 141 | 47 |

Both improved on llama.cpp, the 12B by more. No sign of an export problem.

## Speed (llama-bench, tokens per second generated)

| Model | File | RTX 5080 | Ryzen 7 9700X, CPU only, 8 threads |
|---|---|---|---|
| Pack 64/layer Q3_K_M | 7.37 GB | **225** | **20.5** |
| 12B Q4_K_M | 7.38 GB | 98 | 6.6 |
| Pack 64/layer Q4_K_M | 9.21 GB | **200** | **17.5** |
| 12B Q6_K | 9.79 GB | 78 | 5.0 |
| Pack 96/layer Q4_K_M | 13.0 GB | 187 | 17.7 |
| Full 26B Q3_K_M | 13.29 GB | **223** | **20.0** |
| 12B Q8_0 | 12.67 GB | 62 | 3.9 |

Raw output: `results/llamacpp_speed.txt`.

## Stopped by the Gemini spend cap

The judge hit HTTP 429 ("rate-limited to prevent exceeding its monthly spending cap"). Still owed, all resumable: `scripts/run_pending_judging.ps1`

- pack 96 vs 12B Q8_0: the last 3 subjects
- pack 96 vs full 26B Q3_K_M (drop experts or drop bits, at 13 GB)
- pack 64 Q4_K_M vs full 26B Q3_K_M (149 of 300 done)
- full 26B Q3_K_S (12.22 GB) vs 12B Q8_0

## A reframed headline the data supports (for discussion)

> Keep only the experts a subject uses and a 26B MoE runs in 7–9 GB, where the full model can't be squeezed, at 3× the speed of the dense 12B that fits there. The 12B still answers somewhat better at equal size; from ~13 GB the 26B wins on both.

## Not tried (would test the remaining gap)

- Importance-matrix (imatrix) quantisation, for both packs and the 12B, and for the full 26B at IQ2/Q2 sizes (it might rescue the full model at ~10 GB, which would compete with packs).
- Packs at 80 per layer (~11 GB) against the 12B at Q6_K/Q8_0.
