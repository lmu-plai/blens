## Appendix: Implementation Details

To complete our paper, we report our hyper-parameters in the table below.

| **Hyper-parameter**                    | **Value**                |
|----------------------------------------|--------------------------|
| Token dimension ($d$)                  | 768                      |
| Batch size                             | 512                      |
| Function name words                    | 20                       |
| Attention heads                        | 32                       |
| Head dimension                         | 24                       |
|                                        |                          |
| CLAP patches                           | 16                       |
| Dexter patches                         | 16                       |
| PalmTree basic blocks                  | 50                       |
| Function patches ($k_1$)               | 82                       |
| Function tokens ($k_2$)                | 64                       |
|                                        |                          |
| Text encoder transformer blocks        | 6                        |
| Multim. text enc. transf. blocks       | 6                        |
| Lord transformer blocks                | 12                       |
|                                        |                          |
| Optimizer                              | AdamW                    |
| Decoupled weight decay                 | 0.01                     |
| Gradient running averages coefficients | (0.9, 0.999)             |
| Gradient clip                          | 1.0                      |
| Warming steps                          | 2                        |
| Learning rate schedule                 | Cosine decay to 0        |
|                                        |                          |
| Learning rate (pre-training)           | 5e-5                     |
| Learning rate (fine-tuning-A†)         | 1e-5                     |
| Learning rate (fine-tuning-B‡)         | 5e-5                     |
| Label smoothing (fine-tuning)          | 0.1                      |
|                                        |                          |
| Epochs                                 | 200                      |
| Threshold fine-tuning period           | Every 10 epochs          |
| Epochs (Ablation)                      | 80                       |
| Threshold fine-tuning period (Ablation)| Every 4 epochs           |

†: Ensemble encoder and function encoder, ‡: Lord decoder.
