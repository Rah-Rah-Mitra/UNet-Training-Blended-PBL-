## Table A: final results vs the paper's Table 2 (PSNR dB / SSIM, RGB)

| dataset | scale | bicubic paper | bicubic ours | UnetSR paper | UnetSR ours | UnetSR Δ dB | UnetSR+ paper | UnetSR+ ours | UnetSR+ Δ dB | UnetSR / UnetSR+ ours, BSD300-trained | best other method (paper) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BSD300 | x2 | 26.65 / 0.7924 | 26.68 / 0.7938 | 29.42 / 0.8813 | 29.04 / 0.8726 (300 ep) | -0.38 (close) | 29.84 / 0.8816 | 29.06 / 0.8735 (300 ep) | -0.78 (close) | - | DBPN 29.87 / 0.8834 |
| BSD300 | x4 | 23.51 / 0.6157 | 23.54 / 0.6178 | 24.83 / 0.6843 | 24.58 / 0.6755 (300 ep) | -0.25 (matches) | 24.95 / 0.6901 | 24.55 / 0.6753 (300 ep) | -0.40 (close) | - | DBPN 25.06 / 0.6967 |
| BSD300 | x8 | 21.31 / 0.4933 | 21.34 / 0.4951 | 21.99 / 0.5231 | 22.03 / 0.5255 (300 ep) | +0.04 (matches) | 22.04 / 0.5235 | 22.04 / 0.5263 (300 ep) | +0.01 (matches) | - | DBPN 22.06 / 0.5229 |
| SET14 | x2 | 24.45 / 0.8482 | 24.45 / 0.8482 | 26.72 / 0.8735 | 24.51 / 0.8441 (300 ep) | -2.21 (differs) | 28.40 / 0.9198 | 24.43 / 0.8413 (300 ep) | -3.96 (differs) | 28.84 / 28.92 | VDSR 28.66 / 0.9269 |
| SET14 | x4 | 19.72 / 0.6089 | 19.72 / 0.6089 | 20.89 / 0.6693 | 18.55 / 0.5826 (300 ep) | -2.34 (differs) | 21.68 / 0.7112 | 18.57 / 0.5819 (300 ep) | -3.11 (differs) | 21.96 / 21.87 | DBPN 21.77 / 0.7171 |
| SET14 | x8 | 16.11 / 0.3673 | 16.11 / 0.3673 | 16.70 / 0.4093 | 16.23 / 0.3727 (300 ep) | -0.47 (close) | 17.83 / 0.4103 | 16.08 / 0.3689 (300 ep) | -1.75 (differs) | 17.10 / 17.12 | VDSR 16.80 / 0.4095 |

## Table B: PSNR [dB] on the blur sweep (σ in LR pixels; σ = 0 is the paper's test set)

| dataset | scale | model | run | σ=0 | σ=0.25 | σ=0.5 | σ=0.75 | σ=1 |
|---|---|---|---|---|---|---|---|---|
| BSD300 | x2 | bicubic | - | 26.68 | 26.31 | 25.25 | 24.26 | 23.45 |
| BSD300 | x2 | UnetSR, fixed degradation | BSD300_x2_mse | 29.04 | 28.80 | 26.85 | 25.01 | 23.82 |
| BSD300 | x2 | UnetSR+, fixed degradation | BSD300_x2_mixge | 29.06 | 28.82 | 26.87 | 25.03 | 23.83 |
| BSD300 | x2 | UnetSR+, random blur, fine-tuned | BSD300_x2_mixge_rand_ft_lr0.0001 | 28.90 | 28.68 | 27.11 | 25.31 | 24.01 |
| BSD300 | x4 | bicubic | - | 23.54 | 23.27 | 22.62 | 21.91 | 21.29 |
| BSD300 | x4 | UnetSR, fixed degradation | BSD300_x4_mse | 24.58 | 24.43 | 23.52 | 22.41 | 21.56 |
| BSD300 | x4 | UnetSR+, fixed degradation | BSD300_x4_mixge | 24.55 | 24.42 | 23.53 | 22.43 | 21.57 |
| BSD300 | x4 | UnetSR+, random blur, fine-tuned | BSD300_x4_mixge_rand_ft_lr0.0001 | 24.55 | 24.40 | 23.60 | 22.52 | 21.63 |
| BSD300 | x8 | bicubic | - | 21.34 | 21.13 | 20.60 | 20.00 | 19.45 |
| BSD300 | x8 | UnetSR, fixed degradation | BSD300_x8_mse | 22.03 | 21.92 | 21.24 | 20.37 | 19.66 |
| BSD300 | x8 | UnetSR+, fixed degradation | BSD300_x8_mixge_300ep_gpu | 22.04 | 21.95 | 21.28 | 20.40 | 19.68 |
| SET14 | x2 | bicubic | - | 24.45 | 23.89 | 22.29 | 20.75 | 19.48 |
| SET14 | x2 | UnetSR, fixed degradation | SET14_x2_mse | 24.51 | 24.23 | 23.01 | 21.42 | 19.96 |
| SET14 | x2 | UnetSR+, fixed degradation | SET14_x2_mixge | 24.43 | 24.17 | 22.99 | 21.45 | 20.02 |
| SET14 | x4 | bicubic | - | 19.72 | 19.25 | 18.17 | 17.04 | 16.12 |
| SET14 | x4 | UnetSR, fixed degradation | SET14_x4_mse | 18.55 | 18.36 | 17.73 | 16.79 | 15.88 |
| SET14 | x4 | UnetSR+, fixed degradation | SET14_x4_mixge | 18.57 | 18.38 | 17.75 | 16.81 | 15.90 |
| SET14 | x8 | bicubic | - | 16.11 | 15.84 | 15.22 | 14.59 | 14.12 |
| SET14 | x8 | UnetSR, fixed degradation | SET14_x8_mse | 16.23 | 16.08 | 15.52 | 14.81 | 14.23 |
| SET14 | x8 | UnetSR+, fixed degradation | SET14_x8_mixge | 16.08 | 15.92 | 15.36 | 14.67 | 14.11 |
