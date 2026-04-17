# Eliminating Clipping Bias in DP-SGD

This repository contains the code for our project on Differentially Private SGD (DP-SGD). Currently, it holds the baseline implementation trained on the MNIST dataset using Meta's Opacus library.

Deep learning models can sometimes memorize parts of their training data, making them vulnerable to privacy attacks. Standard DP-SGD solves this by clipping individual gradients to a fixed threshold and adding noise. Though the clipping operation guarantees a strict privacy bound, it is also a source of severe bias in the optimization trajectory because it permanently discards gradient magnitude. 

Our main goal for this project is to implement Dice-SGD, an error-feedback algorithm that stores the residual error in a memory buffer rather than discarding it permanently, effectively eliminating the clipping bias.

## What's inside
* `Baseline_DP_SGD.ipynb`: This is the Google Colab notebook containing our basic DP-SGD training loop. We used this to validate our computational pipeline and get a formal control group before we build the custom Dice-SGD optimizer.

## Baseline Hyperparameters
For the baseline DP-SGD model on MNIST, we used these exact settings based on standard practices:
* **Clipping Norm Threshold (C):** 1.0
* **Noise Multiplier ($\sigma$):** 1.0
* **Learning Rate:** 0.05
* **Momentum:** 0.5
* **Batch Size:** 64
* **Epochs:** 5

With these settings, the baseline model successfully converged to approximately 91.0% accuracy with a strict privacy budget of `$\epsilon \approx 0.35$`.

## How to run
1. Upload the `Baseline_DP_SGD.ipynb` notebook to Google Colab.
2. Go to **Runtime** -> **Change runtime type** and select the **T4 GPU** to avoid out-of-memory exceptions.
3. Run all the cells. The Opacus Privacy Engine will automatically track the RDP moments and calculate the privacy budget as it trains.

## Next Steps (Phase 2)
Now that the baseline is deployed, our next task is to subclass the core `DPOptimizer` to write a custom PyTorch optimizer for Dice-SGD. After that, we will conduct a rigorous benchmark analysis against this baseline using both MNIST and CIFAR-10.
