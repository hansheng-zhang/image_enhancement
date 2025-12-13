# Final Report: Low-Light Image Enhancement

## 1. Introduction and Motivation

Images captured under low-light conditions often suffer from poor visibility, low contrast, and amplified noise. This is challenging for consumer smartphones and edge devices, because hardware sensor size is limited. While modern deep learning methods have achieved impressive capabilities in low-light enhancement, they are often computationally demanding and difficult to deploy in real-time or resource-constrained environments.

The motivation of this project is to evaluate and compare classical, lightweight image enhancement algorithms, **CLAHE (Contrast Limited Adaptive Histogram Equalization)** and **Retinex (MSRCR)**, to understand their trade-offs between image quality, computational efficiency, and robustness in real-world scenarios. Unlike deep neural networks that function as black boxes, these classical methods are more interpretable and tunable, which are essential for industrial applications.

## 2. Related Works

Many deep learning-based low-light enhancement methods have been proposed in recent years. **Zero-DCE** predicts pixel-wise curve parameters to adjust exposure without relying on paired training data. **EnlightenGAN** uses an unpaired GAN architecture to enhance images in a fully unsupervised way and generalizes effectively across various scenes. Decomposition models such as **KinD** further separate reflectance and illumination components, incorporating Retinex theory into a learnable framework.

Classical methods remain widely utilized. **Gamma correction** imposes a global nonlinear mapping, though it often excessively brightens highlights while leaving shadows dark. **Homomorphic filtering** operates in the frequency domain to reduce illumination variations, but it relies on filter design and can distort details. **Wavelet-based enhancement** and **tone-mapping** approaches have also been explored, though they typically require careful parameter tuning.

Overall, there remains a need for simple, interpretable, and efficient approaches, especially when deployment resources are limited.

## 3. Method
Overall Framework
![System Block Diagram](report_images/block_diagram.jpg)

We implemented two primary algorithms:

### 3.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE enhances local contrast by processing the image in small, non-overlapping tiles.
**Algorithm Steps:**
1.  **Color Space Conversion**: Convert the input RGB image to a luminance-chrominance space (e.g., LAB or YCrCb) and extract the Luminance (L) channel.
2.  **Tiling**: Partition the L-channel into a grid of $M \times N$ tiles.
3.  **Histogram Equalization**: For each tile, compute the histogram.
4.  **Clipping**: Clip the histogram at a pre-defined threshold $\alpha$ (Clip Limit). The pixels above the threshold are redistributed uniformly across all bins to prevent noise amplification.
5.  **Mapping**: Build the Cumulative Distribution Function (CDF) and map pixel intensities.
6.  **Interpolation**: Bilinearly interpolate the mappings between tile centers to remove block artifacts.
7.  **Recombination**: Merge the enhanced L-channel with the original chroma channels and convert back to RGB.

### 3.2 Retinex (MSRCR)
We implemented Multi-Scale Retinex with Color Restoration (MSRCR) to balance dynamic range compression and color consistency.
**Theoretical Foundation**:
An image $I(x,y)$ is modeled as $I(x,y) = R(x,y) \cdot L(x,y)$, where $R$ is reflectance and $L$ is illumination.
**Algorithm Steps:**
1.  **Multi-Scale Decomposition**: Compute the log-domain reflectance for multiple scales using Gaussian kernels $G_{\sigma_i}$:
    $$ R_{MSR}(x,y) = \sum_{i=1}^N w_i (\log I(x,y) - \log (I(x,y) * G_{\sigma_i}(x,y))) $$
    We used 3 scales (small, medium, large) to capture both fine details and global brightness.
2.  **Color Restoration**: Apply a color restoration function (CRF) to correct color distortions caused by independent channel processing:
    $$ I_{MSRCR}(x,y) = C(x,y) \cdot R_{MSR}(x,y) $$
    where $C(x,y)$ adjusts the ratio of color channels relative to the spectral sum.
3.  **Post-Processing**: Linearly scale the result to the $[0, 255]$ display range.

#### Optimization for Real-World Images (Contrast Stretching)
During experiments with iPhone photos, we observed that the standard MSRCR output sometimes lacked sufficient contrast and visual impact. To address this, we implemented a linear contrast stretch (normalization) step as the final post-processing operation:
```python
if stretch:
    # Stretch to 0-255 based on min/max intensity
    denom = high_val - low_val
    if denom == 0: denom = 1e-6
    c = (c - low_val) / denom * 255.0
```
This simple method ensures the enhanced image has the full available bit-depth, which was found to differ significantly in visual quality for mobile photography compared to the LOL dataset.

## 4. Evaluations

### 4.1 Experimental Setup
**Datasets:**
1.  **LOL Dataset (v1)**: 500 paired low-light/normal-light images. Used for quantitative reference-based evaluation.
2.  **Custom Smartphone Dataset (iPhone)**: 13 real-world low-light images collected with an iPhone. Used for qualitative and no-reference evaluation.

**Metrics:**
*   **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level fidelity (Higher is better).
*   **SSIM (Structural Similarity Index)**: Measures structural preservation (Higher is better).
*   **BRISQUE**: No-reference metric evaluating naturalness (Lower is usually better).
*   **Runtime**: Processing time per image (Seconds).

### 4.2 Quantitative Results

#### A. LOL Dataset Analysis (Paired Data)
We evaluated the methods on the processed LOL dataset.

| Method | Mean PSNR (dB) | Mean SSIM | Mean BRISQUE | Avg Runtime (s) |
| :--- | :--- | :--- | :--- | :--- |
| **CLAHE** | 9.07 | 0.35 | 22.26 | **0.002** |
| **Retinex** | **15.41** | **0.58** | 28.00 | 0.848 |

**Findings:**
*   **Fidelity**: Retinex is better than CLAHE in terms of PSNR (+6.3 dB) and SSIM (+0.23). This indicates MSRCR is much more effective at recovering the underlying ground truth signal in extreme low-light conditions.
*   **Speed**: CLAHE is extremely fast (2ms), making it suitable for 60fps real-time video processing. Retinex requires more time (~0.85s), which may require optimization for real-time use.

#### B. iPhone Dataset Analysis (Real-World Unpaired)
We evaluated performance on high-resolution smartphone photos. We compared two versions of Retinex: standard and with the **Contrast Stretch** optimization enabled.

| Method | Mean BRISQUE | Avg Runtime (s) |
| :--- | :--- | :--- |
| **CLAHE** | **13.79** | **0.010** |
| **Retinex (Standard)** | 16.56 | 3.561 |
| **Retinex (With Stretch)** | 15.60 | 3.592 |

**Findings:**
*   **Perceptual Quality (Metric vs. Visual)**: 
    *   Numerically, **CLAHE** still achieves the best (lowest) BRISQUE score (13.79). 
    *   Adding the **Stretch** operation to Retinex improved its BRISQUE score from 16.56 to 15.60. 
    *   **However**, as discussed in the Qualitative section, the visual improvement of "Retinex + Stretch" is far more significant than the metric suggests. BRISQUE often penalizes slight noise amplification or saturation boosts that are actually desirable for visibility in human perception.
*   **Scalability**: Runtime remains a bottleneck for Retinex (~3.6s) compared to CLAHE (0.01s).

### 4.3 Qualitative Comparisons results

#### Visual Analysis
To better understand the perceptual differences, we present a side-by-side comparison.

**Example 1: LOL Dataset (Low Light)**
![Results on LOL - CLAHE](report_images/lol_clahe.png)
![Results on LOL - Retinex](report_images/lol_retinex.png)
*Observation*: Retinex recovers more shadow detail but avoids over-amplifying noise compared to CLAHE in extremely dark regions.

**Example 2: iPhone Real-world Data (Impact of Contrast Stretching)**
We specifically examined the effect of our "Stretch" optimization on iPhone photos.

| **Retinex (Standard)** | **Retinex (With Stretch)** | **CLAHE** |
| :---: | :---: | :---: |
| ![No Stretch](report_images/iphone_retinex_no_stretch.png) | ![With Stretch](report_images/iphone_retinex_with_stretch.png) | ![CLAHE](report_images/iphone_clahe.png) |

*Observation*: 
*   **Standard Retinex**: Enhances brightness but can look slightly washed out (Left).
*   **Retinex + Stretch**: Significantly improves global contrast and visual "pop" (Middle). Subjectively, this result is often preferred over CLAHE.
*   **CLAHE**: Natural and balanced, but sometimes lacks the "brightening" power of Retinex for very dark scenes (Right).

#### Analysis Plots
The following plots visualize the statistical distribution of our metrics.

**LOL Dataset (PSNR & SSIM):**
![PSNR Barplot](report_images/barplot_psnr_with_gt.png)
![SSIM Barplot](report_images/barplot_ssim_with_gt.png)

**iPhone Dataset (BRISQUE & Runtime):**
![BRISQUE Barplot](report_images/barplot_brisque_no_gt.png)
![Runtime Comparison](report_images/barplot_runtime_sec_no_gt.png)

Based on visual inspection of the results:
*   **CLAHE**: Effectively improves local contrast and visibility. However, it tends to amplify noise in the darkest regions and can sometimes produce a flat look if the clip limit is too high. It preserves edges well but cannot recover color information in near-black areas.
*   **Retinex**: Recovers significantly more brightness and color details from the shadows. The images look much brighter and vivid. However, it has some "halo effect" around high-contrast edges and can lead to unnatural color shifts if the Color Restoration Function is not perfectly tuned.


## 5. Discussion

Our results highlight a clear trade-off:
1.  **Quality vs. Efficiency**: Retinex is the clear winner for restoration quality on the standard benchmark (LOL), recovering lost details that CLAHE misses. However, CLAHE is much faster (400x on iPhone data).
2.  **Domain Sensitivity**: CLAHE performs well on the iPhone dataset according to BRISQUE. This might be because modern smartphone sensors already apply some ISP processing, and CLAHE's gentle local contrast boost works better with this pre-processed data than Retinex's heavy-handed physical model assumption.
3.  **Limitations**:
    *   **CLAHE**: Noise amplification in dark areas is a major issue.
    *   **Retinex**: High computational cost and susceptibility to halo artifacts.

## 6. Future Work
*   **Denoising Integration**: Integrating a denoising step (e.g., BM3D or Bilateral Filter) before CLAHE could mitigate noise amplification.
*   **Optimization**: Retinex could be accelerated using GPU implementation (CUDA) or Fast Fourier Transform (FFT) for the convolution steps to make it viable for mobile applications.
*   **Deep Learning on Edge**: Investigating lightweight CNNs (like Zero-DCE) that might work as good as Retinex with inference speeds closer to CLAHE.

## 7. Conclusion
In this project, we successfully implemented and evaluated CLAHE and MSRCR Retinex for low-light image enhancement. Quantitative evaluation on the LOL dataset shows that **Retinex offers superior restoration fidelity (15.41 dB PSNR)** compared to CLAHE (9.07 dB). However, **CLAHE demonstrates exceptional efficiency**, processing high-resolution iPhone images in 0.01 seconds, compared to 3.5 seconds for Retinex. For real-time mobile applications, CLAHE remains a pragmatic choice, while Retinex is better for offline post-processing.
