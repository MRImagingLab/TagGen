# TagGen
This is the project page of TagGen published on *Magnetic Resonance in Medicine*: [paper](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30422).

## TagGen: Diffusion-based generative model for cardiac MR tagging super resolution
We developed a diffusion-based generative super-resolution model for MR tagging images and demonstrated its potential to integrate with parallel imaging to reconstruct highly accelerated cine MR tagging images acquired in three heartbeats with enhanced tag grid quality.

<p align="center">
  <img src="figures/Main Architecture.png" width="70%" alt="figures/Main Architecture" />
</p>

## Results

### Evaluation on synthetic data

- Synthetic low-resolution MR tagging images for a patient with a rate-3.3 acceleration of the two-chamber view at end-diastole and end-systole.
<p align="center">
  <img src="figures/Figure 2.png" width="70%" alt="figures/Figure 2" />
</p>

- Quantitative comparisons of REGAIN and TagGen for super resolution of prospective MR tagging images with central 30% k-space and GRAPPA-3 (generalized autocalibrating partially parallel acquisitions 3). The evaluation was performed on 18 slices from 6 independent subjects and scored by two radiologists on a 5-point Likert scale.
<p align="center">
  <img src="figures/Figure 5.png" width="70%" alt="figures/Figure 5" />
</p>

### Evaluation on prospective data
- Prospectively acquired low-resolution MR tagging images for a volunteer with a rate-10 acceleration of the two-chamber view on a 1.5T scanner.
<p align="center">
  <img src="figures/Figure 3.png" width="70%" alt="figures/Figure 3" />
</p>

- Prospectively acquired low-resolution MR tagging images for a patient with a nominal rate-10 acceleration of the four-chamber view on a 3T scanner.
<p align="center">
  <img src="figures/Figure 4.png" width="70%" alt="figures/Figure 4" />
</p>

- TagGen in enhancing prospectively acquired low-resolution cine tagging MRI images for a patient with a rate-10 acceleration of two-chamber, three-chamber, and four-chamber views acquired on a 1.5T scanner.
<p align="center">
  <img src="figures/Figure S1.gif" width="60%" alt="figures/Figure S1" />
</p>

## Citation
If this work is helpful for your research, please consider citing:

Sun, C., et al. (2025). "TagGen: Diffusion-based generative model for cardiac MR tagging super resolution." Magn Reson Med.
	PURPOSE: The aim of the work is to develop a cascaded diffusion-based super-resolution model for low-resolution (LR) MR tagging acquisitions, which is integrated with parallel imaging to achieve highly accelerated MR tagging while enhancing the tag grid quality of low-resolution images. METHODS: We introduced TagGen, a diffusion-based conditional generative model that uses low-resolution MR tagging images as guidance to generate corresponding high-resolution tagging images. The model was developed on 50 patients with long-axis-view, high-resolution tagging acquisitions. During training, we retrospectively synthesized LR tagging images using an undersampling rate (R) of 3.3 with truncated outer phase-encoding lines. During inference, we evaluated the performance of TagGen and compared it with REGAIN, a generative adversarial network-based super-resolution model that was previously applied to MR tagging. In addition, we prospectively acquired data from 6 subjects with three heartbeats per slice using 10-fold acceleration achieved by combining low-resolution R = 3.3 with GRAPPA-3 (generalized autocalibrating partially parallel acquisitions 3). RESULTS: For synthetic data (R = 3.3), TagGen outperformed REGAIN in terms of normalized root mean square error, peak signal-to-noise ratio, and structural similarity index (p < 0.05 for all). For prospectively 10-fold accelerated data, TagGen provided better tag grid quality, signal-to-noise ratio, and overall image quality than REGAIN, as scored by two (blinded) radiologists (p < 0.05 for all). CONCLUSIONS: We developed a diffusion-based generative super-resolution model for MR tagging images and demonstrated its potential to integrate with parallel imaging to reconstruct highly accelerated cine MR tagging images acquired in three heartbeats with enhanced tag grid quality.


