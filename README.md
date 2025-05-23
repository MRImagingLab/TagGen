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

> Sun C, Thornburgh C, Wang Y, Kumar S, Altes TA.
> TagGen: Diffusion-based generative model for cardiac MR tagging super resolution.
> Magnetic Resonance in Medicine. 2025 Mar 1.
