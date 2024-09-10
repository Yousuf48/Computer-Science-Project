# Training and Sampling using PixelCNN.
Note: the code only runs on a GPU with PyTorch framwork installed.

The project has two directories, one for training and the other for sampling.

### The "training" directory contains nine models to train, included below:
* '**colourisation_rgb.py**': for RGB colourisation with the additional channel.
* '**colourisation_hsl.py**': for HSL colourisation without the additional channel. 
* '**colourisation_hsl_add.py**': for HSL colourisation without the additional channel. 
* '**colourisation_rg.py**': for RG colourisation without the additional channel. 
* '**colourisation_rg_add.py**': for RG colourisation without the additional channel. 
* '**denoising.py**': for denoising low-level noise images in MNIST datasets. 
* '**denoising_fashion.py**': for denoising low-level noise images in Fashion MNIST datasets. 
* '**denoising_more_level.py**': for denoising higher-level noise images in Fashion MNIST datasets.  


### The "sampling" directory contains eight models to sample from with metrics evaluation, included below:
* '**denoising_samplingl.py**': produces denoise samples using two trained models on low-level noise.
* '**denoising_sampling_more_level.py**': produces denoise samples using two trained models on low-level noise.
* '**rgb_sampling.py**' produces colourised samples using the trained model in RGB colour space.
* '**rgb_add_sampling.py**': produces colourised samples using the trained model in RGB colour space with the additional channel.
* '**hsl_sampling.py**': produces colourised samples using the trained model in HSL colour space.
* '**hsl_sampling_add.py**':  produces colourised samples using the trained model in HSL colour space with the additional channel.
* '**rg_sampling.py**': produces colourised samples using the trained model in RG.
* '**rg_sampling_add.py**': produces colourised samples using the trained model in RG with the additional channel.


To run any Python file, navigate to the correct directory and run
``` python3 <name of the files> ``` or ``` python <name of the files> ```



[Project's report](./Final%20Report.pdf)