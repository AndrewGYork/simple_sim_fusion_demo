# simple_sim_fusion_demo
Simple demo of structured illumination microscopy [image fusion via Richardson-Lucy deconvolution](https://www.ncbi.nlm.nih.gov/pubmed/24436314). To run the code yourself, download [`sim_fusion.py`](https://github.com/AndrewGYork/simple_sim_fusion_demo/blob/master/sim_fusion.py) and [`np_tif.py`](https://github.com/AndrewGYork/simple_sim_fusion_demo/blob/master/np_tif.py), put them in the same directory, and execute them in a [Python 3 environment that includes Numpy and Scipy](https://www.scipy.org/install.html). To see the results of running the code, scroll down.

Given a 2D x-z object:

<img src="./images/1_true_density.png" alt="True density" width="200">

Illuminated with a series of 2D x-z intensity patterns like this:

<img src="./images/5_illumination_intensity.gif" alt="Illumination" width="200">

And blurred with a 2D x-z PSF like this:

<img src="./images/3_psf_intensity.png" alt="Point spread function" width="200">

Yielding simulated data like this:

<img src="./images/6_noisy_measurement.gif" alt="Measurement" width="200">

We process this simulated data into an estimate of the true density (truth is in red, estimate is in green):

<img src="./images/8_final_estimate.png" alt="Estimate vs. truth" width="200">

Via iterative Richardson-Lucy deconvolution:

<img src="./images/7_estimate_history.gif" alt="Iterative convergence" width="200">

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

