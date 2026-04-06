# FFT (Fast Fourier Transform) Examples

This directory contains examples demonstrating the Fast Fourier Transform (FFT) and its applications in various domains.

## Overview

The Fast Fourier Transform (FFT) is an efficient algorithm to compute the Discrete Fourier Transform (DFT) and its inverse. It is widely used in signal processing, image processing, data compression, and many other fields.

## Files Description

### [basic_fft_example.py](file:///d:/git/py_process/fft/basic_fft_example.py)
A basic example showing how to perform FFT on simple synthetic signals. This example demonstrates:
- Creating composite signals from multiple sine waves
- Performing FFT to transform signals from time domain to frequency domain
- Visualizing amplitude and phase spectra
- Interpreting FFT results

### [audio_fft_example.py](file:///d:/git/py_process/fft/audio_fft_example.py)
An example showing FFT applications in audio signal processing. This example covers:
- Creating sample audio signals with multiple frequencies
- Analyzing audio spectrum using FFT
- Visualizing frequency components in linear and logarithmic scales
- Identifying dominant frequencies in audio signals

### [image_fft_example.py](file:///d:/git/py_process/fft/image_fft_example.py)
An example showing FFT applications in image processing. This example includes:
- Converting images to frequency domain using 2D FFT
- Visualizing image frequency spectrum
- Applying different types of filters in frequency domain:
  - Ideal Low Pass Filter
  - Butterworth Low Pass Filter
  - Gaussian Low Pass Filter
- Comparing filtering effects in both frequency and spatial domains

### [fft_principle_demo.py](file:///d:/git/py_process/fft/fft_principle_demo.py)
A demonstration of FFT principles and advanced concepts:
- Basic FFT transformation process
- Signal reconstruction from frequency components
- Effects of different numbers of frequency components on signal reconstruction
- Impact of window functions on FFT results:
  - Rectangular window
  - Hann window
  - Hamming window

## Requirements

To run these examples, you'll need the following Python packages:
```
numpy
matplotlib
scipy
pillow
opencv-python
```

You can install them using pip:
```bash
pip install numpy matplotlib scipy pillow opencv-python
```

## Usage

Run any example directly with Python:
```bash
python basic_fft_example.py
python audio_fft_example.py
python image_fft_example.py
python fft_principle_demo.py
```

Each example is self-contained and will display visualizations of the results.

## Theory

The Fourier Transform converts a signal from its original domain (often time or space) to a representation in the frequency domain and vice versa. The Fast Fourier Transform (FFT) is an algorithm that computes the Discrete Fourier Transform (DFT) in O(N log N) time, compared to O(N²) for the direct computation.

### Key Concepts:
1. **Time Domain vs Frequency Domain**: Signals can be represented either as variations of amplitude over time or as a combination of frequencies with different amplitudes and phases.
2. **Complex Numbers**: FFT results are generally complex numbers, where the magnitude represents amplitude and the angle represents phase.
3. **Sampling Rate and Frequency Resolution**: The sampling rate determines the maximum detectable frequency, while the total observation time determines the frequency resolution.
4. **Windowing**: Applied to reduce spectral leakage when analyzing finite-length signals.
5. **Filtering**: Operations performed in the frequency domain can achieve efficient filtering compared to convolution in the time domain.

## Applications

- **Signal Processing**: Audio processing, telecommunications, radar systems
- **Image Processing**: Filtering, compression, feature extraction
- **Data Analysis**: Spectral analysis, pattern recognition
- **Engineering**: Vibration analysis, structural analysis
- **Physics**: Quantum mechanics, optics, acoustics

These examples provide a foundation for understanding and applying FFT in various practical scenarios.