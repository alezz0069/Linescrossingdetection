
# Corrección Intersecciones

This repository contains scripts that use image processing techniques to detect and analyze line segments in a P&ID.

## Overview

1. `Correccion_intersecciones.py`: This script loads and preprocesses an image, applies Canny edge detection for identifying edges, uses the Hough Line Transformation to detect line segments, and then analyzes and filters the detected segments based on their orientation and length.
2. `pdf_to_png_converter.py`: A utility script to convert the first page of a PDF file to a PNG image.
3. `Correccion_intersecciones.ipynb`: A Jupyter notebook that demonstrates how to use the scripts for image processing. This serves as an example of how the scripts can be used.

## Setup and Usage

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. If you are running the code on a system without a pre-installed version of Poppler, you can set it up using:
   ```bash
   ./setup.sh
   ```

4. Use the scripts as required. For instance:
   ```bash
   python Correccion_intersecciones.py
   ```

5. Alternatively, launch Jupyter Notebook or Jupyter Lab and open the `Correccion_intersecciones.ipynb` notebook to see an example of how the scripts can be used.

## Dependencies

- OpenCV (specifically `opencv-python-headless==4.5.3.56` for headless environments)
- NumPy
- Pandas
- Matplotlib
