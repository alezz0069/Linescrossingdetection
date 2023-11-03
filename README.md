
# Corrección Intersecciones

Jupyter notebook that uses image processing techniques to detect and analyze line segments in an P&ID.

## Overview

The notebook (`Correccion_intersecciones.ipynb`) performs the following tasks:

1. Load and preprocess an image.
2. Apply Canny edge detection for identifying edges. (Saves the image and CSV)
3. Use Hough Line Transformation to detect line segments. (Saves CSV)
4. Analyze and filter the detected segments based on their orientation and length. (Saves the image)

## Setup and Usage

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the setup script to install the necessary dependencies. This will install both system packages and Python libraries:
```bash
chmod +x setup.sh
./setup.sh

3. Launch Jupyter Notebook or Jupyter Lab and open the `Correccion_intersecciones.ipynb` notebook.

4. Follow the instructions in the notebook to run the code cells.

## Dependencies

- OpenCV (specifically opencv-python-headless==4.5.3.56 for headless environments)
- NumPy
- Pandas
- Matplotlib
- Poppler-utils (for pdf2image)
