# Line Segment Detection in Grayscale Images

This project implements a complete pipeline for detecting line segments in grayscale images using the **Canny edge detector** and **Hough Transform**. It was developed as part of a university project at the Technical University of Cluj-Napoca.

![Screenshot 2025-06-22 155815](https://github.com/user-attachments/assets/562ccd85-7186-40f0-816c-3edbc0d957f6)

![image](https://github.com/user-attachments/assets/3b0d6397-e354-436f-a0e3-9b81730c5db2)


## 📌 Features

- Gaussian filtering for noise reduction
- Custom implementation of the Canny edge detection algorithm
- Hough Transform for detecting straight lines
- Local maxima detection in the accumulator space
- Visualization of:
  - Input grayscale image
  - Edge-detected image
  - Detected lines overlayed on the input image
  - Hough accumulator with maxima

## 🧠 Algorithms Used

### 1. Canny Edge Detection

- **Gaussian Blur** (5x5 kernel) to reduce noise
- **Sobel Operator** to compute image gradients
- **Non-Maximum Suppression** to thin edges
- **Adaptive double thresholding** using parameters `p` and `k`
- **Edge tracking by hysteresis** (connecting weak edges to strong ones)

### 2. Hough Transform

- Parametrization using `ρ = x*cos(θ) + y*sin(θ)`
- Voting in an accumulator for each edge pixel
- Local maxima detection using a sliding window (`windowSize = 7`)
- Lines ranked by number of votes and top `N` lines selected

## 🛠️ Requirements

- C++17
- OpenCV 4.x
- Windows (uses `openFileDlg` for file dialog, which may be platform-specific)

## ▶️ How to Run

1. **Build the project** using a C++ compiler and link against OpenCV.
2. **Run the executable.**
3. **Choose a grayscale `.bmp` image** via the file dialog.
4. **Enter `N`**, the number of top lines to be displayed.
5. The following windows will appear:
   - Original image
   - Edges (after Canny detection)
   - Detected lines (green lines with magenta edge highlights)
   - Hough accumulator with red crosses on local maxima

## 🧪 Example

```bash
Enter N (number of lines to detect): 5
Top 5 local maxima (theta, rho, value):
(45, 150, 122)
(90, 123, 117)
...
