#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
using namespace std;


//------------------------- Canny detection ---------------------------------


Mat applyGaussianFilter(const cv::Mat& input) {

    // gaussian kernel (5x5)
    float sigma = 0.8f;
    float kernel[5][5];
    int center = 2;
 
    // G(x,y) = 1/(pi*sigma^2) * exp(-((x-x0)^2 + (y-y0)^2)/sigma^2)
    float sum = 0.0f;
    for (int y = -center; y <= center; y++) {
        for (int x = -center; x <= center; x++) {
            float exponent = -(x * x + y * y) / (2 * sigma * sigma);
            kernel[y + center][x + center] = exp(exponent) / (2 * PI * sigma * sigma);
            sum += kernel[y + center][x + center];
        }
    }

    // normalize the kernel => sum of all elements is 1
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            kernel[i][j] /= sum;
        }
    }

    int height = input.rows;
    int width = input.cols;
    Mat output = input.clone();

    // apply convolution with border replication
    // pad because we are using a 5x5 kernel and it extends 2 pixels in all directions from the center
    int pad = 2;
    for (int y = pad; y < height - pad; y++) {
        for (int x = pad; x < width - pad; x++) {
            float sum = 0.0f;

            // loops over kernel rows from -2 to +2
            for (int ky = -pad; ky <= pad; ky++) {
                for (int kx = -pad; kx <= pad; kx++) {
                    // computes the weighted sum of the neighborhood
                    sum += input.at<uchar>(y + ky, x + kx) *
                        kernel[ky + pad][kx + pad];
                }
            }

            // stores the computed value in output at pixel (y, x)
            // saturate_cast -> clips sum to the range [0, 255]
            output.at<uchar>(y, x) = saturate_cast<uchar>(sum);
        }
    }

    return output;
}


tuple<Mat, Mat, Mat> applySobel(const Mat& input) {
    // sobel kernels
    // detects horizontal and vertical edges
    int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobelY[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    // initialize gradient matrices: horizontal, vertical, and magnitude
    Mat gradX = Mat::zeros(input.size(), CV_32F);
    Mat gradY = Mat::zeros(input.size(), CV_32F);
    Mat magnitude = Mat::zeros(input.size(), CV_32F);

    int height = input.rows;
    int width = input.cols;

    // apply convolution with border replication
    // pad because we are using a 3x3 kernel and it extends 1 pixel in all directions from the center
    int pad = 1;
    for (int y = pad; y < height - pad; y++) {
        for (int x = pad; x < width - pad; x++) {
            // compute gradients using Sobel kernels
            // loops over kernel rows from -1 to +1
            // gx and gy are the gradients in x and y directions
            float gx = 0, gy = 0;
            for (int ky = -pad; ky <= pad; ky++) {
                for (int kx = -pad; kx <= pad; kx++) {
                    // computes the weighted sum of the neighborhood
                    gx += input.at<uchar>(y + ky, x + kx) * sobelX[ky + pad][kx + pad];
                    gy += input.at<uchar>(y + ky, x + kx) * sobelY[ky + pad][kx + pad];
                }
            }
            gradX.at<float>(y, x) = gx;
            gradY.at<float>(y, x) = gy;

            // compute the magnitude of the gradient by formula sqrt(gx^2 + gy^2)
            magnitude.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        }
    }


    return make_tuple(gradX, gradY, magnitude);
}


Mat applyNonMaxSuppression(const Mat& gradMag, const Mat& gradX, const Mat& gradY) {

    Mat suppressed = Mat::zeros(gradMag.size(), CV_32F);

    int height = gradMag.rows;
    int width = gradMag.cols;

    // loop through each pixel in the gradient magnitude image
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            float val = gradMag.at<float>(y, x);
            bool isMax = true;

            // check the gradient direction
            float angle = atan2(gradY.at<float>(y, x), gradX.at<float>(y, x)) * 180 / PI;
            if (angle < 0) angle += 180;

            // determine the neighboring pixels to compare based on the gradient direction
            float p1, p2;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                // Horizontal edge (0 degrees)
                p1 = gradMag.at<float>(y, x - 1);
                p2 = gradMag.at<float>(y, x + 1);
            }
            else if (angle >= 22.5 && angle < 67.5) {
                // Diagonal edge (45 degrees)
                p1 = gradMag.at<float>(y + 1, x - 1);
                p2 = gradMag.at<float>(y - 1, x + 1);
            }
            else if (angle >= 67.5 && angle < 112.5) {
                // Vertical edge (90 degrees)
                p1 = gradMag.at<float>(y - 1, x);
                p2 = gradMag.at<float>(y + 1, x);
            }
            else { // angle >= 112.5 && angle < 157.5
                // Diagonal edge (135 degrees)
                p1 = gradMag.at<float>(y - 1, x - 1);
                p2 = gradMag.at<float>(y + 1, x + 1);
            }

            // compare with neighbors
            if (val < p1 || val < p2) {
                isMax = false;
            }

            // if the current pixel is a local maximum, keep it; otherwise, suppress it
            if (isMax) {
                suppressed.at<float>(y, x) = val;
            }
           
        }
    }
    return suppressed;
}


pair<int, int> adaptiveThresholding(Mat& gradMag, double p, double k) {

    //histogram
    int histogram[256] = { 0 };
    int totalPixels = 0;

    int height = gradMag.rows;
    int width = gradMag.cols;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uchar val = gradMag.at<uchar>(i, j);

           // val = val / (4 * sqrt(2));
            histogram[val]++;

            if (val > 0) {
                totalPixels++;
            }
        }
    }

    //number of non-edge pixels
    int nonEdgePixels = (1.0 - p) * totalPixels;

    //find adaptive threshold
    int sum = 0;
    int highThreshold = 0;

    for (int i = 1; i <= 255; i++) {
        sum += histogram[i];

        if (sum > nonEdgePixels) {
            highThreshold = i;
            break;
        }
    }

    int lowThreshold = k * highThreshold;

    return make_pair(highThreshold, lowThreshold);
}


Mat connectEdges(Mat& edgeMat) {

    int height = edgeMat.rows;
    int width = edgeMat.cols;

    queue<Point> queue;

    //first pas
    //find all strong edges and put them in the queue
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            //if i find a strong edge
            if (edgeMat.at<uchar>(i, j) == 255) {
                //add it to the queue
                queue.push(Point(j, i));
            }
        }
    }

    //8-neighborhood direction
    int dx8[] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    int dy8[] = { 0, -1, -1, -1, 0, 1, 1, 1 };

    //process queue
    while (!queue.empty()) {

        //get first point from the queue
        Point p = queue.front();
        queue.pop();

        //find all its weak edge neighbours
        for (int i = 0; i < 8; i++) {
            int nx = p.x + dx8[i];
            int ny = p.y + dy8[i];

            // check bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {

                // if neighbor is a weak edge, make it a strong edge
                if (edgeMat.at<uchar>(ny, nx) == 128) {
                  
                    // convert to strong edge
                    edgeMat.at<uchar>(ny, nx) = 255;
                    // add to queue to extend further
                    queue.push(Point(nx, ny)); 
                }
            }
        }
    }


    //second pass: remove all weak edges
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (edgeMat.at<uchar>(i, j) == 128) {
                //remove if weak edge
                edgeMat.at<uchar>(i, j) = 0;
            }
        }
    }

    return edgeMat;
}


Mat applyHysteresis(Mat& gradMag, int highThreshold, int lowThreshold) {
    //edge map
    Mat edgeMat = Mat::zeros(gradMag.size(), CV_8U);


    int height = gradMag.rows;
    int width = gradMag.cols;

    //mark strong and weak edges
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uchar val = gradMag.at<uchar>(i, j);

            //strong edge
            if (val >= highThreshold) {
                edgeMat.at<uchar>(i, j) = 255;
            }

            //weak edge
            else if (val >= lowThreshold) {
                edgeMat.at<uchar>(i, j) = 128;
            }

            //if weak edge it remains 0 -> eliminated
        }
    }

    //connect weak edges to strong edges
    connectEdges(edgeMat);

    return edgeMat;
}


Mat cannyEdgeDetection(Mat& input, double p, double k) {
    // Step 1: apply gaussian filter 
    Mat blurred = applyGaussianFilter(input);

    // Step 2: apply sobel operator 
    auto result = applySobel(blurred);
    Mat gradX = std::get<0>(result);
    Mat gradY = std::get<1>(result);
    Mat gradMag = std::get<2>(result);
    

    // Step 3: non-maximum suppression
    Mat suppressed = applyNonMaxSuppression(gradMag, gradX, gradY);

    // create a copy for thresholding
    Mat gradMag8U;
    //normalize(suppressed, suppressed, 0, 255, NORM_MINMAX);

    suppressed.convertTo(gradMag8U, CV_8U);

    // Step 4: double threshold with hysteresis
    auto thresholds = adaptiveThresholding(gradMag8U, p, k);
    int highThreshold = thresholds.first;
    int lowThreshold = thresholds.second;

   
    //Step 5: hysteresis to connect edges
    Mat edges = applyHysteresis(gradMag8U, highThreshold, lowThreshold);


    return edges;
}




//---------------------------- Hough Transform ---------------------------

Mat drawLines(Mat& src, vector<Vec3i> lines, Mat& edges) {
    Mat result;
    cvtColor(src, result, COLOR_GRAY2BGR);

    int height = src.rows;
    int width = src.cols;

    // for each line in lines
    for (const Vec3i& line : lines) {
        // line[0] - vote count
        // line[1] - rho
        // line[2] - theta
        int rho = line[1];
        int theta = line[2];

        //convert to radians
        double thetaRad = theta * CV_PI / 180.0;
        double cosTheta = cos(thetaRad);
        double sinTheta = sin(thetaRad);

        Point pt1, pt2;

        //fabs = float abs
        // if line is almost vertical - sin ~ 0
        if (fabs(sinTheta) < 1e-10) {
            pt1 = Point(rho, 0);
            pt2 = Point(rho, height);
        }
        else {
            double x0 = rho * cosTheta;
            double y0 = rho * sinTheta;

            // alpha = offset so that the line is long enough to cover the image
            double alpha = 1000;

            pt1 = Point(cvRound(x0 + alpha * (-sinTheta)),
                cvRound(y0 + alpha * cosTheta));
            pt2 = Point(cvRound(x0 - alpha * (-sinTheta)),
                cvRound(y0 - alpha * cosTheta));
        }

        // draw green line
        cv::line(result, pt1, pt2, Scalar(0, 255, 0), 1);

        // color in magenta the pixels in canny
        // grabs pixels from line p1 p2, 8-bit, 3 channel image
        LineIterator it(result, pt1, pt2, 8);
        for (int j = 0; j < it.count; ++j, ++it) {

            // pos returns coordinates of the current pixel
            Point p = it.pos();

            //check if its inside image bounds
            if (p.y >= 0 && p.y < edges.rows && p.x >= 0 && p.x < edges.cols) {

                //check if it corresponds to an edge pixel from canny
                if (edges.at<uchar>(p) > 0) {
                    result.at<Vec3b>(p) = Vec3b(255, 0, 255); // color magenta
                }
            }
        }
    }

    return result;
}


Mat houghTransform(Mat& edges, Mat& src, int windowSize, int threshold, int numLines) {
    int height = edges.rows;
    int width = edges.cols;

    // calculate the diagonal length of the image
    int diagonalLength = cvRound(sqrt(height * height + width * width));

    // create accumulator (θ in [0, 359), ρ in [0, diagonalLength])
    Mat accumulator = Mat::zeros(360, diagonalLength + 1, CV_32S);

    // fill the accumulator
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // process only edge pixels
            if (edges.at<uchar>(y, x) > 0) {

                // for each possible θ value (0 to 359 degrees)
                for (int theta = 0; theta < 360; theta++) {
                    // convert theta to radians
                    double thetaRad = theta * CV_PI / 180.0;

                    // calculate ρ = x*cos(θ) + y*sin(θ)
                    int rho = cvRound(x * cos(thetaRad) + y * sin(thetaRad));

                    // ensure rho is within the valid range
                    if (rho >= 0 && rho <= diagonalLength) {

                        // increment the accumulator
                        accumulator.at<int>(theta, rho)++;
                    }
                }
            }
        }
    }

    Mat accDisplay;
    normalize(accumulator, accDisplay, 0, 255, NORM_MINMAX);
    accDisplay.convertTo(accDisplay, CV_8U);
    cvtColor(accDisplay, accDisplay, COLOR_GRAY2BGR);


    // find local maxima in the accumulator
    vector<Vec3i> lines; // (value, rho, theta)

    // half window size for local maxima detection
    int halfWindow = windowSize / 2;

    for (int theta = 0; theta < 360; theta++) {
        for (int rho = 0; rho <= diagonalLength; rho++) {
            int value = accumulator.at<int>(theta, rho);

            // skip if below threshold
            if (value < threshold) continue;

            bool isLocalMax = true;

            // check if it's a local maximum in a windowSize x windowSize window
            for (int dTheta = -halfWindow; dTheta <= halfWindow && isLocalMax; dTheta++) {
                for (int dRho = -halfWindow; dRho <= halfWindow && isLocalMax; dRho++) {
                    // skip the center point
                    if (dTheta == 0 && dRho == 0) continue;

                    // wrap around for theta (circular)
                    int checkTheta = (theta + dTheta + 360) % 360;
                    int checkRho = rho + dRho;

                    // check bounds for rho
                    if (checkRho >= 0 && checkRho <= diagonalLength) {
                        // if there's a larger value in the window, this is not a local max
                        if (accumulator.at<int>(checkTheta, checkRho) > value) {
                            isLocalMax = false;
                        }
                    }
                }
            }

            // if it's a local maximum and above threshold, add it to lines
            if (isLocalMax) {
                lines.push_back(Vec3i(value, rho, theta));
            }
        }
    }

    // sort lines by accumulator value (highest first)
    sort(lines.begin(), lines.end(),
        [](const Vec3i& a, const Vec3i& b) { return a[0] > b[0]; });

    // keep only the top numLines lines
    if (lines.size() > numLines) {
        lines.resize(numLines);
    }


    // Step 4: Draw crosses on accumulator image
    for (const auto& line : lines) {
        int rho = line[1], theta = line[2];
        Point center(theta, rho);
        cv::line(accDisplay, Point(theta - 3, rho), Point(theta + 3, rho), Scalar(0, 0, 255), 1);
        cv::line(accDisplay, Point(theta, rho - 3), Point(theta, rho + 3), Scalar(0, 0, 255), 1);
    }

    imshow("Hough Accumulator", accDisplay);

    cout << "Top " << lines.size() << " local maxima (theta, rho, value):" << endl;
    for (const auto& line : lines) {
        cout << "(" << line[2] << ", " << line[1] << ", " << line[0] << ")\n";
    }

    Mat result = drawLines(src, lines, edges);

    return result;
}




//--------------------------- Main ---------------------------------------

int main() {

    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat image = imread(fname, IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Could not open image!" << std::endl;
            return -1;
        }

        int N;
        cout << "Enter N (number of lines to detect): ";
        cin >> N;

        // Step 1: Edge detection
        Mat edges = cannyEdgeDetection(image, 0.08, 0.4);

        // Step 2: Line detection using Hough transform
        Mat result = houghTransform(edges, image, 7, 10, N);

        imshow("Original image", image);
        imshow("Edges", edges);
        imshow("Detected lines", result);
        waitKey(0);
    }

    return 0;
}