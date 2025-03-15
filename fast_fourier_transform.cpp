#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>


using ComplexType = std::complex<double>;


void fft(std::vector<ComplexType>& x, bool inverse = false) 
{
    const size_t N = x.size();
    if (N <= 1) return;


    std::vector<ComplexType> even(N / 2), odd(N / 2);
    for (size_t i = 0; i < N / 2; i++) 
    {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    fft(even, inverse);
    fft(odd, inverse);


    double angle_sign = inverse ? 1.0 : -1.0;
    for (size_t k = 0; k < N / 2; k++) 
    {
        double angle = angle_sign * 2.0 * M_PI * k / N;
        ComplexType t = std::polar(1.0, angle) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }


    if (inverse && N == x.size()) 
    {
        for (size_t i = 0; i < N; i++) 
        {
            x[i] /= N; 
        }
    }
}


void fft2D(std::vector<std::vector<ComplexType>>& image, bool inverse = false) 
{
    int rows = image.size();
    int cols = image[0].size();


    for (int r = 0; r < rows; r++) 
    {
        fft(image[r], inverse);
    }


    for (int c = 0; c < cols; c++) 
    {
        std::vector<ComplexType> column(rows);
        for (int r = 0; r < rows; r++) 
        {
            column[r] = image[r][c];
        }
        
        fft(column, inverse);
        
        for (int r = 0; r < rows; r++) 
        {
            image[r][c] = column[r];
        }
    }


    if (inverse) 
    {
        double factor = 1.0 / std::sqrt(rows * cols);
        for (int r = 0; r < rows; r++) 
        {
            for (int c = 0; c < cols; c++) 
            {
                image[r][c] *= factor;
            }
        }
    }
}


void applyLowPassFilter(std::vector<std::vector<ComplexType>>& frequencyDomain, double cutoffFrequency) 
{
    int rows = frequencyDomain.size();
    int cols = frequencyDomain[0].size();
    int centerY = rows / 2;
    int centerX = cols / 2;
    

    double radius = cutoffFrequency * std::min(centerX, centerY);
    
    for (int y = 0; y < rows; y++) 
    {
        for (int x = 0; x < cols; x++) 
        {
            int shiftedY = (y < centerY) ? y : y - rows;
            int shiftedX = (x < centerX) ? x : x - cols;
            double distance = std::sqrt(shiftedX * shiftedX + shiftedY * shiftedY);

            double filter = std::exp(-distance * distance / (2 * radius * radius));
            if (filter < 0.01) filter = 0; 
            
            frequencyDomain[y][x] *= filter;
        }
    }
}


void fftShift(std::vector<std::vector<ComplexType>>& image) {
    int rows = image.size();

    int cols = image[0].size();
    

    auto temp = image;

    for (int y = 0; y < rows; y++) 
    {
        for (int x = 0; x < cols; x++) 
        {
            int newY = (y + rows / 2) % rows;
            int newX = (x + cols / 2) % cols;
            image[newY][newX] = temp[y][x];
        }
    }
}


std::vector<std::vector<ComplexType>> matToComplex(const cv::Mat& input) 
{
    int rows = input.rows;
    int cols = input.cols;
    std::vector<std::vector<ComplexType>> result(rows, std::vector<ComplexType>(cols));
    
    for (int y = 0; y < rows; y++) 
    {
        for (int x = 0; x < cols; x++) 
        {

            result[y][x] = ComplexType(static_cast<double>(input.at<uchar>(y, x)), 0);
        }
    }
    
    return result;
}


cv::Mat complexToMat(const std::vector<std::vector<ComplexType>>& input) {
    int rows = input.size();
    int cols = input[0].size();
    cv::Mat result(rows, cols, CV_8UC1);
    
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::min();
    
    // First pass: find min and max values
    for (int y = 0; y < rows; y++) 
    {
        for (int x = 0; x < cols; x++)
         {
            double value = std::abs(input[y][x]);
            if (value < minVal) minVal = value;
            if (value > maxVal) maxVal = value;
        }
    }
    
    // Ensure we don't divide by zero
    if (maxVal == minVal) maxVal = minVal + 1;
    

    for (int y = 0; y < rows; y++) 
    {
        for (int x = 0; x < cols; x++) 
        {
            // Get the magnitude (we could also use real part but magnitude is safer)
            double value = std::abs(input[y][x]);
            
            // Normalize to range [0,255]
            value = 255.0 * (value - minVal) / (maxVal - minVal);
            
            result.at<uchar>(y, x) = cv::saturate_cast<uchar>(value);
        }
    }
    
    return result;
}


cv::Mat visualizeSpectrum(const std::vector<std::vector<ComplexType>>& frequencyDomain) 
{
    int rows = frequencyDomain.size();
    int cols = frequencyDomain[0].size();
    cv::Mat spectrum(rows, cols, CV_8UC1);
    
    double maxMagnitude = 0;
    
    // Find maximum magnitude for normalization
    for (int y = 0; y < rows; y++) 
    {
        for (int x = 0; x < cols; x++) 
        {
            double magnitude = std::abs(frequencyDomain[y][x]);
            if (magnitude > maxMagnitude) 
            {
                maxMagnitude = magnitude;
            }
        }
    }
    
    // Prevent division by zero
    if (maxMagnitude == 0) maxMagnitude = 1;
    
    // Convert to logarithmic scale and normalize
    for (int y = 0; y < rows; y++) 
    {
        for (int x = 0; x < cols; x++) 
        {
            double magnitude = std::abs(frequencyDomain[y][x]);
            // Add small value to prevent log(0)
            double logValue = std::log(1 + magnitude);
            spectrum.at<uchar>(y, x) = cv::saturate_cast<uchar>(255 * logValue / std::log(1 + maxMagnitude));
        }
    }
    
    return spectrum;
}

// Find the next power of 2
int nextPowerOf2(int n) 
{
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// Pad image to the nearest power of 2 in both dimensions
cv::Mat padImage(const cv::Mat& input) 
{
    int paddedRows = nextPowerOf2(input.rows);
    int paddedCols = nextPowerOf2(input.cols);
    
    cv::Mat padded;
    cv::copyMakeBorder(input, padded, 0, paddedRows - input.rows, 0, paddedCols - input.cols, 
                      cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    return padded;
}

int main() 
{

    cv::Mat image = cv::imread("Lena.png", cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) 
    {
        std::cout << "Error loading image" << std::endl;
        return -1;
    }
    
    int originalRows = image.rows;
    int originalCols = image.cols;
    
    std::cout << "Original image dimensions: " << originalRows << "x" << originalCols << std::endl;
    

    cv::imshow("Original Image", image);
    
    cv::Mat paddedImage = padImage(image);
    int rows = paddedImage.rows;
    int cols = paddedImage.cols;
    
    std::cout << "Padded image dimensions: " << rows << "x" << cols << std::endl;
    

    auto complexImage = matToComplex(paddedImage);
    

    fft2D(complexImage);
    

    auto complexImageCopy = complexImage;
    fftShift(complexImageCopy);
    

    cv::Mat spectrumImage = visualizeSpectrum(complexImageCopy);
    cv::imshow("Frequency Spectrum", spectrumImage);
    

    double cutoffFrequency = 0.05; 
    applyLowPassFilter(complexImage, cutoffFrequency);
    

    auto filteredCopy = complexImage;
    fftShift(filteredCopy);
    

    cv::Mat filteredSpectrumImage = visualizeSpectrum(filteredCopy);
    cv::imshow("Filtered Spectrum", filteredSpectrumImage);
    

    fft2D(complexImage, true);

    cv::Mat resultImage = complexToMat(complexImage);
    
    resultImage = resultImage(cv::Rect(0, 0, originalCols, originalRows));

    cv::imshow("Blurred Image", resultImage);
    

    cv::Mat opencvBlurred;
    int blurSize = static_cast<int>(std::max(5.0, 0.05 * std::min(originalRows, originalCols)));
    if (blurSize % 2 == 0) blurSize++; 
    cv::GaussianBlur(image, opencvBlurred, cv::Size(blurSize, blurSize), 0);
    cv::imshow("OpenCV Gaussian Blur", opencvBlurred);
    
    cv::waitKey(0);
    return 0;
}