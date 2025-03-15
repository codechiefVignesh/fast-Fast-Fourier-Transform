#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <omp.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Function to zero-pad the kernel and arrange it for FFT (origin at corners)
void pad_kernel(double *kernel, double *padded_kernel, int k_size, int width, int height) 
{
    // Initialize padded kernel with zeros
    for (int i = 0; i < width * height; i++) 
    {
        padded_kernel[i] = 0.0;
    }

    // Place kernel in FFT-ready format (origin at corners, not center)
    for (int y = 0; y < k_size; y++) 
    {
        for (int x = 0; x < k_size; x++) 
        {
            int dst_y = (y < k_size/2 + 1) ? y : (height + y - k_size);
            int dst_x = (x < k_size/2 + 1) ? x : (width + x - k_size);
            
            padded_kernel[dst_y * width + dst_x] = kernel[y * k_size + x];
        }
    }
}

// Debug function to print min/max values
void print_minmax(unsigned char *image, int size, const char *label) 
{
    unsigned char min = 255, max = 0;
    for (int i = 0; i < size; i++) 
    {
        if (image[i] < min) min = image[i];
        if (image[i] > max) max = image[i];
    }
    printf("%s - Min: %d, Max: %d\n", label, min, max);
}

// Function to apply FFT-based convolution
void apply_filter_fft_parallel(unsigned char *image, int width, int height, int channels, double *kernel, int k_size, int num_threads) 
{
    int size = width * height;
    fftw_complex *in, *out, *kernel_fft, *image_fft;
    fftw_plan forward_plan, backward_plan;

    // Print input image stats for debugging
    print_minmax(image, size, "Input image");

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    image_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    kernel_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);

    omp_set_num_threads(num_threads);  // Set OpenMP thread count

    // Load image into FFTW input array
    #pragma omp parallel for
    for (int i = 0; i < size; i++) 
    {
        in[i][0] = (double)image[i];  
        in[i][1] = 0.0;               
    }

    // Create and execute forward FFT plan for image
    forward_plan = fftw_plan_dft_2d(height, width, in, image_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(forward_plan);

    // Zero-pad kernel and prepare for FFT
    double *padded_kernel = (double*) calloc(size, sizeof(double));
    pad_kernel(kernel, padded_kernel, k_size, width, height);

    // Load kernel into FFTW input array
    #pragma omp parallel for
    for (int i = 0; i < size; i++) 
    {
        in[i][0] = padded_kernel[i];
        in[i][1] = 0.0;
    }

    // Create and execute forward FFT plan for kernel
    fftw_plan kernel_plan = fftw_plan_dft_2d(height, width, in, kernel_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(kernel_plan);
    free(padded_kernel);

    // Multiply FFT(image) * FFT(kernel) in parallel
    #pragma omp parallel for
    for (int i = 0; i < size; i++) 
    {
        double real = image_fft[i][0] * kernel_fft[i][0] - image_fft[i][1] * kernel_fft[i][1];
        double imag = image_fft[i][0] * kernel_fft[i][1] + image_fft[i][1] * kernel_fft[i][0];
        out[i][0] = real;
        out[i][1] = imag;
    }

    // Perform inverse FFT
    backward_plan = fftw_plan_dft_2d(height, width, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(backward_plan);

    // Create a temporary buffer for the result
    unsigned char *result = (unsigned char*) malloc(size * sizeof(unsigned char));

    // Normalize and store result
    double min_val = 0, max_val = 0;
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < size; i++) 
    {
        double val = in[i][0] / size;  // Proper normalization for IFFT
        if (i == 0 || val < min_val) min_val = val;
        if (i == 0 || val > max_val) max_val = val;
    }
    
    printf("After IFFT - Min value: %f, Max value: %f\n", min_val, max_val);
    
    // Apply scaling and clamping
    #pragma omp parallel for
    for (int i = 0; i < size; i++) 
    {
        double val = in[i][0] / size;  // Proper normalization for IFFT
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        result[i] = (unsigned char)val;
    }
    
    // Print output image stats for debugging
    print_minmax(result, size, "Output image");
    
    // Copy result back to the original image buffer
    memcpy(image, result, size * sizeof(unsigned char));
    free(result);

    // Clean up FFTW resources
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(kernel_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(image_fft);
    fftw_free(kernel_fft);
}

int main(int argc, char *argv[]) 
{
    if (argc != 3) 
    {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    // Load as grayscale (force channels=1)
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!image) 
    {
        printf("Error loading image %s\n", argv[1]);
        return 1;
    }
    
    printf("Image loaded: %dx%d with %d channels (converted to grayscale)\n", 
           width, height, channels);

    // Gaussian Kernel (already normalized)
    // double kernel[25] = { 
    //     1/273.0,  4/273.0,  7/273.0,  4/273.0, 1/273.0, 
    //     4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0, 
    //     7/273.0, 26/273.0, 41/273.0, 26/273.0, 7/273.0, 
    //     4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0, 
    //     1/273.0,  4/273.0,  7/273.0,  4/273.0, 1/273.0
    // };

// Stronger Laplacian 3x3
    double kernel[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
};
    // double kernel[441] = {
    //     0,  0,  0,  0,  1,  1,  2,  2,  2,  3,  3,  2,  2,  2,  1,  1,  0,  0,  0,  0,  0,
    //     0,  0,  1,  1,  2,  3,  4,  5,  6,  7,  7,  6,  5,  4,  3,  2,  1,  1,  0,  0,  0,
    //     0,  1,  1,  2,  3,  5,  6,  8, 10, 11, 11, 10,  8,  6,  5,  3,  2,  1,  1,  0,  0,
    //     0,  1,  2,  3,  5,  7,  9, 11, 13, 15, 15, 13, 11,  9,  7,  5,  3,  2,  1,  0,  0,
    //     1,  2,  3,  5,  7, 10, 13, 16, 18, 20, 20, 18, 16, 13, 10,  7,  5,  3,  2,  1,  0,
    //     1,  3,  5,  7, 10, 14, 18, 22, 25, 27, 27, 25, 22, 18, 14, 10,  7,  5,  3,  1,  0,
    //     2,  4,  6,  9, 13, 18, 23, 28, 31, 33, 33, 31, 28, 23, 18, 13,  9,  6,  4,  2,  0,
    //     2,  5,  8, 11, 16, 22, 28, 33, 37, 39, 39, 37, 33, 28, 22, 16, 11,  8,  5,  2,  0,
    //     2,  6, 10, 13, 18, 25, 31, 37, 42, 44, 44, 42, 37, 31, 25, 18, 13, 10,  6,  2,  0,
    //     3,  7, 11, 15, 20, 27, 33, 39, 44, 47, 47, 44, 39, 33, 27, 20, 15, 11,  7,  3,  0,
    //     3,  7, 11, 15, 20, 27, 33, 39, 44, 47, -470, 44, 39, 33, 27, 20, 15, 11,  7,  3,  0,
    //     2,  6, 10, 13, 18, 25, 31, 37, 42, 44, 44, 42, 37, 31, 25, 18, 13, 10,  6,  2,  0,
    //     2,  5,  8, 11, 16, 22, 28, 33, 37, 39, 39, 37, 33, 28, 22, 16, 11,  8,  5,  2,  0,
    //     2,  4,  6,  9, 13, 18, 23, 28, 31, 33, 33, 31, 28, 23, 18, 13,  9,  6,  4,  2,  0,
    //     1,  3,  5,  7, 10, 14, 18, 22, 25, 27, 27, 25, 22, 18, 14, 10,  7,  5,  3,  1,  0,
    //     1,  2,  3,  5,  7, 10, 13, 16, 18, 20, 20, 18, 16, 13, 10,  7,  5,  3,  2,  1,  0,
    //     0,  1,  2,  3,  5,  7,  9, 11, 13, 15, 15, 13, 11,  9,  7,  5,  3,  2,  1,  0,  0,
    //     0,  1,  1,  2,  3,  5,  6,  8, 10, 11, 11, 10,  8,  6,  5,  3,  2,  1,  1,  0,  0,
    //     0,  0,  1,  1,  2,  3,  4,  5,  6,  7,  7,  6,  5,  4,  3,  2,  1,  1,  0,  0,  0,
    //     0,  0,  0,  0,  1,  1,  2,  2,  2,  3,  3,  2,  2,  2,  1,  1,  0,  0,  0,  0,  0
    // };

    // For testing: simple 3x3 box blur kernel
    // double kernel[9] = {
    //     1/9.0, 1/9.0, 1/9.0,
    //     1/9.0, 1/9.0, 1/9.0,
    //     1/9.0, 1/9.0, 1/9.0
    // };
    // int kernel_size = 3;

    int kernel_size = 3;
    
    printf("Using %dx%d kernel for convolution\n", kernel_size, kernel_size);

    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 20, 24, 32, 64};
    // int thread_counts[] = {1};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_tests; i++) 
    {
        int num_threads = thread_counts[i];
        
        // Make a copy of the original image for this test
        unsigned char *img_copy = (unsigned char*)malloc(width * height * sizeof(unsigned char));
        memcpy(img_copy, image, width * height * sizeof(unsigned char));
        
        double start_time = omp_get_wtime();
        apply_filter_fft_parallel(img_copy, width, height, 1, kernel, kernel_size, num_threads);
        double end_time = omp_get_wtime();
        
        printf("Threads: %d, Execution Time: %.6f seconds\n", num_threads, end_time - start_time);
        
        // Save result from the fastest run
        if (i == num_tests - 1) {
            printf("Saving final result to %s\n", argv[2]);
            if (!stbi_write_png(argv[2], width, height, 1, img_copy, width)) {
                printf("Error writing output image\n");
            }
        }
        
        free(img_copy);
    }

    stbi_image_free(image);
    return 0;
}