#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
using namespace std;

#define WIDTH 4096
#define HEIGHT 4096
#define MaxRGB 256

typedef struct {
        int red;
        int green;
        int blue;
} RGB;

typedef struct {
        RGB* image;
        int width;
        int height;
} Mandelbrot;

__constant__ double ci_s[HEIGHT];
__constant__ double cr_s[WIDTH];

__global__ void kernel(Mandelbrot mandelbrot) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row > mandelbrot.height || col > mandelbrot.width) return;

        int index = row * mandelbrot.width + col;
        int i = 0;
        double zr = 0.0;
        double zi = 0.0;
        const int maxIterations = 500;

        while (i < maxIterations && zr * zr + zi * zi < 4.0) {
                double fz = zr * zr - zi * zi + cr_s[col];
                zi = 2.0 * zr * zi + ci_s[row];
                zr = fz;
                i++;
        }

        int r, g, b;
        int maxRGB = 256;
        int max3 = maxRGB;
        double t = (double)i / (double)maxIterations;
        i = (int)(t * (double)max3);  // mapez la 0...255
        b = i / (maxRGB * maxRGB);
        r = (i - b * maxRGB) / maxRGB;
        g = (i - b * maxRGB) - r * maxRGB;
        mandelbrot.image[index].red = r;
        mandelbrot.image[index].green = g;
        mandelbrot.image[index].blue = b;
}

int getValues(double* c, int state, double beginRange, double endRange, double minVal, double maxVal) {
        if (state < endRange) {
                c[state] = ((state - beginRange) / (endRange - beginRange))*(maxVal - minVal) + minVal;
                return getValues(c, state + 1, beginRange, endRange, minVal, maxVal);
        }
        else return 0;
}

void mandelbrotSet(Mandelbrot mandelbrot, double* cr, double* ci)
{
        int width = mandelbrot.width;
        int height = mandelbrot.height;

        Mandelbrot mandelbrot_d;
        mandelbrot_d.width = width;
        mandelbrot_d.height = height;
        size_t  mandelbrotSize = width * height * sizeof(RGB);

        size_t CRealSize = width * sizeof(double);
        size_t CImagSize = height * sizeof(double);

        cudaSetDevice(0);

        cudaMalloc((void**)&mandelbrot_d.image, mandelbrotSize);

        cudaMemcpyToSymbol(cr_s, cr, CRealSize);
        cudaMemcpyToSymbol(ci_s, ci, CImagSize);

        dim3 numBlocks(128, 128);
        dim3 threadsPerBlock(32, 32);

        clock_t begin = clock();
        kernel <<<numBlocks, threadsPerBlock>>> (mandelbrot_d);
        cudaDeviceSynchronize();
        clock_t end = clock();
        printf("Duration (kernel & synchronize): %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

        cudaMemcpy(mandelbrot.image, mandelbrot_d.image, mandelbrotSize, cudaMemcpyDeviceToHost);
        cudaFree(mandelbrot_d.image);
}

int main()
{
        Mandelbrot mandelbrot;
        double *cr, *ci;

        int width = WIDTH;
        int height = HEIGHT;
        int maxRGB = MaxRGB;

        mandelbrot.width = width;
        mandelbrot.height = height;
        mandelbrot.image = (RGB*)malloc(width * height * sizeof(RGB));

        cr = (double*)malloc(width * sizeof(double));
        ci = (double*)malloc(height * sizeof(double));

        getValues(cr, 0, 0, width, -2, 1); // cr, state, begin, end, min, max
        getValues(ci, 0, 0, height, -1.5, 1.5); // ci, state, begin, end, min, max

        mandelbrotSet(mandelbrot, cr, ci);

        printf("Generating image...\n");
        ofstream fout("output_image.ppm");
        fout << "P3" << endl;
        fout << mandelbrot.width << " " << mandelbrot.height << endl;
        fout << maxRGB << endl;
        for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                        int index = h * width + w;
                        fout << mandelbrot.image[index].red << " " << mandelbrot.image[index].green << " " << mandelbrot.image[index].blue << " ";
                }
                fout << endl;
        }
        fout.close();
        printf("Succes!\n");
        return 0;
}
