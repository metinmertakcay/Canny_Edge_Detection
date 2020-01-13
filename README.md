# Canny Edge Detection
## HOW TO RUN PROGRAM?
- Go to directory where you download the project.
- Open console screen and enter “python main.py” command to run.

## REQUIREMENT
You can install necessary libraries using “pip install”.
- Matplotlib
- Opencv
- Numpy

## HOW TO CHOOSE IMAGE?
You have to put all images in “images” folder because all images are taken using “os” library. You can enter number for select image.

## EDGE DETECTION
>OpenCv library is used to read image as grayscale (intensity values of the pixels are 8 bit and range from 0 to 255). After reading image, 5 x 5 gauss filter was used to eliminate noise in the image.

<p align="center">
	<img src="/output/gauss.JPG" alt="Gauss kernel" width="400" height="120">
</p>

> After noise reduction, gradients determined by using Sobel filter. Then magnitude and angle of the gradient are calculated.

<p align="center">
	<img src="/output/sobel.JPG" alt="Sobel" width="400" height="80">
</p>

> The following chart is used for non-maximum supression. The pixel value compared with neighboring pixels corresponding to the angle value. If the displayed pixel is larger than the neigboring pixels, the value is retained, otherwise the value is 0.

<p align="center">
	<img src="/output/chart.png" alt="Chart" width="300" height="300">
</p>

>Thresholding is used to identify the weak and strong edge. If pixel value is greater than high threshold, the edge is defined as strong. If pixel value is less than low threshold, the pixel value is 0. If the pixel value is between two threshold values, hysteresis method is applied.

> Hysteresis method is used to determine whether to erase weak edges. Whether or not the weak pixels are deleted is determined by looking at the surrounding pixels. If there is a strong edge at neigbours, the weak edge is defined as the strong edge, otherwise the pixel value is 0.

### OUTPUTS ###
[Click to see results.](/images)