import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Function for performing Fast Marching segmentation
def fast_marching_segmentation(inputFilename, outputFilename, seedX, seedY, Sigma, SigmoidAlpha, SigmoidBeta, TimeThreshold, StoppingTime):
    # Read the input image
    inputImage = sitk.ReadImage(inputFilename, sitk.sitkFloat32)

    # Apply smoothing using Curvature Anisotropic Diffusion
    smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
    smoothing.SetTimeStep(0.125)
    smoothing.SetNumberOfIterations(5)
    smoothing.SetConductanceParameter(9.0)
    smoothingOutput = smoothing.Execute(inputImage)

    # Calculate gradient magnitude using Recursive Gaussian
    gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradientMagnitude.SetSigma(Sigma)
    gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)

    # Apply sigmoid transformation
    sigmoid = sitk.SigmoidImageFilter()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(SigmoidAlpha)
    sigmoid.SetBeta(SigmoidBeta)
    sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)

    # Initialize Fast Marching
    fastMarching = sitk.FastMarchingImageFilter()

    # Add seed point for Fast Marching
    seedValue = 0
    trialPoint = (seedX, seedY, seedValue)
    fastMarching.AddTrialPoint(trialPoint)

    # Set stopping value for Fast Marching
    fastMarching.SetStoppingValue(StoppingTime)

    # Execute Fast Marching
    fastMarchingOutput = fastMarching.Execute(sigmoidOutput)

    # Apply binary thresholding
    thresholder = sitk.BinaryThresholdImageFilter()
    thresholder.SetLowerThreshold(0.0)
    thresholder.SetUpperThreshold(TimeThreshold)
    thresholder.SetOutsideValue(0)
    thresholder.SetInsideValue(255)

    # Threshold the result
    result = thresholder.Execute(fastMarchingOutput)

    # Return the segmented result
    return result

