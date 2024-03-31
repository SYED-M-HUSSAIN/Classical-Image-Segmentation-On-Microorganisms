import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def fast_marching_segmentation(inputFilename, outputFilename, seedX, seedY, Sigma, SigmoidAlpha, SigmoidBeta, TimeThreshold, StoppingTime):
    inputImage = sitk.ReadImage(inputFilename, sitk.sitkFloat32)

    smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
    smoothing.SetTimeStep(0.125)
    smoothing.SetNumberOfIterations(5)
    smoothing.SetConductanceParameter(9.0)
    smoothingOutput = smoothing.Execute(inputImage)

    gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradientMagnitude.SetSigma(Sigma)
    gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)

    sigmoid = sitk.SigmoidImageFilter()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(SigmoidAlpha)
    sigmoid.SetBeta(SigmoidBeta)
    sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)

    fastMarching = sitk.FastMarchingImageFilter()

    seedValue = 0
    trialPoint = (seedX, seedY, seedValue)

    fastMarching.AddTrialPoint(trialPoint)

    fastMarching.SetStoppingValue(StoppingTime)

    fastMarchingOutput = fastMarching.Execute(sigmoidOutput)

    thresholder = sitk.BinaryThresholdImageFilter()
    thresholder.SetLowerThreshold(0.0)
    thresholder.SetUpperThreshold(TimeThreshold)
    thresholder.SetOutsideValue(0)
    thresholder.SetInsideValue(255)

    result = thresholder.Execute(fastMarchingOutput)

    return result
