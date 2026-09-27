"""Rigid CT registration and multi-label mask resampling."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk


def read_image(path: Path, pixel_type: int | None = None) -> sitk.Image:
    image = sitk.ReadImage(str(path))
    if pixel_type is not None:
        image = sitk.Cast(image, pixel_type)
    return image


def rigid_register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """Estimate one global 3D Euler transform from moving to fixed space."""
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=150,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    return method.Execute(fixed, moving)


def resample_image(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    return sitk.Resample(
        image,
        reference,
        transform,
        interpolator,
        default_value,
        image.GetPixelID(),
    )


def resample_mask(
    mask: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    """Resample a label map without creating fractional labels."""
    return resample_image(
        mask,
        reference,
        transform,
        sitk.sitkNearestNeighbor,
        default_value=0,
    )
