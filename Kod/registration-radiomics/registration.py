"""Rigid CT registration and multi-label mask resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_image(path: Path, pixel_type: int | None = None) -> sitk.Image:
    image = sitk.ReadImage(str(path))
    if pixel_type is not None:
        image = sitk.Cast(image, pixel_type)
    return image


def rigid_register(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
    initial_transform: sitk.Transform | None = None,
    metric: str = "correlation",
) -> sitk.Transform:
    """Estimate one global 3D Euler transform from moving to fixed space."""
    initial = (
        initial_transform
        if initial_transform is not None
        else sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    )

    method = sitk.ImageRegistrationMethod()
    if metric == "correlation":
        method.SetMetricAsCorrelation()
    elif metric == "mattes_mi":
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        raise ValueError(f"Unknown registration metric: {metric}")
    method.SetMetricSamplingStrategy(method.REGULAR)
    method.SetMetricSamplingPercentage(0.2)
    if fixed_mask is not None:
        method.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        method.SetMetricMovingMask(moving_mask)
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


def centroid_initial_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
) -> sitk.Euler3DTransform:
    """Initialize rigid registration from physical mask centroids."""
    stats_fixed = sitk.LabelShapeStatisticsImageFilter()
    stats_moving = sitk.LabelShapeStatisticsImageFilter()
    stats_fixed.Execute(sitk.Cast(fixed_mask > 0, sitk.sitkUInt8))
    stats_moving.Execute(sitk.Cast(moving_mask > 0, sitk.sitkUInt8))
    if not stats_fixed.HasLabel(1) or not stats_moving.HasLabel(1):
        return sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    fixed_centroid = np.asarray(stats_fixed.GetCentroid(1), dtype=float)
    moving_centroid = np.asarray(stats_moving.GetCentroid(1), dtype=float)
    center_index = [(size - 1) / 2.0 for size in fixed.GetSize()]
    transform = sitk.Euler3DTransform()
    transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(center_index))
    # SimpleITK resampling evaluates the moving image at transformed fixed
    # points, so the initializer uses the inverse physical shift.
    transform.SetTranslation(tuple(moving_centroid - fixed_centroid))
    return transform


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


def resample_for_registration(
    image: sitk.Image,
    spacing: float,
    interpolator: int,
) -> sitk.Image:
    """Create a lower-resolution registration-only copy in the same world space."""
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    size = [
        max(1, int(round((old_size[index] - 1) * old_spacing[index] / spacing)) + 1)
        for index in range(3)
    ]
    return sitk.Resample(
        image,
        size,
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        image.GetOrigin(),
        [spacing] * 3,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )
