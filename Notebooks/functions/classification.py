import warnings
import numpy as np
from rasterio.features import rasterize
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf

def extract_training_samples(raster_src, training_gdf, class_field="class"):
    """
    Extract training pixels from polygon training samples.

    Parameters
    ----------
    raster_src : rasterio.DatasetReader
        Open rasterio dataset.

    training_gdf : geopandas.GeoDataFrame
        Polygon training samples in the same CRS as the raster.

    class_field : str, default="class"
        Name of the attribute containing the class labels.

    Returns
    -------
    X : ndarray of shape (n_pixels, n_bands)
        Spectral values of all extracted pixels.

    y : ndarray of shape (n_pixels,)
        Corresponding class labels.
    """

    # Check that the class field exists
    if class_field not in training_gdf.columns:
        raise ValueError(f"Field '{class_field}' not found in training polygons.")

    # Check that raster and polygons use the same coordinate reference system
    if training_gdf.crs != raster_src.crs:
        raise ValueError("Training polygons and raster must have the same CRS.")

    # Read the raster once (all bands)
    raster_data = raster_src.read()

    samples = []
    labels = []

    # Iterate over each training polygon
    for idx, row in training_gdf.iterrows():

        geometry = row.geometry
        label = row[class_field]

        # Rasterize the current polygon to create a binary mask
        mask = rasterize(
            [(geometry, 1)],
            out_shape=(raster_src.height, raster_src.width),
            transform=raster_src.transform,
            fill=0,
            dtype="uint8",
        )

        mask = mask.astype(bool)

        # Skip polygons that do not overlap the raster
        if not mask.any():
            warnings.warn(
                f"ROI {idx} produced an empty mask on the raster extent. Skipping."
            )
            continue

        # Extract spectral values for all pixels inside the polygon
        pixels = raster_data[:, mask].T

        # Remove NoData pixels, if present
        if raster_src.nodata is not None:
            valid = ~(pixels == raster_src.nodata).any(axis=1)
            pixels = pixels[valid]

        # Skip polygons containing only NoData pixels
        if pixels.size == 0:
            warnings.warn(
                f"No valid pixels found in ROI {idx} after NoData filtering. Skipping."
            )
            continue

        samples.append(pixels)
        labels.append(np.full(pixels.shape[0], label))

    # Ensure that at least one training sample has been extracted
    if len(samples) == 0:
        raise RuntimeError(
            "No training pixels extracted. Check CRS and geometry extents."
        )

    X = np.vstack(samples)
    y = np.concatenate(labels)

    return X, y

from sklearn.covariance import LedoitWolf

def estimate_class_statistics(X, y, add_cov_epsilon=1e-6):
    """
    Estimate the statistical parameters required for Maximum Likelihood
    Classification.

    Parameters
    ----------
    X : ndarray of shape (n_pixels, n_bands)
        Spectral values of the training pixels.

    y : ndarray of shape (n_pixels,)
        Class labels corresponding to the training pixels.

    add_cov_epsilon : float, default=1e-6
        Small value added to the diagonal of each covariance matrix to
        improve numerical stability.

    Returns
    -------
    classes_info : dict
        Dictionary containing the mean vector and covariance matrix for
        each class. Each entry has the form:

            {
                class_label: {
                    "mean": ndarray,
                    "cov": ndarray
                }
            }
    """

    classes_info = {}

    # Estimate statistics independently for each class
    for cls in np.unique(y):

        # Select the training pixels belonging to the current class
        class_pixels = X[y == cls]

        # Compute the mean spectral signature
        mean = np.mean(class_pixels, axis=0)

        # Estimate the covariance matrix using the Ledoit-Wolf shrinkage estimator
        covariance = LedoitWolf().fit(class_pixels).covariance_

        # Add a small value to the diagonal to improve numerical stability
        covariance += np.eye(covariance.shape[0]) * add_cov_epsilon

        classes_info[cls] = {
            "mean": mean,
            "cov": covariance,
        }

    return classes_info

def maximum_likelihood_classifier(image, classes_info, nodata_value=None):
    """
    Classify a multispectral image using the Maximum Likelihood Classifier (MLC).

    Parameters
    ----------
    image : ndarray of shape (n_bands, height, width)
        Multispectral image.

    classes_info : dict
        Dictionary containing the mean vector and covariance matrix for each class.
        Each entry must have the form:
            {
                class_label: {
                    "mean": ndarray,
                    "cov": ndarray
                }
            }

    nodata_value : float or int, optional
        NoData value in the input image. If provided, pixels containing NoData
        in any band are excluded from the classification.

    Returns
    -------
    classified : ndarray of shape (height, width)
        Classified raster containing the original class labels.
    """

    # Convert the image from (bands, rows, cols) to (n_pixels, n_bands)
    bands, height, width = image.shape
    image_2d = image.reshape(bands, -1).T

    # Identify valid pixels
    if nodata_value is not None:
        valid_pixels = ~(image_2d == nodata_value).any(axis=1)
    else:
        valid_pixels = np.ones(image_2d.shape[0], dtype=bool)

    valid_data = image_2d[valid_pixels]

    class_labels = list(classes_info.keys())
    log_likelihoods = np.full(
        (image_2d.shape[0], len(class_labels)),
        -np.inf,
        dtype=float,
    )

    # Compute the log-likelihood for each class
    for i, cls in enumerate(class_labels):

        mean = classes_info[cls]["mean"]
        cov = classes_info[cls]["cov"]

        try:
            ll = multivariate_normal.logpdf(
                valid_data,
                mean=mean,
                cov=cov,
            )

        except Exception:
            # Add a tiny diagonal regularization if numerical problems occur
            cov_safe = cov + np.eye(cov.shape[0]) * 1e-8

            ll = multivariate_normal.logpdf(
                valid_data,
                mean=mean,
                cov=cov_safe,
            )

        log_likelihoods[valid_pixels, i] = ll

    # Assign each pixel to the class with the highest likelihood
    assigned = np.argmax(log_likelihoods[valid_pixels], axis=1)

    classified = np.full(
        image_2d.shape[0],
        0,
        dtype=object,
    )

    classified[valid_pixels] = np.array(class_labels)[assigned]

    # Restore the original raster shape
    classified = classified.reshape(height, width)

    return classified