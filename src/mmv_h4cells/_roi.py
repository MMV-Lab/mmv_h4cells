import numpy as np
import pandas as pd
from typing import Tuple
from napari.qt.threading import thread_worker


@thread_worker
def analyse_roi(
    data: np.ndarray,
    y: Tuple[int, int],
    x: Tuple[int, int],
    size_threshold: int,
    paths: Tuple[str, str],
):
    cropped_mask = data[y[0]:y[1], x[0]:x[1]]
    
    # Get unique ids and counts
    ids, counts = np.unique(cropped_mask, return_counts=True)

    if ids[0] == 0: # Remove background
        ids = ids[1:]
        counts = counts[1:]    

    # Get centroids
    centroids = []
    for id in ids:
        mask_id = cropped_mask == id
        y_id, x_id = np.where(mask_id)
        centroid = (int(np.mean(y_id)) + y[0], int(np.mean(x_id)) + x[0]) 
        centroids.append(centroid)     

    # Create dataframe
    df = pd.DataFrame({
        'id': ids,
        'count [px]': counts,
        'centroid (y,x)': centroids
    })

    # Filter ids by size threshold
    df = df[df['count [px]'] > size_threshold]

    # Get full mask and ignore labels outside ROI
    mask_outside_range = np.ones_like(data, dtype=bool)
    mask_outside_range[y[0]:y[1], x[0]:x[1]] = False
    data[mask_outside_range] = 0

    # Filter mask based on size_threshold
    filtered_mask = np.isin(data, df[df['count [px]'] > size_threshold]['id'])
    data[~filtered_mask] = 0

    return data, df, paths, size_threshold