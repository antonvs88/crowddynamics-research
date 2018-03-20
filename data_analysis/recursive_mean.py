import numpy as np

def recursive_mean(data, chunk):
    """
    Calculate mean of data array recursively by averaging a "chunk" of the data.

    Parameters
    ----------
    data : array, float
        Input data
    chunk : integer
        Size of chunk

    Returns
    -------
    time_sample_average : array
        Data array averaged over time.

    """

    # Return the largest integer smaller or equal to the division of the first dimension of data array and chunk size.
    divider = np.floor_divide(data.shape[0], chunk)
    # Computes the remainder complementary to the floor_divide function.
    remainder = np.remainder(data.shape[0], chunk)

    # Initialize array that is returned.
    time_sample_average = np.zeros((data.shape[1], data.shape[2]), dtype=np.float16)

    # Calculate mean of data array recursively by taking averaging a "chunk" of the data.
    for zzz in range(0,divider+1):
        # If remainder only left, calculate take the average of the remainder.
        if zzz == divider:
            if remainder == 0:
                break
            elif remainder == 1:
                temp_mean = data[chunk * zzz + remainder - 1, :, :]
            else:
                temp_mean = np.mean(data[chunk * zzz:chunk * zzz + remainder -1, :, :], axis=0, dtype=np.float16)
            time_sample_average = (time_sample_average * chunk * zzz + temp_mean * remainder) /\
                                        (chunk * zzz + remainder)
        else:
            if chunk == 1:
                temp_mean = data[chunk * zzz, :, :]
            else:
                temp_mean = np.mean(data[chunk * zzz:chunk * (zzz+1)-1, :, :], axis=0, dtype=np.float16)
            time_sample_average = (time_sample_average * chunk * zzz + temp_mean * chunk) / \
                                        (chunk * (zzz + 1))

    return time_sample_average
