import numpy as np
def transform_to_hu(medical_image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    image = medical_image.pixel_array
    hu_image = image * slope + intercept

    return hu_image

def apply_windowing(ds, window_center, window_width, normalized=False, zero_centered=True):
    
    # Get the pixel data and rescale to Hounsfield units (HU)
    pixel_array = ds.pixel_array.astype(np.float32)
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    hu_array = pixel_array * slope + intercept
    
    # Apply windowing
    window_min = window_center - (window_width / 2)
    window_max = window_center + (window_width / 2)
    windowed_array = np.clip(hu_array, window_min, window_max)
    if(zero_centered):
        windowed_array = zero_center(windowed_array)
    if(normalized):
    # Normalize the windowed array to [0, 1]
        normalized_array = (windowed_array - window_min) / (window_max - window_min)
        return normalized_array
    else: 
        return windowed_array

def set_window(img, hu=[-800.,1200.]):
    window = np.array(hu)
    newimg = (img-window[0]) / (window[1]-window[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def zero_center(hu_image):
    hu_image = hu_image - np.mean(hu_image)
    return hu_image