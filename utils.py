import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def overlay_bbox(image: np.ndarray, bbox: np.ndarray, slice_idx: int, output_path: str):
    """
    Overlay the bounding box on the image.
    """
    image = image[slice_idx]
    x_min, y_min, x_max, y_max = bbox

    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display the image
    ax.imshow(image, cmap='gray')
    
    # Draw rectangle on the plot
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return image

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array


def mask2D_to_bbox(gt2D, file):
    try:
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        bbox_shift = np.random.randint(0, 6, 1)[0]
        scale_y, scale_x = gt2D.shape
        bbox_shift_x = int(bbox_shift * scale_x/256)
        bbox_shift_y = int(bbox_shift * scale_y/256)
        x_min = max(0, x_min - bbox_shift_x)
        x_max = min(W-1, x_max + bbox_shift_x)
        y_min = max(0, y_min - bbox_shift_y)
        y_max = min(H-1, y_max + bbox_shift_y)
        boxes = np.array([x_min, y_min, x_max, y_max])
        return boxes
    except Exception as e:
        raise Exception(f'error {e} with file {file} and sum of gts is {gt2D.sum()}')


def mask3D_to_bbox(gt3D, file):
    z_indices, _, _ = np.where(gt3D > 0)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    # add perturbation to bounding box coordinates
    D, H, W = gt3D.shape

    z_mid = np.median(z_indices).astype(int)
    gt_mid = gt3D[z_mid]

    box_2d = mask2D_to_bbox(gt_mid, file)
    x_min, y_min, x_max, y_max = box_2d

    z_min = max(0, z_min)
    z_max = min(D-1, z_max)
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes3d

def preprocess(
        image_data: np.ndarray, 
        window_level: float | None = None, 
        window_width: float | None = None
    ) -> np.ndarray:
    """
    Preprocess the image data.

    Parameters:
        image_data (np.ndarray): Input image data.
        window_level (float): Window level.
        window_width (float): Window width.

    Returns:
        np.ndarray: Preprocessed image data.
    """
    if window_level is None or window_width is None:
        return image_data
    
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre))
        * 255.0
    )
    
    #image_data_pre[image_data == 0] = 0
    return image_data_pre

