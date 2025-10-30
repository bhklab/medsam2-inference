from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def dice_multi_class(preds, targets, smooth=0.0):
    """
    Compute the Dice score for multi-class segmentation.
    
    Parameters:
        preds (np.ndarray): Predicted segmentation mask of shape (H, W, C).
        targets (np.ndarray): Ground truth segmentation mask of shape (H, W, C).
    """

    assert preds.shape == targets.shape, "preds and targets must have the same shape"

    dices = []
    labels = np.unique(targets)[1:] 
    for label in labels:
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices.append((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    return np.mean(dices)

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


def pad_bbox(box:np.array,
             mask:np.array, 
             padding:int,
             spacing:np.array = None
             ):
    # Get full image dimensions to keep padding within image size
    # D, H, W (z, y, x)
    mask_shape = mask.shape

    if spacing is not None: # Use the actual image spacing to calculate the padding
        # Check that spacing can be applied to this mask's bounding box dimensions
        if len(spacing) < len(mask_shape):
            spacing = spacing[0, len(mask_shape)]
        elif len(spacing) > len(mask_shape):
            message = "Spacing for padding has more dimensions than the mask image."
            raise ValueError(message)

        # calculate the number of voxels to pad based on the actual image spacing
        padding = np.round(padding / spacing)
    else:
        # Convert padding into an array with the same length as image dimensions
        # This matches the behaviour of the spacing option
        padding = padding * np.ones(len(mask_shape))
    
    pad_x_min = max(0, box[0] - padding[0])
    pad_y_min = max(0, box[1] - padding[1])
    # Handling 2D bounding box
    if len(box) == 4:
        mask_H, mask_W = mask_shape[0], mask_shape[1]
        pad_x_max = min(mask_W, box[2] + padding[0])
        pad_y_max = min(mask_H, box[3] + padding[1])

        padded_box = np.array([pad_x_min, pad_y_min,
                               pad_x_max, pad_y_max])

    # Handling 3D bounding box
    if len(box) == 6:
        mask_D, mask_H, mask_W = mask_shape[0], mask_shape[1], mask_shape[2]
        pad_z_min = max(0, box[2] - padding[2])
        pad_x_max = min(mask_W, box[3] + padding[0])
        pad_y_max = min(mask_H, box[4] + padding[1]) 
        pad_z_max = min(mask_D, box[5] + padding[2])
        
        padded_box = np.array([pad_x_min, pad_y_min, pad_z_min,
                               pad_x_max, pad_y_max, pad_z_max])

    return padded_box.astype(int)


def mask2D_to_bbox(gt2D:np.array, 
                   mask_path:Path, 
                   padding:int | None = None,
                   spacing:np.array = None):
    try:
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        boxes = np.array([x_min, y_min, x_max, y_max])

        if padding:
            boxes = pad_bbox(box = boxes,
                             mask = gt2D,
                             padding = padding,
                             spacing = spacing)
        
        return boxes.astype(int)
    except Exception as e:
        raise Exception(f'error {e} with file {mask_path} and sum of gts is {gt2D.sum()}')


def mask3D_to_bbox(gt3D:np.array, 
                   mask_path:Path, 
                   padding:int | None = None,
                   spacing:np.array = None):
    try:
         # Find slices in the mask that have label voxels to get z boundaries
        z_indices, _, _ = np.where(gt3D > 0)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
    except Exception as e:
        raise Exception(f'error {e} with file {mask_path} and sum of gts is {gt3D.sum()}')
   
    # Find the centre slice of the mask
    z_mid = np.median(z_indices).astype(int)
    # Select out this slice from the array
    gt_mid = gt3D[z_mid]

    # Get the x and y mask boundaries using the 2D function
    box_2d = mask2D_to_bbox(gt_mid, mask_path)
    x_min, y_min, x_max, y_max = box_2d
    boxes3D = np.array([x_min, y_min, z_min, x_max, y_max, z_max])

    if padding:
        # Apply padding to bounding box if requested
        boxes3D = pad_bbox(box = boxes3D,
                           mask = gt3D,
                           padding = padding,
                           spacing = spacing)
    
    return boxes3D.astype(int)

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

def getEdgeOfMask(mask):
    '''
    Computes and returns edge of a segmentation mask
    '''
    # edge has the pixels which are at the edge of the mask
    edge = np.zeros_like(mask)
    
    # mask_pixels has the pixels which are inside the mask of the automated segmentation result
    mask_pixels = np.where(mask > 0)

    for idx in range(0,mask_pixels[0].size):

        x = mask_pixels[0][idx]
        y = mask_pixels[1][idx]
        z = mask_pixels[2][idx]

        # Count # pixels in 3x3 neighborhood that are in the mask
        # If sum < 27, then (x, y, z) is on the edge of the mask
        if mask[x-1:x+2, y-1:y+2, z-1:z+2].sum() < 27:
            edge[x,y,z] = 1
            
    return edge

def AddedPathLength(auto, gt):
    '''
    Returns the added path length, in pixels
    
    Steps:
    1. Find pixels at the edge of the mask for both auto and gt
    2. Count # pixels on the edge of gt that are not in the edge of auto
    '''
    
    # Check if auto and gt have same dimensions. If not, then raise a ValueError
    if auto.shape != gt.shape:
        raise ValueError('Shape of auto and gt must be identical!')

    # edge_auto has the pixels which are at the edge of the automated segmentation result
    edge_auto = getEdgeOfMask(auto)
    # edge_gt has the pixels which are at the edge of the ground truth segmentation
    edge_gt = getEdgeOfMask(gt)
    
    # Count # pixels on the edge of gt that are on not in the edge of auto
    apl = (edge_gt > edge_auto).astype(int).sum()
    
    return apl 
