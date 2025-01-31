import numpy as np
import time
import pycocotools.mask as cocomask

def generate_combined_mask(
    rgb_image, model, frame_index=None, device='cuda:0', imgsz=(480, 640), conf=0.5, first_frame=True
):
    """
    Generate a combined binary mask and its bounding box from YOLO segmentation results.

    Args:
        rgb_image (numpy.ndarray): The input RGB image.
        model: The YOLO segmentation model.
        frame_index (int, optional): The index of the frame for logging purposes.
        device (str): The device to run the model on ('cpu' or 'cuda:X').
        imgsz (tuple): The size of the image for inference.
        conf (float): Confidence threshold for segmentation.
        first_frame (bool): If True, convert the mask to JSON format.

    Returns:
        combined_mask (numpy.ndarray): The combined binary mask.
        segmentation_time (float): The time taken for segmentation.
        json_data (list, optional): If `first_frame` is True, returns the JSON data structure.
    """
    # Start timer
    print(f"Frame {frame_index}: Starting Segmentation")
    start_time = time.time()

    # Run inference
    results = model.predict(rgb_image, imgsz=imgsz, conf=conf, device=device)
    result = results[0]

    # Combine masks and calculate the bounding box
    height, width = result.masks.data[0].shape[-2:]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Initialize bounding box variables
    x_min, y_min = width, height
    x_max, y_max = 0, 0

    for mask, box in zip(result.masks.data, result.boxes.xyxy):
        binary_mask = mask.cpu().numpy().astype(np.uint8)
        combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)

        # Update the bounding box based on the individual detection
        x1, y1, x2, y2 = box.cpu().numpy()
        x_min = min(x_min, int(x1))
        y_min = min(y_min, int(y1))
        x_max = max(x_max, int(x2))
        y_max = max(y_max, int(y2))

    # Convert to binary format
    combined_mask = (combined_mask > 0).astype(np.uint8)

    # Final bounding box: [x, y, width, height]
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    # End timer
    segmentation_time = time.time() - start_time

    # Logging
    if frame_index is not None:
        print(f"Frame {frame_index}: Segmentation time: {segmentation_time:.2f}s")
    print(f"Combined mask shape: {combined_mask.shape}")
    print(f"Bounding box: {bbox}")
    print(f"Number of non-zero pixels in mask: {np.sum(combined_mask)}")

    # If first_frame is True, convert to JSON format
    if first_frame:
        # Convert binary mask to COCO RLE
        rle = cocomask.encode(np.asfortranarray(combined_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")

        # Create JSON data
        json_data = [
            {
                "scene_id": 0,
                "image_id": 0,
                "category_id": 1,
                "bbox": bbox,
                "score": 1.0,
                "segmentation": {
                    "size": [height, width],
                    "counts": rle["counts"]
                }
            }
        ]

        return combined_mask, json_data

    # If not first_frame, return only the combined mask and segmentation time
    return combined_mask
