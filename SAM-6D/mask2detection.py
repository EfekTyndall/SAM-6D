import cv2
import numpy as np
import pycocotools.mask as cocomask
import json

# 1) Load the PNG as a grayscale image
mask_gray = cv2.imread("/home/martyn/Thesis/pose-tracking/data/frames/frames_part/scene_04/masks/1737368176499.png", cv2.IMREAD_GRAYSCALE)

# 2) Threshold/binarize if needed:
#    If your mask is strictly 0 (background) or 255 (object), this step ensures you get a bool array.
mask_bool = mask_gray > 128  # True where object is

# 3) Encode as COCO RLE
#    'encode' requires Fortran-contiguous array, so use np.asfortranarray:
rle = cocomask.encode(np.asfortranarray(mask_bool.astype(np.uint8)))

# 4) The 'counts' field is bytes; decode it to ASCII to store in JSON
rle["counts"] = rle["counts"].decode("ascii")

height, width = mask_bool.shape

data = [
  {
    "scene_id": 0,
    "image_id": 0,
    "category_id": 1,
    "bbox": [186, 143, 235, 270],  # optional or you can supply a real bounding box
    "score": 1.0,                  # must be > det_score_thresh so it won't be filtered out
    "segmentation": {
      "size": [height, width],
      "counts": rle["counts"]
    }
  }
]

# 5) Save this list to a JSON file
with open("first_frame_detection.json", "w") as f:
    json.dump(data, f)