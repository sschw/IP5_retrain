# 
from skimage.transform import resize, rescale
from skimage.filters.thresholding import threshold_otsu
from skimage.io import imread, imsave, imshow
from skimage.color import rgb2hsv
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import numpy as np
import os
from joblib import Parallel, delayed
import time

def get_bounding_box_by_lab_thresholding(im, scalefactor=8, min_pix_area=10):
    # Go through all pixels and make a bounding box around pixels that are not 
    # in the hsv range.
    
    # image to a smaller hsv values
    pix = rgb2hsv(rescale(im, 1/scalefactor, mode='constant'))
    pix_h = pix[:, :, 0]
    pix_s = pix[:, :, 1]

    # make a binary image by filtering by h and s values
    # make a closing operation
    # remove regions at the border of the image. (might be the scan box)
    thr = clear_border(
      closing(
        np.logical_or(
            np.logical_or(pix_h < 0.18, pix_h > 0.5), 
            pix_s < 0.2
        ), square(3)
      )
    )
    
    # label the regions
    label_img = label(thr)
    
    minx = pix.shape[1]
    miny = pix.shape[0]
    maxx = 0
    maxy = 0
    # find the bounding box around all regions that are bigger than 10px
    for region in regionprops(label_img):
      if region.area > min_pix_area:
        miy, mix, may, max = region.bbox
        miny = miy if miy < miny else miny
        minx = mix if mix < minx else minx
        maxy = may if may > maxy else maxy
        maxx = max if max > maxx else maxx
    
    # scale the bounding box to the real size
    return minx*scalefactor, miny*scalefactor, maxx*scalefactor, maxy*scalefactor

def scale_on_object(im, padding=120):
    min_x, min_y, max_x, max_y = get_bounding_box_by_lab_thresholding(im)
    
    # find longer side for crop
    length_x = max_x - min_x
    length_y = max_y - min_y
    
    if length_x > length_y:
        center_y = min_y + ((max_y - min_y) >> 1)
        min_y = center_y - (length_x >> 1)
        max_y = center_y + (length_x >> 1)
    else:
        center_x = min_x + ((max_x - min_x) >> 1)
        min_x = center_x - (length_y >> 1)
        max_x = center_x + (length_y >> 1)
    # add padding and check if valid
    if min_x - padding < 0 or min_y - padding < 0 or max_x + padding >= im.shape[1] and max_y + padding >= im.shape[0]:
      min_x = 0
      min_y = 0
      max_x = im.shape[1]
      max_y = im.shape[0]
    else:
      min_x -= padding
      min_y -= padding
      max_x += padding
      max_y += padding
    
    # crop the image by the object with a padding
    return im[min_y : max_y, min_x : max_x]

def scale_and_resize_object(fn, root, dir, dest):
    try:
        fn = os.path.join(root, fn)
        
        # prepare output name
        new_fn = fn.replace(dir, dest).replace(" ", "_")
        new_fn, _ = os.path.splitext(new_fn)
        new_fn = new_fn + ".PNG"

        if not os.path.isfile(new_fn) \
                and not os.path.isfile(new_fn.replace(os.sep + 'train', os.sep + 'test')) \
                and not os.path.isfile(new_fn.replace(os.sep + 'train', os.sep + 'validation')) \
                and (fn.endswith(".jpg") or fn.endswith(".JPG") or fn.endswith(".PNG")):
            # Load image
            im = imread(fn)
            # Pre-crop - increases overall performance
            center_x, center_y = im.shape[1] >> 1, im.shape[0] >> 1
            center = center_x
            if center_x > center_y:
                center = center_y
            new_im = im[center_y - center : center_y + center, center_x - center : center_x + center]
            
            # Crop on object
            new_im = scale_on_object(new_im)
            new_im = resize(new_im, (224, 224, 3), mode='constant')
            # create directory in training dir if it doesn't already exist
            if not os.path.exists(os.path.dirname(new_fn)):
                try:
                    os.makedirs(os.path.dirname(new_fn))
                except OSError as e:
                    if e.errno != 17:
                        raise
                    pass
            imsave(new_fn, new_im)
    except:
        print("object not found in " + fn)

def scale_and_resize(dir, dest):
    # scales and resizes in a folder with parallelization
    for root, dirs, files in os.walk(dir):
        Parallel(n_jobs=8)(
            delayed(scale_and_resize_object)(fn=fn, root=root, dir=dir, dest=dest) for fn in files)
