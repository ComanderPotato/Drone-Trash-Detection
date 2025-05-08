from imgaug import augmenters as iaa

def get_augmentation_pipeline():
    """
    Returns an image augmentation pipeline using imgaug.
    The pipeline applies random augmentations such as noise, blur, flips, 
    brightness/contrast changes, and affine transformations.

    Returns:
        iaa.Sequential: Configured image augmentation pipeline.
    """
    augmentation_pipeline = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="AWGN"),
        iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"),
        iaa.Fliplr(0.5),
        iaa.Add((-20, 20), name="Add"),
        iaa.Multiply((0.8, 1.2), name="Multiply"),
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
    ], random_order=True)
    
    return augmentation_pipeline
