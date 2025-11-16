import albumentations as A

def get_mild_augmentations():
    """
    Defines a minimal augmentation pipeline for the majority class (No DR).
    This prevents the model from seeing too many altered healthy images.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
    ])

def get_aggressive_augmentations():
    """
    Defines a robust augmentation pipeline for the minority DR classes.
    This creates more diverse training examples to help the model learn
    the subtle features of early-stage diabetic retinopathy.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        # CLAHE can significantly improve visibility of microaneurysms
        A.CLAHE(clip_limit=4.0, p=0.7),
        # Add distortions that mimic real-world variations in retina shape
        A.GridDistortion(p=0.5),
        A.ElasticTransform(p=0.5),
        # Randomly black out parts of the image to force the model
        # to learn from different features
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5)
    ])

