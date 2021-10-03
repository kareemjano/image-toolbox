from imgaug import augmenters as iaa

def getImageAug():
    return iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.Fliplr(0.5),
        iaa.AddToBrightness((-30, 30)),
    ])