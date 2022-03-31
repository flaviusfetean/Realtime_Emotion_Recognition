from sklearn.feature_extraction.image import extract_patches_2d
#this imported function literally returns an array of images
#that represent random crops of a given image, of a given size
#and the number of elements in the array (nr of samples) is
# given as a param, as below

class PatchPreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):

        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]