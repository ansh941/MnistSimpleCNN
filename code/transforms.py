import random
import torchvision.transforms.functional as F

class RandomRotation(object):
    def __init__(self, degrees, seed=1):
        self.degrees = (-degrees, degrees)
        random.seed(seed)
    
    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, False, False, None, None)
