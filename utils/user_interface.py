import cv2


class UserInterface:
    @staticmethod
    def get_key():
        return cv2.waitKey(1)
