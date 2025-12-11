

class Resources:

    BASE_IMAGE = 'resources/images/'

    @classmethod
    def get_image_path(cls, image_name: str) -> str:
        return cls.BASE_IMAGE + image_name
