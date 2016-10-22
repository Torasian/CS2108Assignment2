
class Video:
    def __init__(self, file, feature_vector):
        self.file = file
        self.feature_vector = feature_vector

    def __str__(self):
        return "{}:{}".format(self.file, self.feature_vector)

    def __repr__(self):
        return "{}:{}".format(self.file, self.feature_vector)