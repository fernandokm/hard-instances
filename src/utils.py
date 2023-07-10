import sys


class Tee:
    def __init__(self, dst1, dst2):
        self.dst1 = dst1
        self.dst2 = dst2

    def write(self, data):
        self.dst1.write(data)
        self.dst2.write(data)

    def flush(self):
        self.dst1.flush()
        self.dst2.flush()

    @staticmethod
    def save_stdout_stderr(filename: str, mode="w+"):
        f = open(filename, mode)
        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)
