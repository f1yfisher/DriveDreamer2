import time


class Timer:
    def __init__(self, prefix='Cost'):
        self.prefix = prefix
        self.start_time = None

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        print('{}: {}'.format(self.prefix, time.time() - self.start_time))

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc()


def get_cur_time():
    return time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
