import time

class Timer:

    def __init__(self, verbose=True):
        self.verbose = verbose

    def start(self):
        self.s = time.time()

    def format(self, t):
        if t < 1:
            return '{:.2f} ms'.format(t * 1000)
        else:
            res = []

            second = t % 60
            t = int(t // 60)
            res.append('{:.2f}s'.format(second))

            if t > 0:
                miniute = t % 60
                t = t // 60
                res.append('{}min'.format(miniute))

            if t > 0:
                hour = t % 60
                t = t // 24
                res.append('{}h'.format(hour))

            if t > 0:
                res.append('{}day'.format(t))

        return ' '.join(res[::-1])

    def stop(self, total=0):
        self.e = time.time()
        self.t = self.e - self.s

        if self.verbose:
            print('Total time used:', self.format(self.t))
            if total > 0:
                print('Average of {} execution(s): {}/exec'.format(
                    total, self.format(self.t / total)))
        return self.t


def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False
