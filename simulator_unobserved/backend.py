np = __import__("numpy")


def set_backend(use_gpu=False):
    global np
    np = __import__("cupy" if use_gpu else "numpy")
