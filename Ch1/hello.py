def add(x, y):
    """Let's kick it up a notch"""

    return (x ^ y) + ((x & y) << 1)

print(add(1, 1))