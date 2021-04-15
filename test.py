import numpy as np

class BaseClass():
    """
    """
    def __init__(self, func1, func2):
        self.func1 = func1
        self.func2 = func2

    def test_func1(self, q):
        self.func1(q)

    def test_func2(self, q):
        self.func2(q)

class DerivedClass(BaseClass):
    """
    """
    def __init__(self):
        self.a = 0

        def func1(b):
            print("here")
            self.a = b+1

        def func2(b):
            self.a = b*5

        super().__init__(func1, func2)

if __name__ == "__main__":
    import argparse

    dclass = DerivedClass()

    print("a is {}".format(dclass.a))

    dclass.test_func1(10)

    print("a is {}".format(dclass.a))

    dclass.test_func2(10)

    print("a is {}".format(dclass.a))
