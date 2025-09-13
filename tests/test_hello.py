import kernels
import unittest 
import torch

class TestHello(unittest.TestCase):
    def test_hello(self):
        kernels.hello_world()

if __name__ == "__main__":
    unittest.main()