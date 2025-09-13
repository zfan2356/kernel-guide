import kernels_cpp
import unittest 

class TestHello(unittest.TestCase):
    def test_hello(self):
        kernels_cpp.hello_world()

if __name__ == "__main__":
    unittest.main()