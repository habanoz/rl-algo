import numpy as np
from numpy import ndarray
import io


def serialize_values(v: ndarray, name: str):
    np.savetxt(f"./env/baselines/{name}.txt", v)


def deserialize_values(name: str):
    return np.loadtxt(f"./env/baselines/{name}.txt")


if __name__ == '__main__':
    arr = np.full((3, 3), 0.123456)

    serialize_values(arr, "dummy")
    arr_recovered = deserialize_values("dummy")
    print(np.equal(arr, arr_recovered))
