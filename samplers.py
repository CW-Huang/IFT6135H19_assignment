import numpy as np
import random


def distribution1(x, batch_size=1):
    # Distribution defined as (x, U(0,1)). Can be used for question 3
    while True:
        yield(np.array([(x, random.uniform(0, 1)) for _ in range(batch_size)]))


def distribution2(batch_size=1):
    # High dimension uniform distribution
    while True:
        yield(np.random.uniform(0, 1, (batch_size, 20)))


def distribution3(batch_size=1):
    # High dimension gaussian distribution
    while True:
        yield(np.random.normal(0, 1, (batch_size, 20)))


if __name__ == '__main__':
    # Example of usage
    dist = iter(distribution1(0, 100))
    samples = next(dist)
