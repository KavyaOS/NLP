import numpy as np 


def main():
    num_iter = 20
    for dim in range(2, 30):
        entropies = []
        for i in range(num_iter):
            probs = np.exp(np.random.rand(dim))
            probs /= probs.sum(0)
            entropy = - np.sum(probs * np.log(probs))
            entropies.append(entropy)
        print('dim = {}, entropy: {}'.format(dim, np.mean(entropies)))


if __name__ == "__main__":
    main()