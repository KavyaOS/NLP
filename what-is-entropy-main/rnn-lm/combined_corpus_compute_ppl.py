import numpy as np

with open('output_ppl/combined_corpus_subwordtk.txt', 'r') as f:
    numbers = []
    for line in f:
        numbers.append(float(line[:-1]))
    print("\n\nMean of Combined corpus: ", np.mean(numbers))
    print("\n\nStandard deviation of Combined corpus: ", np.std(numbers), "\n\n")