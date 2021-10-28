from tokenizers import Tokenizer
import pandas as pd

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_tokenizer(tokenizer_path: str, text_file):
    num_unks = 0

    tokenizer = Tokenizer.from_file(tokenizer_path)

    lines = open(text_file,'r').read().split('\n')
    outputs = tokenizer.encode_batch(lines)
    
    for i in range(len(outputs)):
        for id in outputs[i].ids:
            if id == 8:
                num_unks = num_unks + 1
    
    return num_unks

data = ["bnc", "maptask", "switchboard"]
df = pd.DataFrame()
for tknzr in data:
    result = {}
    for filename in data:
        result[filename] = test_tokenizer('trained_tokenizers/' + tknzr + '.json', 'data/' + filename + '/train.csv')
    series = pd.Series(result)
    df_new = pd.DataFrame(series)
    df = pd.concat([df, df_new], axis=1)

df.columns = data

print("\n\n\n------------Displaying number of UNKs-----------\n\n")
print("Column: Tokenizer used\nRow: Text file tested\n\n")
print(df)
print("\n\n")