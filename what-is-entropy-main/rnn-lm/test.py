from run import *
from tqdm import tqdm
import sys


def unit_test():
    corpus.print_info()
    # It matches the corpus statistics here: https://pytorch.org/text/stable/_modules/torchtext/datasets/wikitext2.html

    outputs = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
    print(outputs[0].tokens)
    print(outputs[1].tokens)

    # test iter
    # count = 0
    # for item in train_iter:
    #     print(len(item.strip()))
    #     count += 1
    #     if count >= 5:
    #         break

    # test loader
    count = 0
    for batch in train_loader:
        print(len(batch))

        ids, mask, lengths = batch
        print('mask', mask)
        print('lengths', lengths)

        ones_count = torch.count_nonzero(mask, dim=1)
        print('ones_count', ones_count)

        data, lengths, targets = process_batch(batch)
        # print('data', data)
        # print(len(data))
        # print(type(batch))
        print('ids.shape', ids.shape)
        # print(batch.size())
        count += 1
        if count >= 1:
            break

def test_loader():
    step = 0
    for i in range(3):
        print(f'epoch {i+1}')
        for batch in train_loader:
            step += 1
            # sys.stdout.write(f'\rstep {step}, train_data.pos {train_data.pos()}')
            sys.stdout.write(f'\rstep {step}')
            sys.stdout.flush()
            # print(batch)
            # if step > 0:
            #     break
        print()


if __name__ == '__main__':
    # unit_test()
    test_loader()