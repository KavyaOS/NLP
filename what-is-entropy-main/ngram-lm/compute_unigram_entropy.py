import sys
import os
import colibricore
import numpy as np


def encode_file(base_name, ext_name='.txt'):
    text_file = base_name + ext_name

    classencoder = colibricore.ClassEncoder()
    classencoder.build(text_file)
    class_file = base_name + '.colibri.cls'
    classencoder.save(class_file)
    classdecoder = colibricore.ClassDecoder(class_file)

    corpus_file = base_name + '.colibri.dat'
    classencoder.encodefile(text_file, corpus_file)

    return class_file, corpus_file, classencoder, classdecoder


def unigram_entropy(word, model, classencoder, classdecoder, all_bigrams=None):
    """
    word: a str or a pattern
    model: an instance of colibricore.UnindexedPatternModel
    """
    if isinstance(word, str):
        word = classencoder.buildpattern(word)
    if word not in model:
        # print('"{}" not in model'.format(word))
        return None

    if all_bigrams:
        all_bigrams = all_bigrams[:] # make a copy of all_bigrams list
        bigram_counts = []
        remove_indice = []
        for i, (pattern, count) in enumerate(all_bigrams):
            if pattern[0] == word:
                bigram_counts.append(count)
                remove_indice.append(i)
        bigram_counts = np.asarray(bigram_counts)
        for idx in sorted(remove_indice, reverse=True):
            del all_bigrams[idx]
    else:
        bigram_counts = np.asarray([count for pattern, count in model.filter(0, colibricore.Category.NGRAM, 2) if pattern[0] == word])
    probs = bigram_counts / np.sum(bigram_counts)
    entropy = - np.sum(probs * np.log2(probs))

    return entropy


def get_unigram_surprisal(model, classdecoder):
    """
    return: {word:str -> surprisal:float}, all unigram surprisals
    """
    results = {}
    count_sum = 0.0
    for pattern, count in model.filter(0, colibricore.Category.NGRAM, 1):
        results[pattern.tostring(classdecoder)] = count
        count_sum += count
    for word in results.keys():
        prob = results[word] / count_sum
        results[word] = -np.log2(prob)

    return results


def uni_ent_exp1(base_name):
    # Build or load encoder and decoder
    class_file = base_name + '.colibri.cls'
    corpus_file = base_name + '.colibri.dat'
    if not os.path.exists(class_file) or not os.path.exists(corpus_file):
        _, _, classencoder, classdecoder = encode_file(base_name)
    else:
        classencoder = colibricore.ClassEncoder(class_file)
        classdecoder = colibricore.ClassDecoder(class_file)

    # Build or load pattern model
    options = colibricore.PatternModelOptions(mintokens=2,maxlength=3)
    patternmodel_file = base_name + '.colibri.patternmodel'
    if os.path.exists(patternmodel_file):
        model = colibricore.UnindexedPatternModel(patternmodel_file, options)
    else:
        model = colibricore.UnindexedPatternModel()
        model.train(corpus_file, options)
        model.write(patternmodel_file)

    # Compute the entropy for all the unigrams in the corpus
    all_bigrams = list(model.filter(0, colibricore.Category.NGRAM, 2))
    entropy_results = {}
    count = 0
    for pattern, _ in model.filter(0, colibricore.Category.NGRAM, 1):
        entropy = unigram_entropy(pattern, model, classencoder, classdecoder, all_bigrams=all_bigrams)
        try:
            entropy_results[pattern.tostring(classdecoder)] = entropy
        except UnicodeDecodeError as e:
            pass
        count += 1
        if count % 10 == 0:
            sys.stdout.write('\r{} unigram computed'.format(count))
            sys.stdout.flush()

    output_file = base_name + '.unigram.entropy'
    with open(output_file, 'w') as f:
        for pattern, _ in sorted(model.filter(0, colibricore.Category.NGRAM, 1), key=lambda x:x[1], reverse=True):
            try:
                word = pattern.tostring(classdecoder)
            except UnicodeDecodeError as e:
                pass
            else:
                f.write(word + ' ' + str(entropy_results[word]) + '\n')

    # Compute unigram surprisals
    surprisal_results = get_unigram_surprisal(model, classdecoder)
    output_file2 = base_name + '.unigram.surprisal'
    with open(output_file2, 'w') as f:
        for pattern, _ in sorted(model.filter(0, colibricore.Category.NGRAM, 1), key=lambda x:x[1], reverse=True):
            try:
                word = pattern.tostring(classdecoder)
            except UnicodeDecodeError as e:
                pass
            else:
                f.write(word + ' ' + str(surprisal_results[word]) + '\n')


def entropy_in_sentence(entropy_file, surprisal_file, sentence_file):
    output_file1 = sentence_file + '.ent_in_sent'
    output_file2 = sentence_file + '.surp_in_sent'
    # read entropy and surprisal
    entropy = {}
    with open(entropy_file, 'r') as f:
        for line in f:
            key, val = line.strip().split()
            entropy[key] = val
    surprisal = {}
    with open(surprisal_file, 'r') as f:
        for line in f:
            key, val = line.strip().split()
            surprisal[key] = val
    # read from sentence_file
    with open(sentence_file, 'r') as fr, open(output_file1, 'w') as fw1, open(output_file2, 'w') as fw2:
        sent_id = 0
        for line in fr:
            words = line.strip().split()
            for i, w in enumerate(words):
                if w in entropy:
                    ent = entropy[w]
                    fw1.write(' '.join([str(sent_id), str(i), ent]) + '\n')
                if w in surprisal:
                    surp = surprisal[w]
                    fw2.write(' '.join([str(sent_id), str(i), surp]) + '\n')
            sent_id += 1
            if sent_id % 100 == 0:
                sys.stdout.write('\r{} sentences written'.format(sent_id))
                sys.stdout.flush()


if __name__ == '__main__':
    # uni_ent_exp1('wsj_raw_repl_lower')
    entropy_in_sentence('wsj_raw_repl_lower.unigram.entropy', 'wsj_raw_repl_lower.unigram.surprisal', 'wsj_raw_repl_lower.txt')
