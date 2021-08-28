import spacy

nlp = spacy.load('en_core_web_sm', disable=['ner'])

doc = nlp(open('transcripts_with_timestamps/transcripts_from_punctuator/1.txt').read())

output_file = open("transcripts_with_timestamps/tanscripts_from_spacy/1.txt", "w")

for sent in doc.sents:
    output_file.write(str(sent)+"\n")

output_file.close()