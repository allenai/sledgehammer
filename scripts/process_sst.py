#!/usr/bin/env python


#############################################################################
#                                                                           #
# Process the SST dataset, downloaded from                                  #
# http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip            #
#                                                                           #
#############################################################################

import os
import sys
import re

encoding='utf-8'
def main(args):
    """
    A script that processes a downloaded SST treebank:
    1. Convert 5 ways labels to binary labels
    2. Remove constituents
    """
    if len(args) < 3:
        print("Usage: {} <input dir> <output dir>".format(args[0]))
        return -1
    
    in_dir = args[1]

    doc2id = {}
    id2doc = {}

    # 1. Read sentences and their ids (for train/dev/test splitting)
    with open(in_dir+'/datasetSentences.txt', encoding=encoding) as ifh:
                # skip header line
        ifh.readline()
        for l in ifh:
            data = l.rstrip().split("\t")
            assert len(data) == 2, "line {} in datasetSentences.txt doesn't have two elements, can't parse it".format(l)

            data[1] = clean_str(data[1])
            id = data[0]

            id2doc[id] = data[1]
            doc2id[data[1]] = id

    
    dictionary_ids = {}

    found_docs = {}

    # A second mapping of documents (for labels)
    with open(in_dir+'/dictionary.txt', encoding=encoding) as ifh:
        for l in ifh:
            data = l.rstrip().split("|")

            assert len(data) == 2, "line {} in dictionary.txt doesn't have two elements, can't parse it".format(l)

            doc = clean_str(data[0])
            if doc in doc2id:
                found_docs[doc] = 1
                dictionary_ids[data[1]] = doc


    doc2label = {}

    # Reading labels
    with open(in_dir+'/sentiment_labels.txt') as ifh:
        # skip header line
        ifh.readline()
        for l in ifh:
            data = l.rstrip().split("|")
            assert len(data) == 2, "line {} in sentiment_labels.txt doesn't have two elements, can't parse it".format(l)

            if data[0] in dictionary_ids:
                label = int(5*float(data[1]))

                # Binary labels: ignoring middle label
                if label < 2:
                    label = 0
                    doc2label[dictionary_ids[data[0]]] = 0
                elif label > 2:
                    doc2label[dictionary_ids[data[0]]] = 1

   
    splits = [[], [], []]

    # Reading splits
    with open(in_dir+'/datasetSplit.txt') as ifh:
        # skip header line
        ifh.readline()

        for l in ifh:
            data = l.rstrip().split(",")
            assert len(data) == 2, "line {} in datasetSplit.txt doesn't have two elements, can't parse it".format(l)
            if data[0] in id2doc and id2doc[data[0]] in doc2label:
                d0 = data[0]
                index = int(data[1])-1
                doc = id2doc[data[0]]
                label = doc2label[doc]
                out = [doc2label[doc], revert_doc(doc)]

                splits[index].append(out)
            
    
    # Saving
    save_file(args[2]+'/train', splits[0])
    save_file(args[2]+'/test', splits[1])
    save_file(args[2]+'/dev', splits[2])

    print("Done")
    return 0

def clean_str(s):
    # Replace non-ascii chars
    pairs = {"Ã³": "ó","Ã­": "í", "-LRB-": "(", '-RRB-': ')', 
            'Ã©': 'é', "Ã¢": "â", "Ã¯":"ï", "Ã¡": "á", "Ã£": "ã",
            "Ã¦": "æ", "Ã¨": "è", "Ã¶": "ö", "Ã±": "ñ", "Ã¼": "ü",
            "Â": "", "Ã§": "ç", "Ã´": "ô", "Ã»": "û"}

    for p in pairs:
        s = s.replace(p, pairs[p])

    # Last excpetion replacement
    s = re.sub("Ã.", "à", s)

    return s

def revert_doc(doc):
    # Replace '(' and ')' with -[lr]rb-, and lowering case
    return doc.replace("(", "-lrb-").replace(')', '-rrb-').lower()

def save_file(ofile, docs):
    with open(ofile, 'w') as ofh:
        for l in docs:
            ofh.write("{}\t{}\n".format(l[0], l[1]))



if __name__ == '__main__':
    sys.exit(main(sys.argv))