#!/usr/bin/env bash

#############################################################################
#                                                                           #
# A script for downloading the datasets for the paper                       #
# "The Right Tool for the Job: Matching Model and Instance Complexities"    #
# (Schwartz et al., ACL 2020)                                               #
# The script downloads the following datasets:                              #
# AG news, IMDB, SST binary, SNLI and MultiNLI                              #
#                                                                           #
#############################################################################

delete_tmp=1

if [ $# -lt 1 ]; then
    echo "Usage: $0 <target directory> <tmp directory (optional)>"
    exit -1
elif [ $# -gt 1 ]; then
    tmp_dir=$2
    delete_tmp=0
else
    tmp_dir=$(mktemp -d)
fi

out_dir=$1

# Creating directory structure
mkdir -p $out_dir/text_cat/sst
if [ $? != 0 ]; then
        echo "Cannot create $out_dir"
        exit -2
fi

mkdir -p $out_dir/text_cat/imdb
mkdir -p $out_dir/text_cat/ag

mkdir -p $out_dir/nli/snli
mkdir -p $out_dir/nli/mnli


# Helper function 1: extract NLI files
function extract_nli {
    split_in=$1
    split_out=$2
    name=$3
    long_name=$4

    ofile=$out_dir/nli/${name}/$split_out

    if [ ! -e $ofile ]; then
        tail -n+2 $tmp_dir/${name}/${long_name}_1.0/${long_name}_1.0_${split_in}.txt | grep -vE '^-'  | cut -d $'\t' -f1,6,7 | sed 's/^neutral/0/' | sed 's/^contradiction/1/' | sed 's/^entailment/2/' > $ofile
    fi
}

# Helpfer function 2: download NLI dataset
function download_nli {
    url=$1
    name=$2
    long_name=$3
    dev_name=$4
    test_name=$5

    wget -O $tmp_dir/${name}.zip $url 

    unzip $tmp_dir/${name}.zip -d $tmp_dir/$name/

    extract_nli train train $name $long_name
    extract_nli $dev_name dev $name $long_name
    extract_nli $test_name test $name $long_name
}

# Helper function 3: download text classification dataset from AllenNLP repo
function download_allennlp {
    local name=$1
    local remote_name=$2
    local label_delta=$3

    mkdir -p $tmp_dir/$name/

    for i in train dev test; do
        json_file=$tmp_dir/$name/${i}.jsonl
        curl -Lo $json_file https://s3-us-west-2.amazonaws.com/allennlp/datasets/$remote_name/${i}.jsonl
        cat $json_file  | jq -r '[.label, .text] | @tsv' | awk -v label_delta=$label_delta -F $'\t' '{print $1-label_delta"\t"$2}' | sed 's/\\\\/\\/g' > $out_dir/text_cat/$name/$i
    done
}

#### SNLI and MultiNLi
download_nli https://nlp.stanford.edu/projects/snli/snli_1.0.zip snli snli dev test
download_nli https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip mnli multinli dev_matched dev_mismatched

#### AG news and IMDB
download_allennlp ag ag-news 1
download_allennlp imdb imdb 0


#### SST
mkdir $tmp_dir/sst
wget -O $tmp_dir/sst/sst.zip http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip $tmp_dir/sst/sst.zip -d $tmp_dir/sst/

base_dir=$(dirname $0)
python $base_dir/process_sst.py $tmp_dir/sst/stanfordSentimentTreebank $out_dir/text_cat/sst/

if [ $delete_tmp -eq 1 ]; then
    /bin/rm -rf $tmp_dir
fi

exit 0