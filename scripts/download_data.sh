#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <target directory>"
    exit -1
elif [ $# -gt 1 ]; then
    tmp_dir=$2
else
    tmp_dir=$(mktemp -d)
fi

out_dir=$1

mkdir -p $out_dir/text_cat/sst
if [ $? != 0 ]; then
        echo "Cannot create $out_dir"
        exit -2
fi

mkdir -p $out_dir/text_cat/imdb
mkdir -p $out_dir/text_cat/ag


mkdir -p $out_dir/nli/snli
mkdir -p $out_dir/nli/mnli


# 1. SNLI

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

function download_nli {
    url=$1
    name=$2
    long_name=$3
    dev_name=$4
    test_name=$5

#    wget -O $tmp_dir/${name}.zip $url 

 #   unzip $tmp_dir/${name}.zip -d $tmp_dir/$name/

    extract_nli train train $name $long_name
    extract_nli $dev_name dev $name $long_name
    extract_nli $test_name test $name $long_name
}

function download_allennlp {
    local name=$1
    local remote_name=$2

    mkdir -p $tmp_dir/$name/

    for i in train dev test; do
        json_file=$tmp_dir/$name/${i}.jsonl
        curl -Lo $json_file https://s3-us-west-2.amazonaws.com/allennlp/datasets/$remote_name/${i}.jsonl
        cat $json_file  | jq -r '[.label, .text] | @tsv' | awk -F $'\t' '{print $1-1"\t"$2}' | sed 's/\\\\/\\/g' > $out_dir/text_cat/$name/$i
    done
}

download_nli https://nlp.stanford.edu/projects/snli/snli_1.0.zip snli snli dev test

download_nli https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip mnli multinli dev_matched dev_mismatched

#### AG news and IMDB
download_allennlp ag ag-news 
download_allennlp imdb imdb

echo done with $tmp_dir
# /bin/rm -rf $tmp_dir

exit 0