# Attention-Visualization
Visualization for simple attention and Google's multi-head attention.

## Requirements

- python
- jdk1.8

## Usage

1\. Python version (for simple attention only):

``` bash
python exec/plot_heatmap.py --input xxx.attention
```

2\. Java version (for both simple attention and Google's multi-head attention):
``` bash
java -jar exec/plot_heatmap.jar
```
then select the attention file on the GUI.


## Data Format

The name of the attention file should end with ".attention" extension especially when using exec/plot_heatmap.jar and the file should be json format:

``` python
{ "0": {
    "source": " ",       # the source sentence (without <s> and </s> symbols)
    "translation": " ",      # the target sentence (without <s> and </s> symbols)
    "attentions": [         # various attention results
        {
            "name": " ",        # a unique name for this attention
            "type": " ",        # the type of this attention (simple or multihead)
            "value": [...]      # the attention weights, a json array
        },      # end of one attention result
        {...}, ...]     # end of various attention results
    },   # end of the first sample
  "1":{
        ...
    },       # end of the second sample
    ...
}       # end of file
```

Note that due to the hard coding, the `name` of each attention should contain "encoder_decoder_attention", "encoder_self_attention" or "decoder_self_attention" substring on the basis of its real meaning.

The `value` has shape [length_queries, length_keys] when `type`=simple and has shape [num_heads, length_queries, length_keys] when `type`=multihead.

For more details, see [attention.py](https://github.com/zhaocq-nlp/NJUNMT-tf/blob/master/njunmt/inference/attention.py).

## Demo

The `toydata/toy.attention` is generated by a NMT model with a self-attention encoder, Bahdanau's attention and a RNN decoder using [NJUNMT-tf](https://github.com/zhaocq-nlp/NJUNMT-tf).

1\. Execute the python version (for simple attention only):

``` bash
python exec/plot_heatmap.py --input toydata/toy.attention
```
It will plot the traditional attention heatmap:

<div align="center">
  <img src="https://github.com/zhaocq-nlp/Attention-Visualization/blob/master/toydata/figures/py-heatmap1.png"><br><br>
</div>


2\. As for java version (for both simple attention and Google's multihead attention), execute
``` bash
java -jar exec/plot_heatmap.jar
```
then select the `toydata/toy.attention` on the GUI.

<div align="center">
  <img src="https://github.com/zhaocq-nlp/Attention-Visualization/blob/master/toydata/figures/java-heatmap1.png"><br><br>
</div>
<div align="center">
  <img src="https://github.com/zhaocq-nlp/Attention-Visualization/blob/master/toydata/figures/java-heatmap2.png"><br><br>
</div>

The words on the left side are attention "queries" and attention "keys" are on the right. Click on the words on the left side to see the heatmap:

<div align="center">
  <img src="https://github.com/zhaocq-nlp/Attention-Visualization/blob/master/toydata/figures/java-heatmap3.png"><br><br>
</div>

Here shows the traditional `encoder_decoder_attention` of word "obtained". The color depth of lines and squares indicate the degree of attention.

Next, select `encoder_self_attention0` under the menu bar. Click on the "获得" on the left.

<div align="center">
  <img src="https://github.com/zhaocq-nlp/Attention-Visualization/blob/master/toydata/figures/java-heatmap4.png"><br><br>
</div>

It shows the multi-head attention of the word "获得". Attention weights of head0 - head7 are displayed on the right.