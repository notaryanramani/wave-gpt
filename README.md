# WaveGPT: A GPT Model Enhanced with WaveNet

# Contents
- [What is a Wave Net](#Introduction)
- [Archicture](#architecture)
- [Data](#data)
- [Usage](#usage)
- [Contribution](#contributing)


## Introduction

A WaveNet is a mini-gpt model enhaced with the use of a wavenet.

#### What is a GPT?  
 GPT, or Generative Pre-trained Transformer, is a type of deep learning model known for generating human-like text based on input prompts. It uses a Transformer architecture and is trained on vast amounts of text data to understand and generate natural language

#### What is a WaveNet?  
 WaveNet is a deep generative model developed by DeepMind for producing high-quality raw audio waveforms(here we are using wavenet for text data generation). It uses dilated convolutions to capture long-range dependencies in data, allowing it to capture more information to generate text with a long context.


## Architecture

### WaveGPT
<center>
    <figure>
        <img src="img/WaveGPT.png" width=400><br>
    <figcaption>WaveGPT Architecture</figcation>
    </figure>
</center>

### GPT
<center>
    <figure>
        <img src="img/GPT.png" width=300><br>
    <figcaption >GPT Architecture <br><i><a href="https://en.m.wikipedia.org/wiki/File:Full_GPT_architecture.png">image source</a></i> </figcation>
    </figure>
</center>


**Note:** We do not have the linear layer and output probablities at the end because we add the output of WaveNet.



### WaveNet
<center>
    <figure>
        <img src="img/WaveNet.png" width=300><br>
    <figcaption >WaveNet Architecture</figcation>
    </figure>
</center>

### WaveNet Layer
<center>
    <figure>
        <img src="img/WaveNetLayer.png" width=300><br>
    <figcaption >A single WaveNet Layer</figcation>
    </figure>
</center>
<br>
The diagram above illustrates the architecture of WaveGPT. The process involves:

1. **Input Sequence**: The input sequence of shape (B, T).
2. **Embedding Layer**: Converts the input sequence into embeddings of shape (B, T, C).
3. **WaveNet Block**: Processes the embeddings using dilated convolutions and produces an output of shape (B, 1, C).
4. **Transformer Block**: Processes the embeddings using multihead masked attention and produces an output of shape (B, T, C).
5. **Broadcasting and Addition**: The WaveNet output is broadcasted and added to the Transformer output, resulting in a shape of (B, T, C).
6. **Linear Layer**: The combined output is passed through a linear layer, producing logits of shape (B, T, vocab_size).
7. **Output Probabilities**: The final output probablities after applying softmax.


## Data

The data used to train the model is [OpenWebText Corpus](https://huggingface.co/datasets/Skylion007/openwebtext) that was used by used by OpenAI to train [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). For this project only first 250,000 documents have been taken. The data text file has not been uploaded to this repository as it is more than 1GB. Please refer to `data.py` to download the dataset to your local machine.

## Usage

This model is used to train a GPT and WaveNet from scratch. Please refer to `main.py` file for sample code.

## Contributing

Contributions are welcome, just raise a pull request. Feel free to raise an issue if you encounter an error or bug!