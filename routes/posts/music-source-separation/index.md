---
template: post.html
title: Using Deep Learning to isolate vocals from songs
author: Julian Vossen
img: /posts/music-source-separation/vocalstripper.jpg
abstract: A short post on isolating vocals from songs by training a WaveNet-like model on raw audio
width: 2
---

Splitting audio tracks into individual audio sources, such as vocals or instruments, is also known as source separation.
This problem has been approached with Deep Learning by first transforming the audio signal from the time domain into the frequency domain, applying a 2D convolutional network to mask the frequencies and, finally, transforming the masked frequencies back into the time domain.

If you are now wondering what time domain and frequency domain are, here is a short explanation: In the time domain, an audio signal is a one-dimensional vector, where each element corresponds to the amplitude of the signal at a given time. Typical audio signals are sampled at around 44 thousand samples per second. You can think of the amplitude of an audio signal as the displacement of a speaker membrane playing the sound. We can transform this time series of samples into the frequency domain, by chopping the series into small batches and determining for each of the batches, sine waves of which frequencies we would need to overlay to obtain the same pattern. A signal in the frequency domain is represented as a 2D frequency-over-time matrix. Each element of this matrix determines, how much of a sine wave of this frequency is in the signal at a given time step. For a more mathematical description, have a look at [Fourier transforms](https://en.wikipedia.org/wiki/Fourier_transform).

## Processing Raw Audio
Starting with the introduction of [WaveNet](https://arxiv.org/pdf/1609.03499.pdf), the trend in speech generation seems to be to generate raw audio, i.e. a time domain signal. Therefore, I was wondering if we could approach source separation similarly, by simply feeding a 1D time series of samples into a model and train it to output the 1D series of samples for the vocal signal.

The challenge, when working with raw audio, is the vast number of samples per second. The signal of a short drum hit of 0.5 seconds spans across 22 thousand samples in a 44.1KHz audio signal. Therefore, a model identifying this entire drum hit would ideally be able to combine the information from 22 thousand input samples. This is a large context to consider for a neural network. The spacial extend of inputs (e.g. the number of samples) used to generate an output, is also called receptive field. WaveNet uses __dilated convolutions__ to increase the receptive field, while preserving the spacial resolution. Strided convolution (typical for classification) uses dense kernel matrices, but skips convolutions to increase the receptive field, resulting in decreased spacial resolution. Dilated convolutions, on the other hand, use larger but sparse kernel matrices to increase the receptive field while preserving the spacial resolution. The difference is shown in these animation borrowed from [vdumoulin](https://github.com/vdumoulin):

<div class="flex flex-row">
<div class="w-1/2">
<img src="stride.gif" style="width: 60%; margin-left: 20%;"/>
<p style="text-align: center">Strided convolution</p>
</div>

<div class="w-1/2">
<img src="dilation.gif" style="width: 60%; margin-left: 20%;"/>
<p style="text-align: center">Dilated convolution</p>
</div>
</div>


Apart from the repeated blocks of layers of dilated convolutions, key features of the WaveNet architecture include: 

* __Skip connections__ and __residual connections__, to allow direct information and gradient flow between the output and early layers in the network
* __Gated activations__: Single layers with single activations are replaced by two parallel layers, one with a _tanh_ activation and the other with a _sigmoid_. The output of both activations is multiplied to obtain the final activation. My intuitive explanation of the motivation behind this is to factorise a layer into a _tanh_ layer which learns what a feature's impact would be if it had any impact and a _sigmoid_ layer which learns if a feature has an impact, without learning what the impact would be. Similar to LSTMs or GRUs, the _sigmoid_ layer effectively helpts to protect the information in the _tanh_ layer.
* __Causal convolutions__: Normal convolutional kernels are symmetric meaning the output of a kernel is computed from inputs to all sides of the output. In causal convolutions, the kernels are masked such that the output depends only on inputs on one side of it. This way, WaveNet can generate new samples from past samples, without peeking into the future.

For the source separation model, we use 3 blocks of 10 convolutional layers each, with dilation rates of [1, 2, 4, ..., 512]. We use residual connections, skip connections and gated activations. However, we don't need causal convolutions, as the complete source signal is given before the model evaluation and we would like the model to be able to exploit context to both sides of the output. The model output is the sum of outputs of all the skip layers subject to a final convolutional layer to obtain the 2 output channels needed for stereo audio. We use a linear activation in the final layer as this should result in better gradients when combined with the MAE loss between the target vocal signal and the model output. Here is the implementation of the model in Keras:

```python
from keras.models import Model
from keras.layers import Input, Conv1D, Activation, Add, Multiply

def VocNet(nb_filters=128, nb_skip_filters=192, 
           nb_layers=30, nb_layers_per_stage=10):
    inputs = Input((None, 2))
    h = Conv1D(nb_filters, 1, activation='relu', padding='same')(inputs)
    
    skips = []
    for lay in range(nb_layers):
        dil_rate = 2 ** (lay % nb_layers_per_stage)

        tanh = Conv1D(nb_filters, 2, dilation_rate=dil_rate, 
                      activation='tanh', padding='same')(h)
        sigm = Conv1D(nb_filters, 2, dilation_rate=dil_rate, 
                      activation='sigmoid', padding='same')(h)

        gated = Multiply()([tanh, sigm])
        
        residual = Conv1D(nb_filters, 1, activation='relu', padding='same')(gated)
        h = Add()([h, residual])

        skip = Conv1D(nb_skip_filters, 1, activation='relu', padding='same')(gated)
        skips.append(skip)
        
    out = Add()(skips)
    out = Conv1D(2, 1, activation='linear', padding='same')(out)
    
    return Model(inputs=[inputs], outputs=[out])
```

The training data is gathered by downloading instrumental music and acapellas from YouTube. We can generate arbitrarily many training examples by overlaying random snippets of instrumentals and acapellas. Note that we do not require the instrumentals and vocals to stem from the same song, which makes creating a large data set much easier.

## Results
Here are a few model outputs for songs where neither the acapella nor the instrumental was part of the training set. For each of the examples, the first audio file is the original song and the second file is the model output.

#### Adele - Skyfall
<div>
<audio controls>
  <source src="skyfall.mp3" type="audio/mpeg"/>
</audio>
</div>
<div>
<audio controls>
  <source src="skyfall_acapella.mp3" type="audio/mpeg"/>
</audio>
</div>

#### Portugal. The Man - Feel It Still
<div>
<audio controls>
  <source src="feel_it_still.mp3" type="audio/mpeg"/>
</audio>
</div>
<div>
<audio controls>
  <source src="feel_it_still_acapella.mp3" type="audio/mpeg"/>
</audio>
</div>

#### Pharrell Williams - Happy
<div>
<audio controls>
  <source src="happy.mp3" type="audio/mpeg"/>
</audio>
</div>
<div>
<audio controls>
  <source src="happy_acapella.mp3" type="audio/mpeg"/>
</audio>
</div>

#### Red Hot Chili Peppers - Otherside
<div>
<audio controls>
  <source src="otherside.mp3" type="audio/mpeg"/>
</audio>
</div>
<div>
<audio controls>
  <source src="otherside_acapella.mp3" type="audio/mpeg"/>
</audio>
</div>

The results are clearly not good enough to be practically useful. However, these results also show that performing source separation on raw audio signals is viable. Interestingly, the kind of error that the model makes, sounds qualitatively different from the kind of error that seems typical for source separation in the frequency domain. Often, the output of frequency-domain models has a flanger-like sound to it. Therefore, it could be interesting to combine both approaches.

## Potential for improvement: more context, perceptual loss
I suspect there are two main potentials for improving this raw audio approach to source separation:

* The receptive field of the above model is 3 x 512 = 1536 samples or 0.035s at a sample rate of 44.1KHz. This feels like quite little information for telling apart vocals from instrumentals. For increasing the context, we could use a model with more layers and higher dilation rates or introduce a form of global context. The global context could be a compact vector representation of the whole song. The source separation network would then be conditioned on this global representation. 

* In this example, we used the MAE loss function. I imagine that an output with a low MAE to the target waveform is not necessarily an output which sounds good to us. E.g. if we imagine a wave being offset by a constant distance over time, it might sound better to us then a wave wiggling around the target with the same distance, even though the MAE would be the same. Therefore, a GAN architecture with a perceptual loss using a discriminator network, might yield much better results. 

<br>