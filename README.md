# RT_sum README

Text summarization using tensorflow seq2seq library.

Input is XML file of academic research articles (e.g. Nature papers).

Target output is list of article abstracts.



## Model

LSTM Encoder-decoder pair with attention mechanism.

```
=== LSTM with a forget gate ===
The compact forms of the equations for the forward pass of an LSTM unit with a forget gate are:<ref name="lstm1997" /><ref name="lstm2000">{{Cite journal 
 | author = Felix A. Gers
 | author2 = Jürgen Schmidhuber
 | author3 = Fred Cummins
 | title = Learning to Forget: Continual Prediction with LSTM
 | journal = [[Neural Computation (journal)|Neural Computation]]
 | volume = 12
 | issue = 10
 | pages = 2451–2471
 | year = 2000
 | doi=10.1162/089976600300015015
| citeseerx = 10.1.1.55.5709
 }}</ref>

:<math>
\begin{align}
f_t &= \sigma_g(W_{f} x_t + U_{f} h_{t-1} + b_f) \\
i_t &= \sigma_g(W_{i} x_t + U_{i} h_{t-1} + b_i) \\
o_t &= \sigma_g(W_{o} x_t + U_{o} h_{t-1} + b_o) \\
c_t &= f_t \circ c_{t-1} + i_t \circ \sigma_c(W_{c} x_t + U_{c} h_{t-1} + b_c) \\
h_t &= o_t \circ \sigma_h(c_t)
\end{align}
</math>

where the initial values are <math>c_0 = 0</math> and <math>h_0 = 0</math> and the operator <math>\circ</math> denotes the [[Hadamard product (matrices)|Hadamard product]] (element-wise product). The subscript <math>t</math> indexes the time step.

==== Variables ====
*<math>x_t \in \mathbb{R}^{d}</math>: input vector to the LSTM unit
*<math>f_t \in \mathbb{R}^{h}</math>: forget gate's activation vector
*<math>i_t \in \mathbb{R}^{h}</math>: input gate's activation vector
*<math>o_t \in \mathbb{R}^{h}</math>: output gate's activation vector
*<math>h_t \in \mathbb{R}^{h}</math>: hidden state vector also known as output vector of the LSTM unit
*<math>c_t \in \mathbb{R}^{h}</math>: cell state vector
*<math>W \in \mathbb{R}^{h \times d}</math>, <math>U \in \mathbb{R}^{h \times h} </math> and <math>b \in \mathbb{R}^{h}</math>: weight matrices and bias vector parameters which need to be learned during training

where the superscripts <math>d</math> and <math>h</math> refer to the number of input features and number of hidden units, respectively.

==== [[Activation function]]s ====
* <math>\sigma_g</math>: [[sigmoid function]].
* <math>\sigma_c</math>: [[hyperbolic tangent]] function.
* <math>\sigma_h</math>: hyperbolic tangent function or, as the peephole LSTM paper<ref name="peepholeLSTM"/><ref name="peephole2002"/> suggests, <math>\sigma_h(x) = x</math>.

```

