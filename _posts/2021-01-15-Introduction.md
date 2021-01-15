---
toc: false
layout: post
description: My introduction to Deep Learning.
categories: [markdown]
title: Introduction to deep learning(3rd Semester)
---  

Recently I came across a deep learning course that has changed my life forever!!! I am talking about fast.ai's [FastAI](https://fast.ai) "Practical Deep Learning for Coders". I was looking for a course that could explain the "coding" part of deep learning. One of my friend suggested MITs ["Introduction to deep learning"](https://introtodeeplearning.com), its nice but could have been nicer if the code in lab part was also explained. FastAI does that and I call it my spirit course.

## First misunderstanding: Deep Learning only for PhDs  
    
I started deep learning in my 3rd semester minor project(currently in 6th semester) and I went through a lot of courses on youtube and coursera. At first, it was very exciting to know the maths behind all of this but when I tried to implement it I was unable to do so. I found myself searching tutorials and guides for tensorflow, keras, pytorch but nothing made sense. I was frustrated as I was unable to implement something which I understood. For example, the project that I was working on had the equation for the prediction as  
y(m,t) = Σσ<sub>1</sub>(b+m\*e<sup>w1</sup>) \* σ<sub>2</sub>(b-t\*e<sup>w2</sup>) \* e<sup>w3</sup>  
where σ<sub>1</sub> is softplus function and σ<sub>2</sub> is a sigmoid function.  
The above equation clearly states that I have to create all these layers from scratch since all the predefined models do computation of the form b+(some input)\*w. Making above architecture seemed so difficult to me and I came to the conclusion that only PhDs understood intricacies of deep learning. So, at last I went to tensorflow keras tutorial, randomly added lots of dense and embedding(lol, it didn't require any embedding layer, I didn't even know back then what an embedding layer was) layers for this model, got an accuracy so bad(mape of 1 lakh) and lost all interest in deep learning.  
For those who are looking for the code to implement above architecture:  
```python
  class DenseLayer1(tf.keras.layers.Layer):
  def __init__(self, n_output_nodes):
    super(DenseLayer1, self).__init__()
    self.n_output_nodes=n_output_nodes

  def build(self, input_shape):
    self.W = self.add_weight("weight", shape=[1, self.n_output_nodes])
    self.b = self.add_weight("bias", shape=[1, self.n_output_nodes]) 

  def call(self, x):
    z = self.b - x*(tf.math.exp(self.W))
    y = tf.math.softplus(z)
    return y
    
  class DenseLayer2(tf.keras.layers.Layer):
  def __init__(self, n_output_nodes):
    super(DenseLayer2, self).__init__()
    self.n_output_nodes=n_output_nodes

  def build(self, input_shape):
    self.W = self.add_weight("weight", shape=[1, self.n_output_nodes])
    self.b = self.add_weight("bias", shape=[1, self.n_output_nodes]) 

  def call(self, x):
    z = self.b + x*(tf.math.exp(self.W))
    y = tf.sigmoid(z)
    return y
    
  class DenseLayer3(tf.keras.layers.Layer):
  def __init__(self, n_output_nodes):
    super(DenseLayer3, self).__init__()
    self.n_output_nodes=n_output_nodes

  def build(self, input_shape):
    self.W = self.add_weight("weight", shape=[1, self.n_output_nodes])

  def call(self, x):
    y = x*(tf.math.exp(self.W))
    return tf.reduce_sum(y,1)
```

Let me explain how easy it is to implement above architecture which seemed like a gigantic task back then. I am doing tensorflow because currently I am enrolled in a deep learning course in my college which is taught in tensorflow, but it doesn't matter, the difference is only in the syntax and one can easily convert to pytorch in a week or so.

In tensorflow, every layer is a defined as a class and the first function is same for all of the 3 layers which defines the number of neurons in that respective layer, a parameter that you have to pass in the variable n_output_nodes. Second function is also same, randomly initializes weights and biases. The shape of these values can change depending on your data. For our case, we require only one weight for one input and we have n_output_nodes neurons so shape is defined like that. If, on the other hand we had images as inputs then for one input(one image) we may keep weights for each of the pixels in the image, therefore we would have defined the shape as \[no. of pixels, n_out_nodes]. For those who do not understand this, what we generally do is convert image into 2d matrix and then convert it to 1d array by keeping all the values from the matrix contiguously for easy computation. Then the third function, which defines what function we want to apply to inputs and parameters(other name for weights and biases combined). You can perform operations on these custom layers now as you perform on predefined layers. That is it, that is what I had to do back then, if only I found a course which explained the coding part rather than the math part.

## Second misunderstanding: More epochs means more accuracy  
I remember I used to train the above model for 250 epochs and leave my laptop for hours thinking that the error would decrease. Back then I thought that infinite epochs could make a perfect 100% accurate model. After taking the fast.ai course, I've come across exceptional accuracy by training for only 3 epochs. That is why 250 seems a very large number.  

So that was my introduction to deep learning which was not very pleasant as evident.    

  
