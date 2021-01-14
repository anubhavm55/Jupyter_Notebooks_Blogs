---
toc: false
layout: post
description: My introduction to Deep Learning.
categories: [markdown]
title: Introduction to deep learning(3rd Semester)
---  

Recently I came across a deep learning course that has changed my life forever!!! I am talking about fast.ai's[FastAI](https://fast.ai) "Practical Deep Learning for Coders".
## First misunderstanding: Deep Learning only for PhDs  
    
I started deep learning in my 3rd semester minor project(currently in 6th semester) and I went through a lot of courses on youtube and coursera. At first, it was very exciting to know the maths behind all of this but when I tried to implement it I was unable to do so. I found myself searching tutorials and guides for tensorflow, keras, pytorch but nothing made sense. I was frustrated as I was unable to implement something which I understood. For example, the project that I was working on had the equation for the prediction as  
y(m,t) = Σσ<sub>1</sub>(b+m*e<sup>w1</sup>) * σ<sub>2</sub>(b-t*e<sup>w2</sup>) * e<sup>w3</sup>  
where σ<sub>1</sub> is softplus function and σ<sub>2</sub> is a sigmoid function.  
The above equation clearly states that I have to create all these layers from scratch since all the predefined models do computation of the form b+(some input)*w. Making above architecture seemed so difficult to me and I came to the conclusion that only PhDs understood intricacies of deep learning. So, at last I went to tensorflow keras tutorial, randomly added lots of dense and embedding(lol, it didn't require any embedding layer, I didn't even know back then what an embedding layer was) layers for this model, got an accuracy so bad(mape of 1 lakh) and lost all interest in deep learning.  

## Second misunderstanding: More epochs means more accuracy  
I remember I used to train the above model for 250 epochs(lmao, its funny now that I know) and leave my laptop for hours thinking that the error would decrease. After taking the fast.ai course, I've come across exceptional accuracy by training for only 3 epochs. That is why 250 seems a very large number.  

So that was my introduction to deep learning which was not very pleasant as evident.    

  
