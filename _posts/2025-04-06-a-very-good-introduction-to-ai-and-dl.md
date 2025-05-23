# fastai - a very good introduction to AI and Deep Learning

In the recent weeks I have learnt the remarkable power of fastai, a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains. It is highly useful for both practitioners and researchers, since it allows low-level components to be mixed and matched up to build new approaches.

fastai has a book, and also a free course. The [Github repository](https://github.com/fastai/fastai) contains most of the information to get started, including the link to the free course and the book on Amazon.

The book looks like this and can be read for free online [here](https://github.com/fastai/fastbook). The whole book is written as Jupyter notebooks. This makes the reader able to execute all the code oneself, which improves learning and understanding.
![fastai book](https://course.fast.ai/images/book.png)

## The course's teacher
The course's teacher is Jeremy Howard, a machine learning teacher with around 30 years experience of teaching with a history of very high ranking on Kaggle, which he now is President and Chief Scientist of. He is now a deep learning researcher here at UQ, which is very cool. The course is very pedagogical and does not require much prior knowledge.

## the vision_learner
One particular thing I want to mention in this post is the fastai `vision_learner` class. It creates a CNN with a pretrained backbone (e.g. ResNet18). It is very simple to setup and here is a setup example:

`
from fastai.vision.all import *
path = untar_data(URLs.PETS) # download fastai dataset of pets
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2) # create a dataloader, with 20% validation set
learn = vision_learner(dls, resnet18, metrics=accuracy) # create the learner/model, pretrained from resnet18 and optimized for accuracy
learn.fine_tune(1) # fine-tune it for 1 epoch
`
ResNet is the fastest widely used computer vision model. So even for classifying in images, this can be done with a couple lines of code. Some years ago, this was a *very* difficult task.

## Interesting topics from the fastai course
The fastai course is comprehensive, so there is no point of me for explaining all the things I have learnt from the course. Also, as you can see on my [blog](https://filiporestav.github.io/), I have specific posts where I dive deeper into the specific topics I have learnt.

Nonetheless, some interestings topics I have learnt about are:
* What a neural network is, and the history behind it
* How to train a neural network, and different training techniques
* Challenging issues when training neural networks (e.g. overfitting)
* Metrics for evaluating models
* [What fine-tuning is and why to use pretrained models](https://github.com/filiporestav/filiporestav.github.io/blob/master/_posts/2025-04-06-what-is-finetuning-and-why-pretrained-models.md)
* Importance of having a valiation and test sets
* What to do with time series data
* [The large amount of areas that deep learning can be applied to](https://filiporestav.github.io/2025/04/06/the-large-amount-of-areas-deep-learning-can-be-applied-to.html)

Some of the topics above have links, where I have written more comprehensive about my learnings. Stay posted, more links will be added as I progress in my learnings and in the course! 😎
