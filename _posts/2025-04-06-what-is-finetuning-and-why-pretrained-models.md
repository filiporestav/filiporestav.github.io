# What is finetuning and why should we use pretrained models?
Training deep learning models is often not hard, but as the networks are deep they can have a large amount of parameters which have to be trained. This training requires a lot of computations and can limit the normal person who does not have access to powerful GPUs. Luckily, there is a solution.

## Pretrained models
Pretrained models are models, usually large, which have been trained on a large amount of data so that they are generalizable. For example, the *vision_learner* from fastai has a parameter *pretrained* which defaults to True. This sets the model's weights (basically the importance of each neuron) to values that have already been trained by experts to recognize a thousand different categories across 1.3 million photos, using the famous [ImageNet dataset](https://www.image-net.org/).
There are numerous benefits of this:
1. We don't have to train a network from scratch, and instead finetune it (more on that below), which reduces the amount of computations needed.
2. We get a trained model faster.

Using a pretrained model allows us to, before even showing it to any of our data, be capable of delivering good results. This is because the pretrained models have capabilities that are important regardless of dataset, such as edge, gradient and color detection. This is the beauty of transfer learning, transfering the knowledge from a general model to another task.

![Transfer learning](https://upload.wikimedia.org/wikipedia/commons/6/6f/Transfer_learning.svg)

Image from [Wikipedia](https://www.google.com/url?sa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FTransfer_learning&psig=AOvVaw0SgnhUPvgRw7ByjakMHDd2&ust=1743996082419000&source=images&cd=vfe&opi=89978449&ved=0CBcQjhxqFwoTCJjvwum5wowDFQAAAAAdAAAAABAE)

## Finetuning
So what is finetuning? In fastai, when using a pretrained model, vision_learner will remove the last layer, since that is always specifically customized to the original training task. The last layer will then be replaced with one or more new layers with randomized weights with an appropriate size for the dataset at hand. These are the weights which the model primarily is going to focus on when it is training. This last part of the model is called the *head*.

Finetuning in fastai is very easy. You can just use `learn.fine_tune(n_epochs)` on your learner object (your model). It works like this:
1. Use one epoch to fit the new random head to work correctly with the dataset (freeze the other weights).
2. Use the number of requested epochs when calling the method to fit the entire model, updating the later layers (especially on the head) faster than the earlier layers.

Thus, it is the head that is the part that is newly added to be specific to the new dataset.

This technique is not only for images, but for most models in general. Large Language Models with billions amount of parameters always have to be finetuned if you do not own a billion dollar company yourself and have the GPUs, time, money and knowledge to train a model from scratch.

Please guess how much it costed to train ChatGPT-4!
<details>
<summary>See answer here</summary>
  It costed more than $100 million dollars, according to Sam Altman.
</details>
