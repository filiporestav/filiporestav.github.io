# 👋 Welcome to my blog!

<img src="https://media.giphy.com/media/26AHONQ79FdWZhAI0/giphy.gif" width="200" align="right">

Hey there! I'm **Filip**, and this blog is my personal space to document my learning journey in:

- 🧠 `ELEC4630` — *Computer Vision and Deep Learning* at the University of Queensland
- 🐣 The **fast.ai** course — a deep dive into practical, cutting-edge deep learning

Here, you'll find:
- 🔬 Projects and experiments I've worked on
- 🤖 Learnings, insights, and cool tricks
- 📉 Model results, visualizations, and failure cases
- 💭 Reflections on what clicked (and what didn’t!)

---

## 🧭 Why this blog?

Writing is thinking. This blog helps me:
- Track my growth 🚀  
- Explain complex topics in a simple way 🧠  
- Share resources with others 📚  
- Learn by teaching ✍️  

> _"If you can't explain it simply, you don't understand it well enough."_  
> — *Albert Einstein*

---

## 🔍 Topics I explore

- 📷 **Computer Vision**: CNNs, object detection, segmentation  
- 🐍 **Python & PyTorch**: From basics to production-ready code  
- 📚 **Fast.ai** magic: Quick and easy ways to build production-ready models by taking advantage of transfer learning, data blocks, augmentation  
- 📊 **Visualization techniques**: Confusion matrices, t-SNE embeddings  
- 🧪 **Experimentation**: Hyperparameter tuning, overfitting, and underfitting

---

## 🧠 Sample insight: making the most of `vision_learner`
One thing that really stood out to me early in the fastai course was just how much is packed into a single line of code. When I first wrote:
```python
learn = vision_learner(dls, resnet18, metrics=accuracy)
```

...I didn’t realize what a big deal that was.

This line isn’t just creating a model - it’s:
* loading a pretrained ResNet18,
* attaching a new head tailored to the number of classes in my dataset,
* applying sensible default loss functions and optimizers,
* and setting up data augmentation, normalization, and tracking metrics.

It abstracts away so much boilerplate, letting you focus on experimenting and learning.

### Transfer learning done right
The idea behind this is transfer learning: instead of training a model from scratch, we start with one that’s already been trained on a massive dataset (ImageNet). That means the model already "knows" what edges, textures, and basic shapes look like - so when we fine-tune it on a new task, we’re just teaching it how to combine those building blocks in a new way.

In my own project where I did a [multiclass classifier](https://filiporestav.github.io/2025/04/16/building-a-multiclass-classifier-with-fastai.html) for five different classes, this meant I could get surprisingly good results on relatively small datasets, even with just a few epochs of training. For example, using fine_tune(3) gave me solid performance with minimal effort:

```python
learn.fine_tune(3)
```

What’s happening here is clever too: fastai first "freezes" the pretrained layers so only the new head trains for a bit. Then it "unfreezes" the rest and trains the whole model with a lower learning rate - this avoids wrecking the pretrained weights too quickly.

### Reflecting on it
Coming from a background where I’m used to doing a lot of manual setup, such as defining layers, tensors, etc., fastai’s design feels like a cheat code. But it's not just convenience - it reflects a deep understanding of how deep learning works in practice. I’ve started to realize that abstraction, when done right, doesn’t take away from learning—it actually lets you go deeper, faster.

---

## 🔗 Explore my posts

👉 Check out why finetuning is very useful: **[What is finetuning and why should we use pretrained models?](https://filiporestav.github.io/2025/04/06/what-is-finetuning-and-why-pretrained-models.html)**  
👉 Or dive into: **[Training a multi-class image classifier with fast.ai](https://filiporestav.github.io/2025/04/16/building-a-multiclass-classifier-with-fastai.html)**

---

## 🚀 Stay Connected

Got ideas, feedback, or memes? Open an issue, fork the repo, or just drop a message.

Happy learning! ✌️  
— *Filip*
