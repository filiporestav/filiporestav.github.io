# Building a multiclass classifier with fastai: airplanes, cars, and pets

The fastai course has shown me how deep learning can tackle complex tasks, like recognizing different objects in images. Inspired by a course example, I built a multiclass classifier to identify five categories: airplanes, automobiles, birds, cats, and dogs. Using fastai’s tools, I scraped images from the web, trained a model, and analyzed its performance. This post shares my journey, methods, results, and what I learned.

## Why multiclass classification?

Classifying images into multiple categories is a powerful skill. A few years ago, distinguishing a dog from an airplane was a major challenge, but fastai makes it approachable. My goal was to train a model to recognize five distinct classes, testing deep learning’s ability to handle diverse objects.

## Experiment: Classifying five categories

I followed these steps:
1. Scraped images of airplanes, automobiles, birds, cats, and dogs using DuckDuckGo.
2. Organized the data with fastai’s `DataBlock`, which holds the training and validation data.
3. Trained a model with `vision_learner`.
4. Analyzed results with a confusion matrix and t-SNE visualization.

### Step 1: Collecting images
I used the `duckduckgo_search` library to download images for each class. To ensure variety and generalization of the model, I searched for “photo,” “sun photo,” and “shade photo” (e.g. “dog photo”, “dog sun photo”). Here’s the code:

```python
from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *

def search_images(term, max_images=3):
    return L(DDGS().images(term, max_results=max_images)).itemgot('image')

searches = 'airplane', 'automobile', 'bird', 'cat', 'dog'
path = Path('classification_images')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    download_images(dest, urls=search_images(f'{o} sun photo'))
    download_images(dest, urls=search_images(f'{o} shade photo'))
    resize_images(path/o, max_size=400, dest=path/o)
```

I aimed for ~15 images per class (5 per search), just to make it quick as it is a proof-of-concept. However, for much better performance more images are needed. I also resized the images to 400 pixels to save space and make them the same size. I removed broken images to prevent errors:

```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
```

**Note:** My code used `max_images=5` for testing, but in practice, I increased it to gather more images, totaling ~100–200 per class.

### Step 2: Preparing data
Fastai’s DataBlock organized the images into training (80%) and validation (20%) sets:

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)
```

This code:
* Takes images as inputs and categories (airplane, automobile, bird, cat, dog) as outputs.
* Finds all images in the classification_images folder.
* Splits data randomly with a fixed seed for reproducibility.
* Resizes images to 192x192 pixels by squishing.

Here is a sample batch from my data:
![batch](/images/batch_example.png)

### Step 3: Training the model
I used vision_learner with ResNet18, a pretrained model from ImageNet, and trained it for 3 epochs:
```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

**Method selection:** I chose vision_learner for its pretrained weights, which let me train on a small dataset without starting from scratch. ResNet18 is fast and effective, as noted in the fastai course, balancing speed and accuracy for my ~500–1000 images.

**Algorithm:** The model is a convolutional neural network (CNN), which scans images for patterns like shapes or textures. Fine-tuning adjusts weights to minimize errors, using a loss function called cross-entropy loss, ideal for multiple classes:

$L = -\sum_{c=1}^My_{i}\log(\hat{y_{i}})$

where (M) is the number of classes (5), $y_{i}$ is the true label (1 for the correct class, 0 otherwise), and $\hat{y}_i$ is the predicted probability. Fastai automatically selects this loss for multiclass tasks.

### Step 4: Analyzing results
After training, I evaluated the model’s performance.

**Accuracy:** The model achieved ~97% accuracy (error rate ~3%) on the validation set, meaning it correctly classified 97% of unseen images. This is impressive given our very small dataset and shows the [power of finetuning](https://filiporestav.github.io/2025/04/06/what-is-finetuning-and-why-pretrained-models.html).

**Confusion matrix:** I visualized errors with a confusion matrix:

```python
from fastai.interpret import ClassificationInterpretation
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

![small_cf](/images/small_cf.png)

I also did a t-SNE visualization, which allowed me to explore how the model "sees" images. This is done by projecting features from the penultimate layer into 2D. I won't paste my code here. But we can clearly see that the penultimate layer has learnt distinct features for each class.

![t-sne](/images/tsne-second-to-last.png)

### Challenges
The main challenges here are the time for training and for downloading images. Ideally one should have high network speed and a GPU for training. I had remote access to my university's GPUs (RTX 2080) but I could not always use them since they were either busy or the PC was turned off. However, running the model at a smaller scale still worked on my laptop.
