# NLP-MultiTask-Transformer
This repository contains a simplified, toy multi-task sentence transformer model

Step 1: Implement a Sentence Transformer Model

This model architecture was designed with the goal of creating a runnable sentence transformer including key aspects of transformer architectures, such as embeddings, positional encodings, self-attention, and feedforward networks with residual connections and normalization. It's a simplified toy version. 

we could increase curernt model capacity by try the following: implement a more advanced tokenizer, use sinusoidal positional encodings used in the original transformer paper, stack multiple transformer encoder layers to capture more complex patterns and hierarchical representations in the data, introduce dropout layers to prevent overfitting, try other pooling strategies like Use CLS token's embedding as the sentence representation etc. Btw use gpu instead of cpu here. 

Step 2 Multi-Task Learning Expansion

Task A: Classify online news into 3 predefined categories (sports, politics, and others).


Task B: Named Entity Recognition (NER), which outputs a sequence of labels for each token in the input sentence.

Key Changes to Implement:


for Task A (Sentence Classification):Add a Sentence Classification Head(a fully connected layer followed by softmax layer for the classification head to output probabilities over the 3 predefined categories)


Task B (NER): Add a token classification head that outputs a label for each token(a fully connected layer followed by softmax layer for the classification head to output probabilities over the 5 predefined entities)

Next Steps: 

First define loss functions for both tasks.For example, for Task A (Sentence Classification), use CrossEntropyLoss for each category. For Task B (NER), use CrossEntropyLoss applied at each token position. 

Second set up an optimizer and implement a training loop to train the model on labeled data.

Step 3: Discussion Questions

1 Situations for freezing the transformer backbone and training only the task-specific layers:

a. Pre-trained backbone on a large corpus: If the transformer backbone has already been pre-trained on a large corpus at a high computational cost, it likely captures rich semantic features. In this case, we only need to fine-tune the task-specific layers for downstream tasks, without modifying the backbone.

b. Preventing catastrophic forgetting: Continually training the transformer backbone for specific downstream tasks can lead to catastrophic forgetting, where the model loses its general language understanding. This is particularly problematic when the downstream task has a small dataset, as it increases the risk of overfitting. Freezing the backbone ensures that the model retains its general capabilities while adapting to the specific task.

2. Discuss how you would decide when to implement a multi-task model like the one in this assignment and when it would make more sense to use two completely separate models for each task.

When to use a Multi-Task model:


a. Relevant tasks. If those downsteam tasks are related tasks especially they can share common features like entence classification and NER tasks. 


b. Data limitation. When some tasks have very limited data while others has massive data, a multi-task model can share knowledge by unified architecture and transfer learning to improve limited data tasks performance.


c Parameter Efficiency training and faster inference. A shared model reduces the total number of parameters compared to two separate models when training. Also faster inference becaues running a single model is more efficient than running two separate models sequentially, also save resources cost. 

When to use two completely separate models:
a. Unrelated Tasks. For exmaple image classification and NER may not benefit from shared representations. 


b. Task-Specific Architectures


c. One task possibly negatively impacting the learning of another.  for example when optimizing for one task degrades performance on the other due to conflicting gradients.


3. Handling Data Imbalance in Multi-Task Learning  

a. Adjust Loss Weights： Increase the Loss Weight for Task B to prevents the model from being dominated by Task A's abundant data, and it can also balances the gradient contributions from both tasks.


b balance datasets by sampling etc. Over-Sample Task B Data to increase datasets size, or under-sample Task A Data to reduce its size(or use synthetic data generation to augment task B datasets ). 


c train downsteam tasks sequentially based on their datasets size. For example train the shared backbone using Task A's abundant data with  freezeing Task B's head, then freeze the backbone and train only Task B's head.

