# GenAI
This repo includes various tasks that I have tried out regarding basic Generative AI exercises, such as building a multimodal LM or fine-tuning a classifier. 


# Multimodal LM

You can start by running this command:

```shell script
cd src/multimodal_lm/
```
You can then follow the instructions outlined in the directory's README!

**Why Our Solution Stands Out**:

1.	**Multimodal Learning**: Traditional ML approaches typically handle one type of data, but our model thrives on diversity. By integrating textual data from reviews with numerical and categorical data (like user behavior and product details), we ensure a more holistic understanding of user sentiment.

2.	**State-of-the-Art Text Processing**: We've incorporated [DistilBERT](https://arxiv.org/abs/1910.01108), a lightweight variant of the BERT model. DistilBERT retains 97% of BERT's performance but runs 60% faster and is 40% smaller in size! This ensures you get the power of deep learning without the associated overhead.

3.	**Scalable & Efficient**: The model is designed with scalability in mind. As your data grows, our solution will continue to accommodate and process efficiently.

4.	**Modularity**: We've deliberately separated out the tokenization process, making it easier for future updates or modifications without disturbing the primary model architecture.

5.	**Optimized for Real-world Scenarios**: We've incorporated the Standard Scaler from Scikit-learn for numerical data. This ensures that the model isn't biased by varying scales across different numerical features, leading to more accurate predictions.

6.	**Cloud Integration**: The solution is tailored for easy deployment on the cloud. Whether you're thinking of training on SageMaker or deploying via AWS Lambda, we've got you covered!

# Fine-tune classifier on AWS 

Please refer to [this section](https://github.com/jamesliounis/HuggingFace-TakeHome/tree/build_model_scripts/src/fine_tune_classifier) of the repository for all sub-questions, or run this command:

```shell script
cd src/fine_tune_classifier/
```
The MLOps workflow can also be found in the README of that section. 



