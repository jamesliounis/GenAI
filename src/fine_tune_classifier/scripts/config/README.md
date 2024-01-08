# Overview
This directory contains configuration files and scripts essential for setting up and training the machine learning models.

# Contents

`ds_config.json`:
- This is the configuration file for DeepSpeed, a deep learning optimization library that makes distributed training easy and efficient.
- It defines various parameters such as batch sizes, learning rates, optimizer settings, and memory optimization techniques.
- **Important**: This configuration file is for illustration purposes. You can / need to modify this file to adjust the distributed training settings as per your requirements. You can find dozens of DeepSpeed configuration examples that address various practical needs in the DeepSpeedExamples repo:
```shell script
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
```
Also, feel free to check out [Hugging Face's official DeepSpeed integration](https://huggingface.co/docs/transformers/main/main_classes/deepspeed) page for additional guidance!

`model_and_training_config.py`:
- A Python script that encapsulates the configuration for the tokenizer, model, training arguments, and data collator.
- It initializes the Roberta tokenizer and model for sequence classification and sets training arguments.
- The script also points to the `ds_config.json` for DeepSpeed integration.
Usage
To modify training configurations:

Adjust parameters in `ds_config.json` if you want to change distributed training settings.
Modify `model_and_training_config.py` for changes related to the model, tokenizer, or other training arguments.

**Important**: while DeepSpeed offers several memory and compute optimizations even in non-distributed training settings, the main motivation for using it here is the certainty of the project transitioning to a distributed setup. This will then simplify the transition between the two scenarios.
However, if your requirements change, you may find yourself wishing to *not* use it for simplicity purposes. In that case, simply comment out the following line in `model_and_training_config.py`:

`deepspeed=DEEPSPEED_CONFIG_PATH,`
