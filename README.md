<p align="center">
  <a href="https://github.com/flowritecom/flow-merge">
   <img src="https://github.com/flowritecom/flow-merge/assets/52296520/4ee27017-7eb4-4d1f-8bae-34e43fb5f0a8" alt="Logo">

  </a>
  
  <p align="center">
    Merge Language Models with Ease
    <br />
    <a href="https://github.com/flowritecom/flow-merge#-getting-started">Getting Started</a>
    -
    <a href="https://github.com/flowritecom/flow-merge/blob/main/CONTRIBUTING.md/">Contributing</a>
    -
    <a href="https://github.com/flowritecom/flow-merge/issues">Issues</a>
    -
    <a href="http://flow-merge.com/">Website</a>
  </p>
</p>

# **👋 Welcome**

Model merging is an innovative technique that allows you to combine pre-trained and fine-tuned language models (LMs) into new models with unique capabilities.

By merging existing LMs, you can potentially create a new model that inherits the strengths and capabilities of its constituent models. This way, you can explore new model variations and experiment with different combinations without the need for expensive GPU resources or extensive training from scratch.

`flow-merge` is a fully open-source library written in Python that implements some of the most popular merge methods such as model soups, SLERP, ties-MERGING or DARE. The library is built on top of the Hugging Face `transformers` library and the deep learning framework Pytorch, and provides a simple and easy-to-use interface to merge models and upload them to the Hugging Face Hub.

# **⭐️ Features**

`flow-merge` has been designed to serve both beginners and experts in merging transformer-based language models (LMs). You don't need prior experience with merge methods or advanced knowledge of LMs; a basic understanding of LMs and the command-line interface (CLI) is sufficient.

The library walks you through the merging process, so you can focus on finding the best possible merges without getting bogged down in details of the complex merge methods. Our ultimate goal is to make language model merging simple, flexible, and customizable to your specific needs.

The key features of the library consists of:

- **Default parameter settings**: Sane default values for the most important parameters based on the experiments in the papers.
- **Input validations**: `flow-merge` validates all the user inputs before starting the merge and provides helpful error messages if something is wrong.
- **CLI and Library**: A command-line interface (CLI) for easy merging and uploading of models to the Hugging Face Hub. Also a library that you can use in your own projects.
- **Memory efficient**: `flow-merge` is designed to be memory efficient, so you can merge large models without running out of memory or without a GPU.

# **🎉 Getting started**

## **💻 Installation**
Clone the repository and navigate to the root directory:
```bash
# via ssh
git clone git@github.com:flowritecom/flow-merge.git

cd flow-merge
```

Create a new python environment and activate it. For example, with `conda`:
>Note `flow-merge` requires `python>=3.10`

```bash
conda create python>=3.10 && conda activate flow-merge
```

`flow-merge` can be installed with running `pip` inside the project directory (-e for editable install):

```bash
pip install -e .
```

## **🏎️💨 Quick start**

### Write a `flow-merge` config

A merge config is a YAML file that defines the models you want to merge and how you want to merge them.

Below is an example of a merge config that merges three models using the `addition-task-arithmetic` method and saves the merged model to the `./merged_model` directory:

```yaml
method: addition-task-arithmetic
method_global_parameters:
  scaling_coefficient: 0.7
  normalize: False
base_model: Qwen/Qwen1.5-0.5B
models:
  - model: Qwen/Qwen1.5-0.5B
  - model: Qwen/Qwen1.5-0.5B-chat
  - model: minghaowu/Qwen1.5-0.5B-OpenHermes-2.5
tokenizer:
  mode: base
  interpolation_method: linear
directory_settings:
  cache_dir: null
  local_dir: ./models
  output_dir: ./merged_model
hf_token:
  token: null
  trust_remote_code: False
device: cpu
```

The only required fields are `method`, and `models`. The `method` field specifies the merge method you want to use, and `models` is a list of models you want to merge. The rest of the fields are optionally and `flow-merge` will use the default values if they are not provided. For a complete list of the default values, see the [config file documentation](./docs/config.md).

Save the config to a file, for example `my_first_merge.yaml`.

### Run a merge

Merging models with `flow-merge` is as simple as choosing a YAML template from the [examples](./examples) folder, modifying the paths to the models you want to merge, and running the following command:

```bash
flow-merge run --config my_first_merge.yaml --model_name qwen_merge
```

### Upload the merged model to the Hugging Face Hub

After the merge is complete, you can easily upload the merged model to the Hugging Face Hub by running the following command:

```bash
flow-merge upload --model_dir ./merged_model --username <hf_user_id> --model_name qwen_merge --token <hf_token> --private <True/False>
```

# Usage

## CLI

You can check the available commands and options by running:

```bash
flow-merge --help
```

You can display the config yaml schema and the default values by running:

```bash
flow-merge schema
# extra tip: pipe to highlighted json with 'flow-merge schema | jq' or 'flow-merge schema | fx'
# where you require either 'jq' or 'fx' installed beforehand
```

You can optionally validate your config file before running the merge:

```bash
flow-merge validate --config my_first_merge.yaml
```

# **🛠️ Supported Merge methods**

Currently `flow-merge` supports most of the popular and proven merge methods.

| Method                   | Identifier                 | Paper |
| ------------------------ | -------------------------- | ---------- |
| Linear or Model Soups    | `model-soup`               | [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482) |
| SLERP                    | `slerp`                    | -                                     |
| Addition Task Arithmetic | `addition-task-arithmetic` | [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) |
| Ties-MERGING             | `ties-merging`             | [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) |
| DARE Ties-MERGING        | `dare-ties`                | [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099) |

> 📢 _We are working hard on adding more methods to the library._

## Limitations of the methods

| Method                   | Description                 | Uses a Base Model | Can Merge Multiple Models | Supports Weighted Merge |
| ------------------------ | -------------------------- | ---------- | --------------- | -------|
| Linear or Model Soups    | Averages the weights of the models | ❌         | 🟢              | 🟢 |
| SLERP                    | Smoothly interpolates between the weights of two models using spherical linear interpolation | ❌         | ❌              | ❌ |
| Addition Task Arithmetic | Obtains task vectors or deltas and applies them to the base model | 🟢         | 🟢              | 🟢 |
| Ties-MERGING             | It addresses the problem of interference between parameters from different models before merging with addition task arithmetic             | 🟢         |  🟢 |🟢              |
| DARE Ties-MERGING        | Similar to Ties-MERGING but it uses a different approach that prunes the task vectors and rescale them. | 🟢         | 🟢              | 🟢 |

# Supported LLM Architectures

`flow-merge` currently supports merging models that are based on the following architectures:

| Model type | Architecture         |
| ---------- | -------------------- |
| `qwen`     | `QwenForCausalLM`    |
| `mistral`  | `MistralForCausalLM` |
| `llama`    | `LlamaForCausalLM`   |

> 📢 _We plan to support many models and architectures more, including encoder models such as BERT-Family models too._

# Tokenizers

When merging language models, it's crucial to consider the tokenizers involved, as they convert text into tokens that the models can process.

`flow-merge` currently supports two modes for constructing the tokenizer that is used by the resulting merged model:
- `base`: Default mode. The merged model utilizes the tokenizer of the base model. If no base model is specified in the merged configuration, the first model in the models list is used as the base model.
- `merged`: If the tokenizers of the models use different vocabularies, a common vocabulary is created, and a new tokenizer is constructed based on this vocabulary.

## Interpolation of embedding and language modeling layers
If the tokenizers of the models use different vocabularies, `flow-merge` creates `input_ids` mappings for the models and linearly interpolates the embedding and language modeling layers.

Currently, only `linear` interpolation is supported.

## Special tokens
Conflicts can arise from special tokens used by different models' tokenizers, such as differing `eos_token` tokens. In such cases, `flow-merge` uses the special token of the last model in the list.

# 🚧 WIP 🚧 **📚 Additional resources**

Here we have prepared some additional resources to help developers understand the supported merge methods better.

- [Model soups](./docs/model_soups.md)
- [SLERP](./docs/slerp.md)
- [Task vectors](./docs/task_vectors.md)
- [Ties-MERGING](./docs/ties.md)
- [DARE](./docs/dare.md)

# **🗺️ `flow-merge` Roadmap**

Coming soon..

# **✨ Project showcase**

Coming soon..

# **🤝 Contributing**

Wanna pitch in? We're totally open to contributions for the core flow-merge library as well as any cool integrations built on top of it! Check out our [Contribution Guide](./CONTRIBUTING.md) for all the details on how to get started.

# **🙏 Acknowledgments**

Special thanks to these amazing projects that helped us build `flow-merge`:

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
- [Pytorch](https://pytorch.org/)

Also, a big shoutout to the authors of the papers of the merge methods implemented in `flow-merge`, and to Charles O. Goddard, creator of [mergekit](https://github.com/arcee-ai/mergekit), who inspired us to create our own merging toolkit.

Finally, thanks to Derrick Schultz for the pytorch-tensor-slerp.py gist that helped us implement the SLERP method.

# **✍️ Citation**

```
@misc{flowrite_2024_flow_merge,
  author = {The Flowrite Team},
  title = {flow-merge},
  howpublished = {\url{https://https://github.com/flowritecom/flow-merge}},
  year = {2024}
}
```
