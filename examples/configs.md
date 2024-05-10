Title: configs refactor

# Working with full models using models

```yaml
method: model-soups
method_global_parameters:
  normalize: true
base_model: Qwen/Qwen1.5-0.5B
models:
  - model: Qwen/Qwen1.5-0.5B
    weight: 0.5
  - model: Qwen/Qwen1.5-0.5B-chat
    weight: 0.9
```

# Using definition

```yaml
definition:
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B
        range: [0, 32]
        weight: 0.5
        base: true
      - model: Qwen/Qwen1.5-0.5B-chat
        range: [0, 32]
        weight: 0.9
    method:
      name: model-soups
      normalize: true
```
>This is exactly the same config file as the one above but using different attributes

# Piece-wise assembly of models

## Same method for all models

```yaml
method: model-soups
method_global_parameters:
  normalize: true
definition:
    slice:
      sources:
        - model: Qwen/Qwen1.5-0.5B
          range: [0, 16]
          weight: 0.5
          base: true
        - model: Qwen/Qwen1.5-0.5B-chat
          range: [0, 16] # Number of total layers should be the same for all sources
          weight: 0.9
    # If base is not defined, the first model will be considered as base and the rest of the layers populated with the base model
```

Another example:
```yaml
method: ties-merging
method_global_parameters:
  top_k: 0.2
  normalize: true
definition:
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B
        range: [0, 12]
        weight: 0.5
      - model: Qwen/Qwen1.5-0.5B-chat
        range: [10, 22]
        weight: 0.9
      - model: Qwen/Qwen1.5-0.5B-chat
        range: [10, 22]
        weight: 0.3
```

## Different methods for different models

```yaml
definition:
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B
        range: [0, 12]
        weight: 0.5
      - model: Qwen/Qwen1.5-0.5B-chat
        range: [10, 22]
        weight: 0.9
    method:
      name: slerp
      t: 0.5
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B
        range: [0, 12]
        weight: 0.5
        base: true
      - model: Qwen/Qwen1.5-0.5B-chat
        range: [10, 22]
        weight: 0.9
    layers: [self_attn, mlp]
    method:
      name: addition-task-arithmetic
      scaling_coefficient: 0.4
      normalize: true
```

# Layer stacking
```yaml
method: layer-stacking
definition:
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B
        range: [0, 12]
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B-chat
        range: [10, 22]
```

or combined with other methods:
```yaml
definition:
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B
        range: [0, 12]
        weight: 0.5
      - model: Qwen/Qwen1.5-0.5B-chat
        range: [10, 22]
        weight: 0.9
    method:
      name: slerp
      t: 0.5
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B
        range: [0, 12]
        weight: 0.5
        base: true
      - model: Qwen/Qwen1.5-0.5B-chat
        range: [10, 22]
        weight: 0.9
    layers: [self_attn, mlp]
    method:
      name: addition-task-arithmetic
      scaling_coefficient: 0.4
      normalize: true
  slice:
    sources:
      - model: Qwen/Qwen1.5-0.5B
        range: [0, 12]
    method:
      name: layer-stacking
```

