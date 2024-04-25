def has_config_json(file_list):
    return "config.json" in file_list


def has_tokenizer_config(file_list):
    return "tokenizer_config.json" in file_list


def has_tokenizer_file(file_list):
    return "tokenizer.json" in file_list or any(
        file.endswith("tokenizer.vocab") for file in file_list
    )


def has_safetensors_files(file_list):
    safetensors_files = [file for file in file_list if file.endswith(".safetensors")]
    num_shards = len(safetensors_files)
    if num_shards == 1 and "model.safetensors" in file_list:
        return True
    return all(
        f"model-{i:05d}-of-{num_shards:05d}.safetensors" in file_list
        for i in range(1, num_shards + 1)
    )


def has_safetensors_index(files):
    return any(file.endswith(".safetensors.index.json") for file in files)


def has_pytorch_bin_files(file_list):
    pytorch_bin_files = [
        file
        for file in file_list
        if file.endswith(".bin") and not file.endswith(".index.json")
    ]
    num_shards = len(pytorch_bin_files)
    if num_shards == 1 and "pytorch_model.bin" in file_list:
        return True
    return all(
        f"pytorch_model-{i:05d}-of-{num_shards:05d}.bin" in file_list
        for i in range(1, num_shards + 1)
    )


def has_pytorch_bin_index(files):
    return any(file.endswith(".bin.index.json") for file in files)


def has_adapter_files(file_list):
    return any(
        file.startswith("adapter_")
        and (file.endswith(".bin") or file.endswith(".safetensors"))
        for file in file_list
    )


integrity_checks = [
    (has_config_json, "Missing config.json file"),
]

tokenizer_checks = [
    (has_tokenizer_config, "Missing tokenizer_config.json file"),
    (has_tokenizer_file, "Missing tokenizer vocabulary file"),
]

safetensor_checks = [
    (has_safetensors_files, "Missing .safetensors files"),
    (has_safetensors_index, "Missing model.safetensors.index.json file"),
]

pytorch_bin_checks = [
    (has_pytorch_bin_files, "Missing pytorch_model .bin files"),
    (has_pytorch_bin_index, "Missing pytorch_model.bin.index.json file"),
]

adapter_checks = [
    (has_adapter_files, "Missing adapter files"),
]
