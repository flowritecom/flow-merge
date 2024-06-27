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
