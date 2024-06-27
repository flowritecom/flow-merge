def has_pytorch_bin_files(file_list):
    pytorch_bin_files = [
        file
        for file in file_list
        if file.endswith(".bin") and not file.endswith(".index.json")
    ]
    num_shards = len(pytorch_bin_files)
    if not num_shards:
        return False
    
    if num_shards == 1 and "pytorch_model.bin" in file_list:
        return True
    
    return all(
        f"pytorch_model-{i:05d}-of-{num_shards:05d}.bin" in file_list
        for i in range(1, num_shards + 1)
    )


def has_pytorch_bin_index(files):
    return any(file.endswith(".bin.index.json") for file in files)
