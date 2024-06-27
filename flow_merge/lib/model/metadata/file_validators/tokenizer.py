def has_tokenizer_config(file_list):
    return "tokenizer_config.json" in file_list


def has_tokenizer_file(file_list):
    return "tokenizer.json" in file_list or any(
        file.endswith("tokenizer.vocab") for file in file_list
    )