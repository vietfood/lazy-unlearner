import pandas as pd
from datasets import concatenate_datasets, load_dataset

# fmt: off

# This is created by using "Gemini-2.5-Pro" to classify each data on AdvBench and MaliciousInstruct datastes to two categories "Biology" and "Cyber Security" based on original WMDP categories.
BIOLOGY_HARZARDOUS_INDICES = [
    3, 7, 20, 21, 29, 36, 37, 50, 54, 57, 71, 79, 100, 108, 110, 140, 143, 145,
    152, 153, 155, 196, 217, 274, 300, 323, 335, 337, 348, 353, 358, 363, 364,
    371, 383, 399, 404, 424, 455, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514
]

CYBER_CRIME_INDICES = [
    0, 1, 4, 8, 10, 12, 14, 16, 17, 18, 28, 38, 41, 43, 44, 46, 47, 48, 52, 53,
    55, 56, 59, 62, 63, 66, 67, 70, 73, 75, 78, 80, 81, 82, 83, 85, 86, 87, 89,
    91, 94, 97, 98, 101, 102, 105, 106, 107, 108, 111, 113, 114, 116, 118, 119,
    120, 122, 123, 129, 130, 131, 134, 135, 138, 144, 147, 148, 151, 157, 158,
    159, 161, 162, 163, 164, 168, 171, 173, 174, 176, 177, 180, 182, 183, 184,
    186, 190, 191, 192, 195, 196, 197, 198, 199, 201, 202, 204, 206, 207, 210,
    211, 213, 216, 218, 219, 221, 224, 225, 227, 228, 230, 231, 233, 234, 236,
    242, 243, 244, 245, 246, 249, 250, 254, 255, 257, 260, 261, 262, 263, 265,
    266, 267, 269, 270, 271, 272, 274, 276, 277, 280, 281, 283, 284, 285, 286,
    287, 290, 291, 292, 293, 295, 296, 298, 300, 301, 303, 304, 305, 306, 307,
    308, 309, 310, 313, 314, 316, 317, 318, 319, 320, 323, 326, 329, 331, 332,
    333, 334, 335, 337, 340, 341, 342, 343, 350, 356, 357, 358, 360, 363, 364,
    366, 367, 368, 370, 372, 373, 375, 378, 379, 381, 382, 385, 386, 390, 393,
    397, 399, 400, 401, 405, 409, 410, 414, 419, 425, 427, 429, 430, 431, 432,
    435, 436, 437, 438, 441, 443, 447, 450, 455, 456, 457, 461, 463, 464, 465,
    467, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504
]


# fmt: on
def sample_data(data, sample_size, seed, max_len=512, key="question"):
    if max_len is not None:
        data = data.filter(lambda example: len(example[key]) < max_len)

    if sample_size is not None:
        if sample_size <= len(data):
            data = data.shuffle(seed=seed).select(range(sample_size))
        else:
            data = data.shuffle(seed=seed)

    return data


def format_qa(data, reason=True):
    def format_choice(choices):
        return "\n".join([f"{key}: {choice}\n" for key, choice in choices.items()])

    if reason:
        return """Question:\n{question}\n\nChoices:\n\n{choices}\nAnswer:{answer}\n\nReason:""".format(
            question=data["question"],
            choices=format_choice(
                {chr(idx + 65): choice for idx, choice in enumerate(data["choices"])}
            ),
            answer=chr(data["answer"] + 65),
        )
    else:
        return """Question:\n{question}\n\nChoices:\n\n{choices}\nAnswer:""".format(
            question=data["question"],
            choices=format_choice(
                {chr(idx + 65): choice for idx, choice in enumerate(data["choices"])}
            ),
        )


def load_harmbench_completion(completion_path):
    df = pd.read_csv(completion_path)
    return (
        df[df["category"] == "chemical_biological"],
        df[df["category"] == "cybercrime_intrusion"],
    )


def load_harmbench():
    harmbench_ds = load_dataset("walledai/HarmBench", "standard", split="train")

    harmbench_cyber = harmbench_ds.filter(
        lambda example: example["category"] == "cybercrime_intrusion"
    )
    harmbench_cyber = harmbench_cyber.remove_columns("category")

    harmbench_bio = harmbench_ds.filter(
        lambda example: example["category"] == "chemical_biological"
    )
    harmbench_bio = harmbench_bio.remove_columns("category")

    return harmbench_bio, harmbench_cyber


def load_adv_malicious_instruct():
    # Load AdvBench
    adv = load_dataset("walledai/AdvBench", split="train")
    adv = adv.remove_columns("target")

    # Load MaliciousInstruct
    malicious = load_dataset("walledai/MaliciousInstruct", split="train")

    # Merge two datatset
    extra_ds = concatenate_datasets([adv, malicious])

    # return (bio, cyber) dataset
    return extra_ds.select(BIOLOGY_HARZARDOUS_INDICES), extra_ds.select(
        CYBER_CRIME_INDICES
    )


def to_instructions(data, format_fn):
    if isinstance(data, list):
        result = []
        for d in data:
            result.append([format_fn(val) for val in d])
        return result
    else:
        return [format_fn(val) for val in data]


def load_wmdp(sample_size, seed=42, bio_max_len=512, cyber_max_len=512, filter_fn=None):
    wmdp_bio = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    wmdp_cyber = load_dataset("cais/wmdp", "wmdp-cyber", split="test")

    if filter_fn:
        wmdp_bio = wmdp_bio.filter(filter_fn)
        wmdp_cyber = wmdp_cyber.filter(filter_fn)

    if sample_size is not None:
        wmdp_bio = sample_data(wmdp_bio, sample_size, seed, max_len=bio_max_len)
        wmdp_cyber = sample_data(wmdp_cyber, sample_size, seed, max_len=cyber_max_len)

    return wmdp_bio, wmdp_cyber


def split(dataset, test_size):
    splits = dataset.train_test_split(test_size)
    return splits["train"], splits["test"]
