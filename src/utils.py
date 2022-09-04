import torch

label_mapping = lambda l: torch.tensor([1 if ll in [2,3,5,6,7,8,9,10,11,12,13,14,15,16,20,22,23,24,25,27] else 0 for ll in l])

DEFAULT_GROUPS = [0,1,8,8,2,9,9,9,9,10,10,9,9,10,10,8,9,3,4,5,13,6,11,11,14,12,7,15]
DEFAULT_FACTORS = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.2,0.2,1,1,1,1,0.5,0.2,0.2]

def calculate_weights(labels, groups=DEFAULT_GROUPS, factors=DEFAULT_FACTORS):
    content = groups % 8
    weights = torch.ones(len(torch.unique(labels)))

    for c in torch.unique(content):
        content_labels = labels[content[labels]==c]
        content_binary_sample_count = torch.unique(label_mapping(content_labels), return_counts=True)[1]
        content_binary_weight = content_binary_sample_count.min().pow(0.2) / content_binary_sample_count
        fake_content_labels = labels[groups[labels]==c+8]
        fake_content_sample_count = torch.unique(fake_content_labels, return_counts=True)[1]
        fake_content_weight = fake_content_sample_count.pow(0.2)

        sum_before = fake_content_sample_count.sum()
        sum_after = (fake_content_sample_count * fake_content_weight).sum()

        weights[groups==c] = content_binary_weight[0]
        weights[groups==c+8] = content_binary_weight[1] * fake_content_weight / sum_after * sum_before

    return weights * factors