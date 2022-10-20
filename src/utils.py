import torch

label_mapping = lambda l: torch.tensor([1 if ll in [2,3,5,6,7,8,9,10,11,12,13,14,15,16]+[20,22,23,24,25,27]+[29,31,33,36,37]+[41,42,43] else 0 for ll in l])

DEFAULT_GROUPS = torch.tensor([0,1,16,16,2,17,17,17,17,18,18,17,17,18,18,16,17,3,4,5,21,6,19,19,22,20,7,23,8,24,9,26,10,27,11,12,25,28,13,14,15,29,30,31])
DEFAULT_FACTORS = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.2,0.2,1,1,1,1,0.5,0.2,0.2]+[0.3,0.3,1,0.3,0.3,0.3,0.3,1,1,1]+[1,1,1,1,1,1])

def calculate_weights(labels, groups=DEFAULT_GROUPS, factors=DEFAULT_FACTORS):
    n_contents = ((DEFAULT_GROUPS.max().item()+1)//2)
    content = groups % n_contents
    weights = torch.ones(len(torch.unique(labels)))

    for c in torch.unique(content):
        content_labels = labels[content[labels]==c]
        content_binary_sample_count = torch.unique(label_mapping(content_labels), return_counts=True)[1]
        content_binary_weight = content_binary_sample_count.min().pow(0.2) / content_binary_sample_count
        fake_content_labels = labels[groups[labels]==c+n_contents]
        fake_content_sample_count = torch.unique(fake_content_labels, return_counts=True)[1]
        fake_content_weight = fake_content_sample_count.pow(0.2) / fake_content_sample_count

        sum_before = fake_content_sample_count.sum()
        sum_after = (fake_content_sample_count * fake_content_weight).sum()

        weights[groups==c] = content_binary_weight[0]
        weights[groups==c+n_contents] = content_binary_weight[1] * fake_content_weight / sum_after * sum_before

    return weights * factors