import os
from collections import defaultdict
import matplotlib.pyplot as plt

def get_class_distribution_per_subset(label_dir='preprocessed_taco/labels'):
    subsets = ['train', 'val', 'test']
    all_distributions = {}

    for subset in subsets:
        class_counts = defaultdict(int)
        subset_path = os.path.join(label_dir, subset)
        if not os.path.exists(subset_path):
            continue

        for file_name in os.listdir(subset_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(subset_path, file_name), 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            class_counts[class_id] += 1

        all_distributions[subset] = dict(sorted(class_counts.items()))

    return all_distributions

def plot_class_distributions(distributions):
    subsets = list(distributions.keys())
    num_subsets = len(subsets)

    fig, axs = plt.subplots(1, num_subsets, figsize=(6 * num_subsets, 6), sharey=True)

    if num_subsets == 1:
        axs = [axs]  # make it iterable

    for ax, subset in zip(axs, subsets):
        class_counts = distributions[subset]
        class_ids = list(class_counts.keys())
        counts = list(class_counts.values())

        ax.bar(class_ids, counts, color='skyblue')
        ax.set_title(f'{subset.capitalize()} Set')
        ax.set_xlabel('Class ID')
        ax.set_xticks(class_ids)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    axs[0].set_ylabel('Number of Instances')
    plt.suptitle('Class Distribution per Dataset Split', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

from pprint import pprint
if __name__ == '__main__':
    distributions = get_class_distribution_per_subset('preprocessed_taco/labels')
    pprint(distributions)
    plot_class_distributions(distributions)
