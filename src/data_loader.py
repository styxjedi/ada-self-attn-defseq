import torch
import torch.utils.data as data
import numpy as np


class DefseqDataset(data.Dataset):
    def __init__(self, root, mode='train'):
        """Set the path for dataset.
        Args:
            root: data directory.
        """
        if mode not in ['train', 'valid', 'test']:
            raise ValueError("mode must be train/valid/test")
        self.mode = mode
        data = np.load(root)
        self.word_sememes = data['word_sememes']
        if self.mode != 'test':
            self.definitions = data['definitions']

    def __getitem__(self, index):
        """Returns one data pair ( word, sememe, definition )."""
        word_sememes = torch.LongTensor(self.word_sememes[index])
        if self.mode != 'test':
            definition = torch.LongTensor(self.definitions[index])
            item = (word_sememes, definition)
        else:
            item = (word_sememes)
        return item

    def __len__(self):
        return len(self.word_sememes)


def get_loader(root, batch_size, shuffle=True, num_workers=4, mode='train'):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    defseq = DefseqDataset(root=root, mode=mode)

    if mode == 'test':
        data_loader = data.DataLoader(
            dataset=defseq,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers)
    else:
        data_loader = data.DataLoader(
            dataset=defseq,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers)
    return data_loader
