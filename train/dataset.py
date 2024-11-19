from torch.utils.data import Dataset

class SFTAudioQADataset(Dataset):
    """
    Dataset for pre-processed prompt-based audio QA data.

    Each item is a single pre-processed prompt line from the dataset file.
    """

    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to the dataset file containing preprocessed QA prompts.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            str: A single line from the preprocessed dataset file.
        """
        return self.data[idx].strip()  # Each line is a single QA formatted prompt