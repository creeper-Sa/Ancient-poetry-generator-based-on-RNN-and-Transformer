import re
import torch
from torch.utils.data import Dataset
from torch import device


class PoetryData(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        device: device,
        *,
        token_length: int = 48,
        poetry_file: str = "poem_transformer/archive/chinese_poems.txt",
        # poetry_file: str = "archive/chinese_poems.txt",
        max_lines: int = 12000,
    ) -> None:
        super().__init__()
        self.corpus: list[tuple[str, str]] = []
        self.token_length = token_length
        self.idx2word = ["<bos>", "<eos>", "<pad>"]
        self.word2idx = {v: k for k, v in enumerate(self.idx2word)}
        idx = len(self.idx2word)
        loaded_lines = 0
        self.device = device

        with open(poetry_file, "r", encoding="utf-8") as file:
            while loaded_lines < max_lines or max_lines == -1:
                line = file.readline().strip()
                if not line:
                    break
                loaded_lines += 1

                # 按逗号和句号分割，成句对
                sentences = re.split(r"[，,。\.]", line)
                sentences = [s for s in sentences if s]

                for i in range(0, len(sentences) - 1, 2):
                    up = sentences[i]
                    down = sentences[i + 1]
                    if 2 <= len(up) <= self.token_length and 2 <= len(down) <= self.token_length:
                        self.corpus.append((up, down))
                        for ch in up + down:
                            if ch not in self.word2idx:
                                self.word2idx[ch] = idx
                                self.idx2word.append(ch)
                                idx += 1

        self.vocab_size = len(self.word2idx)

    def word2token(self, words: str) -> torch.Tensor:
        t = [0]  # <bos>
        for x in words[: self.token_length - 2]:
            if x in self.word2idx:
                t.append(self.word2idx[x])
            else:
                t.append(self.word2idx["<pad>"])  # 或者跳过未识别字符
        t.append(1)  # <eos>
        # 补齐 <pad>
        t.extend(2 for _ in range(max(0, self.token_length - len(t))))
        return torch.LongTensor(t).to(self.device)

    def token2word(self, tokens: list[int]) -> str:
        words = []
        for idx in tokens:
            if idx in [0, 1, 2]:
                continue
            if idx < len(self.idx2word):
                words.append(self.idx2word[idx])
        return "".join(words)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        up, down = self.corpus[index]
        token = self.word2token(up)
        token_res = self.word2token(down)
        return (token, token_res)
