import torch
import torch.nn.functional as F

from poem_transformer.dataset import PoetryData
from poem_transformer.model import PoetryNet


class PoetryGen:
    def __init__(self):
        self.lr = 0.0001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = PoetryData(self.device, max_lines=50000, token_length=12)
        self.vocab_size = self.dataset.vocab_size

        self.net = PoetryNet(self.vocab_size, self.device, embed_size=512).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 256)

        self.loaded_checkpoint_file = None
        self.load_checkpoint('poem_transformer/model/checkpoint-250707-0922.pth')
        # self.load_checkpoint('model/checkpoint-250707-0922.pth')

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]
        self.loaded_checkpoint_file = path
        self.scheduler.last_epoch = self.epoch

    def nucleus_sampling(self, logits, top_p=0.9):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        cutoff = cumulative_probs > top_p
        cutoff_idx = cutoff.nonzero()[0, 0] + 1
        filtered_logits = sorted_logits[:, :cutoff_idx]
        filtered_indices = sorted_indices[:, :cutoff_idx]
        final_probs = F.softmax(filtered_logits, dim=-1)

        next_word_idx = torch.multinomial(final_probs, 1)
        return filtered_indices[0, next_word_idx].item()

    def generate_one(self, pre_sentence: str, start_words: str = "", temperature=1.0, top_k=10, top_p=None):
        self.net.eval()
        start_words_token = [0]  # <bos>
        for ch in start_words:
            start_words_token.append(self.dataset.word2idx.get(ch, self.dataset.word2idx["<pad>"]))

        src = self.dataset.word2token(pre_sentence).unsqueeze(0)
        tgt = torch.LongTensor([start_words_token]).to(self.device)
        memo = self.net.encode(src)
        res = []
        max_gen_len = 20

        for _ in range(max_gen_len):
            out = self.net.decode(tgt, memo)
            logits = out[:, -1, :] / temperature

            if top_p:
                next_id = self.nucleus_sampling(logits, top_p)
            else:
                topk_logits, topk_indices = torch.topk(logits, top_k)
                probs = F.softmax(topk_logits, dim=-1)
                next_word_idx = torch.multinomial(probs, 1)
                next_id = topk_indices[0, next_word_idx].item()

            if next_id in (1, 2):
                break
            res.append(next_id)
            tgt = torch.cat((tgt, torch.tensor([[next_id]], device=self.device)), dim=1)

        result = self.dataset.token2word(res)
        return start_words + result

    def generate_by_start(self, start_words: str, pre_style: str) -> str:
        res = []
        start_words_list = start_words.split("/")
        if not start_words_list:
            return ""
        for i, s in enumerate(start_words_list):
            pre = pre_style if i == 0 else res[-1]
            generated = self.generate_one(pre, s)
            res.append(generated)
        return "/".join(res)

def gen_transformer_poem(input_char, input_style):
    model = PoetryGen()
    if input_style == "七言绝句":
        poem = model.generate_by_start(input_char, "绿蔓如藤不用栽")
    elif input_style == "五言绝句":
        poem = model.generate_by_start(input_char,"床前明月光")
    else:
        poem = ""

    return poem

if __name__ == "__main__":
    print(gen_transformer_poem("春", "七言绝句"))
    print(len(gen_transformer_poem("春", "七言绝句")))
