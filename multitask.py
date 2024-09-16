import torch
import torch.nn as nn

class SimpleTokenizer:
    def __init__(self, sentences):
        self.build_vocab(sentences)

    def build_vocab(self, sentences):
        unique_tokens = set()
        for sentence in sentences:
            tokens = sentence.lower().split()
            unique_tokens.update(tokens)

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.special_tokens = [self.pad_token, self.unk_token]
        self.word2idx = {}
        self.idx2word = {}
        idx = 0

        for token in self.special_tokens:
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            idx += 1

        for token in sorted(unique_tokens):
            if token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

        self.vocab_size = idx

    def tokenize(self, sentence):
        return [
            self.word2idx.get(word.lower(), self.word2idx[self.unk_token])
            for word in sentence.split()
        ]

    def encode(self, sentences, max_len):
        tokenized_sentences = [self.tokenize(sentence) for sentence in sentences]
        padded_sentences = []
        for tokens in tokenized_sentences:
            if len(tokens) < max_len:
                tokens = tokens + [self.word2idx[self.pad_token]] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
            padded_sentences.append(tokens)
        return torch.tensor(padded_sentences, dtype=torch.long)

# Multi-Task Sentence Transformer Model
class MultiTaskSentenceTransformer(nn.Module):
    def __init__(
        self, vocab_size, embed_size=128, num_heads=4, hidden_dim=256,
        max_len=10, pad_idx=0, num_classes_task_a=5, num_classes_task_b=5
    ):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.embed_size = embed_size
        self.pad_idx = pad_idx

        # Shared Backbone
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_len, embed_size)
        self.multihead_attn = nn.MultiheadAttention(embed_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_size)
        )
        self.layer_norm2 = nn.LayerNorm(embed_size)

        # Task A: Sentence Classification Head
        self.fc_classification = nn.Linear(embed_size, num_classes_task_a)
        self.softmax_classification = nn.Softmax(dim=-1)

        # Task B: NER Head
        self.fc_ner = nn.Linear(embed_size, num_classes_task_b)
        self.softmax_ner = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len = x.size()

        token_embeddings = self.embedding(x)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        positional_embeddings = self.position_embeddings(positions)

        embedded = token_embeddings + positional_embeddings

        embedded = embedded.permute(1, 0, 2)

        key_padding_mask = (x == self.pad_idx)

        attn_output, attn_weights = self.multihead_attn(
            embedded, embedded, embedded, key_padding_mask=key_padding_mask
        )

        attn_output = self.layer_norm1(attn_output + embedded)

        attn_output = attn_output.permute(1, 0, 2)

        mask = (x != self.pad_idx).unsqueeze(-1).type(torch.float)
        attn_output_masked = attn_output * mask
        sum_embeddings = torch.sum(attn_output_masked, dim=1)
        valid_token_counts = mask.sum(dim=1)
        sentence_embedding = sum_embeddings / valid_token_counts

        sentence_embedding = self.fc(sentence_embedding)
        sentence_embedding = self.layer_norm2(sentence_embedding)

        # Task A: Sentence Classification
        logits_task_a = self.fc_classification(sentence_embedding)
        probabilities_task_a = self.softmax_classification(logits_task_a)

        # Task B: NER (Token-Level Classification)
        logits_task_b = self.fc_ner(attn_output)
        probabilities_task_b = self.softmax_ner(logits_task_b)

        return probabilities_task_a, probabilities_task_b

# Example usage
if __name__ == "__main__":
    sentences = [
        "Here are the best-dressed celebrities on the 2024 Emmy Awards red carpet",
        "The office market is in trouble. Lower interest rates might not be enough to save it",
        "Epic Thanksgiving Leftovers Sandwich. Cooking with Brontez",
        "Tom works at Microsoft in Seattle ."
    ]

    tokenizer = SimpleTokenizer(sentences)

    max_len = 10

    encoded_sentences = tokenizer.encode(sentences, max_len=max_len)
    print("Encoded sentences:\n", encoded_sentences)

    pad_idx = tokenizer.word2idx['<pad>']

    vocab_size = tokenizer.vocab_size
    num_classes_task_a = 3  #  number of classes for Task A (Sentence Classification: sports, entertaiment, and others)
    num_classes_task_b = 5  #  number of NER entity classes: O(Outside), PER(Person), ORG(Organization), LOC(Location), MISC(Miscellaneous)

    model = MultiTaskSentenceTransformer(
        vocab_size=vocab_size,
        embed_size=128,
        num_heads=4,
        hidden_dim=256,
        max_len=max_len,
        pad_idx=pad_idx,
        num_classes_task_a=num_classes_task_a,
        num_classes_task_b=num_classes_task_b
    )

    # Forward pass to get predictions for both tasks
    with torch.no_grad():
        probabilities_task_a, probabilities_task_b = model(encoded_sentences)

    print("\nTask A (Sentence Classification) probabilities:\n", probabilities_task_a)
    print("\nTask B (NER) probabilities (token-level):\n", probabilities_task_b)
