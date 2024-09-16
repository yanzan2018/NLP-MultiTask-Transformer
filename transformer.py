import torch
import torch.nn as nn

class SimpleTokenizer:
    def __init__(self, sentences):
        self.build_vocab(sentences)

    def build_vocab(self, sentences):
        # Collect unique tokens from training corpus
        unique_tokens = set()
        for sentence in sentences:
            tokens = sentence.lower().split()
            unique_tokens.update(tokens)

        # Initialize special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.special_tokens = [self.pad_token, self.unk_token]
        self.word2idx = {}
        self.idx2word = {}
        idx = 0

        # Add special tokens to vocabulary
        for token in self.special_tokens:
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            idx += 1

        # Add unique tokens from the sentences to the vocabulary
        for token in sorted(unique_tokens):  # Sorting for index consistency
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
                # Pad sequences shorter than max_len
                tokens = tokens + [self.word2idx[self.pad_token]] * (max_len - len(tokens))
            else:
                # Truncate sequences longer than max_len
                tokens = tokens[:max_len]
            padded_sentences.append(tokens)
        return torch.tensor(padded_sentences, dtype=torch.long)

# Sentence Transformer Model(BERT-like toy model)
class SentenceTransformerWithAttention(nn.Module):
    def __init__(
        self, vocab_size, embed_size=128, num_heads=4, hidden_dim=256, max_len=10, pad_idx=0
    ):
        super(SentenceTransformerWithAttention, self).__init__()
        self.embed_size = embed_size
        self.pad_idx = pad_idx
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

    def forward(self, x):
        batch_size, seq_len = x.size()

        token_embeddings = self.embedding(x)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        positional_embeddings = self.position_embeddings(positions)

        # Shape: [batch_size, seq_len, embed_size]
        embedded = token_embeddings + positional_embeddings

        # now the sequence length is the first dimension, batch size is the second
        # prepare for multi-head attention layer
        embedded = embedded.permute(1, 0, 2)

        key_padding_mask = (x == self.pad_idx)

        attn_output, attn_weights = self.multihead_attn(
            embedded, embedded, embedded, key_padding_mask=key_padding_mask
        )

        # Residual Connection and Layer Normalization
        attn_output = self.layer_norm1(attn_output + embedded)

        # Permute back to [batch_size, seq_len, embed_size]
        attn_output = attn_output.permute(1, 0, 2)

        mask = (x != self.pad_idx).unsqueeze(-1).type(torch.float)
        attn_output = attn_output * mask
        sum_embeddings = torch.sum(attn_output, dim=1)
        valid_token_counts = mask.sum(dim=1)
        sentence_embedding = sum_embeddings / valid_token_counts

        sentence_embedding = self.fc(sentence_embedding)

        sentence_embedding = self.layer_norm2(sentence_embedding)

        return sentence_embedding

if __name__ == "__main__":
    # training corpus
    sentences = [
        "Here are the best-dressed celebrities on the 2024 Emmy Awards red carpet",
        "The office market is in trouble. Lower interest rates might not be enough to save it",
        "Epic Thanksgiving Leftovers Sandwich. Cooking with Brontez"
    ]

    tokenizer = SimpleTokenizer(sentences)

    max_len = 10

    # Encode the sentences (tokenize and pad/truncate them)
    encoded_sentences = tokenizer.encode(sentences, max_len=max_len)
    print("Encoded sentences:\n", encoded_sentences)

    pad_idx = tokenizer.word2idx['<pad>']

    vocab_size = tokenizer.vocab_size
    model = SentenceTransformerWithAttention(
        vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len
    )

    # Inference
    with torch.no_grad():
        embeddings = model(encoded_sentences)

    print("\nSentence embeddings:\n", embeddings)
