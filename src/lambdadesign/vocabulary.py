"""The checkpoint's SELFIES token vocabulary."""

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class SelfiesVocabulary:
    def __init__(self, token_to_id):
        if not isinstance(token_to_id, dict) or not token_to_id:
            raise ValueError("Checkpoint has no SELFIES vocabulary")
        if any(not isinstance(k, str) or type(v) is not int for k,v in token_to_id.items()):
            raise ValueError("Vocabulary must map strings to integer token IDs")
        if set(token_to_id.values()) != set(range(len(token_to_id))):
            raise ValueError("Vocabulary IDs must be unique and contiguous from zero")
        if any(t not in token_to_id for t in SPECIAL_TOKENS):
            raise ValueError("Vocabulary is missing a required special token")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {v:k for k,v in token_to_id.items()}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = [token_to_id[t] for t in SPECIAL_TOKENS]

    def __len__(self):
        return len(self.token_to_id)

    def decode(self, ids):
        pieces = []
        for index in ids:
            if index == self.eos_id:
                break
            token = self.id_to_token.get(int(index))
            if token is None:
                raise ValueError(f"Decoder emitted unknown token ID {index}")
            if token not in SPECIAL_TOKENS:
                pieces.append(token)
        return "".join(pieces)
