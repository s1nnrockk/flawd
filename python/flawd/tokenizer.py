"""Lossless byte-level baseline tokenizer; BPE can replace it later."""
class ByteTokenizer:
    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258
    VOCAB_SIZE = 259

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.BOS_ID)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        data = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < self.VOCAB_SIZE:
                raise ValueError(f"invalid token id: {token_id!r}")
            if token_id <= 255:
                data.append(token_id)
            elif not skip_special_tokens:
                raise ValueError("special tokens cannot be decoded as UTF-8 bytes")
        return data.decode("utf-8", errors="replace")
