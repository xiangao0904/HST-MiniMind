import unittest

from model.hst_token_types import (
    TOKEN_TYPE_BRACKET,
    TOKEN_TYPE_CJK,
    TOKEN_TYPE_CODE_SYMBOL,
    TOKEN_TYPE_DIGIT,
    TOKEN_TYPE_LATIN,
    TOKEN_TYPE_NEWLINE,
    TOKEN_TYPE_SPECIAL,
    classify_token_text,
)


class TokenTypeTest(unittest.TestCase):
    def test_classifies_common_token_shapes(self):
        self.assertEqual(classify_token_text("123"), TOKEN_TYPE_DIGIT)
        self.assertEqual(classify_token_text("abc"), TOKEN_TYPE_LATIN)
        self.assertEqual(classify_token_text("北京"), TOKEN_TYPE_CJK)
        self.assertEqual(classify_token_text("\n"), TOKEN_TYPE_NEWLINE)
        self.assertEqual(classify_token_text("("), TOKEN_TYPE_BRACKET)
        self.assertEqual(classify_token_text("="), TOKEN_TYPE_CODE_SYMBOL)
        self.assertEqual(classify_token_text("<eos>"), TOKEN_TYPE_SPECIAL)


if __name__ == "__main__":
    unittest.main()
