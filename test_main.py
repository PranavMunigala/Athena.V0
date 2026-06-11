import io
import unittest
from contextlib import redirect_stdout

import main


class MainTests(unittest.TestCase):
    def test_main_prints_expected_message(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            main.main()
        self.assertEqual(output.getvalue(), "what about now\n")


if __name__ == "__main__":
    unittest.main()
