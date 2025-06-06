import os
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from openai import OpenAI, APIError
from main import init_client, ask_chat, main


class TestMainFunctions(unittest.TestCase):

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_init_client_success(self):
        """
        וידוא שהפונקציה init_client מחזירה מופע OpenAI כאשר קיים מפתח API תקין
        """
        client = init_client()
        self.assertIsInstance(client, OpenAI)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_init_client_no_api_key(self):
        """
        וידוא שהפונקציה init_client מעלה RuntimeError כאשר המפתח ריק או חסר
        """
        with self.assertRaises(RuntimeError):
            init_client()

    def test_ask_chat_success(self):
        """
        וידוא ש-ask_chat מחזירה את תוכן ההודעה (content) ומספר הטוקנים (total_tokens),
        כאשר ה־mock מייצר אובייקט עם שדות .choices ו-.usage.
        """
        # יוצרים אובייקט המדמה תשובת OpenAI
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Test response"))],
            usage=SimpleNamespace(total_tokens=5),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        response, token_count = ask_chat(mock_client, "Test prompt")
        self.assertEqual(response, "Test response")
        self.assertEqual(token_count, 5)

    def test_ask_chat_api_error(self):
        """
        וידוא ש-ask_chat מעלה APIError כאשר הלקוח מגדיר side_effect מסוג APIError.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIError(
            "Server error", request=MagicMock(), body={}
        )
        with self.assertRaises(APIError):
            ask_chat(mock_client, "Test prompt")

    @patch("builtins.input", side_effect=["Test Book"])
    @patch("main.ask_chat")
    def test_main_success(self, mock_ask_chat, mock_input):
        """
        וידוא שב-main, אם ChatGPT 'יודע' את הספר (yes) ולא קיימת בעיית זכויות יוצרים (yes),
        הוא ישאל 'כמה פרקים יש?' (10), ואז ישיג כותרת עבור כל אחד מהפרקים (tuple),
        ובסוף ידפיס "Found 10 chapters.".

        הפונקציה fake_ask_chat תחזיר:
          - שלושת הקריאות הראשונות: מחרוזות "yes", "yes", "10"
          - כל קריאה נוספת: tuple ("Chapter N", 5), כאשר N עולה אוטומטית.
        כך לא יתרחש StopIteration גם אם main() תבצע יותר קריאות ממה שעשינו כאן.
        """
        # מונה קריאות בתוך fake_ask_chat
        calls = {"i": 0}

        def fake_ask_chat(client, prompt):
            idx = calls["i"]
            calls["i"] += 1

            # הקריאות הראשונות: מחרוזות
            if idx == 0:
                return "yes"  # תשובה לשאלה "Do you know the book?"
            if idx == 1:
                return "yes"  # תשובה לשאלה "Is there a copyright issue?"
            if idx == 2:
                return "10"  # תשובה לשאלה "How many chapters?"

            # כל קריאה נוספת: מחזירים כותרת פרק כ־tuple
            chapter_number = idx - 2
            return (f"Chapter {chapter_number}", 5)

        mock_ask_chat.side_effect = fake_ask_chat

        with patch("builtins.print") as mock_print:
            main()
            # מאמתים שחלה הדפסה של "Found 10 chapters."
            mock_print.assert_any_call("Found 10 chapters.")

    @patch("builtins.input", side_effect=["Unknown Book"])
    @patch("main.ask_chat")
    def test_main_book_not_known(self, mock_ask_chat, mock_input):
        """
        וידוא שב-main, אם ChatGPT 'לא יודע' את הספר (no), הוא מיד ידפיס:
        ChatGPT states it does not know the book Unknown Book. Exiting. וייצא.

        כאן mock_ask_chat מחזיר מחרוזת 'no' לבדה,
        כי main עושה: knows = ask_chat(...); if knows.lower().startswith("no"): ...
        """
        mock_ask_chat.return_value = "no"
        with patch("builtins.print") as mock_print:
            main()
            # כאן חשוב לשים לב לגרשיים המעוקלות (curly quotes) כפי שה-main מדפיס
            mock_print.assert_called_with(
                "ChatGPT states it does not know the book Unknown Book. Exiting."
            )


if __name__ == "__main__":
    unittest.main()
