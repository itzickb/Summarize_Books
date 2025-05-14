#!/usr/bin/env python
# main.py
# תוכנית לסיכום פרקי ספר באמצעות קריאות ל-ChatGPT

import os  # גישה למשתני סביבה וקבצים
import time  # זמן שינה בין ניסיונות חוזרים
import re  # חיפוש טקסט באמצעות Regex
from openai import OpenAI, RateLimitError, APIError  # לקוח OpenAI וחריגות API


def reverse_text(s: str) -> str:
    """
    הופך את סדר התווים במחרוזת.
    מתאים להצגה בטרמינלים שלא תומכים ב־RTL.
    """
    return s[::-1]


# --------------------------------------------------
# 1. אתחול הלקוח של OpenAI
# --------------------------------------------------
def init_client() -> OpenAI:
    """
    מאתחל את מופע ה-OpenAI עם מפתח מהמשתמש
    בודק שהמשתנה OPENAI_API_KEY מוגדר, אחרת קורא לשגיאה
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # אם לא הוגדר מפתח API, מפסיקים ורוקעים שגיאה
        raise RuntimeError(reverse_text("יש להגדיר את משתנה הסביבה OPENAI_API_KEY"))
    # מחזירים מופע לקוח עם המפתח
    return OpenAI(api_key=api_key)


# --------------------------------------------------
# 2. קריאה ל־Chat Completion עם retry על קצב ושגיאות שרת
# --------------------------------------------------
def ask_chat(
    client: OpenAI, prompt: str, system: str = "You are a helpful assistant."
) -> str:
    """
    שולח בקשה ל-ChatGPT ומטפל בניסיונות חוזרים
    - RateLimitError (429): retry עם exponential backoff
    - APIError (5xx): retry אם שגיאת שרת
    sonst: מעביר הלאה
    """
    backoff = 1  # זמן התחלה בשניות לחזרה על בקשה
    while True:
        try:
            # קריאה לפונקציית יצירת completion
            resp = client.chat.completions.create(
                model="gpt-4o-mini",  # בחירת מודל
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,  # רמת גיוון בתשובות
            )
            # מחזירים את תוכן ההודעה הראשונה
            return resp.choices[0].message.content.strip()

        except RateLimitError:
            # במידה והגענו למגבלת קצב, נחכה וננסה שוב
            print(f"Rate limit hit; retrying in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

        except APIError as e:
            # אם מדובר בשגיאת שרת 5xx, נחזור על הבקשה
            status = getattr(e, "http_status", None)
            if status and 500 <= status < 600:
                print(f"Server error {status}; retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                # לכל שגיאה אחרת, תעבור הלאה לשגיאה חיצונית
                raise


# --------------------------------------------------
# 3. הלוגיקה הראשית
# --------------------------------------------------
def main():
    # 1. אתחול הלקוח
    client = init_client()

    # 2. קבלת שם הספר מהמשתמש
    book_title = input(f"{reverse_text("הזן את שם הספר")} :").strip()

    # 3. בדיקת קיום הספר ב-ChatGPT
    knows = ask_chat(
        client, f"אתה מכיר את הספר שנקרא “{book_title}”? השב פשוט כן או לא."
    )
    if knows.lower().startswith("לא"):
        # אם ChatGPT לא מכיר את הספר, נסיים את הריצה
        print(reverse_text(f"ChatGPT מציין שאינו מכיר את “{book_title}”. יוצא."))
        return

    # 4. בקשה למספר הפרקים (מודגש: רק מספר, בלי טקסט נוסף)
    reply = ask_chat(
        client,
        f"כמה פרקים יש בספר “{book_title}”? אנא השב **רק** במספר שלם (למשל: 24), בלי טקסט נוסף.",
    )
    # מוצאים את המספר הראשון בתשובה
    m = re.search(r"\d+", reply)
    if not m:
        # אם לא נמצא מספר – זורקים שגיאה
        raise ValueError(reverse_text(f"לא נמצאה כמות פרקים בתשובה: {reply!r}"))
    chapter_count = int(m.group())
    print(reverse_text(f"נמצאו {chapter_count} פרקים."))

    # 5. סיכום כל פרק
    summaries = []  # רשימה של tuples: (מספר פרק, טקסט הסיכום)
    for i in range(1, chapter_count + 1):
        print(f"{reverse_text(f"מסכם פרק ")}{i}\{chapter_count}…")
        prompt = (
            f"תסכם בעברית את פרק מספר {i} מהספר “{book_title}” "
            "באופן הכי מפורט שניתן."
        )
        # מבצעים קריאה חוזרת ל-ask_chat עם system מתאים
        chapter_text = ask_chat(
            client, prompt, system="You are an expert Hebrew-language summarizer."
        )
        summaries.append((i, chapter_text))

    # 6. כתיבה לקובץ טקסט
    # מחליפים תווים בעייתיים בשם הקובץ
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in book_title)
    out_path = f"{safe_name}_summaries.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        # כותרת כללית ותאריך יצירה
        f.write(f"סיכומים של “{book_title}”\n")
        f.write(f"נוצר ב־{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        # כותבים סיכום לכל פרק
        for i, text in summaries:
            f.write(f"=== פרק {i} ===\n")
            f.write(text + "\n\n")

    # 7. הודעה על סיום
    print(f"{reverse_text(f"הסיכומים נכתבו לקובץ: '")}{out_path}'")


if __name__ == "__main__":
    main()
