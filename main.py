# Program to summarize book chapters using calls to ChatGPT

import os  # Access to environment variables and files
import time  # Sleep time between retries
import re  # Text searching using Regex
from openai import OpenAI, RateLimitError, APIError  # OpenAI client and API exceptions


# --------------------------------------------------
# 1. Initialize the OpenAI client
# --------------------------------------------------
def init_client() -> OpenAI:
    """
    Initializes an OpenAI instance with a key from the user
    Checks that the OPENAI_API_KEY environment variable is set, otherwise raises an error
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # If the API key is not set, stop and raise an error
        raise RuntimeError("The environment variable OPENAI_API_KEY must be set")
    # Return a client instance with the key
    return OpenAI(api_key=api_key)


# --------------------------------------------------
# 2. Call to Chat Completion with retry on rate limits and server errors
# --------------------------------------------------
def ask_chat(
    client: OpenAI, prompt: str, system: str = "You are a helpful assistant."
) -> tuple[str, int]:
    """
    Sends a request to ChatGPT and handles retries
    - RateLimitError (429): retry with exponential backoff
    - APIError (5xx): retry if server error
    otherwise: propagate the error
    """
    backoff = 1  # Initial time in seconds to retry the request
    while True:
        try:
            # Call the completion creation function
            resp = client.chat.completions.create(
                model="gpt-4o",  # Model selection
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,  # Level of diversity in responses
            )
            token_count = (
                resp.usage.total_tokens
            )  # Get the token count from the response
            # Return the content of the first message and token count
            return resp.choices[0].message.content.strip(), token_count

        except RateLimitError:
            # If we hit the rate limit, wait and try again
            print(f"Rate limit hit; retrying in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

        except APIError as e:
            # If it's a server error 5xx, retry the request
            status = getattr(e, "http_status", None)
            if status and 500 <= status < 600:
                print(f"Server error {status}; retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                # For any other error, propagate the error
                raise


# --------------------------------------------------
# 3. Main logic
# --------------------------------------------------
def main():
    # 1. Initialize the client
    client = init_client()

    total_tokens = 0  # Initialize total token count

    # 2. Get the book title from the user
    book_title = input(f"Enter the book title: ").strip()

    # 3. Check if ChatGPT knows the book
    knows = ask_chat(
        client,
        f"Do you know the book titled “{book_title}”? Please answer simply yes or no.",
    )
    if knows[0].lower().startswith("no"):
        # If ChatGPT does not know the book, exit the run
        print(f"ChatGPT states it does not know the book “{book_title}”. Exiting.")
        return

    total_tokens += knows[1]

    # NEW: Question about translating the book
    can_translate = ask_chat(
        client,
        f"Can you summarize the book “{book_title}”? Please answer with 'yes' or 'no'.",
    )
    if "no" in can_translate[0].lower():
        print(f"There is a copyright issue with the book “{book_title}”. Exiting.")
        return

    total_tokens += can_translate[1]

    # 4. Request the number of chapters (emphasized: only a number, no additional text)
    reply = ask_chat(
        client,
        f"How many chapters are in the book “{book_title}”? Please respond **only** with a whole number (e.g., 24), without additional text.",
    )
    # Extract the first number from the response
    m = re.search(r"\d+", reply[0])
    if not m:
        # If no number is found – raise an error
        raise ValueError(f"No chapter count found in the response: {reply!r}")
    chapter_count = int(m.group())
    print(f"{f"Found "}{chapter_count}{f" chapters."}")

    # 5. Summarize each chapter
    summaries = (
        []
    )  # List of tuples: (chapter number, summary text, chapter title, token count)
    titles = []  # List of chapter titles

    for i in range(1, chapter_count + 1):
        print(f"{f"Summarizing chapter "}{i}/{chapter_count}…")

        # Request the chapter title
        summary_prompt = f"Summarize in Hebrew chapter number {i} from the book “{book_title}” in one line."
        chapter_title, title_tokens = ask_chat(
            client, summary_prompt
        )  # Get a one-line summary
        total_tokens += title_tokens  # Add title tokens to total

        # Save the chapter title
        titles.append(chapter_title)

        prompt = (
            f"Summarize in Hebrew, as detailed and comprehensive as possible, chapter number {i} from the book “{book_title}”. "
            "Include specific examples from the text, analysis of key points, and additional elaborations to clarify every concept or idea until the final summary."
        )
        chapter_text, chapter_tokens = ask_chat(
            client, prompt, system="You are an expert Hebrew-language summarizer."
        )
        total_tokens += chapter_tokens  # Add chapter tokens to total
        summaries.append(
            (
                i,
                chapter_title,
                chapter_text,
                chapter_tokens,
            )  # Save the number, title, summary, and token count
        )

    # 6. Write to a text file
    # Replace problematic characters in the filename
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in book_title)
    out_path = f"{safe_name}_summaries.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        # General title and creation date
        f.write(f"Summaries of “{book_title}”\n")
        f.write(f"Created on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        # Write the summary for each chapter
        for i, title, text, chapter_tokens in summaries:
            f.write(f"=== Chapter {i}: {title} (Tokens: {chapter_tokens}) ===\n")
            f.write(text + "\n\n")

    # 7. Message on completion
    print(f"{f"The summaries have been written to the file: '"}{out_path}'")
    print(f"Total tokens used: {total_tokens}")  # Print total tokens used


if __name__ == "__main__":
    main()
