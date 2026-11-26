"""Visit the live SoundMatch app and wake it if Streamlit put it to sleep.

Run by .github/workflows/keep-awake.yml every 8 hours. Exits 0 when the app is
awake (or was woken), 1 if it couldn't be confirmed, so failures show in Actions.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

APP_URL = os.environ.get("APP_URL", "https://gbadrain-machine-learning.streamlit.app/")
WAKE_TEXT = "Yes, get this app back up"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None)
        page = browser.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(8)

        button = page.get_by_role("button", name=WAKE_TEXT)
        if button.count() > 0:
            print("App was asleep. Clicking the wake-up button…")
            button.first.click()
            time.sleep(10)
        else:
            print("No wake-up button. App looks awake.")

        # Streamlit Cloud shows the app inside an iframe; look for our page text in any frame.
        deadline = time.time() + 240
        while time.time() < deadline:
            for frame in page.frames:
                try:
                    if frame.get_by_text("How it works").count() > 0 or frame.get_by_text("SoundMatch").count() > 0:
                        print("App is up and showing SoundMatch.")
                        browser.close()
                        return 0
                except PWTimeout:
                    pass
            time.sleep(10)

        print("Could not confirm the app loaded within 4 minutes.")
        page.screenshot(path="keep_awake_failure.png", full_page=True)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
