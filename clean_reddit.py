"""Conservative, auditable cleaning of timestamped Reddit text."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import html
import re
import unicodedata


SUBREDDITS = {
    "memes", "dankmemes", "adviceanimals", "me_irl",
    "memeeconomy", "copypasta", "dogelore",
    "shitposting", "okbuddyretard", "2meirl4meirl", "wholesomememes",
    "comedyheaven", "starterpacks", "surrealmemes", "bonehurtingjuice",
}
SENTINELS = {"", "[deleted]", "[removed]"}
URL = re.compile(r"https?://\S+|www\.[^\s.][^\s]*", re.IGNORECASE)
MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
IMAGE_URL = re.compile(r"https?://[^\s)]+\.(?:png|jpe?g|gif|webp)(?:\?[^\s)]*)?", re.IGNORECASE)
IMAGE_PLACEHOLDER = re.compile(r"(?i)(?:!gif|!img|\[?image:\s*(?:img|gif)\]?)")
MEDIA_SOURCE = re.compile(r"(?i)(?:i\.redd\.it|i\.imgur\.com|reddit\.com/gallery|v\.redd\.it|\.(?:png|jpe?g|gif|webp)(?:\?|$))")
BOT_FOOTER = re.compile(
    r"i am a bot,? and this action was performed automatically|"
    r"please contact the moderators of this subreddit if you have any questions",
    re.IGNORECASE,
)
SPACES = re.compile(r"[^\S\n]+")
CANONICAL = re.compile(r"\W+", re.UNICODE)


def normalize_text(value: str) -> str:
    """Fix encoding artifacts without changing ordinary spelling or punctuation."""
    # Older Reddit rows sometimes encode the ampersand of a numeric entity.
    value = unicodedata.normalize("NFC", html.unescape(html.unescape(value)))
    # Retain descriptive image alt text as a caption, but never the image URL.
    value = MARKDOWN_IMAGE.sub(
        lambda match: match.group(1) if match.group(1).casefold() not in {"", "img", "image", "gif"} else "",
        value,
    )
    value = HTML_IMAGE.sub("", value)
    value = IMAGE_URL.sub("", value)
    value = IMAGE_PLACEHOLDER.sub("", value)
    value = "".join(
        char for char in value
        if char in "\n\t" or (unicodedata.category(char) not in {"Cc", "Cf"})
        or char == "\u200d"  # Keep the joiner used in compound emoji.
    )
    lines = [SPACES.sub(" ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def text_from_row(row: dict, kind: str) -> str:
    """Build a document while omitting absent or removed selftext."""
    if kind == "comments":
        return normalize_text(row.get("body") or "")
    title = normalize_text(row.get("title") or "")
    body = normalize_text(row.get("selftext") or "")
    if body.casefold() in SENTINELS:
        body = ""
    return f"{title}\n{body}" if title and body else title or body


class Cleaner:
    """Apply bounded monthly sampling while recording every rejection reason."""

    def __init__(self, year: int, month: int, kind: str):
        self.year, self.month, self.kind = year, month, kind
        self.counts = Counter()
        self.by_subreddit = Counter()
        self.by_subreddit_tokens = Counter()
        self.author_counts = Counter()
        self.thread_counts = Counter()
        self.exact_counts = Counter()
        self.near_counts = Counter()
        self.near_buckets = defaultdict(list)
        self.seen_ids = set()
        self.examples = defaultdict(list)

    def _reject(self, reason: str, row: dict, text: str = "") -> None:
        self.counts[reason] += 1
        if len(self.examples[reason]) < 3:
            self.examples[reason].append({
                "id": row.get("id"), "subreddit": row.get("subreddit"),
                "text": text[:180],
            })

    def _near_key(self, text: str) -> str | None:
        """Group only long, nearly unchanged copies within a month."""
        if len(text) < 200:
            return None
        canonical = CANONICAL.sub(" ", text.casefold()).strip()
        if len(canonical) < 160:
            return None
        # Similar-length texts with a common long prefix are cheap candidates.
        bucket = (len(canonical) // 32, canonical[:24])
        for neighbor in (bucket[0] - 1, bucket[0], bucket[0] + 1):
            for prior, key in self.near_buckets[(neighbor, bucket[1])][:100]:
                if abs(len(prior) - len(canonical)) <= max(3, len(canonical) // 50):
                    if SequenceMatcher(None, prior, canonical, autojunk=False).ratio() >= 0.985:
                        return key
        key = hashlib.blake2b(canonical.encode(), digest_size=12).hexdigest()
        self.near_buckets[bucket].append((canonical, key))
        return key

    def clean(self, row: dict) -> dict | None:
        self.counts["raw"] += 1
        subreddit = (row.get("subreddit") or "").casefold()
        if subreddit not in SUBREDDITS:
            self._reject("other_subreddit", row)
            return None
        self.counts["target_subreddit"] += 1

        try:
            created = int(row["created_utc"])
            date = datetime.fromtimestamp(created, timezone.utc)
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            self._reject("bad_timestamp", row)
            return None
        if (date.year, date.month) != (self.year, self.month):
            self._reject("outside_month", row)
            return None

        item_id = str(row.get("id") or "")
        if not item_id or item_id in self.seen_ids:
            self._reject("missing_or_duplicate_id", row)
            return None
        self.seen_ids.add(item_id)
        if self.kind == "submissions" and MEDIA_SOURCE.search(row.get("url") or ""):
            self.counts["media_source_submissions"] += 1
        raw_text = row.get("body") if self.kind == "comments" else (
            (row.get("title") or "") + "\n" + (row.get("selftext") or "")
        )
        if any(pattern.search(raw_text or "") for pattern in (
            MARKDOWN_IMAGE, HTML_IMAGE, IMAGE_URL, IMAGE_PLACEHOLDER,
        )):
            self.counts["rows_with_image_reference"] += 1
        text = text_from_row(row, self.kind)
        if text.casefold() in SENTINELS:
            self._reject("empty_or_removed", row, text)
            return None
        if (row.get("author") or "").casefold() == "automoderator" or BOT_FOOTER.search(text):
            self._reject("bot_or_moderation", row, text)
            return None
        if URL.search(text):
            without_urls = URL.sub("", text).strip(" \n\t[]()<>-:;,.!")
            if not without_urls or (len(text) >= 30 and len(without_urls) / len(text) < 0.2):
                self._reject("url_dominated", row, text)
                return None
        if len(text) > 12000:
            self._reject("pathological_length", row, text)
            return None

        author = (row.get("author") or "").casefold()
        author_key = (subreddit, author)
        if author and author != "[deleted]" and self.author_counts[author_key] >= 100:
            self._reject("author_cap", row, text)
            return None
        thread_id = row.get("link_id") if self.kind == "comments" else f"t3_{item_id}"
        if self.kind == "comments" and thread_id and self.thread_counts[thread_id] >= 100:
            self._reject("thread_cap", row, text)
            return None

        exact_key = (subreddit, hashlib.blake2b(text.encode(), digest_size=16).digest())
        if self.exact_counts[exact_key] >= 5:
            self._reject("exact_copy_cap", row, text)
            return None
        near_key = self._near_key(text)
        if near_key and self.near_counts[(subreddit, near_key)] >= 5:
            self._reject("near_copy_cap", row, text)
            return None

        self.exact_counts[exact_key] += 1
        if near_key:
            self.near_counts[(subreddit, near_key)] += 1
        if author and author != "[deleted]":
            self.author_counts[author_key] += 1
        if self.kind == "comments" and thread_id:
            self.thread_counts[thread_id] += 1
        self.counts["kept"] += 1
        self.by_subreddit[subreddit] += 1
        return {
            "id": item_id,
            "kind": self.kind,
            "subreddit": subreddit,
            "created_utc": created,
            "text": text,
            "score": row.get("score"),
            "thread_id": thread_id,
            "parent_id": row.get("parent_id") if self.kind == "comments" else None,
        }

    def report(self) -> dict:
        """Return a compact audit; examples stay remote with the clean data."""
        return {
            "year": self.year, "month": self.month, "kind": self.kind,
            "counts": dict(self.counts),
            "by_subreddit": dict(self.by_subreddit),
            "examples": dict(self.examples),
        }
