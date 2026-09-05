from __future__ import annotations

from .config import Config
from .models import Gap, PostSnapshot


def apply_comment_policy(posts: list[PostSnapshot], config: Config) -> list[Gap]:
    gaps: list[Gap] = []
    if config.comments_mode == "off":
        for post in posts:
            post.comments = []
        return gaps
    for post in posts:
        original = post.comments
        if config.comments_mode == "bounded":
            kept = [comment for comment in original if comment.depth is None or comment.depth <= config.comment_depth]
            if len(kept) > config.comment_limit:
                kept = kept[: config.comment_limit]
                gaps.append(Gap("comment", "bounded_limit", entity_id=post.post_id, subreddit=post.subreddit, detail=f"limit={config.comment_limit}"))
            if len(kept) < len(original):
                gaps.append(Gap("comment", "bounded_depth", entity_id=post.post_id, subreddit=post.subreddit, detail=f"depth={config.comment_depth}"))
            post.comments = kept
        elif config.comments_mode == "full":
            post.comments = original
    return gaps
