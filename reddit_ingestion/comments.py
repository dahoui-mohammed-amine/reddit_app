from __future__ import annotations

from .config import Config
from .models import Gap, PostSnapshot


def comment_count_gap(post: PostSnapshot) -> Gap | None:
    if post.num_comments is None or post.num_comments <= len(post.comments):
        return None
    return Gap(
        "comment",
        "unexpanded",
        entity_id=post.post_id,
        subreddit=post.subreddit,
        detail=f"provider exposed {post.num_comments} comments but returned {len(post.comments)}",
    )


def apply_comment_policy(posts: list[PostSnapshot], config: Config) -> list[Gap]:
    gaps: list[Gap] = []
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
        else:
            post.comments = original
        count_gap = comment_count_gap(post)
        if count_gap and not any(gap.entity_id == post.post_id and gap.reason == "unexpanded" for gap in gaps):
            gaps.append(count_gap)
    return gaps
