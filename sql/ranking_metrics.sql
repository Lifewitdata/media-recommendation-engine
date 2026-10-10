-- media-recommendation-engine : ranking metrics
-- Engine: MySQL / MariaDB 10.2+ (window functions used)
-- Tables: users(user_id, age_band, region, device),
--         content(content_id, genre, tags, language, duration_min),
--         interactions(user_id, content_id, event, event_ts)
--         event in ('impression','play','completion','like')

-- Q1: popularity baseline - top content per device segment by completion rate
-- Used as the cold-start / fallback ranker when no user history exists.
SELECT
    device,
    content_id,
    genre,
    plays,
    completions,
    ROUND(completion_rate, 4) AS completion_rate,
    RANK() OVER (PARTITION BY device ORDER BY completion_rate DESC) AS seg_rank
FROM (
    SELECT
        u.device,
        c.content_id,
        c.genre,
        SUM(i.event = 'play') AS plays,
        SUM(i.event = 'completion') AS completions,
        SUM(i.event = 'completion') / NULLIF(SUM(i.event = 'play'), 0) AS completion_rate
    FROM interactions AS i
    JOIN users   AS u ON i.user_id = u.user_id
    JOIN content AS c ON i.content_id = c.content_id
    GROUP BY u.device, c.content_id, c.genre
    HAVING plays >= 50
) AS t
ORDER BY device, seg_rank;

-- Q2: user genre affinity - behavioural features for the ranker
-- One row per user x genre: the training signal for personalization.
SELECT
    u.user_id,
    c.genre,
    SUM(i.event = 'play') AS plays,
    SUM(i.event = 'completion') AS completions,
    SUM(i.event = 'like') AS likes,
    ROUND(SUM(i.event = 'completion') / NULLIF(SUM(i.event = 'play'), 0), 4) AS completion_rate,
    ROUND(SUM(i.event = 'like') / NULLIF(SUM(i.event = 'completion'), 0), 4) AS like_per_completion
FROM interactions AS i
JOIN users   AS u ON i.user_id = u.user_id
JOIN content AS c ON i.content_id = c.content_id
GROUP BY u.user_id, c.genre
HAVING plays >= 5
ORDER BY u.user_id, plays DESC;

-- Q3: exploration candidates - high-impression, low-completion content per genre
-- Surfaces catalog items the popularity ranker buries: candidates for interleaving tests.
SELECT
    c.content_id,
    c.genre,
    c.language,
    SUM(i.event = 'impression') AS impressions,
    SUM(i.event = 'completion') AS completions,
    ROUND(SUM(i.event = 'completion') / NULLIF(SUM(i.event = 'impression'), 0), 4) AS impression_to_completion
FROM content AS c
LEFT JOIN interactions AS i ON i.content_id = c.content_id
GROUP BY c.content_id, c.genre, c.language
HAVING impressions >= 200 AND impression_to_completion < 0.05
ORDER BY impressions DESC
LIMIT 100;
