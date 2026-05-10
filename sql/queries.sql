-- =============================================================================
-- sql/queries.sql
-- Digital Ad Campaign Measurement & Attribution Framework
-- =============================================================================
-- All queries assume the ad_events table has been loaded into a SQL database.
-- Run:  python sql/load_to_sqlite.py  to create ad_events.db from parquet.
-- =============================================================================


-- ── 1. Overall Campaign KPIs ──────────────────────────────────────────────────
SELECT
    COUNT(*)                                            AS impressions,
    SUM(clicked)                                        AS clicks,
    SUM(converted)                                      AS conversions,
    ROUND(1.0 * SUM(clicked)    / COUNT(*),       4)   AS CTR,
    ROUND(1.0 * SUM(converted)  / NULLIF(SUM(clicked), 0), 4) AS CVR,
    ROUND(SUM(order_value_usd)  / NULLIF(SUM(ad_spend_usd), 0), 2) AS ROAS,
    ROUND(SUM(ad_spend_usd)     / NULLIF(SUM(converted), 0), 2)    AS CPA,
    ROUND(SUM(order_value_usd), 2)                      AS total_revenue,
    ROUND(SUM(ad_spend_usd),    2)                      AS total_spend
FROM ad_events;


-- ── 2. KPIs by Targeting Strategy ────────────────────────────────────────────
SELECT
    targeting_strategy,
    COUNT(*)                                                         AS impressions,
    SUM(clicked)                                                     AS clicks,
    SUM(converted)                                                   AS conversions,
    ROUND(1.0 * SUM(clicked) / COUNT(*), 4)                         AS CTR,
    ROUND(1.0 * SUM(converted) / NULLIF(SUM(clicked), 0), 4)        AS CVR,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2)   AS ROAS,
    ROUND(SUM(ad_spend_usd)    / NULLIF(SUM(converted), 0), 2)       AS CPA,
    ROUND(SUM(order_value_usd), 2)                                   AS total_revenue
FROM ad_events
GROUP BY targeting_strategy
ORDER BY ROAS DESC;


-- ── 3. A/B Test KPI Comparison ────────────────────────────────────────────────
SELECT
    ab_group,
    targeting_strategy,
    COUNT(*)                                                         AS impressions,
    SUM(clicked)                                                     AS clicks,
    SUM(converted)                                                   AS conversions,
    ROUND(1.0 * SUM(clicked)   / COUNT(*), 4)                       AS CTR,
    ROUND(1.0 * SUM(converted) / NULLIF(SUM(clicked), 0), 4)        AS CVR,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2)   AS ROAS,
    ROUND(SUM(order_value_usd), 2)                                   AS total_revenue
FROM ad_events
GROUP BY ab_group, targeting_strategy
ORDER BY ab_group, ROAS DESC;


-- ── 4. Daily Revenue & Spend Trend ───────────────────────────────────────────
SELECT
    DATE(timestamp)                               AS date,
    campaign_day,
    COUNT(*)                                      AS impressions,
    SUM(clicked)                                  AS clicks,
    SUM(converted)                                AS conversions,
    ROUND(SUM(order_value_usd), 2)                AS revenue,
    ROUND(SUM(ad_spend_usd), 2)                   AS spend,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2) AS ROAS
FROM ad_events
GROUP BY DATE(timestamp), campaign_day
ORDER BY campaign_day;


-- ── 5. CTR by Hour of Day ─────────────────────────────────────────────────────
SELECT
    hour_of_day,
    COUNT(*)                                      AS impressions,
    SUM(clicked)                                  AS clicks,
    ROUND(1.0 * SUM(clicked) / COUNT(*), 4)       AS CTR
FROM ad_events
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- ── 6. ROAS Heatmap: Ad Format × Device ──────────────────────────────────────
SELECT
    ad_format,
    device_type,
    COUNT(*)                                                         AS impressions,
    SUM(converted)                                                   AS conversions,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2)   AS ROAS
FROM ad_events
GROUP BY ad_format, device_type
ORDER BY ROAS DESC;


-- ── 7. Publisher Performance Ranking ─────────────────────────────────────────
SELECT
    publisher_id,
    COUNT(*)                                                          AS impressions,
    SUM(converted)                                                    AS conversions,
    ROUND(1.0 * SUM(converted) / COUNT(*), 4)                        AS CVR,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2)    AS ROAS,
    ROUND(SUM(order_value_usd), 2)                                    AS total_revenue
FROM ad_events
WHERE converted = 1
GROUP BY publisher_id
HAVING impressions >= 500
ORDER BY ROAS DESC
LIMIT 20;


-- ── 8. Attribution: Click-Through vs View-Through Conversions ─────────────────
SELECT
    targeting_strategy,
    SUM(click_through)                            AS click_through_conversions,
    SUM(view_through)                             AS view_through_conversions,
    ROUND(1.0 * SUM(click_through) / NULLIF(SUM(click_through) + SUM(view_through), 0), 4)
                                                  AS click_through_share
FROM ad_events
WHERE converted = 1
GROUP BY targeting_strategy;


-- ── 9. Audience Quality: Recency × Frequency Segments ────────────────────────
SELECT
    CASE
        WHEN recency_days <= 1  THEN '0-1d (hot)'
        WHEN recency_days <= 7  THEN '2-7d (warm)'
        WHEN recency_days <= 30 THEN '8-30d (cool)'
        ELSE '31d+ (cold)'
    END                                                              AS recency_bucket,
    CASE
        WHEN frequency <= 2  THEN 'low'
        WHEN frequency <= 7  THEN 'medium'
        ELSE 'high'
    END                                                              AS frequency_bucket,
    COUNT(*)                                                         AS impressions,
    ROUND(1.0 * SUM(clicked) / COUNT(*), 4)                         AS CTR,
    ROUND(1.0 * SUM(converted) / NULLIF(SUM(clicked), 0), 4)        AS CVR,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2)   AS ROAS
FROM ad_events
GROUP BY recency_bucket, frequency_bucket
ORDER BY ROAS DESC;


-- ── 10. Pre/Post Comparison (campaign day 45 split) ───────────────────────────
SELECT
    CASE WHEN campaign_day < 45 THEN 'pre' ELSE 'post' END           AS period,
    COUNT(*)                                                          AS impressions,
    ROUND(1.0 * SUM(clicked) / COUNT(*), 4)                          AS CTR,
    ROUND(1.0 * SUM(converted) / NULLIF(SUM(clicked), 0), 4)         AS CVR,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2)    AS ROAS,
    ROUND(SUM(order_value_usd), 2)                                    AS revenue
FROM ad_events
GROUP BY period;


-- ── 11. Vertical Performance ──────────────────────────────────────────────────
SELECT
    vertical,
    COUNT(*)                                                          AS impressions,
    SUM(converted)                                                    AS conversions,
    ROUND(1.0 * SUM(converted) / COUNT(*), 4)                        AS CVR_impression_level,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(converted), 0), 2)       AS avg_order_value,
    ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2)    AS ROAS
FROM ad_events
GROUP BY vertical
ORDER BY ROAS DESC;


-- ── 12. Budget Allocation: Current vs Recommended ────────────────────────────
-- Current spend distribution
WITH current_alloc AS (
    SELECT
        targeting_strategy,
        ROUND(SUM(ad_spend_usd), 2)                                  AS current_spend,
        ROUND(SUM(order_value_usd), 2)                               AS current_revenue,
        ROUND(SUM(order_value_usd) / NULLIF(SUM(ad_spend_usd), 0), 2) AS ROAS
    FROM ad_events
    GROUP BY targeting_strategy
),
total_spend AS (
    SELECT SUM(current_spend) AS total FROM current_alloc
)
SELECT
    c.targeting_strategy,
    c.current_spend,
    ROUND(100.0 * c.current_spend / t.total, 1)                      AS spend_pct,
    c.ROAS,
    -- ROAS-proportional reallocation
    ROUND(t.total * c.ROAS / (SELECT SUM(ROAS) FROM current_alloc), 2) AS recommended_spend
FROM current_alloc c, total_spend t
ORDER BY c.ROAS DESC;
