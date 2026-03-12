-- ============================================================
-- 03_traffic_source_analysis.sql
-- Traffic Source Deep-Dive
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- Traffic source volume vs conversion efficiency
-- ────────────────────────────────────────────────────────────
SELECT
    TrafficType,
    COUNT(*) AS total_sessions,
    SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) AS conversions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct,
    ROUND(AVG(PageValues), 2) AS avg_page_value,
    ROUND(AVG(ProductRelated_Duration / 60.0), 2) AS avg_product_time_min,
    ROUND(AVG(BounceRates), 4) AS avg_bounce_rate,
    -- Volume * conversion rate = efficiency score
    ROUND(COUNT(*) * (SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END)::FLOAT / COUNT(*)), 0) AS efficiency_score
FROM online_shoppers
GROUP BY TrafficType
ORDER BY efficiency_score DESC;

-- ────────────────────────────────────────────────────────────
-- Traffic source — returning vs new breakdown
-- ────────────────────────────────────────────────────────────
SELECT
    TrafficType,
    VisitorType,
    COUNT(*) AS sessions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM online_shoppers
WHERE VisitorType IN ('Returning_Visitor', 'New_Visitor')
GROUP BY TrafficType, VisitorType
HAVING COUNT(*) >= 50
ORDER BY TrafficType, VisitorType;

-- ────────────────────────────────────────────────────────────
-- Underperforming traffic sources (high volume, low conversion)
-- ────────────────────────────────────────────────────────────
WITH traffic_stats AS (
    SELECT
        TrafficType,
        COUNT(*) AS sessions,
        ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
    FROM online_shoppers
    GROUP BY TrafficType
),
averages AS (
    SELECT
        AVG(sessions) AS avg_sessions,
        AVG(conversion_rate_pct) AS avg_cvr
    FROM traffic_stats
)
SELECT
    t.TrafficType,
    t.sessions,
    t.conversion_rate_pct,
    CASE
        WHEN t.sessions > a.avg_sessions AND t.conversion_rate_pct < a.avg_cvr
        THEN 'HIGH VOLUME / LOW CVR — Priority for optimization'
        WHEN t.sessions < a.avg_sessions AND t.conversion_rate_pct > a.avg_cvr
        THEN 'LOW VOLUME / HIGH CVR — Opportunity to scale'
        ELSE 'Normal'
    END AS segment_label
FROM traffic_stats t, averages a
ORDER BY t.sessions DESC;
