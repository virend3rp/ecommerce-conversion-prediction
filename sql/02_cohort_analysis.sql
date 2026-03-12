-- ============================================================
-- 02_cohort_analysis.sql
-- Cohort & Segmentation Analysis
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- High-intent session profile (top 20% PageValues)
-- ────────────────────────────────────────────────────────────
WITH page_value_percentiles AS (
    SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY PageValues) AS p80
    FROM online_shoppers
    WHERE PageValues > 0
)
SELECT
    CASE WHEN o.PageValues >= p.p80 THEN 'High Intent' ELSE 'Low Intent' END AS intent_segment,
    COUNT(*) AS sessions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct,
    ROUND(AVG(ProductRelated), 2) AS avg_product_pages,
    ROUND(AVG(BounceRates), 4) AS avg_bounce_rate
FROM online_shoppers o, page_value_percentiles p
GROUP BY intent_segment;

-- ────────────────────────────────────────────────────────────
-- Device / browser cohort conversion
-- ────────────────────────────────────────────────────────────
SELECT
    Browser,
    OperatingSystems,
    COUNT(*) AS sessions,
    SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) AS conversions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM online_shoppers
GROUP BY Browser, OperatingSystems
HAVING COUNT(*) > 100
ORDER BY conversion_rate_pct DESC;

-- ────────────────────────────────────────────────────────────
-- Returning vs New visitor monthly cohort
-- ────────────────────────────────────────────────────────────
SELECT
    Month,
    VisitorType,
    COUNT(*) AS sessions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM online_shoppers
WHERE VisitorType IN ('Returning_Visitor', 'New_Visitor')
GROUP BY Month, VisitorType
ORDER BY
    CASE Month
        WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3
        WHEN 'Apr' THEN 4 WHEN 'May' THEN 5 WHEN 'Jun' THEN 6
        WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8 WHEN 'Sep' THEN 9
        WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12
    END,
    VisitorType;

-- ────────────────────────────────────────────────────────────
-- High-intent non-converters (discount targeting candidates)
-- These are sessions with high PageValues but Revenue = FALSE
-- ────────────────────────────────────────────────────────────
WITH thresholds AS (
    SELECT
        PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY PageValues) AS pv_p80,
        PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY BounceRates) AS br_p20
    FROM online_shoppers
    WHERE PageValues > 0
)
SELECT
    COUNT(*) AS high_intent_non_converters,
    ROUND(AVG(PageValues), 2) AS avg_page_value,
    ROUND(AVG(ProductRelated_Duration / 60.0), 2) AS avg_product_time_min,
    ROUND(AVG(BounceRates), 4) AS avg_bounce_rate,
    -- Estimated revenue left on table (assuming avg order value of $85)
    COUNT(*) * 85 * 0.30 AS estimated_revenue_opportunity
FROM online_shoppers o, thresholds t
WHERE Revenue = FALSE
  AND PageValues >= t.pv_p80
  AND BounceRates <= t.br_p20;
