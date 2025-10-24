DROP VIEW IF EXISTS features_train;
CREATE TEMP VIEW features_train AS
WITH base AS (
  SELECT *,
    ABS(sleep_hours - 8.0) AS sleep_deficit,
    CASE WHEN chronotype='morning' THEN MAX(0, 10 - focus_start_hour)
         WHEN chronotype='evening' THEN MAX(0, focus_start_hour - 13)
         ELSE 0 END AS circadian_alignment,
    (stress_level * (1.0 - (ABS(stress_level - 3) / 2.0))) AS yerkes_arousal,
    (breaks_count * avg_break_minutes) AS break_quality,
    (meetings_minutes + late_meetings_minutes * 1.5) AS meeting_load,
    (notifications + context_switches) AS context_penalty,
    (steps/10000.0) + (CASE WHEN caffeine_mg BETWEEN 50 AND 200 THEN 0.2 ELSE 0.0 END) AS health_score
  FROM events_train
)
SELECT
  user_id, date, sleep_hours, chronotype, focus_start_hour, deep_work_minutes, meetings_minutes,
  late_meetings_minutes, breaks_count, avg_break_minutes, context_switches, notifications,
  steps, stress_level, mood, caffeine_mg, hydration_glasses,
  sleep_deficit, circadian_alignment, yerkes_arousal, break_quality, meeting_load,
  context_penalty, health_score, productivity_score
FROM base;

DROP VIEW IF EXISTS features_candidates;
CREATE TEMP VIEW features_candidates AS
WITH base AS (
  SELECT *,
    ABS(sleep_hours - 8.0) AS sleep_deficit,
    CASE WHEN chronotype='morning' THEN MAX(0, 10 - focus_start_hour)
         WHEN chronotype='evening' THEN MAX(0, focus_start_hour - 13)
         ELSE 0 END AS circadian_alignment,
    (stress_level * (1.0 - (ABS(stress_level - 3) / 2.0))) AS yerkes_arousal,
    (breaks_count * avg_break_minutes) AS break_quality,
    (meetings_minutes + late_meetings_minutes * 1.5) AS meeting_load,
    (notifications + context_switches) AS context_penalty,
    (steps/10000.0) + (CASE WHEN caffeine_mg BETWEEN 50 AND 200 THEN 0.2 ELSE 0.0 END) AS health_score
  FROM events_candidates
)
SELECT
  user_id, date, sleep_hours, chronotype, focus_start_hour, deep_work_minutes, meetings_minutes,
  late_meetings_minutes, breaks_count, avg_break_minutes, context_switches, notifications,
  steps, stress_level, mood, caffeine_mg, hydration_glasses,
  sleep_deficit, circadian_alignment, yerkes_arousal, break_quality, meeting_load,
  context_penalty, health_score
FROM base;
