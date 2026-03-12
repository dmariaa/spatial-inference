SELECT *
FROM MAD_merged_aq_data
WHERE entry_date = "2024-03-10 06:00:00"
AND magnitude_id=7


SELECT DISTINCT(sensor_id)
FROM meteo_data maq
WHERE sensor_id >= 28079102