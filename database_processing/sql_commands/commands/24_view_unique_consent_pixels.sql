-- Select all unique (in name and domain combination) pixel declarations from consent_data
CREATE VIEW IF NOT EXISTS view_unique_consent_pixels AS
SELECT DISTINCT name, domain
FROM view_consent_data_images;

