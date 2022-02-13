--Create a view with consent_data for pixels only
CREATE VIEW IF NOT EXISTS view_consent_data_images AS 
SELECT * FROM consent_data WHERE type_name LIKE 'P%';
