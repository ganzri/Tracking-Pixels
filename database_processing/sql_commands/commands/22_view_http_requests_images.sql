--Create a view containin only http requests for images
CREATE VIEW IF NOT EXISTS view_http_requests_images AS 
SELECT * FROM http_requests where resource_type == 'image';
