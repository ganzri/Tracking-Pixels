-- select only the first http request for an image resource
-- all subsequent requests have the same request_id and visit_id but a later timestamp 
CREATE VIEW IF NOT EXISTS view_first_request_images AS
SELECT *, min(time_stamp) FROM http_requests where resource_type == 'image' group by visit_id, request_id;
