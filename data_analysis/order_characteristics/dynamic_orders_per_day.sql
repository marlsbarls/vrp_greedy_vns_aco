select DATE(ord.created_at), count(1)
from operator.orders_with_deleted ord join operator.business_areas on ord.business_area_id = business_areas.id
where business_areas.name = 'Berlin' and cast(ord.created_at as time) > '09:00:00' or cast(ord.created_at as time) < '17:00:00'
GROUP BY (1)
ORDER BY (1)