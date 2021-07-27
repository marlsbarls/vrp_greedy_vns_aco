select DATE(ord.created_at), count(1)
from operator.orders_with_deleted ord join operator.business_areas on ord.business_area_id = business_areas.id
where business_areas.name = 'Berlin'
GROUP BY (1)
ORDER BY (1)