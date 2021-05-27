select to_char(date_trunc('month', ord.created_at), 'YYYY-MM') as month, count(1)
from operator.orders_with_deleted ord join operator.business_areas on ord.business_area_id = business_areas.id
where business_areas.name = 'Berlin'
GROUP BY month
ORDER BY month;