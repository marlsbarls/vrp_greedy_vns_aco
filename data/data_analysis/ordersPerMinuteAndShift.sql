select to_char(date_trunc('minute', ord.created_at), 'MI')::int as minute, (count(1)) as absolute, (count(1) / (select count(orders_with_deleted.created_at)  from operator.orders_with_deleted)::float)*100 as relative, CASE WHEN to_char(date_trunc('hour', ord.created_at), 'HH24')::int > 17 or to_char(date_trunc('hour', ord.created_at), 'HH24')::int < 9 then 'out_of_shift' ELSE 'inside_shift' End as shift
from operator.orders_with_deleted ord join operator.business_areas on ord.business_area_id = business_areas.id
where business_areas.name = 'Berlin'
GROUP BY shift, minute
ORDER BY shift, minute

