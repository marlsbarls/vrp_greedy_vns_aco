select CASE WHEN to_char(date_trunc('hour', ord.created_at), 'HH24')::int > 17 or to_char(date_trunc('hour', ord.created_at), 'HH24')::int < 9 then 'out_of_shift' ELSE 'inside_shift' End as shift, (count(1)) as absolute, (count(1) / (select count(ord.created_at)  from operator.orders_with_deleted ord join operator.business_areas on ord.business_area_id = business_areas.id
where business_areas.name = 'Berlin' AND ord.created_at > timestamp '01/01/2020')::float)*100 as relative
from operator.orders_with_deleted ord join operator.business_areas on ord.business_area_id = business_areas.id
where business_areas.name = 'Berlin' AND ord.created_at > timestamp '01/01/2020'
GROUP BY shift;