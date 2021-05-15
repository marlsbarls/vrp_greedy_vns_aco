Select date,
       time,
       CASE WHEN cancel_date <= '2020-07-07' AND cancel_time <= '17:00:00' THEN cancel_time ELSE null END,
       order_id,
       task_id,
       task_type,
       longitude,
       latitude
from (Select orders.created_at::date as date,
             orders.created_at::time as time,
             CASE
                 WHEN orders.status = 'canceled' THEN orders.updated_at::date
                 ELSE null END       as cancel_date,
             CASE
                 WHEN orders.status = 'canceled' THEN orders.updated_at::time
                 ELSE null END       as cancel_time,
             tasks.order_id          as order_id,
             tasks.id                as task_id,
             tasks.task_type         as task_type,
             locs.longitude          as longitude,
             locs.latitude           as latitude
      from (operator.tasks tasks join operator.orders on tasks.order_id = orders.id join operator.business_areas ba on orders.business_area_id = ba.id)
               join operator.locations locs on orders.location_id = locs.id
      where orders.created_at >= '2020-07-06 16:00:00'
        and orders.created_at < '2020-07-07 16:00:00'
        and ba.name = 'Berlin'
        and tasks.task_type not in ('relocation', 'shuttle', 'out_of_business_area', 'general_inspection',
                                    'general_inspection_back_relocation')) sub
where sub.cancel_date IS NULL
   or (sub.cancel_date >= '2020-07-07' and sub.cancel_time >= '9:00:00')
ORDER BY date, time;
