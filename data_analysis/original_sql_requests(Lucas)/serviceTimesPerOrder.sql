SELECT sub.task_type, sub.tour_position, sub.service_time
From (SELECT ord.comment,
             ord.tour_position,
             tsk.task_type,
             trs.created_at,
             ord.created_at,
             ord.done_at,
             tsk.created_at,
             tsk.done_at,
             vhc.info,
             vhc.fuel_type,
             vhc.name,
             vhc.license_number,
             ord.done_at - lag(ord.done_at, 2) OVER (ORDER BY ord.tour_id, ord.tour_position) as service_time
      from ((operator.orders_with_deleted ord join operator.tasks tsk on ord.id = tsk.order_id) join operator.vehicles_with_deleted vhc on ord.vehicle_id = vhc.id)
               join operator.tours trs on ord.tour_id = trs.id
      order by tour_id, tour_position
) sub
where sub.service_time > interval '0 hours' and sub.service_time < interval '8 hours';
