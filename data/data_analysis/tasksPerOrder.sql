Select sub.tasks_per_order, count(1)
from (SELECT ord.id                                                                             as id,
             (SELECT COUNT(*) from operator.tasks_with_deleted tsk where tsk.order_id = ord.id) as tasks_per_order
      FROM operator.tasks_with_deleted tsk
               INNER JOIN operator.orders_with_deleted ord ON tsk.order_id = ord.id
               INNER JOIN operator.business_areas_with_deleted ba ON ba.id = business_area_id
      WHERE ba.name = 'Berlin'
        AND ord.status = 'done'
        AND ord.service_type != 'bike_mobile_charge'
      GROUP BY ord.id
      LIMIT 1000
    ) sub
Group by sub.tasks_per_order;