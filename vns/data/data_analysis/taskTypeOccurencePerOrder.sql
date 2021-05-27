SELECT task_type,
       count(1)                                                                      as absolute,
       (count(1) / (select count(*) from operator.orders_with_deleted)::float) * 100 as relative
from operator.tasks_with_deleted
         join operator.orders_with_deleted on tasks_with_deleted.order_id = orders_with_deleted.id
group by task_type
order by absolute DESC;