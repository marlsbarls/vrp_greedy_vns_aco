Select task_type,
       count(1)                                                                     as absolute,
       (count(1) / (select count(*) from operator.tasks_with_deleted)::float) * 100 as relative
from operator.tasks_with_deleted
group by task_type
order by absolute DESC;

