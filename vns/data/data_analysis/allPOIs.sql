SELECT pois.id,
       pois.poi_type,
       loc.latitude,
       loc.longitude
from poi.pois pois
         join operator.business_areas_with_deleted ba on pois.business_area_id = ba.id
         join operator.locations loc on pois.location_id = loc.id
where ba.name = 'Berlin'