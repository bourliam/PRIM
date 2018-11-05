Create table cars_log(
  id text,
  latitude text,
  longitude text,
  heading text,
  speed text,
  ts text
);

COPY cars_log(id, latitude, longitude, heading, speed, ts) FROM 'dummylog.csv' CSV HEADER;

ALTER TABLE cars_log ADD COLUMN geom geometry;

UPDATE cars_log SET geom = ST_SetSRID(ST_MakePoint(longitude::numeric,latitude::numeric) ,4326);

ALTER TABLE planet_osm_line ADD COLUMN wayTF geometry;

UPDATE planet_osm_line SET wayTF = ST_SetSRID(ST_Transform(way,4326),4326);

CREATE INDEX cars_log_index ON cars_log USING GIST(geom);
CREATE INDEX planet_osm_line_indextf ON planet_osm_line USING GIST(waytf);

create or replace function closestSegment(geometry, numeric)
RETURNS geometry AS $$
DECLARE rec geometry;
BEGIN
  select into rec waytf from planet_osm_line
  where ST_Expand($1, $2) && waytf
  order by ST_Distance($1,waytf)
  limit 1;
  return rec;
END;
$$ IMMUTABLE LANGUAGE plpgsql;

ALTER TABLE cars_log add COLUMN geomOnSeg geometry;

UPDATE cars_log SET geomOnSeg = closestSegment(geom, 0.0005);

ALTER TABLE cars_log add COLUMN geomLink geometry;

UPDATE cars_log SET geomLink = ST_MakeLine(ST_ClosestPoint(geomOnSeg, geom), geom);
