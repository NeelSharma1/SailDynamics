"""Physical constants in SI; no Earth-fixed rotating frame is used."""
from datetime import datetime, timezone

AU = 149_597_870_700.0
EARTH_RADIUS = 6_371_000.0
SUN_RADIUS = 695_700_000.0
EARTH_MU = 3.986004418e14
LIGHT_SPEED = 299_792_458.0
SOLAR_CONSTANT = 1361.0
J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
