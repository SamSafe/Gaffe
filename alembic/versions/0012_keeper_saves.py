"""Retain goalkeeper penalty saves in per-fixture observations."""
import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("fact_player_match", sa.Column("penalties_saved", sa.SmallInteger(), nullable=True))


def downgrade():
    op.drop_column("fact_player_match", "penalties_saved")
