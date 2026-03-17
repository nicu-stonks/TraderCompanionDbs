from django.db import migrations


def remove_unique_constraint(apps, schema_editor):
    # Constraint removed via schema operation below; this function placeholder kept for clarity.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('post_analysis', '0004_unique_single_analysis_per_trade'),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name='posttradeanalysis',
            name='uniq_post_analysis_trade',
        ),
    ]
