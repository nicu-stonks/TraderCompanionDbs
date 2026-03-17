from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('ticker_data', '0002_remove_providersettings_fetch_outside_hours_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='historicalprice',
            name='ingested_at',
            field=models.DateTimeField(db_index=True, default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='historicalprice5m',
            name='ingested_at',
            field=models.DateTimeField(db_index=True, default=django.utils.timezone.now),
        ),
    ]
