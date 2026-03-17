from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('post_analysis', '0010_metricpercentbasesetting'),
    ]

    operations = [
        migrations.AddField(
            model_name='metricoptionrecommendation',
            name='is_minimum',
            field=models.BooleanField(default=True),
        ),
    ]
