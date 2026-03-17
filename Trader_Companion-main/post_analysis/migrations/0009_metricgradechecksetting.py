from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('post_analysis', '0008_metricoptionrecommendation'),
    ]

    operations = [
        migrations.CreateModel(
            name='MetricGradeCheckSetting',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('required_metrics', models.TextField(default='')),
                ('exclude_metric', models.CharField(blank=True, default='', max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['-updated_at'],
            },
        ),
    ]
